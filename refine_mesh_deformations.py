"""
refine_mesh_deformations.py — Fix vertex-swap artifacts in ActionMesh deformations.

Some vertices in the ActionMesh output get misassigned to the wrong limb in later
frames, causing edges to become anomalously long and erroneous bridging faces to
appear.

Strategy: sequential (causal) optimisation.

Each frame t is initialised from the PREVIOUS refined frame (t-1) rather than
from the original (potentially jumped) positions. A jumped vertex therefore starts
on the correct side and only needs the edge-length loss to follow natural motion.
A small temporal anchor (--temporal_weight) prevents the optimizer from straying
far from the previous frame, but is weak enough to allow genuine corgi motion.

Loss per frame:
  edge_loss     = mean( (current_edge_lengths - rest_edge_lengths)^2 )
  temporal_loss = mean( (pos - prev_refined)^2 )
  total = edge_weight * edge_loss + temporal_weight * temporal_loss

Output: <input_mesh>/deformations_vertices_refined.npy  (T, V, 3)

Usage:
  python refine_mesh_deformations.py \\
    --input_mesh output/2026.03.03/actionmesh_gs_replace_corgi \\
    --n_steps 300
"""

import os
import sys
import argparse
import numpy as np
import torch
import trimesh


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mesh", required=True,
                        help="ActionMesh output directory containing "
                             "deformations_vertices.npy and deformations_faces.npy.")
    parser.add_argument("--n_steps", type=int, default=300,
                        help="Optimiser steps per frame (default: 300).")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Adam learning rate (default: 5e-4).")
    parser.add_argument("--edge_weight", type=float, default=1.0,
                        help="Weight for edge length preservation loss (default: 1.0).")
    parser.add_argument("--temporal_weight", type=float, default=1.0,
                        help="Weight anchoring each frame to the previous refined frame "
                             "(default: 1.0). Prevents large jumps while allowing natural "
                             "motion.")
    return parser.parse_args()


def get_edges(faces_np):
    """Extract unique undirected edges from face array. Returns (E, 2) int64."""
    all_edges = np.concatenate([
        faces_np[:, [0, 1]],
        faces_np[:, [1, 2]],
        faces_np[:, [0, 2]],
    ], axis=0)
    all_edges = np.sort(all_edges, axis=1)
    return np.unique(all_edges, axis=0).astype(np.int64)


def print_edge_stretch_stats(deformations, edges):
    """Diagnostic: report edge stretch stats across all frames."""
    T, V, _ = deformations.shape
    ei, ej = edges[:, 0], edges[:, 1]
    pos_0        = deformations[0]
    rest_lengths = np.maximum(
        np.linalg.norm(pos_0[ei] - pos_0[ej], axis=-1), 1e-8
    )
    max_stretch = np.ones(edges.shape[0], dtype=np.float32)
    for t in range(T):
        pos_t   = deformations[t]
        lengths = np.linalg.norm(pos_t[ei] - pos_t[ej], axis=-1)
        np.maximum(max_stretch, lengths / rest_lengths, out=max_stretch)
    print(f"  Edge stretch (original) — max: {max_stretch.max():.1f}x, "
          f"median: {np.median(max_stretch):.2f}x, "
          f"95th pct: {np.percentile(max_stretch, 95):.2f}x, "
          f"edges >3x: {int((max_stretch > 3).sum())}")


def main():
    args = parse_args()

    deformations_path = os.path.join(args.input_mesh, "deformations_vertices.npy")
    faces_path        = os.path.join(args.input_mesh, "deformations_faces.npy")
    obj_path          = os.path.join(args.input_mesh, "texture", "mesh_00.obj")
    output_path       = os.path.join(args.input_mesh, "deformations_vertices_refined.npy")

    print(f"\nLoading deformations from {deformations_path}")
    deformations = np.load(deformations_path)   # (T, V, 3)
    T, V, _      = deformations.shape
    print(f"  {T} frames, {V} vertices")

    # Load face topology — prefer deformations_faces.npy, fall back to mesh_00.obj
    if os.path.exists(faces_path):
        faces = np.load(faces_path)   # (F, 3) indices into V
        print(f"  Faces loaded from {faces_path}: {faces.shape[0]}")
    else:
        print(f"  deformations_faces.npy not found, loading topology from {obj_path}")
        tm = trimesh.load(obj_path, process=False, force="mesh")
        mesh_verts = np.array(tm.vertices, dtype=np.float32)  # (V_obj, 3)
        mesh_faces = np.array(tm.faces,    dtype=np.int64)    # (F, 3)

        # NN mapping: obj verts → deformation verts (frame 0 is in saved coord system)
        # Deformation verts are saved as [-z, x, y] — undo to match obj space
        def0 = deformations[0].copy()
        def0_xyz = np.zeros_like(def0)
        def0_xyz[:, 0] = def0[:, 1]
        def0_xyz[:, 1] = def0[:, 2]
        def0_xyz[:, 2] = -def0[:, 0]

        device_cpu = "cuda" if torch.cuda.is_available() else "cpu"
        v_obj  = torch.from_numpy(mesh_verts).float().to(device_cpu)
        v_def  = torch.from_numpy(def0_xyz).float().to(device_cpu)
        dists  = torch.cdist(v_obj.unsqueeze(0), v_def.unsqueeze(0)).squeeze(0)
        _, nn  = torch.min(dists, dim=1)
        nn_np  = nn.cpu().numpy()  # (V_obj,) maps obj vert → deformation vert

        avg_d = dists.min(dim=1).values.mean().item()
        print(f"  NN mapping avg distance: {avg_d:.6f}")
        if avg_d > 1e-3:
            print("  WARNING: High avg distance — face indices may be inaccurate.")

        faces = nn_np[mesh_faces]   # (F, 3) indices into deformation verts
        print(f"  Faces remapped: {faces.shape[0]}")

    print(f"  Total: {T} frames, {V} verts, {faces.shape[0]} faces")

    print("\nBuilding edges...")
    edges = get_edges(faces)
    E = edges.shape[0]
    print(f"  {E} unique edges")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    print("\nEdge stretch diagnostics (original deformations)...")
    print_edge_stretch_stats(deformations, edges)

    edges_t  = torch.from_numpy(edges).long().to(device)
    ei, ej   = edges_t[:, 0], edges_t[:, 1]

    # Rest-pose edge lengths (frame 0)
    pos_0        = torch.from_numpy(deformations[0]).float().to(device)
    rest_lengths = torch.norm(pos_0[ei] - pos_0[ej], dim=1)  # (E,)

    # Sequential optimisation: each frame initialised from previous refined frame
    print(f"\nRefining {T-1} frames sequentially  "
          f"({args.n_steps} steps, lr={args.lr}, "
          f"edge_weight={args.edge_weight}, temporal_weight={args.temporal_weight})")
    refined    = np.zeros_like(deformations)
    refined[0] = deformations[0]   # frame 0 is the rest pose — keep unchanged

    for t in range(1, T):
        # Initialise from previous refined frame (not the original jumped positions)
        pos_prev = torch.from_numpy(refined[t-1]).float().to(device)
        pos      = pos_prev.clone().requires_grad_(True)
        opt      = torch.optim.Adam([pos], lr=args.lr)

        for _ in range(args.n_steps):
            opt.zero_grad()
            lengths       = torch.norm(pos[ei] - pos[ej], dim=1)
            edge_loss     = ((lengths - rest_lengths) ** 2).mean()
            temporal_loss = ((pos - pos_prev) ** 2).mean()
            loss          = args.edge_weight * edge_loss + args.temporal_weight * temporal_loss
            loss.backward()
            opt.step()

        refined[t] = pos.detach().cpu().numpy()

        if t % 10 == 0 or t == T - 1:
            print(f"  Frame {t:3d}/{T-1}  "
                  f"edge={edge_loss.item():.2e}  temporal={temporal_loss.item():.2e}")

    np.save(output_path, refined.astype(np.float32))
    print(f"\nSaved refined deformations: {output_path}")
    print(f"  Shape: {refined.shape}")


if __name__ == "__main__":
    main()
