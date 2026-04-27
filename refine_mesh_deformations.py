"""
refine_mesh_deformations.py — Fix vertex-swap artifacts in ActionMesh deformations.

Some vertices in the ActionMesh output get misassigned to the wrong limb in later
frames, causing edges to become anomalously long and erroneous bridging faces to
appear.

Strategy: complementary per-vertex anchoring using frame-to-frame stretch.

The key metric is the maximum FRAME-TO-FRAME edge stretch ratio
max_t( length(t) / length(t-1) ). This is invariant to global rigid motion
(a whole limb swinging does not change its internal edge lengths), so it only
flags vertices that suddenly jump relative to their mesh neighbours.

Each vertex gets a weight w = 1 / max_ft_stretch^alpha:
  - Stable / smoothly-deforming vertex (w ≈ 1): anchored to deformations[t]
  - Suddenly-jumped vertex              (w ≈ 0): anchored to refined[t-1]

Loss per frame (no edge loss — it fights legitimate deformation):
  data_loss     = mean( w     * (pos - deformations[t])^2 )
  temporal_loss = mean( (1-w) * (pos - refined[t-1])^2 )
  total         = data_weight * data_loss + temporal_weight * temporal_loss

Output: <input_mesh>/deformations_vertices_refined.npy  (T, V, 3)

Usage:
  python refine_mesh_deformations.py \\
    --input_mesh output/2026.03.03/actionmesh_gs_replace_corgi \\
    --n_steps 300
"""

import os
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
    parser.add_argument("--data_weight", type=float, default=50.0,
                        help="Weight for data anchor (stable vertices → deformations[t]). "
                             "(default: 50.0).")
    parser.add_argument("--temporal_weight", type=float, default=50.0,
                        help="Weight for temporal anchor (jumped vertices → refined[t-1]). "
                             "(default: 50.0).")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Exponent for weight: w = 1/stretch^alpha. "
                             "Higher values more sharply redirect jumped vertices. "
                             "(default: 2.0).")
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


def compute_vertex_weights(deformations, edges, alpha):
    """
    Per-vertex weight w = 1 / max_frame_to_frame_stretch^alpha.

    Uses max_t(length(t)/length(t-1)) — invariant to global rigid motion so
    only flags vertices that jump suddenly relative to their mesh neighbours.
    Also prints rest-pose stretch stats for comparison.
    """
    T, V, _ = deformations.shape
    E = edges.shape[0]
    ei, ej = edges[:, 0], edges[:, 1]

    # --- Rest-pose stretch (diagnostic only) ---
    pos_0        = deformations[0]
    rest_lengths = np.maximum(np.linalg.norm(pos_0[ei] - pos_0[ej], axis=-1), 1e-8)
    max_rest_stretch = np.ones(E, dtype=np.float32)
    for t in range(T):
        pos_t   = deformations[t]
        lengths = np.linalg.norm(pos_t[ei] - pos_t[ej], axis=-1)
        np.maximum(max_rest_stretch, lengths / rest_lengths, out=max_rest_stretch)
    print(f"  Rest-pose edge stretch    — max: {max_rest_stretch.max():.1f}x, "
          f"median: {np.median(max_rest_stretch):.2f}x, "
          f"95th pct: {np.percentile(max_rest_stretch, 95):.2f}x, "
          f"edges >3x: {int((max_rest_stretch > 3).sum())}")

    # --- Frame-to-frame stretch (used for weights) ---
    max_ft_stretch = np.ones(E, dtype=np.float32)
    prev_lengths   = rest_lengths.copy()  # start from frame 0 lengths
    for t in range(1, T):
        pos_t   = deformations[t]
        lengths = np.maximum(np.linalg.norm(pos_t[ei] - pos_t[ej], axis=-1), 1e-8)
        stretch   = lengths / prev_lengths
        stretch_r = prev_lengths / lengths  # catch compression→stretch transitions
        np.maximum(max_ft_stretch, stretch,   out=max_ft_stretch)
        np.maximum(max_ft_stretch, stretch_r, out=max_ft_stretch)
        prev_lengths = lengths
    print(f"  Frame-to-frame edge stretch — max: {max_ft_stretch.max():.1f}x, "
          f"median: {np.median(max_ft_stretch):.2f}x, "
          f"95th pct: {np.percentile(max_ft_stretch, 95):.2f}x, "
          f"edges >3x: {int((max_ft_stretch > 3).sum())}")

    # Per-vertex: max stretch over all incident edges
    max_ft_stretch_per_vertex = np.ones(V, dtype=np.float32)
    np.maximum.at(max_ft_stretch_per_vertex, ei, max_ft_stretch)
    np.maximum.at(max_ft_stretch_per_vertex, ej, max_ft_stretch)

    weights = (1.0 / np.maximum(max_ft_stretch_per_vertex, 1.0) ** alpha).astype(np.float32)
    print(f"  Vertex weights (ft-based) — min: {weights.min():.4f}, "
          f"median: {np.median(weights):.4f}, "
          f"vertices below 0.1: {int((weights < 0.1).sum())} / {V}")
    return weights


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
        faces = np.load(faces_path)
        print(f"  Faces loaded from {faces_path}: {faces.shape[0]}")
    else:
        print(f"  deformations_faces.npy not found, loading topology from {obj_path}")
        tm = trimesh.load(obj_path, process=False, force="mesh")
        mesh_verts = np.array(tm.vertices, dtype=np.float32)
        mesh_faces = np.array(tm.faces,    dtype=np.int64)

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
        nn_np  = nn.cpu().numpy()

        avg_d = dists.min(dim=1).values.mean().item()
        print(f"  NN mapping avg distance: {avg_d:.6f}")
        if avg_d > 1e-3:
            print("  WARNING: High avg distance — face indices may be inaccurate.")

        faces = nn_np[mesh_faces]
        print(f"  Faces remapped: {faces.shape[0]}")

    print(f"  Total: {T} frames, {V} verts, {faces.shape[0]} faces")

    print("\nBuilding edges...")
    edges = get_edges(faces)
    print(f"  {edges.shape[0]} unique edges")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    print("\nComputing per-vertex weights (frame-to-frame stretch)...")
    weights       = compute_vertex_weights(deformations, edges, args.alpha)
    w_t           = torch.from_numpy(weights).float().to(device).unsqueeze(1)
    one_minus_w_t = 1.0 - w_t

    print(f"\nRefining {T-1} frames  "
          f"({args.n_steps} steps, lr={args.lr}, "
          f"data_weight={args.data_weight}, temporal_weight={args.temporal_weight})")
    refined    = np.zeros_like(deformations)
    refined[0] = deformations[0]

    for t in range(1, T):
        pos_orig = torch.from_numpy(deformations[t]).float().to(device)
        pos_prev = torch.from_numpy(refined[t-1]).float().to(device)
        pos      = pos_orig.clone().requires_grad_(True)
        opt      = torch.optim.Adam([pos], lr=args.lr)

        for _ in range(args.n_steps):
            opt.zero_grad()
            data_loss     = (w_t           * (pos - pos_orig) ** 2).mean()
            temporal_loss = (one_minus_w_t * (pos - pos_prev) ** 2).mean()
            loss          = args.data_weight * data_loss + args.temporal_weight * temporal_loss
            loss.backward()
            opt.step()

        refined[t] = pos.detach().cpu().numpy()

        if t % 10 == 0 or t == T - 1:
            print(f"  Frame {t:3d}/{T-1}  "
                  f"data={data_loss.item():.2e}  temporal={temporal_loss.item():.2e}")

    np.save(output_path, refined.astype(np.float32))
    print(f"\nSaved refined deformations: {output_path}")
    print(f"  Shape: {refined.shape}")


if __name__ == "__main__":
    main()
