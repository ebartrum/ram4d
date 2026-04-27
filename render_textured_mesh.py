"""
render_textured_mesh.py — Render a textured dynamic mesh as a multi-view video.

Loads the OBJ + texture from an ActionMesh texture/ directory and renders a
4-view side-by-side video, using either the original or a refined deformations
file. Use this to quickly verify the effect of refine_mesh_deformations.py
without re-running the full mesh_to_4dgs pipeline.

Usage:
  # Render original deformations
  python render_textured_mesh.py \
    --input_mesh output/2026.03.03/actionmesh_gs_replace_corgi

  # Render refined deformations
  python render_textured_mesh.py \
    --input_mesh output/2026.03.03/actionmesh_gs_replace_corgi \
    --deformations_path output/2026.03.03/actionmesh_gs_replace_corgi/deformations_vertices_refined.npy \
    --output_path output/2026.04.27/textured_dynamic_mesh_refined.mp4
"""

import sys
import os
import glob

if "LIBGOMP_PRELOADED" not in os.environ:
    preload_files = []
    libgomp_files = glob.glob(
        "/home/ubuntu/miniconda3/envs/mvadapter/lib/python*/site-packages/scikit_learn.libs/libgomp*.so*"
    )
    if libgomp_files:
        preload_files.append(libgomp_files[0])
    libgl_path = "/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0"
    if os.path.exists(libgl_path):
        preload_files.append(libgl_path)
    if preload_files:
        os.environ["LD_PRELOAD"] = ":".join(preload_files)
        os.environ["LIBGOMP_PRELOADED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "actionmesh"))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "actionmesh", "third_party", "TripoSG"))

import argparse
import numpy as np
import torch
from PIL import Image
import imageio

from mvadapter.utils.mesh_utils import load_mesh, get_orthogonal_camera
from mvadapter.utils.mesh_utils.render import NVDiffRastContextWrapper
from mesh_render_utils import (load_and_transform_deformations,
                                compute_vertex_mapping,
                                render_dynamic_mesh_video)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mesh", required=True,
                        help="ActionMesh output dir containing texture/ subdir.")
    parser.add_argument("--deformations_path", type=str, default=None,
                        help="Path to deformations_vertices .npy file. "
                             "Defaults to <input_mesh>/deformations_vertices.npy.")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output MP4 path. Defaults to "
                             "<input_mesh>/texture/textured_dynamic_mesh_render.mp4.")
    parser.add_argument("--size", type=int, default=512,
                        help="Render resolution per view in pixels (default: 512).")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--context_type", type=str, default="cuda",
                        choices=["gl", "cuda"],
                        help="nvdiffrast context type (default: cuda).")
    return parser.parse_args()


def main():
    args = parse_args()

    texture_dir       = os.path.join(args.input_mesh, "texture")
    mesh_path         = os.path.join(texture_dir, "mesh_00.obj")
    texture_img_path  = os.path.join(texture_dir, "mesh_00_texture.png")
    deformations_path = args.deformations_path or os.path.join(
        args.input_mesh, "deformations_vertices.npy"
    )
    output_path = args.output_path or os.path.join(
        texture_dir, "textured_dynamic_mesh_render.mp4"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load mesh + texture ---
    print("\n--- Loading mesh ---")
    texture_img = Image.open(texture_img_path).convert("RGB")
    uv_size = texture_img.width
    mesh, offset, scale = load_mesh(
        mesh_path,
        rescale=True, move_to_center=True, front_x_to_y=True,
        default_uv_size=uv_size, device=device,
        return_transform=True, merge_vertices=False,
    )
    mesh.texture = torch.from_numpy(
        np.array(texture_img).astype(np.float32) / 255.0
    ).to(device)
    print(f"  Vertices: {mesh.v_pos.shape[0]}, Faces: {mesh.t_pos_idx.shape[0]}")

    # --- Load deformations ---
    print(f"\n--- Loading deformations ---")
    deformations_orig = load_and_transform_deformations(deformations_path, offset, scale)
    print(f"  Shape: {deformations_orig.shape}")

    mapping_idx = compute_vertex_mapping(mesh.v_pos, deformations_orig[0], device)

    # --- Cameras ---
    cameras = get_orthogonal_camera(
        elevation_deg=[0] * 4, distance=[2.2] * 4,
        left=-0.55, right=0.55, bottom=-0.55, top=0.55,
        azimuth_deg=[-90, 0, 90, 180],
        device=device,
    )

    # --- Render ---
    ctx = NVDiffRastContextWrapper(device=device, context_type=args.context_type)
    H = W = args.size
    print(f"\n--- Rendering {deformations_orig.shape[0]} frames at {W}×{H} ---")
    video_frames = render_dynamic_mesh_video(
        ctx, mesh, cameras, deformations_orig, mapping_idx,
        height=H, width=W,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    imageio.mimsave(output_path, video_frames, fps=args.fps)
    print(f"\nDone. Saved: {output_path}")


if __name__ == "__main__":
    main()
