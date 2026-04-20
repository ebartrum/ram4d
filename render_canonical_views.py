"""
render_canonical_views.py — Render 4 canonical views (front/left/back/right) of the composite scene.

Uses create_orbit_cameras with n_frames=4 to generate evenly-spaced orbit poses, then renders
frame 0 (static) of the foreground from each pose.

Usage:
  python render_canonical_views.py \\
      --output_path output/2026.04.16/corgi_gaussians_200k_v2 \\
      --gs_scene_path Inpaint360GS/data/inpaint360/bag \\
      --gs_model_path output/2026.02.26/inpainted_scene/point_cloud_object_inpaint_virtual \\
      --placement_path output/2026.04.16/corgi_refined/placement_refined.json \\
      --render_output_dir output/2026.04.20/corgi_canonical
"""

import sys
import os
import glob
import math

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
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "Inpaint360GS"))

import argparse
import json
import numpy as np
import torch
import imageio
from plyfile import PlyData

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from camera_utils import create_orbit_cameras, make_raster_camera_at_resolution

C0 = 0.28209479177387814

VIEW_NAMES = ["right", "back", "left", "front"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True,
                        help="Gaussians dir (contains gaussians.ply and fg_positions_world.npy)")
    parser.add_argument("--gs_scene_path", required=True,
                        help="Inpaint360GS scene dir (contains sparse/0/)")
    parser.add_argument("--gs_model_path", required=True,
                        help="Inpaint360GS model dir (contains iteration_N/point_cloud.ply)")
    parser.add_argument("--placement_path", default=None,
                        help="Override path to placement.json (default: <output_path>/placement.json)")
    parser.add_argument("--fg_positions_path", default=None,
                        help="Override path to fg_positions_world.npy")
    parser.add_argument("--render_output_dir", default=None,
                        help="Directory to save output PNGs (default: <output_path>)")
    parser.add_argument("--render_scale", type=float, default=0.25,
                        help="Scale factor applied to COLMAP camera resolution (default 0.25)")
    parser.add_argument("--rendering_mode", choices=["static", "dynamic"], default="dynamic",
                        help="Render a single static frame (static) or full animation videos (dynamic, default)")
    parser.add_argument("--frame_idx", type=int, default=0,
                        help="Foreground animation frame to render in static mode (default 0)")
    parser.add_argument("--fps", type=int, default=16,
                        help="Frames per second for dynamic video output (default 16)")
    return parser.parse_args()


def find_latest_ply(model_path):
    pc_dirs = glob.glob(os.path.join(model_path, "iteration_*"))
    if not pc_dirs:
        pc_dirs = glob.glob(os.path.join(model_path, "point_cloud", "iteration_*"))
    if not pc_dirs:
        raise FileNotFoundError(f"No iteration_* in {model_path}")
    pc_dirs.sort(key=lambda p: int(p.split("_")[-1]))
    ply = os.path.join(pc_dirs[-1], "point_cloud.ply")
    if not os.path.exists(ply):
        raise FileNotFoundError(f"PLY not found: {ply}")
    return ply


def load_ply_gs(ply_path):
    v = PlyData.read(ply_path).elements[0]
    xyz     = np.stack([v['x'],       v['y'],       v['z']      ], axis=1).astype(np.float32)
    f_dc    = np.stack([v['f_dc_0'],  v['f_dc_1'],  v['f_dc_2'] ], axis=1).astype(np.float32)
    opacity = v['opacity'].astype(np.float32)
    log_sc  = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1).astype(np.float32)
    rot     = np.stack([v['rot_0'],   v['rot_1'],   v['rot_2'],   v['rot_3']], axis=1).astype(np.float32)
    return xyz, f_dc, opacity, log_sc, rot


def render_frame(means3D, colors, opacities, scales, rotations,
                 viewmatrix, full_proj, campos, tanfovx, tanfovy, W, H, device):
    N = means3D.shape[0]
    means2D = torch.zeros(N, 3, device=device, dtype=torch.float32)
    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=float(tanfovx),
        tanfovy=float(tanfovy),
        bg=torch.zeros(3, device=device, dtype=torch.float32),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=full_proj,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered, _, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )
    img = rendered.clamp(0.0, 1.0).permute(1, 2, 0)
    return (img.cpu().numpy() * 255).astype(np.uint8)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gaussians_dir     = args.output_path
    fg_positions_path = args.fg_positions_path or os.path.join(gaussians_dir, "fg_positions_world.npy")
    fg_ply_path       = os.path.join(gaussians_dir, "gaussians.ply")
    placement_path    = args.placement_path or os.path.join(gaussians_dir, "placement.json")
    render_dir        = args.render_output_dir or gaussians_dir
    os.makedirs(render_dir, exist_ok=True)

    # --- Foreground positions ---
    print("\n--- Loading fg_positions_world.npy ---")
    fg_positions_world = np.load(fg_positions_path)   # (T_fg, N_fg, 3)
    T_fg, N_fg, _ = fg_positions_world.shape
    print(f"  Shape: {fg_positions_world.shape}")

    # --- Foreground attributes ---
    print("\n--- Loading foreground Gaussian attributes ---")
    _, f_dc_fg, op_fg, log_sc_fg, rot_fg = load_ply_gs(fg_ply_path)
    with open(placement_path) as f:
        scale = float(json.load(f)["scale"])
    scales_fg = np.exp(log_sc_fg) * scale
    print(f"  {N_fg:,} Gaussians  scale={scale:.4f}")

    # --- Background GS ---
    print("\n--- Loading background 3DGS ---")
    bg_ply = find_latest_ply(args.gs_model_path)
    print(f"  PLY: {bg_ply}")
    xyz_bg, f_dc_bg, op_bg, log_sc_bg, rot_bg = load_ply_gs(bg_ply)
    print(f"  {len(xyz_bg):,} Gaussians")
    xyz_bg_t   = torch.from_numpy(xyz_bg).float().to(device)
    f_dc_all   = np.concatenate([f_dc_bg,  f_dc_fg ], axis=0)
    op_all     = np.concatenate([op_bg,    op_fg   ], axis=0)
    rot_all    = np.concatenate([rot_bg,   rot_fg  ], axis=0)
    scales_all = np.concatenate([np.exp(log_sc_bg), scales_fg], axis=0)

    # --- GPU attribute tensors (composite) ---
    print("\n--- Building GPU tensors ---")
    rgb_t    = torch.from_numpy(np.clip(f_dc_all * C0 + 0.5, 0.0, 1.0)).float().to(device)
    alpha_t  = torch.from_numpy(1.0 / (1.0 + np.exp(-op_all))).float().to(device).unsqueeze(1)
    scales_t = torch.from_numpy(scales_all).float().to(device)
    rot_t    = torch.from_numpy(rot_all).float().to(device)
    print(f"  Gaussians: bg={len(xyz_bg):,}  fg={N_fg:,}")

    # --- fg-only tensors for alpha matte (white colors, fg opacities) ---
    alpha_fg_t  = torch.from_numpy(1.0 / (1.0 + np.exp(-op_fg))).float().to(device).unsqueeze(1)
    white_fg_t  = torch.ones(N_fg, 3, device=device, dtype=torch.float32)
    scales_fg_t = torch.from_numpy(scales_fg).float().to(device)
    rot_fg_t    = torch.from_numpy(rot_fg).float().to(device)

    # --- 4 canonical orbit cameras ---
    print("\n--- Creating 4 canonical cameras ---")
    poses_w2c, FoVx, FoVy, cam_W, cam_H = create_orbit_cameras(args.gs_scene_path, n_frames=4)
    W = max(1, int(cam_W * args.render_scale))
    H = max(1, int(cam_H * args.render_scale))
    print(f"  Native {cam_W}×{cam_H} → render {W}×{H}  (scale={args.render_scale})")

    # --- Precompute per-view camera matrices ---
    cameras = [
        make_raster_camera_at_resolution(poses_w2c[i], FoVx, W, H, device)
        for i in range(4)
    ]

    # --- Render and save ---
    if args.rendering_mode == "static":
        t_fg = args.frame_idx % T_fg
        print(f"\n--- Rendering 4 static views (fg frame {t_fg}) ---")
        fg_pos_t = torch.from_numpy(fg_positions_world[t_fg]).float().to(device)
        means3D  = torch.cat([xyz_bg_t, fg_pos_t], dim=0)
        for (viewmat, full_proj, campos, tanfovx, tanfovy), name in zip(cameras, VIEW_NAMES):
            img = render_frame(
                means3D, rgb_t, alpha_t, scales_t, rot_t,
                viewmat, full_proj, campos, tanfovx, tanfovy, W, H, device,
            )
            alpha_img = render_frame(
                fg_pos_t, white_fg_t, alpha_fg_t, scales_fg_t, rot_fg_t,
                viewmat, full_proj, campos, tanfovx, tanfovy, W, H, device,
            )
            imageio.imwrite(os.path.join(render_dir, f"{name}.png"), img)
            imageio.imwrite(os.path.join(render_dir, f"{name}_alpha.png"), alpha_img)
            print(f"  {name}: {name}.png  {name}_alpha.png")
    else:
        print(f"\n--- Rendering 4 dynamic views ({T_fg} frames each @ {args.fps} fps) ---")
        for (viewmat, full_proj, campos, tanfovx, tanfovy), name in zip(cameras, VIEW_NAMES):
            frames, alpha_frames = [], []
            for t in range(T_fg):
                fg_pos_t = torch.from_numpy(fg_positions_world[t]).float().to(device)
                means3D  = torch.cat([xyz_bg_t, fg_pos_t], dim=0)
                frames.append(render_frame(
                    means3D, rgb_t, alpha_t, scales_t, rot_t,
                    viewmat, full_proj, campos, tanfovx, tanfovy, W, H, device,
                ))
                alpha_frames.append(render_frame(
                    fg_pos_t, white_fg_t, alpha_fg_t, scales_fg_t, rot_fg_t,
                    viewmat, full_proj, campos, tanfovx, tanfovy, W, H, device,
                ))
            imageio.mimsave(os.path.join(render_dir, f"{name}.mp4"), frames, fps=args.fps)
            imageio.mimsave(os.path.join(render_dir, f"{name}_alpha.mp4"), alpha_frames, fps=args.fps)
            print(f"  {name}: {name}.mp4  {name}_alpha.mp4")


if __name__ == "__main__":
    main()
