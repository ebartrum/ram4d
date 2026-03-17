"""
render_distillation_inputs.py — Render static frame 0 from N training camera viewpoints at 1024×1024.

Loads fg_positions_world.npy (frame 0) and the background 3DGS, then renders
composite images from evenly-spaced COLMAP training cameras at a fixed 1024×1024
resolution using make_raster_camera_at_resolution() from camera_utils.py.

Usage:
  python render_training_views.py \\
      --output_path output/2026.03.03/actionmesh_gs_replace_corgi \\
      --gs_scene_path Inpaint360GS/data/inpaint360/bag \\
      --gs_model_path output/2026.02.26/inpainted_scene/point_cloud_object_inpaint_virtual \\
      [--n_views 5] [--render_size 1024]
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
from scipy.ndimage import binary_dilation

from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from camera_utils import make_raster_camera_at_resolution

C0 = 0.28209479177387814


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True,
                        help="ram4d output dir (contains gaussians/fg_positions_world.npy)")
    parser.add_argument("--gs_scene_path", required=True,
                        help="Inpaint360GS scene dir (contains sparse/0/)")
    parser.add_argument("--gs_model_path", required=True,
                        help="Inpaint360GS model dir (contains iteration_N/point_cloud.ply)")
    parser.add_argument("--n_views", type=int, default=5,
                        help="Number of evenly-spaced training cameras to render (default 5)")
    parser.add_argument("--render_size", type=int, default=1024,
                        help="Output image size in pixels (square, default 1024)")
    parser.add_argument("--fg_positions_path", default=None,
                        help="Override path to fg_positions_world.npy")
    parser.add_argument("--placement_path", default=None,
                        help="Override path to placement.json for fg scale")
    parser.add_argument("--render_output_dir", default=None,
                        help="Directory for output PNGs (default: <output_path>/gaussians/training_views/)")
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


def load_training_cameras(scene_path, n_views):
    """Load evenly-spaced COLMAP training cameras, return list of camera dicts."""
    images_bin  = os.path.join(scene_path, "sparse", "0", "images.bin")
    cameras_bin = os.path.join(scene_path, "sparse", "0", "cameras.bin")
    extrinsics = read_extrinsics_binary(images_bin)
    intrinsics = read_intrinsics_binary(cameras_bin)
    sorted_images = sorted(extrinsics.values(), key=lambda x: x.name)

    total = len(sorted_images)
    indices = [int(round(i * (total - 1) / (n_views - 1))) for i in range(n_views)] if n_views > 1 else [0]
    selected = [sorted_images[i] for i in indices]

    cameras = []
    for img in selected:
        cam = intrinsics[img.camera_id]
        R_w2c = qvec2rotmat(img.qvec)
        tvec  = np.array(img.tvec, dtype=np.float64)
        W, H  = int(cam.width), int(cam.height)

        if cam.model in ("PINHOLE", "OPENCV"):
            fx = cam.params[0]
        elif cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = cam.params[0]
        else:
            raise ValueError(f"Unsupported COLMAP camera model: {cam.model}")

        fovx = 2.0 * math.atan(W / (2.0 * fx))

        pose_w2c = np.eye(4, dtype=np.float64)
        pose_w2c[:3, :3] = R_w2c
        pose_w2c[:3,  3] = tvec

        cameras.append({
            "name": img.name,
            "pose_w2c": pose_w2c,
            "fovx": fovx,
            "native_W": W,
            "native_H": H,
        })
        print(f"  Selected '{img.name}'  native {W}×{H}  fovx={math.degrees(fovx):.1f}°")

    return cameras


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
    S = args.render_size

    gaussians_dir     = os.path.join(args.output_path, "gaussians")
    fg_positions_path = args.fg_positions_path or os.path.join(gaussians_dir, "fg_positions_world.npy")
    fg_ply_path       = os.path.join(gaussians_dir, "gaussians.ply")
    placement_path    = args.placement_path or os.path.join(gaussians_dir, "placement.json")

    render_dir = args.render_output_dir or os.path.join(gaussians_dir, "training_views")
    os.makedirs(render_dir, exist_ok=True)

    # --- Foreground positions (frame 0 only) ---
    print("\n--- Loading fg_positions_world.npy ---")
    fg_positions_world = np.load(fg_positions_path)   # (T_fg, N_fg, 3)
    print(f"  Shape: {fg_positions_world.shape}")
    fg_pos_frame0 = fg_positions_world[0]              # (N_fg, 3)

    # --- Foreground attributes ---
    print("\n--- Loading foreground Gaussian attributes ---")
    _, f_dc_fg, op_fg, log_sc_fg, rot_fg = load_ply_gs(fg_ply_path)
    with open(placement_path) as f:
        scale = float(json.load(f)["scale"])
    scales_fg = np.exp(log_sc_fg) * scale
    N_fg = len(fg_pos_frame0)
    print(f"  {N_fg:,} fg Gaussians  scale={scale:.4f}")

    # --- Background GS ---
    print("\n--- Loading background 3DGS ---")
    bg_ply = find_latest_ply(args.gs_model_path)
    print(f"  PLY: {bg_ply}")
    xyz_bg, f_dc_bg, op_bg, log_sc_bg, rot_bg = load_ply_gs(bg_ply)
    print(f"  {len(xyz_bg):,} bg Gaussians")

    # Concatenate [bg | fg]
    f_dc_all   = np.concatenate([f_dc_bg,           f_dc_fg  ], axis=0)
    op_all     = np.concatenate([op_bg,             op_fg    ], axis=0)
    rot_all    = np.concatenate([rot_bg,            rot_fg   ], axis=0)
    scales_all = np.concatenate([np.exp(log_sc_bg), scales_fg], axis=0)

    xyz_bg_t = torch.from_numpy(xyz_bg).float().to(device)
    fg_pos_t = torch.from_numpy(fg_pos_frame0).float().to(device)
    means3D  = torch.cat([xyz_bg_t, fg_pos_t], dim=0)

    rgb_t    = torch.from_numpy(np.clip(f_dc_all * C0 + 0.5, 0.0, 1.0)).float().to(device)
    alpha_t  = torch.from_numpy(1.0 / (1.0 + np.exp(-op_all))).float().to(device).unsqueeze(1)
    scales_t = torch.from_numpy(scales_all).float().to(device)
    rot_t    = torch.from_numpy(rot_all).float().to(device)
    print(f"  Total Gaussians: {means3D.shape[0]:,}")

    # Fg-only tensors for mask rendering
    N_bg = len(xyz_bg)
    white_fg_t  = torch.ones(N_fg, 3, device=device)
    alpha_fg_t  = alpha_t[N_bg:]
    scales_fg_t = scales_t[N_bg:]
    rot_fg_t    = rot_t[N_bg:]

    # Disk structuring element for 5px dilation
    r = 5
    yy, xx = np.ogrid[-r:r+1, -r:r+1]
    disk = (xx**2 + yy**2) <= r**2

    # --- Cameras ---
    print(f"\n--- Selecting {args.n_views} training cameras ---")
    cameras = load_training_cameras(args.gs_scene_path, args.n_views)

    # --- Render ---
    print(f"\n--- Rendering {args.n_views} views at {S}×{S} ---")
    for i, cam in enumerate(cameras):
        viewmat, full_proj, campos, tanfovx, tanfovy = make_raster_camera_at_resolution(
            cam["pose_w2c"], cam["fovx"], S, S, device
        )
        img = render_frame(
            means3D, rgb_t, alpha_t, scales_t, rot_t,
            viewmat, full_proj, campos, tanfovx, tanfovy, S, S, device,
        )
        stem = os.path.splitext(cam["name"])[0]
        out_path = os.path.join(render_dir, f"view_{i:02d}_{stem}.png")
        imageio.imwrite(out_path, img)

        # Fg mask: render fg-only white-on-black, threshold, dilate
        mask_raw = render_frame(
            fg_pos_t, white_fg_t, alpha_fg_t, scales_fg_t, rot_fg_t,
            viewmat, full_proj, campos, tanfovx, tanfovy, S, S, device,
        )
        mask_binary = mask_raw.mean(axis=2) > 127
        mask_dilated = binary_dilation(mask_binary, structure=disk)
        mask_img = (mask_dilated * 255).astype(np.uint8)
        mask_path = os.path.join(render_dir, f"mask_{i:02d}_{stem}.png")
        imageio.imwrite(mask_path, mask_img)

        print(f"  [{i+1}/{args.n_views}] {out_path}  +  {mask_path}")

    print(f"\nDone. {args.n_views} images saved to {render_dir}/")


if __name__ == "__main__":
    main()
