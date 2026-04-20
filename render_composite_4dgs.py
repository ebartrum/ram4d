"""
render_composite_4dgs.py — Render composite 4DGS from a COLMAP camera or an orbiting trajectory.

Loads fg_positions_world.npy (produced by create_composite_4dgs.py) and renders
the background 3DGS + foreground 4DGS together using the Inpaint360GS CUDA rasterizer.

Run create_composite_4dgs.py first to produce gaussians/fg_positions_world.npy.

Modes:
  Single camera (default):
    Renders T_fg frames (one per 4DGS animation frame) from a fixed COLMAP camera.
    --camera_idx selects which camera.

  Orbit (--orbit):
    Renders a smooth circular orbit around the scene using generate_ellipse_path().
    --n_frames sets the orbit length; the 4DGS animation loops if n_frames > T_fg.

Usage:
  # Single camera
  python render_composite_4dgs.py \\
      --output_path output/2026.03.02/actionmesh_gs_replace_corgi \\
      --gs_scene_path Inpaint360GS/data/inpaint360/bag \\
      --gs_model_path output/2026.02.26/inpainted_scene/point_cloud_object_inpaint_virtual \\
      --camera_idx 28

  # Orbit
  python render_composite_4dgs.py \\
      --output_path output/2026.03.02/actionmesh_gs_replace_corgi \\
      --gs_scene_path Inpaint360GS/data/inpaint360/bag \\
      --gs_model_path output/2026.02.26/inpainted_scene/point_cloud_object_inpaint_virtual \\
      --orbit [--n_frames 240]
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

from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from camera_utils import create_orbit_cameras

C0 = 0.28209479177387814

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True,
                        help="ram4d output dir (contains gaussians/fg_positions_world.npy)")
    parser.add_argument("--gs_scene_path", required=True,
                        help="Inpaint360GS scene dir (contains sparse/0/)")
    parser.add_argument("--gs_model_path", required=True,
                        help="Inpaint360GS model dir (contains iteration_N/point_cloud.ply)")
    parser.add_argument("--camera_idx", type=int, default=28,
                        help="COLMAP camera to render from (single-camera mode, default 28)")
    parser.add_argument("--orbit", action="store_true",
                        help="Render an orbiting camera trajectory instead of a single camera")
    parser.add_argument("--n_frames", type=int, default=240,
                        help="Number of orbit frames (orbit mode only, default 240)")
    parser.add_argument("--render_scale", type=float, default=0.25,
                        help="Scale factor applied to COLMAP camera resolution (default 0.25)")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--output_format", choices=["video", "frames"], default="video",
                        help="Output as video (default) or individual PNG frames in a subdirectory")
    parser.add_argument("--static", action="store_true",
                        help="Freeze fg at frame 0 (static placement check, no animation)")
    parser.add_argument("--fg_only", action="store_true",
                        help="Render foreground Gaussians only (no background 3DGS)")
    parser.add_argument("--fg_positions_path", default=None,
                        help="Override path to fg_positions_world.npy "
                             "(e.g. use fg_positions_world_deformed.npy from refine_deform.py)")
    parser.add_argument("--placement_path", default=None,
                        help="Override path to placement.json for fg scale "
                             "(e.g. placement_refined.json from refine_frame0.py)")
    parser.add_argument("--render_output_dir", default=None,
                        help="Directory to save rendered video and frame0 image "
                             "(default: <output_path>/gaussians/)")
    parser.add_argument("--frame_idx", type=int, default=None,
                        help="If set, render only this animation frame and save as PNG (no video).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Camera loading
# ---------------------------------------------------------------------------

def load_colmap_camera(scene_path, camera_idx):
    """Load a single COLMAP camera's extrinsics and intrinsics."""
    images_bin  = os.path.join(scene_path, "sparse", "0", "images.bin")
    cameras_bin = os.path.join(scene_path, "sparse", "0", "cameras.bin")
    extrinsics = read_extrinsics_binary(images_bin)
    intrinsics = read_intrinsics_binary(cameras_bin)
    sorted_images = sorted(extrinsics.values(), key=lambda x: x.name)

    if camera_idx >= len(sorted_images):
        raise ValueError(f"camera_idx {camera_idx} out of range ({len(sorted_images)} cameras)")

    img = sorted_images[camera_idx]
    cam = intrinsics[img.camera_id]
    print(f"  Camera {camera_idx}: '{img.name}'  COLMAP id={img.id}  model={cam.model}")

    R_w2c = qvec2rotmat(img.qvec)
    tvec  = np.array(img.tvec, dtype=np.float64)
    W, H  = int(cam.width), int(cam.height)

    if cam.model in ("PINHOLE", "OPENCV"):
        fx, fy = cam.params[0], cam.params[1]
    elif cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = cam.params[0]
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {cam.model}")

    FoVx = 2.0 * math.atan(W / (2.0 * fx))
    FoVy = 2.0 * math.atan(H / (2.0 * fy))

    # Single camera: one W2C pose, repeated for each fg animation frame
    pose_w2c = np.eye(4, dtype=np.float64)
    pose_w2c[:3, :3] = R_w2c
    pose_w2c[:3,  3] = tvec
    return [pose_w2c], FoVx, FoVy, W, H

# ---------------------------------------------------------------------------
# PLY loading
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Camera matrices
# ---------------------------------------------------------------------------

def make_raster_camera(pose_w2c, FoVx, FoVy, W, H, device, znear=0.01, zfar=200.0):
    """Build viewmatrix / full_proj / campos from a 4×4 W2C pose matrix."""
    R_w2c = pose_w2c[:3, :3].astype(np.float32)
    t     = pose_w2c[:3, 3].astype(np.float32)
    R_c2w = R_w2c.T

    W2C = np.eye(4, dtype=np.float32)
    W2C[:3, :3] = R_w2c
    W2C[:3,  3] = t

    tanfovx = math.tan(FoVx / 2.0)
    tanfovy = math.tan(FoVy / 2.0)

    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] =  1.0 / tanfovx
    P[1, 1] =  1.0 / tanfovy
    P[3, 2] =  1.0
    P[2, 2] =  zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    viewmatrix = torch.from_numpy(W2C.T).float().to(device)
    proj_t     = torch.from_numpy(P.T  ).float().to(device)
    full_proj  = (viewmatrix.unsqueeze(0).bmm(proj_t.unsqueeze(0))).squeeze(0)
    campos     = torch.tensor(-(R_c2w @ t), dtype=torch.float32, device=device)
    return viewmatrix, full_proj, campos, tanfovx, tanfovy

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_frame(means3D, colors, opacities, scales, rotations,
                 viewmatrix, full_proj, campos, tanfovx, tanfovy, W, H, device, bg_color=None):
    N = means3D.shape[0]
    means2D = torch.zeros(N, 3, device=device, dtype=torch.float32)

    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=float(tanfovx),
        tanfovy=float(tanfovy),
        bg=bg_color if bg_color is not None else torch.zeros(3, device=device, dtype=torch.float32),
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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gaussians_dir     = args.output_path
    fg_positions_path = args.fg_positions_path or os.path.join(gaussians_dir, "fg_positions_world.npy")
    fg_ply_path       = os.path.join(gaussians_dir, "gaussians.ply")
    placement_path    = args.placement_path or os.path.join(gaussians_dir, "placement.json")

    render_dir = args.render_output_dir or gaussians_dir
    os.makedirs(render_dir, exist_ok=True)

    static_suffix  = "_static"  if args.static  else ""
    fg_only_suffix = "_fg_only" if args.fg_only else ""
    suffix = static_suffix + fg_only_suffix
    if args.orbit:
        out_video  = os.path.join(render_dir, f"orbit_{args.n_frames}frames{suffix}.mp4")
        out_frame0 = os.path.join(render_dir, f"orbit_{args.n_frames}frames{suffix}_frame0.png")
    else:
        out_video  = os.path.join(render_dir, f"composite_cam{args.camera_idx}{suffix}.mp4")
        out_frame0 = os.path.join(render_dir, f"composite_cam{args.camera_idx}{suffix}_frame0.png")

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

    # --- Camera sequence ---
    print("\n--- Loading cameras ---")
    if args.orbit:
        poses_w2c, FoVx, FoVy, cam_W, cam_H = create_orbit_cameras(
            args.gs_scene_path, args.n_frames
        )
    else:
        poses_w2c, FoVx, FoVy, cam_W, cam_H = load_colmap_camera(
            args.gs_scene_path, args.camera_idx
        )
    W = max(1, int(cam_W * args.render_scale))
    H = max(1, int(cam_H * args.render_scale))
    print(f"  Native {cam_W}×{cam_H} → render {W}×{H}  (scale={args.render_scale})")

    # --- Background GS ---
    if args.fg_only:
        print("\n--- Skipping background 3DGS (--fg_only) ---")
        xyz_bg_t = None
        f_dc_all   = f_dc_fg
        op_all     = op_fg
        rot_all    = rot_fg
        scales_all = scales_fg
    else:
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

    # --- Build GPU attribute tensors (shared across all frames) ---
    print("\n--- Building GPU tensors ---")
    rgb_t    = torch.from_numpy(np.clip(f_dc_all * C0 + 0.5, 0.0, 1.0)).float().to(device)
    alpha_t  = torch.from_numpy(1.0 / (1.0 + np.exp(-op_all))).float().to(device).unsqueeze(1)
    scales_t = torch.from_numpy(scales_all).float().to(device)
    rot_t    = torch.from_numpy(rot_all).float().to(device)

    if args.fg_only:
        print(f"  Total Gaussians: {N_fg:,}  (fg only)")
    else:
        print(f"  Total Gaussians: {len(f_dc_all):,}  (bg={len(xyz_bg):,}  fg={N_fg:,})")
    print(f"  GPU memory: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB allocated")

    # --- In single-camera mode, render one frame per fg animation frame.
    #     In orbit mode, render one frame per orbit pose, looping the fg animation.
    if args.orbit:
        n_render = len(poses_w2c)
        if n_render > T_fg:
            print(f"\n  4DGS animation ({T_fg} frames) will loop {n_render / T_fg:.1f}x")
    else:
        n_render = T_fg   # one orbit pose (the fixed camera), repeated T_fg times

    # --- Determine which frames to render ---
    if args.frame_idx is not None:
        if args.frame_idx >= T_fg:
            print(f"  WARNING: frame_idx {args.frame_idx} >= T_fg {T_fg}, using {args.frame_idx % T_fg}")
        frames_to_render = [args.frame_idx]
    else:
        frames_to_render = range(n_render)

    # --- Render ---
    print(f"\n--- Rendering {len(list(frames_to_render))} frame(s) at {W}×{H} ---")
    frames = []
    for i in frames_to_render:
        pose_w2c = poses_w2c[i % len(poses_w2c)]
        t_fg     = 0 if args.static else i % T_fg

        viewmat, full_proj, campos, tanfovx, tanfovy = make_raster_camera(
            pose_w2c, FoVx, FoVy, W, H, device
        )
        fg_pos_t = torch.from_numpy(fg_positions_world[t_fg]).float().to(device)
        means3D  = fg_pos_t if args.fg_only else torch.cat([xyz_bg_t, fg_pos_t], dim=0)

        bg_color = torch.ones(3, device=device, dtype=torch.float32) if args.fg_only else None
        img = render_frame(
            means3D, rgb_t, alpha_t, scales_t, rot_t,
            viewmat, full_proj, campos, tanfovx, tanfovy, W, H, device, bg_color=bg_color,
        )
        frames.append(img)

        if i % 20 == 0:
            print(f"  Frame {i}  (fg frame {t_fg})")

    # --- Save ---
    if args.frame_idx is not None:
        # Single-frame mode: save PNG only
        if args.orbit:
            png_name = f"orbit_frame{args.frame_idx}{suffix}.png"
        else:
            png_name = f"composite_cam{args.camera_idx}_frame{args.frame_idx}{suffix}.png"
        png_path = os.path.join(render_dir, png_name)
        imageio.imwrite(png_path, frames[0])
        print(f"\nFrame: {png_path}")
    elif args.output_format == "frames":
        # Derive a frames subdir name from the video name (strip .mp4)
        frames_dir = out_video[:-4] + "_frames"
        os.makedirs(frames_dir, exist_ok=True)
        for idx, frame in enumerate(frames):
            imageio.imwrite(os.path.join(frames_dir, f"{idx:05d}.png"), frame)
        print(f"\nFrames: {frames_dir}  ({len(frames)} PNGs  {W}×{H})")
    else:
        imageio.imwrite(out_frame0, frames[0])
        print(f"\nFrame 0: {out_frame0}")
        imageio.mimsave(out_video, frames, fps=args.fps)
        print(f"Video:   {out_video}  ({len(frames)} frames @ {args.fps} fps  {W}×{H})")


if __name__ == "__main__":
    main()
