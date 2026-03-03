"""
render_composite_4dgs.py — Render background 3DGS + foreground 4DGS composited from a COLMAP camera.

Loads placement.json (from estimate_fg_placement.py) to transform the foreground 4DGS
into the background scene's world space, then renders both together using the
Inpaint360GS CUDA rasterizer.

Foreground transform: p_world = R @ (scale * p_local) + translation

R is computed automatically to align the 4DGS local axes with the COLMAP world:
  - 4DGS local Z (up) → world up (estimated as mean -cam_down across all cameras)
  - 4DGS local -X (corgi front) → direction from corgi toward camera_idx
Foreground scales are also multiplied by scale so splat sizes match the world.

--yaw_deg rotates the corgi horizontally (around world_up) to fine-tune facing.

Usage:
  python render_composite_4dgs.py \\
      --output_path output/2026.03.02/actionmesh_gs_replace_corgi \\
      --gs_scene_path Inpaint360GS/data/inpaint360/bag \\
      --gs_model_path output/2026.02.26/inpainted_scene/point_cloud_object_inpaint_virtual \\
      --camera_idx 28 [--yaw_deg 0]
"""

import sys
import os
import glob
import math
import json

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
import numpy as np
import torch
import imageio
from plyfile import PlyData

from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

C0 = 0.28209479177387814


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True,
                        help="ram4d output dir (contains gaussians/ with placement.json)")
    parser.add_argument("--gs_scene_path", required=True,
                        help="Inpaint360GS scene dir (contains sparse/0/)")
    parser.add_argument("--gs_model_path", required=True,
                        help="Inpaint360GS model dir (contains iteration_N/point_cloud.ply)")
    parser.add_argument("--camera_idx", type=int, default=28)
    parser.add_argument("--render_scale", type=float, default=0.25,
                        help="Scale factor applied to COLMAP camera resolution (default 0.25)")
    parser.add_argument("--yaw_deg", type=float, default=0.0,
                        help="Extra yaw rotation around world_up (degrees, default 0)")
    parser.add_argument("--fps", type=int, default=16)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Camera loading (same as estimate_fg_placement.py)
# ---------------------------------------------------------------------------

def load_colmap_cameras(scene_path, camera_idx):
    """Load camera_idx's parameters plus world_up estimated from all cameras."""
    images_bin  = os.path.join(scene_path, "sparse", "0", "images.bin")
    cameras_bin = os.path.join(scene_path, "sparse", "0", "cameras.bin")
    extrinsics = read_extrinsics_binary(images_bin)
    intrinsics = read_intrinsics_binary(cameras_bin)
    sorted_images = sorted(extrinsics.values(), key=lambda x: x.name)
    if camera_idx >= len(sorted_images):
        raise ValueError(f"camera_idx {camera_idx} out of range ({len(sorted_images)} cameras)")

    # World up: mean of -cam_down (col 1 of R_c2w) across all cameras
    downs = np.stack([qvec2rotmat(si.qvec).T[:, 1] for si in sorted_images])
    mean_down = downs.mean(axis=0)
    world_up = -mean_down / np.linalg.norm(mean_down)
    print(f"  World up (from {len(sorted_images)} cameras): {np.round(world_up, 4)}")

    img = sorted_images[camera_idx]
    cam = intrinsics[img.camera_id]
    print(f"  Camera {camera_idx}: '{img.name}'  COLMAP id={img.id}  model={cam.model}")
    R_w2c = qvec2rotmat(img.qvec)
    tvec  = np.array(img.tvec, dtype=np.float64)
    R_c2w = R_w2c.T
    W, H = int(cam.width), int(cam.height)
    if cam.model in ("PINHOLE", "OPENCV"):
        fx, fy = cam.params[0], cam.params[1]
    elif cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = cam.params[0]
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {cam.model}")
    FoVx = 2.0 * math.atan(W / (2.0 * fx))
    FoVy = 2.0 * math.atan(H / (2.0 * fy))
    return R_c2w, tvec, FoVx, FoVy, W, H, world_up


def compute_fg_rotation(translation, cam_center, world_up, yaw_deg=0.0):
    """
    Build rotation R such that:
      - 4DGS local Z ([0,0,1], up) → world_up
      - 4DGS local -X (corgi front) → toward camera_idx (projected onto horizontal plane)
    Plus an optional yaw around world_up.

    Returns R (3,3) float32.
    """
    world_up = world_up / np.linalg.norm(world_up)

    # Direction from corgi to camera, projected onto the plane perpendicular to world_up
    to_cam = np.array(cam_center, dtype=np.float64) - np.array(translation, dtype=np.float64)
    to_cam_h = to_cam - np.dot(to_cam, world_up) * world_up
    if np.linalg.norm(to_cam_h) < 1e-6:
        to_cam_h = np.array([1.0, 0.0, 0.0])   # fallback
    world_fwd = to_cam_h / np.linalg.norm(to_cam_h)

    # Optional yaw around world_up (Rodrigues)
    if abs(yaw_deg) > 1e-6:
        angle = math.radians(yaw_deg)
        k = world_up
        c, s = math.cos(angle), math.sin(angle)
        # Rodrigues: v_rot = v*c + (k×v)*s + k*(k·v)*(1-c)
        world_fwd = (world_fwd * c
                     + np.cross(k, world_fwd) * s
                     + k * np.dot(k, world_fwd) * (1 - c))
        world_fwd /= np.linalg.norm(world_fwd)

    # Third axis: right = fwd × up  (gives right-handed system with det=+1)
    world_right = np.cross(world_fwd, world_up)
    world_right /= np.linalg.norm(world_right)

    # R columns: local +X → -world_fwd,  local Y → world_right,  local Z → world_up
    # (4DGS front = -X, so -X maps to world_fwd, hence +X maps to -world_fwd)
    R = np.stack([-world_fwd, world_right, world_up], axis=1).astype(np.float32)
    print(f"  Rotation det={np.linalg.det(R):.4f}")
    print(f"  world_fwd : {np.round(world_fwd, 4)}")
    print(f"  world_right: {np.round(world_right, 4)}")
    print(f"  world_up  : {np.round(world_up, 4)}")
    return R


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
    """Load Gaussian Splatting PLY — DC SH only, works for both background and foreground."""
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

def make_colmap_raster_camera(R_c2w, tvec, FoVx, FoVy, W, H, device, znear=0.01, zfar=200.0):
    """Build viewmatrix / full_proj / campos for the Inpaint360GS rasterizer."""
    R_w2c = R_c2w.T.astype(np.float32)
    t     = tvec.astype(np.float32)

    W2C = np.eye(4, dtype=np.float32)
    W2C[:3, :3] = R_w2c
    W2C[:3,  3] = t

    tanfovx = math.tan(FoVx / 2.0)
    tanfovy = math.tan(FoVy / 2.0)

    # Same projection matrix as Inpaint360GS getProjectionMatrix
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] =  1.0 / tanfovx
    P[1, 1] =  1.0 / tanfovy
    P[3, 2] =  1.0
    P[2, 2] =  zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    viewmatrix = torch.from_numpy(W2C.T).float().to(device)
    proj_t     = torch.from_numpy(P.T  ).float().to(device)
    full_proj  = (viewmatrix.unsqueeze(0).bmm(proj_t.unsqueeze(0))).squeeze(0)

    # Camera centre in world space
    campos = torch.tensor(
        -(R_c2w.astype(np.float32) @ t),
        dtype=torch.float32, device=device,
    )

    return viewmatrix, full_proj, campos, tanfovx, tanfovy


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gaussians_dir  = os.path.join(args.output_path, "gaussians")
    placement_path = os.path.join(gaussians_dir, "placement.json")
    yaw_tag = f"_yaw{int(args.yaw_deg):+d}" if args.yaw_deg != 0.0 else ""
    out_video  = os.path.join(gaussians_dir, f"composite_cam{args.camera_idx}{yaw_tag}.mp4")
    out_frame0 = os.path.join(gaussians_dir, f"composite_cam{args.camera_idx}{yaw_tag}_frame0.png")

    # --- Placement ---
    print("\n--- Loading placement.json ---")
    with open(placement_path) as f:
        placement = json.load(f)
    translation = np.array(placement["translation"], dtype=np.float32)
    scale       = float(placement["scale"])
    print(f"  translation: {translation.tolist()}")
    print(f"  scale:       {scale:.4f}")

    # --- COLMAP camera ---
    print("\n--- Loading COLMAP camera ---")
    R_c2w, tvec, FoVx, FoVy, cam_W, cam_H, world_up = load_colmap_cameras(
        args.gs_scene_path, args.camera_idx
    )
    W = max(1, int(cam_W * args.render_scale))
    H = max(1, int(cam_H * args.render_scale))
    print(f"  Native {cam_W}×{cam_H} → render {W}×{H}  (scale={args.render_scale})")

    viewmatrix, full_proj, campos, tanfovx, tanfovy = make_colmap_raster_camera(
        R_c2w, tvec, FoVx, FoVy, W, H, device
    )

    # --- Background GS ---
    print("\n--- Loading background 3DGS ---")
    bg_ply = find_latest_ply(args.gs_model_path)
    print(f"  PLY: {bg_ply}")
    xyz_bg, f_dc_bg, op_bg, log_sc_bg, rot_bg = load_ply_gs(bg_ply)
    print(f"  {len(xyz_bg):,} Gaussians")

    # --- Foreground 4DGS ---
    print("\n--- Loading foreground 4DGS ---")
    fg_ply = os.path.join(gaussians_dir, "gaussians.ply")
    offsets_path = os.path.join(gaussians_dir, "deformation_offsets.npy")
    xyz_fg, f_dc_fg, op_fg, log_sc_fg, rot_fg = load_ply_gs(fg_ply)
    offsets = np.load(offsets_path) if os.path.exists(offsets_path) else None
    T = offsets.shape[0] if offsets is not None else 1
    print(f"  {len(xyz_fg):,} Gaussians  |  {T} frames")

    # --- Foreground rotation ---
    print("\n--- Computing foreground orientation ---")
    cam_center = -(R_c2w.astype(np.float32) @ tvec.astype(np.float32))
    R_fg = compute_fg_rotation(translation, cam_center, world_up, args.yaw_deg)

    # --- Transform foreground to world space ---
    # Base rotated positions (no translation yet — applied per-frame)
    # p_world_t = xyz_fg_rot + R_fg @ (offsets[t] * scale) + translation_t
    print("\n--- Applying placement transform to foreground ---")
    xyz_fg_rot = (R_fg @ (xyz_fg * scale).T).T      # (N, 3) — rotation + scale, centred at origin
    scales_fg_world = np.exp(log_sc_fg) * scale      # activated + scaled

    # Per-frame translations (falls back to constant frame-0 translation)
    per_frame_translations = placement.get("per_frame_translations")
    if per_frame_translations is not None:
        print(f"  Using per-frame translations ({len(per_frame_translations)} frames)")
    else:
        print("  No per_frame_translations in placement.json — using constant translation")

    # --- Concatenate non-position arrays and move to GPU ---
    print("\n--- Building GPU tensors ---")
    f_dc_all  = np.concatenate([f_dc_bg,  f_dc_fg ], axis=0)
    op_all    = np.concatenate([op_bg,    op_fg   ], axis=0)
    rot_all   = np.concatenate([rot_bg,   rot_fg  ], axis=0)
    scales_bg = np.exp(log_sc_bg)
    scales_all = np.concatenate([scales_bg, scales_fg_world], axis=0)

    rgb_t    = torch.from_numpy(np.clip(f_dc_all * C0 + 0.5, 0.0, 1.0)).float().to(device)
    alpha_t  = torch.from_numpy(1.0 / (1.0 + np.exp(-op_all))).float().to(device).unsqueeze(1)
    scales_t = torch.from_numpy(scales_all).float().to(device)
    rot_t    = torch.from_numpy(rot_all).float().to(device)

    xyz_bg_t    = torch.from_numpy(xyz_bg).float().to(device)
    xyz_fg_rot_t = torch.from_numpy(xyz_fg_rot).float().to(device)   # rotated+scaled, no translation

    print(f"  Total Gaussians: {len(f_dc_all):,}  (bg={len(xyz_bg):,}  fg={len(xyz_fg):,})")
    print(f"  GPU memory: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB allocated")

    # --- Render ---
    print(f"\n--- Rendering {T} frames at {W}×{H} ---")
    frames = []
    fg_positions_world_list = []
    R_fg_t = torch.from_numpy(R_fg).float().to(device)
    translation_const = torch.from_numpy(translation).float().to(device)

    for t in range(T):
        # Per-frame translation (or fall back to constant frame-0 translation)
        if per_frame_translations is not None and t < len(per_frame_translations):
            trans_t = torch.tensor(per_frame_translations[t], dtype=torch.float32, device=device)
        else:
            trans_t = translation_const

        # Rotated+scaled canonical positions + rotated offsets + per-frame translation
        if offsets is not None:
            off_world = (R_fg_t @ (torch.from_numpy(offsets[t] * scale).float().to(device)).T).T
            fg_pos = xyz_fg_rot_t + off_world + trans_t
        else:
            fg_pos = xyz_fg_rot_t + trans_t

        fg_positions_world_list.append(fg_pos.cpu().numpy())

        means3D = torch.cat([xyz_bg_t, fg_pos], dim=0)

        img = render_frame(
            means3D, rgb_t, alpha_t, scales_t, rot_t,
            viewmatrix, full_proj, campos, tanfovx, tanfovy, W, H, device,
        )
        frames.append(img)

        if t % 10 == 0:
            print(f"  Frame {t}/{T}")

    # --- Save ---
    import imageio
    imageio.imwrite(out_frame0, frames[0])
    print(f"\nFrame 0: {out_frame0}")

    imageio.mimsave(out_video, frames, fps=args.fps)
    print(f"Video:   {out_video}")
    print(f"  {T} frames @ {args.fps} fps  {W}×{H}")

    fg_positions_world = np.stack(fg_positions_world_list, axis=0)  # (T, N_fg, 3)
    out_positions = os.path.join(gaussians_dir, "fg_positions_world.npy")
    np.save(out_positions, fg_positions_world)
    print(f"FG positions: {out_positions}  shape={fg_positions_world.shape}  dtype={fg_positions_world.dtype}")


if __name__ == "__main__":
    main()
