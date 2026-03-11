"""
refine_sequential.py — Sequential causal frame-by-frame rigid refinement.

Takes the per-frame ActionMesh 4DGS positions (fg_positions_world.npy) and learns a
per-frame rigid correction (delta_q, delta_t) to composite the object into the scene,
warm-started from the previous frame's solution.

Per-frame base positions come from fg_positions_world.npy (create_composite_4dgs.py),
which contains the ActionMesh animated Gaussian positions placed roughly in world space
using depth-estimated translations.  The sequential rigid corrections fix that placement.

Frame 0 initialisation: delta_t is set to the centroid offset between the rough placement
(fg_positions_world[0]) and the refined placement (xyz_world_init from placement_refined.json),
delta_q = identity.  This bootstraps the optimisation near the refine_frame0.py solution
without needing R_orig.  Frame t > 0 warm-starts from frame t-1's solution.

Inputs:
  - placement_refined.json from refine_frame0.py
  - gaussians/gaussians.ply (fg Gaussian attributes)
  - gaussians/fg_positions_world.npy (T, N_fg, 3) from create_composite_4dgs.py
  - WAN target frames   (composite_path/frames/00000.jpg – 00080.jpg)
  - SAM2 per-frame masks (composite_path/sam2_masks/00000.png – 00080.png)
  - Background 3DGS scene + model

Per-frame transform at frame t:
  base_t       = fg_positions_world[t]
  centroid_t   = base_t.mean(0)
  R_delta_t    = quat_to_matrix(normalise(delta_q[t]))
  new_pos_t    = (R_delta_t @ (base_t - centroid_t).T).T + centroid_t + delta_t[t]

Outputs saved to composite_path/gaussians/:
  fg_positions_world_sequential.npy  (T, N_fg, 3)

Outputs saved to output_path/:
  delta_t_sequential.npy  (T, 3)
  delta_q_sequential.npy  (T, 4)
  val/frame_{t:05d}.png   — 5-panel render after each frame's optimisation

Usage:
  python refine_sequential.py \\
    --composite_path output/2026.03.03/actionmesh_gs_replace_corgi \\
    --placement_path output/2026.03.09/corgi_refined_consistent_occl/placement_refined.json \\
    --gs_scene_path  Inpaint360GS/data/inpaint360/bag \\
    --gs_model_path  output/2026.02.26/inpainted_scene/point_cloud_object_inpaint_virtual \\
    --camera_idx     28 \\
    --output_path    output/2026.03.11/corgi_sequential \\
    --n_steps_per_frame 50 --n_frames 15 \\
    --rgb_weight 1.0 --silhouette_weight 1.0 --occlusion_weight 50.0 \\
    --render_scale 0.5
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
import torch.nn.functional as F
from PIL import Image
import imageio
from tqdm import tqdm
from plyfile import PlyData

from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from camera_utils import make_raster_camera_at_resolution

C0 = 0.28209479177387814


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def quat_to_matrix(q):
    """Unit quaternion [w, x, y, z] → rotation matrix (3,3). Normalises q first."""
    q = q / q.norm()
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack([
        1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y),
            2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x),
            2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ]).reshape(3, 3)


def apply_rigid_frame(fg_pos_base, delta_q_t, delta_t_t):
    """
    Apply rigid transform around the centroid of fg_pos_base.

    fg_pos_base : (N_fg, 3) — no grad (frame-0 world-space positions)
    delta_q_t   : (4,)      — has grad (rotation quaternion)
    delta_t_t   : (3,)      — has grad (world-space translation)

    Returns: (N_fg, 3) — gradient flows through delta_q_t and delta_t_t.
    """
    q_norm   = delta_q_t / delta_q_t.norm()
    R        = quat_to_matrix(q_norm)               # (3, 3)
    centroid = fg_pos_base.mean(0)                  # (3,), no grad
    centered = fg_pos_base - centroid               # (N_fg, 3), no grad
    new_pos  = centered @ R.T + centroid + delta_t_t
    return new_pos


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def load_all_colmap_cameras(scene_path):
    """Return list of (pose_w2c 4×4, FoVx, FoVy, W, H) for all COLMAP cameras."""
    images_bin  = os.path.join(scene_path, "sparse", "0", "images.bin")
    cameras_bin = os.path.join(scene_path, "sparse", "0", "cameras.bin")
    extrinsics  = read_extrinsics_binary(images_bin)
    intrinsics  = read_intrinsics_binary(cameras_bin)
    sorted_imgs = sorted(extrinsics.values(), key=lambda x: x.name)

    result = []
    for img in sorted_imgs:
        cam = intrinsics[img.camera_id]
        R_w2c = qvec2rotmat(img.qvec)
        tvec  = np.array(img.tvec, dtype=np.float64)
        W, H  = int(cam.width), int(cam.height)
        if cam.model in ("PINHOLE", "OPENCV"):
            fx, fy = cam.params[0], cam.params[1]
        elif cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = cam.params[0]
        else:
            raise ValueError(f"Unsupported camera model: {cam.model}")
        FoVx = 2.0 * math.atan(W / (2.0 * fx))
        FoVy = 2.0 * math.atan(H / (2.0 * fy))
        pose_w2c = np.eye(4, dtype=np.float64)
        pose_w2c[:3, :3] = R_w2c
        pose_w2c[:3,  3] = tvec
        result.append((pose_w2c, FoVx, FoVy, W, H))
    return result


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
    v      = PlyData.read(ply_path).elements[0]
    xyz    = np.stack([v['x'],       v['y'],       v['z']      ], axis=1).astype(np.float32)
    f_dc   = np.stack([v['f_dc_0'],  v['f_dc_1'],  v['f_dc_2'] ], axis=1).astype(np.float32)
    opacity = v['opacity'].astype(np.float32)
    log_sc = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1).astype(np.float32)
    rot    = np.stack([v['rot_0'],   v['rot_1'],   v['rot_2'],   v['rot_3']], axis=1).astype(np.float32)
    return xyz, f_dc, opacity, log_sc, rot


# ---------------------------------------------------------------------------
# Differentiable rendering
# ---------------------------------------------------------------------------

def render_diff(means3D, colors_t, alpha_t, scales_t, rot_t,
                viewmatrix, full_proj, campos, tanfovx, tanfovy, W, H, device):
    """
    Differentiable render using diff_gaussian_rasterization.
    Returns: rgb (3,H,W), depth (1,H,W), None.
    """
    N = means3D.shape[0]
    means2D = torch.zeros(N, 3, device=device, dtype=torch.float32, requires_grad=True)

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

    color, _, depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_t,
        opacities=alpha_t,
        scales=scales_t,
        rotations=rot_t,
        cov3D_precomp=None,
    )
    return color, depth, None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequential causal frame-by-frame rigid refinement from refine_frame0 output."
    )
    parser.add_argument("--composite_path", required=True,
                        help="actionmesh output dir (contains gaussians/, frames/, sam2_masks/)")
    parser.add_argument("--placement_path", required=True,
                        help="placement_refined.json from refine_frame0.py")
    parser.add_argument("--gs_scene_path", required=True,
                        help="Inpaint360GS scene dir (sparse/0/)")
    parser.add_argument("--gs_model_path", required=True,
                        help="Inpaint360GS model dir (iteration_N/point_cloud.ply)")
    parser.add_argument("--camera_idx", type=int, default=28,
                        help="COLMAP camera index for source view (default 28)")
    parser.add_argument("--fg_positions_world_path", default=None,
                        help="Path to fg_positions_world.npy from create_composite_4dgs.py "
                             "(default: composite_path/gaussians/fg_positions_world.npy)")
    parser.add_argument("--frames_dir", default=None,
                        help="WAN target frames directory (default: composite_path/frames/)")
    parser.add_argument("--masks_dir", default=None,
                        help="SAM2 per-frame masks directory (default: composite_path/sam2_masks/)")
    parser.add_argument("--output_path", required=True,
                        help="Output directory for corrections, val renders")
    # Optimisation
    parser.add_argument("--n_steps_per_frame", type=int, default=50,
                        help="Adam steps per frame (default 50)")
    parser.add_argument("--n_frames", type=int, default=None,
                        help="Only process first N frames (default: all T frames)")
    parser.add_argument("--lr",     type=float, default=1e-3,
                        help="Base learning rate — used if --lr_t / --lr_rot not set (default 1e-3)")
    parser.add_argument("--lr_t",   type=float, default=None,
                        help="Learning rate for delta_t (default: 10× --lr)")
    parser.add_argument("--lr_rot", type=float, default=None,
                        help="Learning rate for delta_q (default: --lr)")
    parser.add_argument("--render_scale", type=float, default=0.5,
                        help="Render scale relative to WAN frame resolution (default 0.5)")
    # Loss weights
    parser.add_argument("--rgb_weight",        type=float, default=1.0)
    parser.add_argument("--silhouette_weight", type=float, default=1.0)
    parser.add_argument("--occlusion_weight",  type=float, default=50.0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Val render helper
# ---------------------------------------------------------------------------

def save_val_frame(t, fg_pos_base, delta_q_t, delta_t_t,
                   xyz_bg_t, colors_bg_t, alpha_bg_t, rot_bg_t, scales_bg_t,
                   colors_fg_t, alpha_fg_t, rot_fg_t, scales_fg_t,
                   N_bg, wan_frames_t, fg_masks_t,
                   src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
                   rW, rH, device, val_dir):
    """Render and save a 5-panel val image for frame t."""
    with torch.no_grad():
        new_pos_v = apply_rigid_frame(fg_pos_base, delta_q_t, delta_t_t)

        means3D_v = torch.cat([xyz_bg_t,    new_pos_v],     dim=0)
        colors_v  = torch.cat([colors_bg_t, colors_fg_t],  dim=0)
        alphas_v  = torch.cat([alpha_bg_t,  alpha_fg_t],   dim=0)
        rots_v    = torch.cat([rot_bg_t,    rot_fg_t],     dim=0)
        scales_v  = torch.cat([scales_bg_t, scales_fg_t],  dim=0)

        color_v, _, _ = render_diff(
            means3D_v, colors_v, alphas_v, scales_v, rots_v,
            src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
            rW, rH, device,
        )
        rendered_np = (color_v.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Silhouette
        sil_cols_v = torch.zeros_like(colors_v)
        sil_cols_v[N_bg:] = 1.0
        sil_render_v, _, _ = render_diff(
            means3D_v, sil_cols_v, alphas_v, scales_v, rots_v,
            src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
            rW, rH, device,
        )
        sil_np  = (sil_render_v.mean(0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        sil_rgb = np.stack([sil_np] * 3, axis=-1)

        # FG-only
        fg_only_v, _, _ = render_diff(
            new_pos_v, colors_fg_t, alpha_fg_t, scales_fg_t, rot_fg_t,
            src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
            rW, rH, device,
        )
        fg_only_np = (fg_only_v.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # GT mask
        gt_mask_np  = (fg_masks_t[t].cpu().numpy() > 0.5).astype(np.uint8) * 255
        gt_mask_rgb = np.stack([gt_mask_np] * 3, axis=-1)

        # WAN target
        target_np = (wan_frames_t[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        panels  = [rendered_np, target_np, sil_rgb, gt_mask_rgb, fg_only_np]
        val_img = np.concatenate(panels, axis=1)
        imageio.imwrite(os.path.join(val_dir, f"frame_{t:05d}.png"), val_img)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_path, exist_ok=True)
    val_dir = os.path.join(args.output_path, "val")
    os.makedirs(val_dir, exist_ok=True)

    gaussians_dir     = os.path.join(args.composite_path, "gaussians")
    frames_dir        = args.frames_dir or os.path.join(args.composite_path, "frames")
    masks_dir         = args.masks_dir  or os.path.join(args.composite_path, "sam2_masks")
    fg_ply_path       = os.path.join(gaussians_dir, "gaussians.ply")
    fg_pos_world_path = args.fg_positions_world_path or os.path.join(
        gaussians_dir, "fg_positions_world.npy"
    )

    lr_t   = args.lr_t   if args.lr_t   is not None else args.lr * 10
    lr_rot = args.lr_rot if args.lr_rot is not None else args.lr

    # -----------------------------------------------------------------------
    # Placement — R, t, s from refine_frame0.py
    # -----------------------------------------------------------------------
    print("\n--- Loading placement_refined.json ---")
    with open(args.placement_path) as f:
        placement = json.load(f)
    t_base = np.array(placement["translation"], dtype=np.float32)
    s_base = float(placement["scale"])
    R_base = np.array(placement["rotation"], dtype=np.float32)   # (3, 3)
    print(f"  t={t_base}  s={s_base:.4f}")

    # -----------------------------------------------------------------------
    # Load fg Gaussian attributes
    # -----------------------------------------------------------------------
    print("\n--- Loading foreground Gaussians ---")
    xyz_fg_canon, f_dc_fg, op_fg, log_sc_fg, rot_fg = load_ply_gs(fg_ply_path)
    N_fg = len(xyz_fg_canon)
    print(f"  {N_fg:,} Gaussians")

    # Frame-0 world positions: R @ (s * xyz_canon) + t
    xyz_world_init_np = (R_base @ (xyz_fg_canon * s_base).T).T + t_base  # (N_fg, 3)

    # -----------------------------------------------------------------------
    # Load background 3DGS
    # -----------------------------------------------------------------------
    print("\n--- Loading background 3DGS ---")
    bg_ply = find_latest_ply(args.gs_model_path)
    print(f"  PLY: {bg_ply}")
    xyz_bg, f_dc_bg, op_bg, log_sc_bg, rot_bg = load_ply_gs(bg_ply)
    N_bg = len(xyz_bg)
    print(f"  {N_bg:,} Gaussians")

    # -----------------------------------------------------------------------
    # GPU tensors
    # -----------------------------------------------------------------------
    print("\n--- Building GPU tensors ---")
    colors_bg_t = torch.from_numpy(np.clip(f_dc_bg * C0 + 0.5, 0.0, 1.0)).float().to(device)
    alpha_bg_t  = torch.from_numpy(1.0 / (1.0 + np.exp(-op_bg))).float().to(device).unsqueeze(1)
    rot_bg_t    = torch.from_numpy(rot_bg).float().to(device)
    scales_bg_t = torch.from_numpy(np.exp(log_sc_bg)).float().to(device)
    xyz_bg_t    = torch.from_numpy(xyz_bg).float().to(device)

    colors_fg_t = torch.from_numpy(np.clip(f_dc_fg * C0 + 0.5, 0.0, 1.0)).float().to(device)
    alpha_fg_t  = torch.from_numpy(1.0 / (1.0 + np.exp(-op_fg))).float().to(device).unsqueeze(1)
    rot_fg_t    = torch.from_numpy(rot_fg).float().to(device)
    scales_fg_t = torch.from_numpy(np.exp(log_sc_fg) * s_base).float().to(device)

    # Frame-0 refined world positions — used to bootstrap frame-0 initialisation
    xyz_world_init_t = torch.from_numpy(xyz_world_init_np).float().to(device)  # (N_fg, 3)

    print(f"  Total: {N_bg + N_fg:,}  (bg={N_bg:,}  fg={N_fg:,})")

    # -----------------------------------------------------------------------
    # Load per-frame ActionMesh positions (T, N_fg, 3)
    # -----------------------------------------------------------------------
    print(f"\n--- Loading fg_positions_world.npy ---")
    if not os.path.exists(fg_pos_world_path):
        raise FileNotFoundError(
            f"Not found: {fg_pos_world_path}\n"
            "Run create_composite_4dgs.py first."
        )
    fg_pos_world_np = np.load(fg_pos_world_path)   # (T_anim, N_fg, 3)
    assert fg_pos_world_np.shape[1] == N_fg, (
        f"N_fg mismatch: PLY has {N_fg}, fg_positions_world has {fg_pos_world_np.shape[1]}"
    )
    T_anim = fg_pos_world_np.shape[0]
    print(f"  Shape: {fg_pos_world_np.shape}")

    # -----------------------------------------------------------------------
    # Source camera
    # -----------------------------------------------------------------------
    print("\n--- Loading source camera ---")
    all_cams = load_all_colmap_cameras(args.gs_scene_path)
    if args.camera_idx >= len(all_cams):
        raise ValueError(f"camera_idx {args.camera_idx} out of range ({len(all_cams)})")
    src_pose_w2c, src_FoVx, _, _, _ = all_cams[args.camera_idx]

    # -----------------------------------------------------------------------
    # Load WAN frames and SAM2 masks
    # -----------------------------------------------------------------------
    print("\n--- Loading WAN frames and SAM2 masks ---")
    frame_files = sorted(
        glob.glob(os.path.join(frames_dir, "*.jpg")) +
        glob.glob(os.path.join(frames_dir, "*.png"))
    )
    mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))

    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    if not mask_files:
        raise FileNotFoundError(f"No masks found in {masks_dir}")

    W_wan, H_wan = Image.open(frame_files[0]).size
    rW = max(1, round(W_wan * args.render_scale))
    rH = max(1, round(H_wan * args.render_scale))
    print(f"  WAN frame size: {W_wan}×{H_wan} → render {rW}×{rH}")

    T = min(len(frame_files), len(mask_files), T_anim)
    if args.n_frames is not None:
        T = min(T, args.n_frames)
    print(f"  Processing {T} frames")

    # GPU tensor for per-frame base positions
    fg_pos_world_t = torch.from_numpy(fg_pos_world_np[:T]).float().to(device)  # (T, N_fg, 3)

    wan_frames_t = []
    fg_masks_t   = []
    for i in range(T):
        img   = Image.open(frame_files[i]).convert("RGB").resize((rW, rH), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
        wan_frames_t.append(img_t)

        mask   = Image.open(mask_files[i]).convert("L").resize((rW, rH), Image.NEAREST)
        mask_t = torch.from_numpy((np.array(mask) > 128).astype(np.float32)).to(device)
        fg_masks_t.append(mask_t)

    print(f"  Loaded {T} frames and masks.")

    src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy = \
        make_raster_camera_at_resolution(src_pose_w2c, src_FoVx, rW, rH, device)
    print(f"  Source cam {args.camera_idx}: fovx={math.degrees(src_FoVx):.1f}°")

    # -----------------------------------------------------------------------
    # Pre-render background depth (for occlusion loss)
    # -----------------------------------------------------------------------
    bg_depth_color_map = None
    bg_occlusion_mask  = None

    if args.occlusion_weight > 0:
        print("\n--- Pre-rendering background depth ---")
        with torch.no_grad():
            bg_hom      = torch.cat([xyz_bg_t, torch.ones(N_bg, 1, device=device)], dim=1)
            bg_z        = (bg_hom @ src_viewmat)[:, 2]
            bg_z_colors = bg_z.unsqueeze(1).expand(-1, 3).float()
            bg_depth_render, _, _ = render_diff(
                xyz_bg_t, bg_z_colors, alpha_bg_t, scales_bg_t, rot_bg_t,
                src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
                rW, rH, device,
            )
        bg_depth_color_map = bg_depth_render[0].detach()
        bg_occlusion_mask  = (bg_depth_color_map > 0).float()
        print(f"  BG depth: non-zero px = {bg_occlusion_mask.sum().int().item()}")

    # -----------------------------------------------------------------------
    # Sequential per-frame optimisation
    # -----------------------------------------------------------------------
    print(f"\n--- Sequential optimisation: {T} frames × {args.n_steps_per_frame} steps ---")
    print(f"  lr_t={lr_t}  lr_rot={lr_rot}")

    # Frame 0 initialisation: centroid offset from rough to refined placement.
    # This puts the object near the refine_frame0.py solution without needing R_orig.
    centroid_rough   = fg_pos_world_t[0].mean(0)   # centroid under rough placement
    centroid_refined = xyz_world_init_t.mean(0)    # centroid under refined placement
    init_delta_t     = (centroid_refined - centroid_rough).detach()
    init_delta_q     = torch.tensor([1., 0., 0., 0.], device=device)

    all_delta_t = []
    all_delta_q = []

    for t in tqdm(range(T), desc="Sequential frames"):
        if t == 0:
            delta_t = torch.nn.Parameter(init_delta_t.clone())
            delta_q = torch.nn.Parameter(init_delta_q.clone())
        else:
            delta_t = torch.nn.Parameter(prev_delta_t.clone())
            delta_q = torch.nn.Parameter(prev_delta_q.clone())

        optimizer = torch.optim.Adam([
            {"params": [delta_t], "lr": lr_t},
            {"params": [delta_q], "lr": lr_rot},
        ])

        fg_pos_base = fg_pos_world_t[t]   # (N_fg, 3), per-frame ActionMesh positions
        target_t    = wan_frames_t[t]
        mask_t      = fg_masks_t[t]

        for _ in range(args.n_steps_per_frame):
            optimizer.zero_grad()

            new_pos = apply_rigid_frame(fg_pos_base, delta_q, delta_t)

            means3D = torch.cat([xyz_bg_t,    new_pos],      dim=0)
            colors  = torch.cat([colors_bg_t, colors_fg_t], dim=0)
            alphas  = torch.cat([alpha_bg_t,  alpha_fg_t],  dim=0)
            rots    = torch.cat([rot_bg_t,    rot_fg_t],    dim=0)
            scales  = torch.cat([scales_bg_t, scales_fg_t], dim=0)

            color_render, _, _ = render_diff(
                means3D, colors, alphas, scales, rots,
                src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
                rW, rH, device,
            )
            rgb_render = color_render.clamp(0, 1)

            # Photo L1 (fg-masked)
            total_loss = args.rgb_weight * (
                (rgb_render - target_t).abs() * mask_t.unsqueeze(0)
            ).mean()

            # Silhouette BCE
            if args.silhouette_weight > 0:
                sil_colors = torch.zeros_like(colors)
                sil_colors[N_bg:] = 1.0
                sil_render, _, _ = render_diff(
                    means3D, sil_colors, alphas, scales, rots,
                    src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
                    rW, rH, device,
                )
                sil_pred = sil_render.mean(0)
                with torch.amp.autocast("cuda", enabled=False):
                    sil_loss = F.binary_cross_entropy(
                        sil_pred.clamp(1e-6, 1 - 1e-6).float(), mask_t.float()
                    )
                total_loss = total_loss + args.silhouette_weight * sil_loss

            # Occlusion: penalise fg Gaussians behind bg
            if args.occlusion_weight > 0 and bg_depth_color_map is not None:
                fg_hom      = torch.cat([new_pos, torch.ones(N_fg, 1, device=device)], dim=1)
                fg_z        = (fg_hom @ src_viewmat)[:, 2]
                fg_z_colors = fg_z.unsqueeze(1).expand(-1, 3).float()
                fg_depth_render, _, _ = render_diff(
                    new_pos, fg_z_colors, alpha_fg_t, scales_fg_t, rot_fg_t,
                    src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
                    rW, rH, device,
                )
                occ_loss = (
                    F.relu(fg_depth_render[0] - bg_depth_color_map) ** 2 * bg_occlusion_mask
                ).sum() / (bg_occlusion_mask.sum() + 1e-6)
                total_loss = total_loss + args.occlusion_weight * occ_loss

            total_loss.backward()
            optimizer.step()

        prev_delta_t = delta_t.detach()
        prev_delta_q = delta_q.detach()
        all_delta_t.append(prev_delta_t.cpu().numpy())
        all_delta_q.append(prev_delta_q.cpu().numpy())

        save_val_frame(
            t, fg_pos_base, prev_delta_q, prev_delta_t,
            xyz_bg_t, colors_bg_t, alpha_bg_t, rot_bg_t, scales_bg_t,
            colors_fg_t, alpha_fg_t, rot_fg_t, scales_fg_t,
            N_bg, wan_frames_t, fg_masks_t,
            src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
            rW, rH, device, val_dir,
        )

    # -----------------------------------------------------------------------
    # Save corrections
    # -----------------------------------------------------------------------
    print("\n--- Saving corrections ---")
    delta_t_arr = np.stack(all_delta_t)   # (T, 3)
    delta_q_arr = np.stack(all_delta_q)   # (T, 4)
    np.save(os.path.join(args.output_path, "delta_t_sequential.npy"), delta_t_arr)
    np.save(os.path.join(args.output_path, "delta_q_sequential.npy"), delta_q_arr)
    print(f"  |Δt| max:     {np.linalg.norm(delta_t_arr, axis=1).max():.4f}")
    print(f"  |Δq - I| max: {np.abs(delta_q_arr - np.array([[1,0,0,0]])).max():.4f}")

    # -----------------------------------------------------------------------
    # Compute and save fg_positions_world_sequential.npy
    # -----------------------------------------------------------------------
    print("\n--- Computing fg_positions_world_sequential.npy ---")
    dynamic_positions = np.zeros((T, N_fg, 3), dtype=np.float32)
    with torch.no_grad():
        for t in range(T):
            dq = torch.from_numpy(delta_q_arr[t]).float().to(device)
            dt = torch.from_numpy(delta_t_arr[t]).float().to(device)
            dynamic_positions[t] = apply_rigid_frame(fg_pos_world_t[t], dq, dt).cpu().numpy()

    seq_path = os.path.join(gaussians_dir, "fg_positions_world_sequential.npy")
    np.save(seq_path, dynamic_positions)
    print(f"  Saved: {seq_path}  shape {dynamic_positions.shape}")

    centroids   = dynamic_positions.mean(axis=1)
    vel_norms   = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    print(f"  Centroid velocity — mean: {vel_norms.mean():.4f},  max: {vel_norms.max():.4f}")

    print("\nDone.")
    print(f"  Sequential positions: {seq_path}")
    print(f"  Val renders:          {val_dir}/")
    print(f"\nNext: render with --fg_positions_path {seq_path}")


if __name__ == "__main__":
    main()
