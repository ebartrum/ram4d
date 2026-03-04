"""
refine_frame0.py — Stage 1: rigid refinement of foreground placement in frame 0.

Optimises 7 parameters: axis-angle Δr (3), translation Δt (3), log-scale Δs (1)
as small corrections applied on top of the coarse placement from placement.json.

Losses (all weighted, only photo is mandatory):
  photo       L1  (fg-weighted)   rendered RGB vs reference frame, source camera
  silhouette  BCE                 rendered fg alpha vs input SAM mask, source camera
  depth       L1  (fg region)     rendered depth vs Marigold prediction (affine-aligned)
  flux        L2  (fg-weighted)   rendered RGB vs Flux Fill pseudo-GT, training cameras

placement_refined.json keys: {translation, scale, rotation} where rotation is a
3×3 matrix. create_composite_4dgs.py reads this if --placement_path is specified.

Usage:
  python refine_frame0.py \\
    --composite_path  output/YYYY.MM.DD/<name>/ \\
    --gs_scene_path   Inpaint360GS/data/inpaint360/bag \\
    --gs_model_path   output/2026.02.26/inpainted_scene/point_cloud_object_inpaint_virtual \\
    --reference_frame data/images/frame0.png \\
    --camera_idx      28 \\
    --mask_path       data/images/corgi_mask.png \\
    --prompt_path     data/captions/corgi.txt \\
    --output_path     output/YYYY.MM.DD/<name>_refined/ \\
    --n_steps         500 \\
    --flux_weight     1.0 \\
    --silhouette_weight 0.1 \\
    --depth_weight    0.1
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import imageio
from plyfile import PlyData

from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

C0 = 0.28209479177387814


# ---------------------------------------------------------------------------
# Differentiable rigid transform helpers
# ---------------------------------------------------------------------------

def rodrigues(r):
    """Axis-angle vector r (3,) → rotation matrix (3,3). Differentiable."""
    theta = r.norm()
    if theta.item() < 1e-8:
        return torch.eye(3, device=r.device, dtype=r.dtype)
    k = r / theta
    K = torch.zeros(3, 3, device=r.device, dtype=r.dtype)
    K[0, 1] = -k[2]; K[0, 2] = k[1]
    K[1, 0] =  k[2]; K[1, 2] = -k[0]
    K[2, 0] = -k[1]; K[2, 1] =  k[0]
    I = torch.eye(3, device=r.device, dtype=r.dtype)
    return I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


def apply_rigid(xyz_canon_t, R_base_t, t_base_t, s_base, delta_r, delta_t, delta_s):
    """
    Apply refined rigid transform to canonical fg positions.
    xyz_canon_t: (N_fg, 3) float32, no grad
    Returns: means3D_fg (N_fg, 3), grad flows through delta_r/delta_t/delta_s.
    """
    R_delta = rodrigues(delta_r)                          # (3,3)
    R = R_delta @ R_base_t                                # (3,3)
    s = s_base * torch.exp(delta_s.squeeze())             # scalar
    # p_world = R @ (s * p_canon) + t_refined
    means3D_fg = (R @ (xyz_canon_t * s).T).T + t_base_t + delta_t   # (N_fg, 3)
    return means3D_fg


# ---------------------------------------------------------------------------
# Camera helpers
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


def load_all_colmap_cameras(scene_path):
    """Return list of (pose_w2c 4x4, FoVx, FoVy, W, H) for all COLMAP cameras."""
    images_bin  = os.path.join(scene_path, "sparse", "0", "images.bin")
    cameras_bin = os.path.join(scene_path, "sparse", "0", "cameras.bin")
    extrinsics = read_extrinsics_binary(images_bin)
    intrinsics = read_intrinsics_binary(cameras_bin)
    sorted_images = sorted(extrinsics.values(), key=lambda x: x.name)

    result = []
    for img in sorted_images:
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


def colmap_world_up_and_cam_center(scene_path, camera_idx):
    """Compute world_up and cam_center for camera_idx, replicating create_composite_4dgs logic."""
    images_bin  = os.path.join(scene_path, "sparse", "0", "images.bin")
    cameras_bin = os.path.join(scene_path, "sparse", "0", "cameras.bin")
    extrinsics = read_extrinsics_binary(images_bin)
    intrinsics = read_intrinsics_binary(cameras_bin)
    sorted_images = sorted(extrinsics.values(), key=lambda x: x.name)

    downs = np.stack([qvec2rotmat(si.qvec).T[:, 1] for si in sorted_images])
    mean_down = downs.mean(axis=0)
    world_up = -mean_down / np.linalg.norm(mean_down)

    img = sorted_images[camera_idx]
    R_c2w = qvec2rotmat(img.qvec).T
    tvec  = np.array(img.tvec, dtype=np.float64)
    cam_center = -(R_c2w @ tvec).astype(np.float32)
    return world_up, cam_center


def compute_fg_rotation(translation, cam_center, world_up, yaw_deg=0.0):
    """Replicate create_composite_4dgs.compute_fg_rotation."""
    world_up = world_up / np.linalg.norm(world_up)
    to_cam = np.array(cam_center, dtype=np.float64) - np.array(translation, dtype=np.float64)
    to_cam_h = to_cam - np.dot(to_cam, world_up) * world_up
    if np.linalg.norm(to_cam_h) < 1e-6:
        to_cam_h = np.array([1.0, 0.0, 0.0])
    world_fwd = to_cam_h / np.linalg.norm(to_cam_h)
    if abs(yaw_deg) > 1e-6:
        angle = math.radians(yaw_deg)
        k = world_up
        c, s = math.cos(angle), math.sin(angle)
        world_fwd = (world_fwd * c + np.cross(k, world_fwd) * s
                     + k * np.dot(k, world_fwd) * (1 - c))
        world_fwd /= np.linalg.norm(world_fwd)
    world_right = np.cross(world_fwd, world_up)
    world_right /= np.linalg.norm(world_right)
    return np.stack([-world_fwd, world_right, world_up], axis=1).astype(np.float32)


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
    Returns: rgb (3,H,W), depth (1,H,W), None — alpha not available from this rasterizer.
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
    return color, depth, None   # (3,H,W), (1,H,W), None


def scale_to_flux_res(W, H, target_long=512):
    """Scale (W, H) so the longest side ≤ target_long, both divisible by 16."""
    scale = target_long / max(W, H)
    W_out = max(16, round(W * scale / 16) * 16)
    H_out = max(16, round(H * scale / 16) * 16)
    return W_out, H_out


# ---------------------------------------------------------------------------
# Marigold depth
# ---------------------------------------------------------------------------

def load_marigold(device):
    from diffusers import MarigoldDepthPipeline
    print("  Loading Marigold (prs-eth/marigold-lcm-v1-0)...")
    pipe = MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-lcm-v1-0",
        torch_dtype=torch.float32,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def predict_marigold(marigold_pipe, rgb_hwc_uint8):
    """
    rgb_hwc_uint8: (H, W, 3) uint8.
    Returns (H, W) float32 affine-invariant depth in [0, 1] (1 = far).
    """
    pil = Image.fromarray(rgb_hwc_uint8)
    with torch.no_grad():
        result = marigold_pipe(pil, num_inference_steps=4, ensemble_size=1)
    # result.prediction: (1, H, W) float32 or similar
    pred = result.prediction
    if hasattr(pred, 'cpu'):
        pred = pred.cpu().numpy()
    pred = np.squeeze(pred).astype(np.float32)
    return pred


def colorize_depth(depth_hw, alpha_hw=None, cmap="turbo"):
    """
    Colorize a (H, W) float32 depth map.
    Normalises over the foreground region (alpha_hw > 0.1) so the full colour
    range is used for the object rather than being washed out by the background.
    Background pixels (outside fg) are shown as mid-grey (128, 128, 128).
    Returns (H, W, 3) uint8.
    """
    import matplotlib.cm as cm
    depth = depth_hw.copy()
    if alpha_hw is not None:
        fg = alpha_hw > 0.1
    else:
        fg = depth > 0

    if fg.sum() > 0:
        lo, hi = depth[fg].min(), depth[fg].max()
    else:
        lo, hi = depth.min(), depth.max()

    denom = hi - lo if hi > lo else 1.0
    norm = np.clip((depth - lo) / denom, 0.0, 1.0)

    colormap = cm.get_cmap(cmap)
    colored = (colormap(norm)[:, :, :3] * 255).astype(np.uint8)  # (H, W, 3)

    # Grey out background where fg is absent
    if alpha_hw is not None:
        bg = ~fg
        colored[bg] = 128

    return colored


def align_depth_affine(depth_pred_np, depth_render_np, fg_mask_np):
    """
    Least-squares: solve (s, b) s.t. s * depth_pred + b ≈ depth_render in fg region.
    Returns aligned_depth (H, W) float32.
    """
    pred_fg = depth_pred_np[fg_mask_np].astype(np.float64)
    rend_fg = depth_render_np[fg_mask_np].astype(np.float64)
    if len(pred_fg) < 10:
        return depth_render_np.copy()  # fallback: no change
    A = np.stack([pred_fg, np.ones_like(pred_fg)], axis=1)
    (s, b), _, _, _ = np.linalg.lstsq(A, rend_fg, rcond=None)
    return (s * depth_pred_np + b).astype(np.float32)


# ---------------------------------------------------------------------------
# Flux Fill pseudo-GT
# ---------------------------------------------------------------------------

def load_flux_fill():
    from diffusers import FluxFillPipeline
    print("  Loading Flux Fill (black-forest-labs/FLUX.1-Fill-dev)...")
    pipe = FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def compute_flux_pseudogt(flux_pipe, bg_rgb_uint8, mask_uint8, prompt, W, H,
                           n_steps=4, seed=42):
    """
    Run Flux Fill on background + mask to get pseudo-GT.
    bg_rgb_uint8: (H, W, 3) uint8 — background image (fg region to be filled)
    mask_uint8: (H, W) uint8, 255 = fill region, 0 = keep
    Returns: (H, W, 3) uint8.
    """
    bg_pil   = Image.fromarray(bg_rgb_uint8)
    mask_pil = Image.fromarray(mask_uint8).convert("L")
    gen = torch.Generator("cpu").manual_seed(seed)
    result = flux_pipe(
        prompt=prompt,
        image=bg_pil,
        mask_image=mask_pil,
        width=W,
        height=H,
        num_inference_steps=n_steps,
        guidance_scale=3.5,
        max_sequence_length=256,
        generator=gen,
    ).images[0]
    return np.array(result)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--composite_path", required=True,
                        help="ram4d output dir (contains gaussians/)")
    parser.add_argument("--gs_scene_path", required=True,
                        help="Inpaint360GS scene dir (sparse/0/)")
    parser.add_argument("--gs_model_path", required=True,
                        help="Inpaint360GS model dir (iteration_N/point_cloud.ply)")
    parser.add_argument("--reference_frame", required=True,
                        help="Reference RGB image for frame 0 (source camera)")
    parser.add_argument("--camera_idx", type=int, default=28,
                        help="COLMAP camera index for the source view (default 28)")
    parser.add_argument("--yaw_deg", type=float, default=0.0,
                        help="Yaw used in create_composite_4dgs (must match, default 0)")
    parser.add_argument("--mask_path", default=None,
                        help="SAM mask PNG (white=fg) for silhouette loss")
    parser.add_argument("--prompt_path", default=None,
                        help="Text prompt file (.txt) for Flux Fill")
    parser.add_argument("--output_path", required=True,
                        help="Output directory for placement_refined.json and debug images")
    # Optimisation
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for all parameters")
    parser.add_argument("--render_scale", type=float, default=0.25,
                        help="Render scale for source camera (default 0.25)")
    # Loss weights
    parser.add_argument("--photo_weight",      type=float, default=1.0)
    parser.add_argument("--silhouette_weight", type=float, default=0.1)
    parser.add_argument("--depth_weight",      type=float, default=0.0,
                        help="Marigold depth loss weight (default 0, loads Marigold if > 0)")
    parser.add_argument("--flux_weight",       type=float, default=0.0,
                        help="Flux pseudo-GT loss weight (default 0, loads Flux if > 0)")
    # Marigold settings
    parser.add_argument("--depth_update_interval", type=int, default=10,
                        help="Update Marigold pseudo-GT every N steps (default 10)")
    # Flux settings
    parser.add_argument("--n_flux_cams", type=int, default=4,
                        help="Number of training cameras for Flux pseudo-GT (default 4)")
    parser.add_argument("--flux_steps", type=int, default=4,
                        help="Flux inference steps for pseudo-GT (default 4)")
    parser.add_argument("--flux_res", type=int, default=512,
                        help="Target long-side resolution for Flux renders (default 512)")
    # Validation
    parser.add_argument("--val_interval", type=int, default=50,
                        help="Save validation image every N steps (default 50)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_path, exist_ok=True)

    gaussians_dir  = os.path.join(args.composite_path, "gaussians")
    placement_path = os.path.join(gaussians_dir, "placement.json")
    fg_ply_path    = os.path.join(gaussians_dir, "gaussians.ply")
    out_json       = os.path.join(args.output_path, "placement_refined.json")
    debug_dir      = os.path.join(args.output_path, "debug")
    val_dir        = os.path.join(args.output_path, "val")
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load placement
    # -----------------------------------------------------------------------
    print("\n--- Loading placement.json ---")
    with open(placement_path) as f:
        placement = json.load(f)
    t_base = np.array(placement["translation"], dtype=np.float32)
    s_base = float(placement["scale"])
    print(f"  translation: {t_base.tolist()}")
    print(f"  scale:       {s_base:.4f}")

    # -----------------------------------------------------------------------
    # Compute base rotation from COLMAP (same logic as create_composite_4dgs)
    # -----------------------------------------------------------------------
    print("\n--- Computing base rotation ---")
    world_up, cam_center = colmap_world_up_and_cam_center(
        args.gs_scene_path, args.camera_idx
    )
    R_base = compute_fg_rotation(t_base, cam_center, world_up, args.yaw_deg)
    print(f"  R_base det={np.linalg.det(R_base):.4f}")

    # -----------------------------------------------------------------------
    # Load foreground Gaussian attributes (canonical positions + appearance)
    # -----------------------------------------------------------------------
    print("\n--- Loading foreground 4DGS ---")
    xyz_fg, f_dc_fg, op_fg, log_sc_fg, rot_fg = load_ply_gs(fg_ply_path)
    N_fg = len(xyz_fg)
    print(f"  {N_fg:,} Gaussians")

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
    # Build GPU attribute tensors (frozen during optimisation)
    # Note: fg scales are kept separate so they can be updated when delta_s changes.
    # -----------------------------------------------------------------------
    print("\n--- Building GPU tensors ---")
    f_dc_all = np.concatenate([f_dc_bg, f_dc_fg], axis=0)
    op_all   = np.concatenate([op_bg,   op_fg],   axis=0)
    rot_all  = np.concatenate([rot_bg,  rot_fg],  axis=0)

    colors_t = torch.from_numpy(
        np.clip(f_dc_all * C0 + 0.5, 0.0, 1.0)
    ).float().to(device)
    alpha_t  = torch.from_numpy(
        1.0 / (1.0 + np.exp(-op_all))
    ).float().to(device).unsqueeze(1)
    rot_t    = torch.from_numpy(rot_all).float().to(device)

    # Bg scales fixed; fg scales base (without placement scale, applied per step)
    scales_bg_t      = torch.from_numpy(np.exp(log_sc_bg)).float().to(device)   # (N_bg, 3)
    scales_fg_base_t = torch.from_numpy(np.exp(log_sc_fg)).float().to(device)   # (N_fg, 3)

    # Fixed bg positions (no grad)
    xyz_bg_t   = torch.from_numpy(xyz_bg).float().to(device)
    # Canonical fg positions (no grad — transform is applied via delta params)
    xyz_fg_t   = torch.from_numpy(xyz_fg).float().to(device)

    # Base transform as tensors
    R_base_t = torch.from_numpy(R_base).float().to(device)
    t_base_t = torch.from_numpy(t_base).float().to(device)

    print(f"  Total: {N_bg + N_fg:,}  (bg={N_bg:,}  fg={N_fg:,})")

    # -----------------------------------------------------------------------
    # Source camera
    # -----------------------------------------------------------------------
    print("\n--- Loading source camera ---")
    all_cams = load_all_colmap_cameras(args.gs_scene_path)
    if args.camera_idx >= len(all_cams):
        raise ValueError(f"camera_idx {args.camera_idx} out of range ({len(all_cams)})")
    src_pose_w2c, src_FoVx, src_FoVy, src_W, src_H = all_cams[args.camera_idx]
    rW = max(1, int(src_W * args.render_scale))
    rH = max(1, int(src_H * args.render_scale))
    src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy = make_raster_camera(
        src_pose_w2c, src_FoVx, src_FoVy, rW, rH, device
    )
    print(f"  Source cam {args.camera_idx}: {src_W}×{src_H} → render {rW}×{rH}")

    # -----------------------------------------------------------------------
    # Reference frame (resized to render resolution)
    # -----------------------------------------------------------------------
    print("\n--- Loading reference frame ---")
    ref_img = np.array(Image.open(args.reference_frame).convert("RGB"))
    ref_img_r = np.array(
        Image.fromarray(ref_img).resize((rW, rH), Image.BILINEAR)
    )
    ref_t = torch.from_numpy(ref_img_r.astype(np.float32) / 255.0).to(device)  # (H, W, 3)
    ref_t = ref_t.permute(2, 0, 1)  # (3, H, W)
    print(f"  Reference: {ref_img.shape[:2]} → {rH}×{rW}")

    # -----------------------------------------------------------------------
    # GT silhouette mask (source camera)
    # -----------------------------------------------------------------------
    mask_src_t = None
    if args.silhouette_weight > 0 and args.mask_path:
        print("\n--- Loading silhouette mask ---")
        mask_np = np.array(Image.open(args.mask_path).convert("L"))
        mask_np_r = np.array(
            Image.fromarray(mask_np).resize((rW, rH), Image.NEAREST)
        )
        fg_mask_gt = (mask_np_r > 128).astype(np.float32)
        mask_src_t = torch.from_numpy(fg_mask_gt).float().to(device)  # (H, W)
        print(f"  Mask fg pixels: {int(fg_mask_gt.sum())} / {rW*rH}")

    # -----------------------------------------------------------------------
    # Flux pseudo-GT precomputation
    # -----------------------------------------------------------------------
    flux_pseudogt_targets = []  # list of (pose_w2c, FoVx, FoVy, W_f, H_f, pseudogt_t)

    if args.flux_weight > 0:
        if args.prompt_path is None:
            raise ValueError("--prompt_path required when --flux_weight > 0")
        with open(args.prompt_path) as f:
            prompt = f.read().strip()
        print(f"\n--- Precomputing Flux pseudo-GT for {args.n_flux_cams} cameras ---")

        # Select training cameras (excluding source), uniformly spaced
        train_indices = [i for i in range(len(all_cams)) if i != args.camera_idx]
        step = max(1, len(train_indices) // args.n_flux_cams)
        selected = train_indices[::step][:args.n_flux_cams]
        print(f"  Selected camera indices: {selected}")

        flux_pipe = load_flux_fill()

        # Initial fg positions at base transform for pseudo-GT rendering
        with torch.no_grad():
            delta_r0 = torch.zeros(3, device=device)
            delta_t0 = torch.zeros(3, device=device)
            delta_s0 = torch.zeros(1, device=device)
            means3D_fg_init = apply_rigid(
                xyz_fg_t, R_base_t, t_base_t, s_base, delta_r0, delta_t0, delta_s0
            )
            means3D_init = torch.cat([xyz_bg_t, means3D_fg_init], dim=0)
            scales_init  = torch.cat([scales_bg_t, scales_fg_base_t * s_base], dim=0)

        for cam_i in selected:
            pose_w2c_i, FoVx_i, FoVy_i, W_i, H_i = all_cams[cam_i]
            W_f, H_f = scale_to_flux_res(W_i, H_i, args.flux_res)
            viewmat_i, fullproj_i, campos_i, tfovx_i, tfovy_i = make_raster_camera(
                pose_w2c_i, FoVx_i, FoVy_i, W_f, H_f, device
            )

            # Render full composite at initial placement to get fg alpha mask
            with torch.no_grad():
                color_i, depth_i, alpha_i = render_diff(
                    means3D_init, colors_t, alpha_t, scales_init, rot_t,
                    viewmat_i, fullproj_i, campos_i, tfovx_i, tfovy_i,
                    W_f, H_f, device,
                )
            # alpha_i is None (rasterizer doesn't return alpha); use depth > 0 as fg proxy
            if alpha_i is not None:
                fg_alpha_np = alpha_i.squeeze(0).cpu().numpy()
            else:
                fg_alpha_np = (depth_i.squeeze(0).cpu().numpy() > 0).astype(np.float32)
            fg_mask_np = (fg_alpha_np > 0.1).astype(np.uint8) * 255

            # Render background only (fg replaced by zeros via mask)
            rgb_full_np = (color_i.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            # Zero out fg region to get background image
            fg_mask_bool = fg_mask_np > 128
            bg_rgb_np = rgb_full_np.copy()
            bg_rgb_np[fg_mask_bool] = 0  # black; Flux Fill will inpaint this

            print(f"  Cam {cam_i}: {W_f}×{H_f}  fg px: {fg_mask_bool.sum()}")
            pseudogt_np = compute_flux_pseudogt(
                flux_pipe, bg_rgb_np, fg_mask_np, prompt, W_f, H_f,
                n_steps=args.flux_steps,
            )
            pseudogt_t = torch.from_numpy(
                pseudogt_np.astype(np.float32) / 255.0
            ).to(device).permute(2, 0, 1)  # (3, H, W)

            # Store camera for per-step rendering
            flux_pseudogt_targets.append({
                "pose_w2c": pose_w2c_i,
                "FoVx": FoVx_i, "FoVy": FoVy_i,
                "W": W_f, "H": H_f,
                "target": pseudogt_t,
            })
            # Save debug image
            imageio.imwrite(
                os.path.join(debug_dir, f"flux_pseudogt_cam{cam_i}.png"), pseudogt_np
            )

        # Free Flux to recover memory
        del flux_pipe
        torch.cuda.empty_cache()
        print(f"  Flux pseudo-GT precomputed for {len(flux_pseudogt_targets)} cameras.")

    # -----------------------------------------------------------------------
    # Marigold
    # -----------------------------------------------------------------------
    marigold_pipe = None
    if args.depth_weight > 0:
        marigold_pipe = load_marigold(device)

    # -----------------------------------------------------------------------
    # Rigid parameters + optimiser
    # -----------------------------------------------------------------------
    print("\n--- Initialising optimisation ---")
    delta_r = nn.Parameter(torch.zeros(3, device=device))
    delta_t = nn.Parameter(torch.zeros(3, device=device))
    delta_s = nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.Adam([delta_r, delta_t, delta_s], lr=args.lr)

    depth_target_t = None   # cached Marigold pseudo-GT (updated every depth_update_interval)
    best_total = float("inf")
    best_state = {"delta_r": delta_r.data.clone(),
                  "delta_t": delta_t.data.clone(),
                  "delta_s": delta_s.data.clone()}

    # -----------------------------------------------------------------------
    # Optimisation loop
    # -----------------------------------------------------------------------
    print(f"\n--- Optimising {args.n_steps} steps ---")
    for step in range(args.n_steps):
        optimizer.zero_grad()

        # Compute refined fg positions and scales (both depend on delta_s)
        means3D_fg = apply_rigid(
            xyz_fg_t, R_base_t, t_base_t, s_base, delta_r, delta_t, delta_s
        )
        means3D    = torch.cat([xyz_bg_t, means3D_fg], dim=0)
        # Fg Gaussian extents also scale with s_refined (same factor as positions)
        s_refined  = s_base * torch.exp(delta_s.squeeze())
        scales_t   = torch.cat([scales_bg_t, scales_fg_base_t * s_refined], dim=0)

        # ---- Render from source camera ----
        color_src, depth_src, alpha_src = render_diff(
            means3D, colors_t, alpha_t, scales_t, rot_t,
            src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
            rW, rH, device,
        )
        rgb_src = color_src.clamp(0, 1)  # (3, H, W)

        # ---- Photo loss (global L1, or fg-weighted if alpha available) ----
        photo_loss = (rgb_src - ref_t).abs()  # (3, H, W)
        if alpha_src is not None:
            fg_weight = alpha_src.detach().clamp(0, 1)  # (1, H, W), no grad
            photo_loss = (photo_loss * fg_weight).mean()
        else:
            photo_loss = photo_loss.mean()

        total_loss = args.photo_weight * photo_loss

        # ---- Silhouette BCE (requires alpha — skipped without it) ----
        sil_loss = torch.tensor(0.0, device=device)
        if args.silhouette_weight > 0 and mask_src_t is not None and alpha_src is not None:
            alpha_sq = alpha_src.squeeze(0).clamp(1e-6, 1 - 1e-6)  # (H, W)
            sil_loss = F.binary_cross_entropy(alpha_sq, mask_src_t)
            total_loss = total_loss + args.silhouette_weight * sil_loss

        # ---- Marigold depth loss ----
        depth_loss = torch.tensor(0.0, device=device)
        if args.depth_weight > 0 and marigold_pipe is not None:
            if step % args.depth_update_interval == 0:
                with torch.no_grad():
                    rgb_np = (rgb_src.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    depth_pred_np = predict_marigold(marigold_pipe, rgb_np)
                    depth_render_np = depth_src.detach().cpu().squeeze().numpy()
                    if alpha_src is not None:
                        fg_mask_d = (alpha_src.detach().cpu().squeeze().numpy() > 0.1)
                    else:
                        fg_mask_d = depth_render_np > 0
                    aligned_np = align_depth_affine(depth_pred_np, depth_render_np, fg_mask_d)
                    depth_target_t = torch.from_numpy(aligned_np).float().to(device).unsqueeze(0)

            if depth_target_t is not None:
                if alpha_src is not None:
                    fg_weight_d = (alpha_src.detach() > 0.1).float()  # (1, H, W)
                else:
                    fg_weight_d = (depth_src.detach() > 0).float()
                depth_loss = ((depth_src - depth_target_t.detach()).abs() * fg_weight_d).mean()
                total_loss = total_loss + args.depth_weight * depth_loss

        # ---- Flux pseudo-GT loss (cycle through training cameras) ----
        flux_loss = torch.tensor(0.0, device=device)
        if args.flux_weight > 0 and flux_pseudogt_targets:
            cam_info = flux_pseudogt_targets[step % len(flux_pseudogt_targets)]
            viewmat_f, fullproj_f, campos_f, tfovx_f, tfovy_f = make_raster_camera(
                cam_info["pose_w2c"], cam_info["FoVx"], cam_info["FoVy"],
                cam_info["W"], cam_info["H"], device,
            )
            color_f, depth_f, alpha_f = render_diff(
                means3D, colors_t, alpha_t, scales_t, rot_t,
                viewmat_f, fullproj_f, campos_f, tfovx_f, tfovy_f,
                cam_info["W"], cam_info["H"], device,
            )
            target_f = cam_info["target"]  # (3, H, W), stop-gradient
            flux_loss = ((color_f.clamp(0, 1) - target_f) ** 2).mean()
            total_loss = total_loss + args.flux_weight * flux_loss

        # ---- Backprop + step ----
        total_loss.backward()
        optimizer.step()

        # Track best
        if total_loss.item() < best_total:
            best_total = total_loss.item()
            best_state = {
                "delta_r": delta_r.data.clone(),
                "delta_t": delta_t.data.clone(),
                "delta_s": delta_s.data.clone(),
            }

        if step % args.val_interval == 0 or step == args.n_steps - 1:
            print(
                f"  step {step:04d}  total={total_loss.item():.4f}"
                f"  photo={photo_loss.item():.4f}"
                f"  sil={sil_loss.item():.4f}"
                f"  depth={depth_loss.item():.4f}"
                f"  flux={flux_loss.item():.4f}"
                f"  |Δr|={delta_r.norm().item():.4f}"
                f"  |Δt|={delta_t.norm().item():.4f}"
                f"  Δs={delta_s.item():.4f}"
            )
            # Save rendered | reference | depth side-by-side
            with torch.no_grad():
                rendered_np = (rgb_src.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                depth_np    = depth_src.detach().cpu().squeeze().numpy()   # (H, W)
                alpha_np    = (alpha_src.detach().cpu().squeeze().numpy()
                               if alpha_src is not None else None)
            depth_colored = colorize_depth(depth_np, alpha_hw=alpha_np)
            val_img = np.concatenate([rendered_np, ref_img_r, depth_colored], axis=1)  # (H, 3W, 3)
            imageio.imwrite(os.path.join(val_dir, f"val_{step:04d}.png"), val_img)

    # -----------------------------------------------------------------------
    # Apply best delta to base transform → refined placement
    # -----------------------------------------------------------------------
    print("\n--- Computing refined placement ---")
    with torch.no_grad():
        R_delta = rodrigues(best_state["delta_r"])
        R_refined = (R_delta @ R_base_t).cpu().numpy().tolist()
        t_refined = (t_base_t + best_state["delta_t"]).cpu().numpy().tolist()
        s_refined = float(s_base * torch.exp(best_state["delta_s"].squeeze()).item())

    print(f"  s_refined:  {s_refined:.4f}  (base {s_base:.4f})")
    print(f"  t_refined:  {[f'{x:.4f}' for x in t_refined]}")
    print(f"  |Δr|:       {best_state['delta_r'].norm().item():.4f}")
    print(f"  |Δt|:       {best_state['delta_t'].norm().item():.4f}")
    print(f"  Δs:         {best_state['delta_s'].item():.4f}")

    # -----------------------------------------------------------------------
    # Save placement_refined.json
    # -----------------------------------------------------------------------
    placement_refined = {
        "translation": t_refined,
        "scale":       s_refined,
        "rotation":    R_refined,
        "notes": (
            "Rigid-refined placement (Stage 1). "
            "Pass --placement_path to create_composite_4dgs.py to use this. "
            "rotation is a 3×3 matrix (row-major list of lists); "
            "if present, create_composite_4dgs skips computing R from COLMAP."
        ),
    }
    with open(out_json, "w") as f:
        json.dump(placement_refined, f, indent=2)
    print(f"\n  Saved: {out_json}")

    # -----------------------------------------------------------------------
    # Save debug render at source camera (frame 0 of refined composite)
    # -----------------------------------------------------------------------
    print("\n--- Saving debug render ---")
    with torch.no_grad():
        dr = best_state["delta_r"].to(device)
        dt = best_state["delta_t"].to(device)
        ds = best_state["delta_s"].to(device)
        means3D_fg_ref = apply_rigid(xyz_fg_t, R_base_t, t_base_t, s_base, dr, dt, ds)
        means3D_ref    = torch.cat([xyz_bg_t, means3D_fg_ref], dim=0)
        s_ref          = s_base * torch.exp(ds.squeeze())
        scales_ref     = torch.cat([scales_bg_t, scales_fg_base_t * s_ref], dim=0)
        color_ref, _, _ = render_diff(
            means3D_ref, colors_t, alpha_t, scales_ref, rot_t,
            src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
            rW, rH, device,
        )
    debug_render = (color_ref.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    imageio.imwrite(os.path.join(debug_dir, "refined_render_src.png"), debug_render)
    imageio.imwrite(os.path.join(debug_dir, "reference_frame.png"), ref_img_r)
    print(f"  Debug renders: {debug_dir}/")

    print("\nDone. Next steps:")
    print(f"  python create_composite_4dgs.py \\")
    print(f"    --output_path {args.composite_path} \\")
    print(f"    --gs_scene_path {args.gs_scene_path} \\")
    print(f"    --placement_path {out_json} \\")
    print(f"    --camera_idx {args.camera_idx}")


if __name__ == "__main__":
    main()
