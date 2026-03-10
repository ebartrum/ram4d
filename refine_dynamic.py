"""
refine_dynamic.py — Dynamic rigid refinement: per-frame translation + rotation corrections.

Learns per-frame corrections — translation Δt_t ∈ ℝ³ and rotation Δq_t ∈ quaternion
(7 DOF total per frame) — to bring each rendered frame into agreement with its
WAN-generated target video, while a temporal consistency loss discourages jerky jumps.

Inputs:
  - fg_positions_world_deformed.npy (T, N_fg, 3) from refine_deform.py
  - WAN target frames   (composite_path/frames/00000.jpg – 00080.jpg)
  - SAM2 per-frame masks (composite_path/sam2_masks/00000.png – 00080.png)
  - Background 3DGS scene + model

Per-frame transform applied at frame t:
  centroid_t  = fg_positions_world_deformed[t].mean(0)
  R_delta_t   = quat_to_matrix(normalise(delta_q[t]))
  new_pos_t   = (R_delta_t @ (fg_pos_deformed[t] - centroid_t).T).T + centroid_t + delta_t[t]


Losses (per step, random frame t):
  1. RGB L1      — fg-masked    — weight --rgb_weight
  2. Silhouette BCE              — weight --silhouette_weight
  3. Occlusion                   — weight --occlusion_weight
  4. Depth pseudo-SDS (optional) — weight --depth_weight
  5. Temporal consistency        — weight --temporal_weight   (all frames, no rendering)
  6. Frame-0 anchor              — weight --anchor_weight

Outputs saved to composite_path/gaussians/:
  fg_positions_world_dynamic.npy  (T, N_fg, 3)

Outputs saved to output_path/:
  delta_t.npy  (T, 3)
  delta_q.npy  (T, 4)
  val/         — validation renders at frames 0, 20, 40, 60, 80

Usage:
  python refine_dynamic.py \\
    --composite_path output/2026.03.03/actionmesh_gs_replace_corgi \\
    --placement_path output/2026.03.09/corgi_refined_consistent_occl/placement_refined.json \\
    --gs_scene_path  Inpaint360GS/data/inpaint360/bag \\
    --gs_model_path  output/2026.02.26/inpainted_scene/point_cloud_object_inpaint_virtual \\
    --camera_idx     28 \\
    --output_path    output/2026.03.09/corgi_dynamic_refine \\
    --n_steps 500 --lr 1e-3 \\
    --rgb_weight 1.0 --silhouette_weight 1.0 --occlusion_weight 50.0 \\
    --temporal_weight 1.0 --anchor_weight 100.0 \\
    --render_scale 0.5 --val_interval 100
"""

import sys
import os
import glob
import math
import json
import random

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


def quat_to_matrix_batch(q):
    """(T, 4) unit quaternions [w, x, y, z] → (T, 3, 3) rotation matrices."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    T = q.shape[0]
    R = torch.stack([
        1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y),
            2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x),
            2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1).reshape(T, 3, 3)
    return R


def apply_rigid_frame(fg_pos_frame, delta_q_t, delta_t_t):
    """
    Apply per-frame rigid correction around the per-frame centroid.

    fg_pos_frame : (N_fg, 3) — no grad (deformed world-space positions)
    delta_q_t    : (4,)      — has grad (per-frame rotation correction quaternion)
    delta_t_t    : (3,)      — has grad (per-frame world-space translation offset)

    Returns: (N_fg, 3) — gradient flows through delta_q_t and delta_t_t.
    """
    q_norm   = delta_q_t / delta_q_t.norm()        # (4,), normalised
    R        = quat_to_matrix(q_norm)               # (3, 3)
    centroid = fg_pos_frame.mean(0)                 # (3,), no grad
    centered = fg_pos_frame - centroid              # (N_fg, 3), no grad
    # (R @ centered.T).T  ==  centered @ R.T
    new_pos  = centered @ R.T + centroid + delta_t_t  # (N_fg, 3)
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


# ---------------------------------------------------------------------------
# Marigold depth (optional)
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


def prepare_sds_components(marigold_pipe, ref_img_np, sds_size, device):
    """Pre-compute fixed SDS components: image latent + text embedding."""
    sds_H = sds_W = sds_size
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=False):
            ref_pil    = Image.fromarray(ref_img_np).resize((sds_W, sds_H), Image.BILINEAR)
            img_np     = np.array(ref_pil).astype(np.float32) / 127.5 - 1.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            img_lat    = marigold_pipe.vae.encode(img_tensor).latent_dist.mode()
            img_lat    = img_lat * marigold_pipe.vae.config.scaling_factor

        try:
            text_out = marigold_pipe.encode_prompt(
                prompt="", device=device, num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            text_emb = text_out[0] if isinstance(text_out, tuple) else text_out
        except Exception:
            toks = marigold_pipe.tokenizer(
                "", return_tensors="pt", padding="max_length",
                max_length=marigold_pipe.tokenizer.model_max_length, truncation=True,
            )
            text_emb = marigold_pipe.text_encoder(toks.input_ids.to(device))[0]
    print(f"  SDS components ready — img_lat {img_lat.shape}, text_emb {text_emb.shape}")
    return img_lat, text_emb


def compute_depth_pseudo_sds(depth_composite_hw, img_lat, text_emb, marigold_pipe,
                              device, sds_H, sds_W, t_min=50, t_max=500):
    """Pixel-space pseudo-SDS depth loss. Returns scalar loss."""
    d   = depth_composite_hw.unsqueeze(0).unsqueeze(0)
    d_r = F.interpolate(d, size=(sds_H, sds_W), mode='bilinear', align_corners=False)

    d_min  = d_r.detach().min()
    d_max  = d_r.detach().max()
    d_norm = 2.0 * (d_r - d_min) / (d_max - d_min + 1e-6) - 1.0
    d_vae  = d_norm.expand(1, 3, sds_H, sds_W)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=False):
            depth_lat = marigold_pipe.vae.encode(d_vae.float()).latent_dist.mode()
            depth_lat = depth_lat * marigold_pipe.vae.config.scaling_factor

    t_val     = random.randint(t_min, t_max)
    t_b       = torch.tensor([t_val], device=device, dtype=torch.long)
    alpha_bar = marigold_pipe.scheduler.alphas_cumprod[t_val].to(device=device, dtype=torch.float32)
    alpha_t   = alpha_bar.sqrt()
    sigma_t   = (1.0 - alpha_bar).sqrt()

    eps         = torch.randn_like(depth_lat)
    noisy_depth = alpha_t * depth_lat + sigma_t * eps

    unet_in = torch.cat([img_lat, noisy_depth], dim=1)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=False):
            noise_pred = marigold_pipe.unet(
                unet_in.float(), t_b, encoder_hidden_states=text_emb.float(),
            ).sample

    pred_type = getattr(marigold_pipe.scheduler.config, 'prediction_type', 'epsilon')
    with torch.no_grad():
        if pred_type == 'v_prediction':
            z0_hat = alpha_t * noisy_depth - sigma_t * noise_pred
        elif pred_type == 'epsilon':
            z0_hat = (noisy_depth - sigma_t * noise_pred) / (alpha_t + 1e-8)
        else:
            z0_hat = noise_pred

        with torch.amp.autocast("cuda", enabled=False):
            pred_depth_px = marigold_pipe.vae.decode(
                z0_hat.float() / marigold_pipe.vae.config.scaling_factor
            ).sample  # (1, 3, sds_H, sds_W)

    loss = F.mse_loss(d_vae, pred_depth_px.detach())
    return loss


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dynamic rigid refinement: per-frame translation + rotation corrections."
    )
    parser.add_argument("--composite_path", required=True,
                        help="actionmesh output dir (contains gaussians/, frames/, sam2_masks/)")
    parser.add_argument("--placement_path", required=True,
                        help="placement_refined.json from refine_frame0.py (scale used for fg sizes)")
    parser.add_argument("--gs_scene_path", required=True,
                        help="Inpaint360GS scene dir (sparse/0/)")
    parser.add_argument("--gs_model_path", required=True,
                        help="Inpaint360GS model dir (iteration_N/point_cloud.ply)")
    parser.add_argument("--camera_idx", type=int, default=28,
                        help="COLMAP camera index for source view (default 28)")
    parser.add_argument("--fg_positions_path", default=None,
                        help="Path to fg_positions_world_deformed.npy "
                             "(default: composite_path/gaussians/fg_positions_world_deformed.npy)")
    parser.add_argument("--frames_dir", default=None,
                        help="WAN target frames directory (default: composite_path/frames/)")
    parser.add_argument("--masks_dir", default=None,
                        help="SAM2 per-frame masks directory (default: composite_path/sam2_masks/)")
    parser.add_argument("--output_path", required=True,
                        help="Output directory for corrections, val renders")
    # Optimisation
    parser.add_argument("--n_steps",       type=int,   default=2000)
    parser.add_argument("--lr",            type=float, default=1e-3,
                        help="Adam learning rate (default 1e-3)")
    parser.add_argument("--render_scale",  type=float, default=0.5,
                        help="Render scale relative to WAN frame resolution (default 0.5)")
    # Loss weights
    parser.add_argument("--rgb_weight",        type=float, default=1.0,
                        help="Photo L1 loss weight, fg-masked (default 1.0)")
    parser.add_argument("--silhouette_weight", type=float, default=1.0,
                        help="Silhouette BCE loss weight (default 1.0)")
    parser.add_argument("--occlusion_weight",  type=float, default=50.0,
                        help="Occlusion depth penalty weight (default 50.0)")
    parser.add_argument("--depth_weight",      type=float, default=0.0,
                        help="Marigold pseudo-SDS depth loss weight (default 0, expensive)")
    parser.add_argument("--temporal_weight",   type=float, default=1.0,
                        help="Temporal velocity smoothness weight (default 1.0)")
    parser.add_argument("--anchor_weight",     type=float, default=100.0,
                        help="Frame-0 anchor weight (default 100.0)")
    # Marigold settings
    parser.add_argument("--sds_t_min", type=int, default=50)
    parser.add_argument("--sds_t_max", type=int, default=500)
    parser.add_argument("--sds_size",  type=int, default=512)
    # Validation
    parser.add_argument("--val_interval", type=int, default=100,
                        help="Save val images every N steps (default 100)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_path, exist_ok=True)
    val_dir = os.path.join(args.output_path, "val")
    os.makedirs(val_dir, exist_ok=True)

    # Resolve default sub-paths
    gaussians_dir  = os.path.join(args.composite_path, "gaussians")
    fg_pos_path    = args.fg_positions_path or os.path.join(
        gaussians_dir, "fg_positions_world_deformed.npy"
    )
    frames_dir     = args.frames_dir or os.path.join(args.composite_path, "frames")
    masks_dir      = args.masks_dir  or os.path.join(args.composite_path, "sam2_masks")
    fg_ply_path    = os.path.join(gaussians_dir, "gaussians.ply")

    # -----------------------------------------------------------------------
    # Placement — we only need the scale for fg Gaussian sizes
    # -----------------------------------------------------------------------
    print("\n--- Loading placement_refined.json ---")
    with open(args.placement_path) as f:
        placement = json.load(f)
    s_base = float(placement["scale"])
    print(f"  scale: {s_base:.4f}")

    # -----------------------------------------------------------------------
    # Load fg deformed positions  (T, N_fg, 3)
    # -----------------------------------------------------------------------
    print(f"\n--- Loading fg_positions_world_deformed.npy ---")
    if not os.path.exists(fg_pos_path):
        raise FileNotFoundError(
            f"Not found: {fg_pos_path}\n"
            "Run refine_deform.py first to produce this file."
        )
    fg_pos_deformed_np = np.load(fg_pos_path)   # (T, N_fg, 3)
    T, N_fg, _ = fg_pos_deformed_np.shape
    print(f"  Shape: {fg_pos_deformed_np.shape}  ({T} frames, {N_fg:,} Gaussians)")

    # -----------------------------------------------------------------------
    # Load fg Gaussian attributes
    # -----------------------------------------------------------------------
    print("\n--- Loading foreground Gaussians ---")
    xyz_fg_canon, f_dc_fg, op_fg, log_sc_fg, rot_fg = load_ply_gs(fg_ply_path)
    assert len(xyz_fg_canon) == N_fg, (
        f"N_fg mismatch: PLY has {len(xyz_fg_canon)}, deformed positions has {N_fg}"
    )
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
    # Build frozen GPU tensors
    # -----------------------------------------------------------------------
    print("\n--- Building GPU tensors ---")
    colors_bg_t  = torch.from_numpy(np.clip(f_dc_bg * C0 + 0.5, 0.0, 1.0)).float().to(device)
    alpha_bg_t   = torch.from_numpy(
        1.0 / (1.0 + np.exp(-op_bg))
    ).float().to(device).unsqueeze(1)
    rot_bg_t     = torch.from_numpy(rot_bg).float().to(device)
    scales_bg_t  = torch.from_numpy(np.exp(log_sc_bg)).float().to(device)
    xyz_bg_t     = torch.from_numpy(xyz_bg).float().to(device)

    colors_fg_t  = torch.from_numpy(np.clip(f_dc_fg * C0 + 0.5, 0.0, 1.0)).float().to(device)
    alpha_fg_t   = torch.from_numpy(
        1.0 / (1.0 + np.exp(-op_fg))
    ).float().to(device).unsqueeze(1)
    rot_fg_t     = torch.from_numpy(rot_fg).float().to(device)
    scales_fg_t  = torch.from_numpy(np.exp(log_sc_fg) * s_base).float().to(device)

    # Deformed fg positions — no grad, indexed per-frame during optimisation
    fg_pos_deformed_t = torch.from_numpy(fg_pos_deformed_np).float().to(device)  # (T, N_fg, 3)

    print(f"  Total: {N_bg + N_fg:,}  (bg={N_bg:,}  fg={N_fg:,})")

    # -----------------------------------------------------------------------
    # Source camera
    # -----------------------------------------------------------------------
    print("\n--- Loading source camera ---")
    all_cams = load_all_colmap_cameras(args.gs_scene_path)
    if args.camera_idx >= len(all_cams):
        raise ValueError(f"camera_idx {args.camera_idx} out of range ({len(all_cams)})")
    src_pose_w2c, src_FoVx, src_FoVy, src_W, src_H = all_cams[args.camera_idx]

    # -----------------------------------------------------------------------
    # Determine render resolution from WAN frames
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

    first_frame = Image.open(frame_files[0])
    W_wan, H_wan = first_frame.size
    rW = max(1, round(W_wan * args.render_scale))
    rH = max(1, round(H_wan * args.render_scale))
    print(f"  WAN frame size: {W_wan}×{H_wan} → render {rW}×{rH}")
    print(f"  Frames: {len(frame_files)},  Masks: {len(mask_files)}")

    src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy = \
        make_raster_camera_at_resolution(src_pose_w2c, src_FoVx, rW, rH, device)
    print(f"  Source cam {args.camera_idx}: fovx={math.degrees(src_FoVx):.1f}°  render {rW}×{rH}")

    # -----------------------------------------------------------------------
    # Load all WAN frames and SAM2 masks
    # WAN generation was guided with latent blending so the background is
    # already fixed — WAN frames are used directly as rgb-loss targets.
    # -----------------------------------------------------------------------
    n_load = min(T, len(frame_files), len(mask_files))
    if n_load < T:
        print(f"  Warning: expected {T} frames/masks, found {n_load}. Truncating to {n_load}.")
        T = n_load

    wan_frames_t = []
    fg_masks_t   = []

    for i in range(T):
        img   = Image.open(frame_files[i]).convert("RGB").resize((rW, rH), Image.BILINEAR)
        img_t = torch.from_numpy(
            np.array(img).astype(np.float32) / 255.0
        ).permute(2, 0, 1).to(device)    # (3, rH, rW)
        wan_frames_t.append(img_t)

        mask   = Image.open(mask_files[i]).convert("L").resize((rW, rH), Image.NEAREST)
        mask_t = torch.from_numpy(
            (np.array(mask) > 128).astype(np.float32)
        ).to(device)                     # (rH, rW)
        fg_masks_t.append(mask_t)

    print(f"  Loaded {T} frames and masks.")

    # -----------------------------------------------------------------------
    # Pre-render background depth (for occlusion loss)
    # -----------------------------------------------------------------------
    bg_depth_color_map = None
    bg_occlusion_mask  = None

    if args.occlusion_weight > 0:
        print("\n--- Pre-rendering background depth (occlusion loss) ---")
        with torch.no_grad():
            bg_hom      = torch.cat([xyz_bg_t, torch.ones(N_bg, 1, device=device)], dim=1)
            bg_z        = (bg_hom @ src_viewmat)[:, 2]
            bg_z_colors = bg_z.unsqueeze(1).expand(-1, 3).float()
            bg_depth_render, _, _ = render_diff(
                xyz_bg_t, bg_z_colors, alpha_bg_t, scales_bg_t, rot_bg_t,
                src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
                rW, rH, device,
            )
        bg_depth_color_map = bg_depth_render[0].detach()        # (rH, rW), no grad
        bg_occlusion_mask  = (bg_depth_color_map > 0).float()   # (rH, rW)
        print(f"  BG depth: non-zero px = {bg_occlusion_mask.sum().int().item()}")

    # -----------------------------------------------------------------------
    # Marigold (optional depth prior)
    # -----------------------------------------------------------------------
    marigold_pipe = None
    sds_img_lat   = None
    sds_text_emb  = None

    if args.depth_weight > 0:
        print("\n--- Loading Marigold ---")
        ref_img_np    = (wan_frames_t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        marigold_pipe = load_marigold(device)
        sds_img_lat, sds_text_emb = prepare_sds_components(
            marigold_pipe, ref_img_np, args.sds_size, device
        )

    # -----------------------------------------------------------------------
    # Learnable per-frame parameters
    # -----------------------------------------------------------------------
    print("\n--- Initialising per-frame parameters ---")
    delta_t      = torch.zeros(T, 3, device=device, requires_grad=True)
    delta_q_init = torch.zeros(T, 4, device=device)
    delta_q_init[:, 0] = 1.0               # identity quaternion [w=1, x=0, y=0, z=0]
    delta_q      = delta_q_init.requires_grad_(True)

    optimizer = torch.optim.Adam([delta_t, delta_q], lr=args.lr)
    print(f"  Parameters: delta_t {tuple(delta_t.shape)}, delta_q {tuple(delta_q.shape)}")
    print(f"  Optimiser: Adam  lr={args.lr}")

    # -----------------------------------------------------------------------
    # Precompute for temporal consistency (fixed, no grad)
    # -----------------------------------------------------------------------
    K           = min(1000, N_fg)
    sample_idx  = torch.randperm(N_fg, device=device)[:K]         # (K,)
    fg_sample_t = fg_pos_deformed_t[:, sample_idx, :]             # (T, K, 3), no grad
    centroids_t = fg_pos_deformed_t.mean(dim=1)                   # (T, 3),    no grad

    # -----------------------------------------------------------------------
    # Fixed validation frames
    # -----------------------------------------------------------------------
    val_frames = sorted(set([0, 20, 40, 60, 80, T - 1]))
    val_frames = [f for f in val_frames if f < T]

    # -----------------------------------------------------------------------
    # Optimisation loop
    # -----------------------------------------------------------------------
    print(f"\n--- Optimising {args.n_steps} steps ---")
    pbar = tqdm(range(args.n_steps), desc="optimising")

    for step in pbar:
        optimizer.zero_grad()

        # Sample a random frame
        t_idx = random.randint(0, T - 1)

        # Apply per-frame rigid correction to deformed positions
        new_pos = apply_rigid_frame(
            fg_pos_deformed_t[t_idx],   # (N_fg, 3), no grad
            delta_q[t_idx],             # (4,), has grad
            delta_t[t_idx],             # (3,), has grad
        )   # (N_fg, 3), grad via delta_q[t_idx] and delta_t[t_idx]

        # Assemble full scene
        means3D  = torch.cat([xyz_bg_t,    new_pos],       dim=0)
        colors   = torch.cat([colors_bg_t, colors_fg_t],  dim=0)
        alphas   = torch.cat([alpha_bg_t,  alpha_fg_t],   dim=0)
        rots     = torch.cat([rot_bg_t,    rot_fg_t],     dim=0)
        scales   = torch.cat([scales_bg_t, scales_fg_t],  dim=0)

        # Composite render
        color_render, _, _ = render_diff(
            means3D, colors, alphas, scales, rots,
            src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
            rW, rH, device,
        )
        rgb_render = color_render.clamp(0, 1)   # (3, rH, rW)

        mask_t   = fg_masks_t[t_idx]            # (rH, rW)
        target_t = wan_frames_t[t_idx]          # (3, rH, rW)

        # ---- Photo L1 (fg-masked) ----
        rgb_loss   = ((rgb_render - target_t).abs() * mask_t.unsqueeze(0)).mean()
        total_loss = args.rgb_weight * rgb_loss

        # ---- Silhouette BCE ----
        sil_loss = torch.tensor(0.0, device=device)
        if args.silhouette_weight > 0:
            sil_colors = torch.zeros_like(colors)
            sil_colors[N_bg:] = 1.0
            sil_render, _, _ = render_diff(
                means3D, sil_colors, alphas, scales, rots,
                src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
                rW, rH, device,
            )
            sil_pred = sil_render.mean(0)       # (rH, rW)
            with torch.amp.autocast("cuda", enabled=False):
                sil_loss = F.binary_cross_entropy(
                    sil_pred.clamp(1e-6, 1 - 1e-6).float(), mask_t.float()
                )
            total_loss = total_loss + args.silhouette_weight * sil_loss

        # ---- Occlusion: penalise fg Gaussians rendered behind bg ----
        occlusion_loss = torch.tensor(0.0, device=device)
        if args.occlusion_weight > 0 and bg_depth_color_map is not None:
            fg_hom      = torch.cat([new_pos, torch.ones(N_fg, 1, device=device)], dim=1)
            fg_z        = (fg_hom @ src_viewmat)[:, 2]
            fg_z_colors = fg_z.unsqueeze(1).expand(-1, 3).float()
            fg_depth_render, _, _ = render_diff(
                new_pos, fg_z_colors, alpha_fg_t, scales_fg_t, rot_fg_t,
                src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
                rW, rH, device,
            )
            fg_depth_map   = fg_depth_render[0]
            occlusion_loss = (
                F.relu(fg_depth_map - bg_depth_color_map) ** 2 * bg_occlusion_mask
            ).sum() / (bg_occlusion_mask.sum() + 1e-6)
            total_loss = total_loss + args.occlusion_weight * occlusion_loss

        # ---- Depth pseudo-SDS (optional, expensive) ----
        depth_loss = torch.tensor(0.0, device=device)
        if args.depth_weight > 0 and marigold_pipe is not None:
            all_hom    = torch.cat([means3D, torch.ones(means3D.shape[0], 1, device=device)], dim=1)
            all_z      = (all_hom @ src_viewmat)[:, 2]
            all_z_cols = all_z.unsqueeze(1).expand(-1, 3).float()
            depth_composite_render, _, _ = render_diff(
                means3D, all_z_cols, alphas, scales, rots,
                src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
                rW, rH, device,
            )
            depth_hw   = depth_composite_render[0]
            depth_loss = compute_depth_pseudo_sds(
                depth_hw, sds_img_lat, sds_text_emb, marigold_pipe,
                device, args.sds_size, args.sds_size, args.sds_t_min, args.sds_t_max,
            )
            total_loss = total_loss + args.depth_weight * depth_loss

        # ---- Temporal consistency: velocity penalty on corrected positions ----
        # Operates on ALL frames each step (no rendering required).
        # Penalises jerky motion in the final trajectory (coarse + delta).
        temporal_loss = torch.tensor(0.0, device=device)
        if args.temporal_weight > 0 and T > 1:
            q_norm    = delta_q / (delta_q.norm(dim=1, keepdim=True) + 1e-8)   # (T, 4)
            R_batch   = quat_to_matrix_batch(q_norm)                           # (T, 3, 3)
            # centered: (T, K, 3) — no grad (deformed positions minus per-frame centroid)
            centered  = fg_sample_t - centroids_t.unsqueeze(1)
            # corrected: (T, K, 3) — grad via R_batch (→ delta_q) and delta_t
            corrected = (
                torch.bmm(centered, R_batch.transpose(-1, -2))
                + centroids_t.unsqueeze(1)
                + delta_t.unsqueeze(1)
            )
            velocity      = corrected[1:] - corrected[:-1]   # (T-1, K, 3)
            temporal_loss = (velocity ** 2).sum(-1).mean()
            total_loss    = total_loss + args.temporal_weight * temporal_loss

        # ---- Frame-0 anchor: keep frame 0 near zero corrections ----
        anchor_loss = torch.tensor(0.0, device=device)
        if args.anchor_weight > 0:
            e0          = torch.tensor([1., 0., 0., 0.], device=device)
            q0_norm     = delta_q[0] / (delta_q[0].norm() + 1e-8)
            anchor_loss = (delta_t[0] ** 2).sum() + (q0_norm - e0).pow(2).sum()
            total_loss  = total_loss + args.anchor_weight * anchor_loss

        # ---- Backprop + step ----
        total_loss.backward()
        optimizer.step()

        pbar.set_postfix(
            total=f"{total_loss.item():.4f}",
            rgb=f"{rgb_loss.item():.4f}",
            sil=f"{sil_loss.item():.4f}",
            occl=f"{occlusion_loss.item():.4f}",
            temp=f"{temporal_loss.item():.4f}",
            anch=f"{anchor_loss.item():.4f}",
            t=t_idx,
        )

        # ---- Validation renders ----
        if step % args.val_interval == 0 or step == args.n_steps - 1:
            with torch.no_grad():
                rows = []
                for vt in val_frames:
                    new_pos_v = apply_rigid_frame(
                        fg_pos_deformed_t[vt], delta_q[vt], delta_t[vt]
                    )
                    means3D_v = torch.cat([xyz_bg_t,    new_pos_v],      dim=0)
                    colors_v  = torch.cat([colors_bg_t, colors_fg_t],   dim=0)
                    alphas_v  = torch.cat([alpha_bg_t,  alpha_fg_t],    dim=0)
                    rots_v    = torch.cat([rot_bg_t,    rot_fg_t],      dim=0)
                    scales_v  = torch.cat([scales_bg_t, scales_fg_t],   dim=0)

                    color_v, _, _ = render_diff(
                        means3D_v, colors_v, alphas_v, scales_v, rots_v,
                        src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
                        rW, rH, device,
                    )
                    rendered_np = (
                        color_v.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                    ).astype(np.uint8)

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
                    fg_only_np = (
                        fg_only_v.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                    ).astype(np.uint8)

                    # GT mask
                    gt_mask_np  = (fg_masks_t[vt].cpu().numpy() > 0.5).astype(np.uint8) * 255
                    gt_mask_rgb = np.stack([gt_mask_np] * 3, axis=-1)

                    # WAN target frame
                    target_np = (
                        wan_frames_t[vt].permute(1, 2, 0).cpu().numpy() * 255
                    ).astype(np.uint8)

                    panels = [rendered_np, target_np, sil_rgb, gt_mask_rgb, fg_only_np]
                    rows.append(np.concatenate(panels, axis=1))

                val_img = np.concatenate(rows, axis=0)
                out_fname = os.path.join(val_dir, f"val_step{step:04d}.png")
                imageio.imwrite(out_fname, val_img)

    # -----------------------------------------------------------------------
    # Save raw per-frame corrections
    # -----------------------------------------------------------------------
    print("\n--- Saving corrections ---")
    delta_t_np = delta_t.detach().cpu().numpy()     # (T, 3)
    delta_q_np = delta_q.detach().cpu().numpy()     # (T, 4)
    np.save(os.path.join(args.output_path, "delta_t.npy"), delta_t_np)
    np.save(os.path.join(args.output_path, "delta_q.npy"), delta_q_np)
    print(f"  |Δt| max:   {np.linalg.norm(delta_t_np, axis=1).max():.4f}")
    print(f"  |Δq - I| max: {np.abs(delta_q_np - np.array([[1,0,0,0]])).max():.4f}")

    # -----------------------------------------------------------------------
    # Apply corrections to all frames → fg_positions_world_dynamic.npy
    # -----------------------------------------------------------------------
    print("\n--- Computing fg_positions_world_dynamic.npy ---")
    dynamic_positions = np.zeros((T, N_fg, 3), dtype=np.float32)
    with torch.no_grad():
        for t_idx in range(T):
            new_pos_t = apply_rigid_frame(
                fg_pos_deformed_t[t_idx], delta_q[t_idx], delta_t[t_idx]
            )
            dynamic_positions[t_idx] = new_pos_t.cpu().numpy()

    dynamic_path = os.path.join(gaussians_dir, "fg_positions_world_dynamic.npy")
    np.save(dynamic_path, dynamic_positions)
    print(f"  Saved: {dynamic_path}  shape {dynamic_positions.shape}")

    # Centroid trajectory stats
    centroids_final   = dynamic_positions.mean(axis=1)                  # (T, 3)
    centroid_velocity = np.diff(centroids_final, axis=0)                # (T-1, 3)
    vel_norms         = np.linalg.norm(centroid_velocity, axis=1)
    print(f"  Centroid velocity — mean: {vel_norms.mean():.4f},  max: {vel_norms.max():.4f}")

    print("\nDone.")
    print(f"  Dynamic positions: {dynamic_path}")
    print(f"  Corrections:       {args.output_path}/delta_t.npy, delta_q.npy")
    print(f"  Val renders:       {val_dir}/")
    print(f"\nNext: render orbit/cam28 video using --fg_positions_path {dynamic_path}")


if __name__ == "__main__":
    main()
