"""
refine_deform.py — Deformation MLP on top of rigid placement.

Learns per-Gaussian position offsets (Δxyz) via a small MLP over normalised
world-space fg positions. Colors / opacities / scales / rotations are frozen.
Only DeformMLP parameters are optimised.

Losses: photo L1 + silhouette BCE + depth pseudo-SDS (Marigold) + L2 deform reg.

After training, the learned deformation is applied to all T frames in
gaussians/fg_positions_world.npy, and the result is saved as
gaussians/fg_positions_world_deformed.npy inside composite_path.

Usage:
  python refine_deform.py \\
    --composite_path  output/2026.03.03/actionmesh_gs_replace_corgi \\
    --placement_path  output/2026.03.06/corgi_refined_consistent/placement_refined.json \\
    --gs_scene_path   Inpaint360GS/data/inpaint360/bag \\
    --gs_model_path   output/2026.02.26/inpainted_scene/point_cloud_object_inpaint_virtual \\
    --reference_frame output/2026.03.06/corgi_consistent_ref_10px/consistent_ref.png \\
    --camera_idx      28 \\
    --seg_prompt_path data/captions/corgi_segmentation.txt \\
    --output_path     output/2026.03.06/corgi_deform \\
    --n_steps 500 --rgb_weight 1.0 --depth_weight 1.0 --sds_type pseudo_sds
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
from tqdm import tqdm
from plyfile import PlyData

from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from camera_utils import make_raster_camera_at_resolution
from pipeline_utils import run_langsam

C0 = 0.28209479177387814


# ---------------------------------------------------------------------------
# Deformation MLP
# ---------------------------------------------------------------------------

class DeformMLP(nn.Module):
    """Small MLP: normalised xyz → Δxyz offsets.

    Final layer is zero-initialised so the deformation starts as identity.
    """
    def __init__(self, hidden=64, n_layers=3):
        super().__init__()
        layers = [nn.Linear(3, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 3)]
        self.net = nn.Sequential(*layers)
        # Small random init on final layer — NOT zero-init.
        # Zero-init blocks gradient to all earlier layers (dL/dh = dL/dout @ W_final = 0),
        # so weights before the final layer never update. Small init gives delta_xyz ≈ 1e-3
        # at step 0 (near-identity) while keeping gradient flow through all layers intact.
        nn.init.normal_(self.net[-1].weight, std=1e-4)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, xyz_norm):
        return self.net(xyz_norm)


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

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


def compute_orig_rotation_from_colmap(scene_path, camera_idx, translation):
    """Reproduce the R_orig that create_composite_4dgs.py computes from COLMAP."""
    images_bin  = os.path.join(scene_path, "sparse", "0", "images.bin")
    extrinsics  = read_extrinsics_binary(images_bin)
    intrinsics  = read_intrinsics_binary(os.path.join(scene_path, "sparse", "0", "cameras.bin"))
    sorted_imgs = sorted(extrinsics.values(), key=lambda x: x.name)
    # world_up = mean -cam_down
    downs     = np.stack([qvec2rotmat(si.qvec).T[:, 1] for si in sorted_imgs])
    world_up  = -downs.mean(0); world_up /= np.linalg.norm(world_up)
    # camera centre for camera_idx
    img       = sorted_imgs[camera_idx]
    R_w2c     = qvec2rotmat(img.qvec)
    cam_center = -(R_w2c.T @ np.array(img.tvec, dtype=np.float64)).astype(np.float32)
    # Build rotation (same as create_composite_4dgs.compute_fg_rotation, yaw_deg=0)
    to_cam   = cam_center - np.array(translation, dtype=np.float64)
    to_cam_h = to_cam - np.dot(to_cam, world_up) * world_up
    if np.linalg.norm(to_cam_h) < 1e-6:
        to_cam_h = np.array([1.0, 0.0, 0.0])
    world_fwd   = to_cam_h / np.linalg.norm(to_cam_h)
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


def colorize_depth(depth_hw, alpha_hw=None, cmap="turbo"):
    """
    Colorize a (H, W) float32 depth map.
    Normalises over the foreground region (alpha_hw > 0.1).
    Background pixels are shown as mid-grey (128, 128, 128).
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
    colored = (colormap(norm)[:, :, :3] * 255).astype(np.uint8)

    if alpha_hw is not None:
        colored[~fg] = 128

    return colored


def prepare_sds_components(marigold_pipe, ref_img_np, sds_size, device):
    """Pre-compute fixed SDS components: image latent + text embedding."""
    sds_H = sds_W = sds_size
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=False):
            ref_pil = Image.fromarray(ref_img_np).resize((sds_W, sds_H), Image.BILINEAR)
            img_np = np.array(ref_pil).astype(np.float32) / 127.5 - 1.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            img_lat = marigold_pipe.vae.encode(img_tensor).latent_dist.mode()
            img_lat = img_lat * marigold_pipe.vae.config.scaling_factor

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


def compute_depth_sds(depth_composite_hw, img_lat, text_emb, marigold_pipe,
                      device, sds_H, sds_W, t_min=50, t_max=500):
    """Clean-target SDS depth loss (latent-space regression)."""
    import random

    d = depth_composite_hw.unsqueeze(0).unsqueeze(0)
    d_r = F.interpolate(d, size=(sds_H, sds_W), mode='bilinear', align_corners=False)

    d_min = d_r.detach().min()
    d_max = d_r.detach().max()
    d_norm = 2.0 * (d_r - d_min) / (d_max - d_min + 1e-6) - 1.0
    d_vae = d_norm.expand(1, 3, sds_H, sds_W)

    with torch.amp.autocast("cuda", enabled=False):
        depth_lat = marigold_pipe.vae.encode(d_vae.float()).latent_dist.mode()
        depth_lat = depth_lat * marigold_pipe.vae.config.scaling_factor

    t_val = random.randint(t_min, t_max)
    t_b = torch.tensor([t_val], device=device, dtype=torch.long)
    alpha_bar = marigold_pipe.scheduler.alphas_cumprod[t_val].to(device=device, dtype=torch.float32)
    alpha_t = alpha_bar.sqrt()
    sigma_t = (1.0 - alpha_bar).sqrt()

    eps = torch.randn_like(depth_lat)
    noisy_depth = alpha_t * depth_lat.detach() + sigma_t * eps

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

    loss = F.mse_loss(depth_lat, z0_hat.detach())
    return loss


def compute_depth_pseudo_sds(depth_composite_hw, img_lat, text_emb, marigold_pipe,
                              device, sds_H, sds_W, t_min=50, t_max=500):
    """Pixel-space pseudo-SDS depth loss. Also returns pred_depth_px for debug viz."""
    import random

    d = depth_composite_hw.unsqueeze(0).unsqueeze(0)
    d_r = F.interpolate(d, size=(sds_H, sds_W), mode='bilinear', align_corners=False)

    d_min = d_r.detach().min()
    d_max = d_r.detach().max()
    d_norm = 2.0 * (d_r - d_min) / (d_max - d_min + 1e-6) - 1.0
    d_vae = d_norm.expand(1, 3, sds_H, sds_W)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=False):
            depth_lat = marigold_pipe.vae.encode(d_vae.float()).latent_dist.mode()
            depth_lat = depth_lat * marigold_pipe.vae.config.scaling_factor

    t_val = random.randint(t_min, t_max)
    t_b = torch.tensor([t_val], device=device, dtype=torch.long)
    alpha_bar = marigold_pipe.scheduler.alphas_cumprod[t_val].to(device=device, dtype=torch.float32)
    alpha_t = alpha_bar.sqrt()
    sigma_t = (1.0 - alpha_bar).sqrt()

    eps = torch.randn_like(depth_lat)
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
    return loss, pred_depth_px


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--composite_path", required=True,
                        help="actionmesh output dir (contains gaussians/)")
    parser.add_argument("--placement_path", required=True,
                        help="placement_refined.json from refine_frame0.py (R, t, s)")
    parser.add_argument("--original_placement_path", default=None,
                        help="Original placement.json used by create_composite_4dgs.py "
                             "(default: <composite_path>/gaussians/placement.json). "
                             "Used to correct motion_coarse from R_orig to R_refined frame.")
    parser.add_argument("--gs_scene_path", required=True,
                        help="Inpaint360GS scene dir (sparse/0/)")
    parser.add_argument("--gs_model_path", required=True,
                        help="Inpaint360GS model dir (iteration_N/point_cloud.ply)")
    parser.add_argument("--reference_frame", required=True,
                        help="Reference RGB image (consistent reference, source camera)")
    parser.add_argument("--camera_idx", type=int, default=28,
                        help="COLMAP camera index for source view (default 28)")
    parser.add_argument("--mask_path", default=None,
                        help="SAM mask PNG (white=fg) for silhouette loss")
    parser.add_argument("--seg_prompt_path", default=None,
                        help="Text prompt file for LangSAM auto-segmentation")
    parser.add_argument("--output_path", required=True,
                        help="Output directory")
    # MLP architecture
    parser.add_argument("--hidden", type=int, default=64,
                        help="MLP hidden layer size (default 64)")
    parser.add_argument("--n_layers", type=int, default=3,
                        help="MLP depth (default 3)")
    # Optimisation
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for MLP parameters (default 1e-3)")
    parser.add_argument("--deform_reg", type=float, default=0.01,
                        help="L2 regularisation weight on Δxyz (default 0.01)")
    parser.add_argument("--render_scale", type=float, default=1.0,
                        help="Render scale for source camera (default 1.0)")
    # Loss weights
    parser.add_argument("--rgb_weight",         type=float, default=1.0)
    parser.add_argument("--silhouette_weight",  type=float, default=1.0)
    parser.add_argument("--depth_weight",       type=float, default=0.0,
                        help="Marigold SDS depth loss weight (default 0)")
    parser.add_argument("--occlusion_weight",   type=float, default=50.0,
                        help="Weight for fg-behind-bg depth ordering penalty (default 50)")
    parser.add_argument("--sds_t_min", type=int, default=50)
    parser.add_argument("--sds_t_max", type=int, default=500)
    parser.add_argument("--sds_size", type=int, default=512)
    parser.add_argument("--sds_type", choices=["sds", "pseudo_sds"], default="pseudo_sds",
                        help="SDS variant (default pseudo_sds)")
    # Validation
    parser.add_argument("--val_interval", type=int, default=50,
                        help="Save val image every N steps (default 50)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_path, exist_ok=True)

    gaussians_dir   = os.path.join(args.composite_path, "gaussians")
    fg_ply_path     = os.path.join(gaussians_dir, "gaussians.ply")
    fg_pos_world_path = os.path.join(gaussians_dir, "fg_positions_world.npy")
    debug_dir       = os.path.join(args.output_path, "debug")
    val_dir         = os.path.join(args.output_path, "val")
    depth_debug_dir = os.path.join(args.output_path, "depth_debug")
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(depth_debug_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load placement_refined.json
    # -----------------------------------------------------------------------
    print("\n--- Loading placement_refined.json ---")
    with open(args.placement_path) as f:
        placement = json.load(f)
    R_base = np.array(placement["rotation"], dtype=np.float32)   # (3,3)
    t_base = np.array(placement["translation"], dtype=np.float32) # (3,)
    s_base = float(placement["scale"])
    print(f"  scale:       {s_base:.4f}")
    print(f"  translation: {t_base.tolist()}")
    print(f"  R det:       {np.linalg.det(R_base):.4f}")

    # -----------------------------------------------------------------------
    # Load original placement.json (from create_composite_4dgs) to correct motion frame
    # -----------------------------------------------------------------------
    orig_placement_path = args.original_placement_path or os.path.join(gaussians_dir, "placement.json")
    print(f"\n--- Loading original placement for motion correction ({orig_placement_path}) ---")
    with open(orig_placement_path) as f:
        orig_placement = json.load(f)
    s_orig = float(orig_placement["scale"])
    if "rotation" in orig_placement:
        R_orig = np.array(orig_placement["rotation"], dtype=np.float32)
        print(f"  R_orig loaded from JSON")
    else:
        print(f"  'rotation' not in original placement JSON — computing from COLMAP")
        t_orig = np.array(orig_placement["translation"], dtype=np.float32)
        R_orig = compute_orig_rotation_from_colmap(args.gs_scene_path, args.camera_idx, t_orig)
    print(f"  s_orig:  {s_orig:.4f}")
    print(f"  R_orig det: {np.linalg.det(R_orig):.4f}")

    # Correction matrix: re-expresses motion from R_orig frame to R_refined frame
    # motion_want[t] = (s_refined/s_orig) * R_refined @ R_orig.T @ motion_raw[t]
    correction = (s_base / s_orig) * (R_base @ R_orig.T)  # (3,3)
    print(f"  correction norm: {np.linalg.norm(correction - np.eye(3)):.4f} (0=no change)")

    # -----------------------------------------------------------------------
    # Load foreground Gaussian attributes (canonical positions + appearance)
    # -----------------------------------------------------------------------
    print("\n--- Loading foreground Gaussians ---")
    xyz_fg_canon, f_dc_fg, op_fg, log_sc_fg, rot_fg = load_ply_gs(fg_ply_path)
    N_fg = len(xyz_fg_canon)
    print(f"  {N_fg:,} Gaussians")

    # Compute world-space positions for frame 0 using refined placement
    # p_world = R @ (s * p_canon) + t
    xyz_world_init_np = (R_base @ (xyz_fg_canon * s_base).T).T + t_base  # (N_fg, 3)

    # -----------------------------------------------------------------------
    # Position normalisation — computed once, saved to deform_meta.json
    # -----------------------------------------------------------------------
    bbox_min = xyz_world_init_np.min(0)  # (3,)
    bbox_max = xyz_world_init_np.max(0)  # (3,)
    center = (bbox_min + bbox_max) / 2.0
    half_extent = float((bbox_max - bbox_min).max()) / 2.0
    print(f"\n--- Position normalisation ---")
    print(f"  center:      {center.tolist()}")
    print(f"  half_extent: {half_extent:.4f}")

    with open(os.path.join(args.output_path, "deform_meta.json"), "w") as f:
        json.dump({
            "center": center.tolist(),
            "half_extent": half_extent,
            "n_fg": N_fg,
            "placement_path": args.placement_path,
        }, f, indent=2)

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
    # BG — frozen throughout
    colors_bg_t = torch.from_numpy(
        np.clip(f_dc_bg * C0 + 0.5, 0.0, 1.0)
    ).float().to(device)                                                    # (N_bg, 3)
    alpha_bg_t  = torch.from_numpy(
        1.0 / (1.0 + np.exp(-op_bg))
    ).float().to(device).unsqueeze(1)                                       # (N_bg, 1)
    rot_bg_t    = torch.from_numpy(rot_bg).float().to(device)               # (N_bg, 4)
    scales_bg_t = torch.from_numpy(np.exp(log_sc_bg)).float().to(device)    # (N_bg, 3)
    xyz_bg_t    = torch.from_numpy(xyz_bg).float().to(device)               # (N_bg, 3)

    # FG attributes — all frozen (no grad)
    colors_fg_t = torch.from_numpy(
        np.clip(f_dc_fg * C0 + 0.5, 0.0, 1.0)
    ).float().to(device)                                                    # (N_fg, 3)
    alpha_fg_t  = torch.from_numpy(
        1.0 / (1.0 + np.exp(-op_fg))
    ).float().to(device).unsqueeze(1)                                       # (N_fg, 1)
    rot_fg_t    = torch.from_numpy(rot_fg).float().to(device)               # (N_fg, 4)
    scales_fg_t = torch.from_numpy(np.exp(log_sc_fg) * s_base).float().to(device)  # (N_fg, 3)

    # World-space fg positions (frame 0, frozen — delta_xyz is added on top)
    xyz_world_init_t = torch.from_numpy(xyz_world_init_np).float().to(device)  # (N_fg, 3)

    # Normalised positions for MLP input (frozen)
    center_t      = torch.from_numpy(center).float().to(device)
    xyz_norm_t    = (xyz_world_init_t - center_t) / (half_extent + 1e-6)   # (N_fg, 3)

    print(f"  Total: {N_bg + N_fg:,}  (bg={N_bg:,}  fg={N_fg:,})")

    # -----------------------------------------------------------------------
    # Reference frame
    # -----------------------------------------------------------------------
    print("\n--- Loading reference frame ---")
    ref_img = np.array(Image.open(args.reference_frame).convert("RGB"))
    W_ref, H_ref = ref_img.shape[1], ref_img.shape[0]
    rW = max(1, round(W_ref * args.render_scale))
    rH = max(1, round(H_ref * args.render_scale))
    ref_img_r = np.array(Image.fromarray(ref_img).resize((rW, rH), Image.BILINEAR))
    ref_t = torch.from_numpy(ref_img_r.astype(np.float32) / 255.0).to(device).permute(2, 0, 1)
    print(f"  Reference: {W_ref}×{H_ref} → render {rW}×{rH}")

    # -----------------------------------------------------------------------
    # Source camera
    # -----------------------------------------------------------------------
    print("\n--- Loading source camera ---")
    all_cams = load_all_colmap_cameras(args.gs_scene_path)
    if args.camera_idx >= len(all_cams):
        raise ValueError(f"camera_idx {args.camera_idx} out of range ({len(all_cams)})")
    src_pose_w2c, src_FoVx, src_FoVy, src_W, src_H = all_cams[args.camera_idx]
    src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy = \
        make_raster_camera_at_resolution(src_pose_w2c, src_FoVx, rW, rH, device)
    print(f"  Source cam {args.camera_idx}: {src_W}×{src_H}, fovx={math.degrees(src_FoVx):.1f}° → render {rW}×{rH}")

    # -----------------------------------------------------------------------
    # GT silhouette mask — run LangSAM BEFORE loop to avoid autocast pollution
    # -----------------------------------------------------------------------
    mask_src_t = None
    if args.mask_path:
        print("\n--- Loading silhouette mask ---")
        mask_np = np.array(Image.open(args.mask_path).convert("L"))
    elif args.seg_prompt_path:
        print("\n--- Auto-segmenting reference frame with LangSAM ---")
        ref_pil = Image.open(args.reference_frame).convert("RGB")
        mask_np = run_langsam(ref_pil, args.seg_prompt_path)
        auto_mask_path = os.path.join(debug_dir, "auto_mask.png")
        Image.fromarray(mask_np).save(auto_mask_path)
        print(f"  Auto mask saved to {auto_mask_path}")
    else:
        mask_np = None

    if mask_np is not None:
        mask_np_r = np.array(Image.fromarray(mask_np).resize((rW, rH), Image.NEAREST))
        fg_mask_gt = (mask_np_r > 128).astype(np.float32)
        mask_src_t = torch.from_numpy(fg_mask_gt).float().to(device)  # (H, W)
        print(f"  Mask fg pixels: {int(fg_mask_gt.sum())} / {rW*rH}")

    # -----------------------------------------------------------------------
    # Depth supervision (Marigold SDS)
    # -----------------------------------------------------------------------
    bg_depth_src_t    = None
    marigold_sds_pipe = None
    sds_img_lat       = None
    sds_text_emb      = None

    if args.depth_weight > 0:
        if mask_np is None:
            print("  Warning: --depth_weight > 0 but no mask available — skipping depth loss")
        else:
            print("\n--- Pre-computing background depth at source camera ---")
            with torch.no_grad():
                _, bg_depth_raw, _ = render_diff(
                    xyz_bg_t, colors_bg_t, alpha_bg_t, scales_bg_t, rot_bg_t,
                    src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
                    rW, rH, device,
                )
            bg_depth_src_t = bg_depth_raw.squeeze(0)  # (H, W), no grad

            print(f"\n--- Loading Marigold (size={args.sds_size}, t=[{args.sds_t_min},{args.sds_t_max}]) ---")
            marigold_sds_pipe = load_marigold(device)
            sds_img_lat, sds_text_emb = prepare_sds_components(
                marigold_sds_pipe, ref_img_r, args.sds_size, device
            )

    # -----------------------------------------------------------------------
    # Background depth map — pre-rendered for occlusion loss (Option B)
    # -----------------------------------------------------------------------
    bg_depth_color_map = None
    bg_occlusion_mask  = None

    if args.occlusion_weight > 0:
        print("\n--- Pre-rendering background depth map (occlusion loss) ---")
        with torch.no_grad():
            bg_hom      = torch.cat([xyz_bg_t, torch.ones(N_bg, 1, device=device)], dim=1)
            bg_z        = (bg_hom @ src_viewmat)[:, 2]                # (N_bg,)
            bg_z_colors = bg_z.unsqueeze(1).expand(-1, 3).float()     # (N_bg, 3)
            bg_depth_render, _, _ = render_diff(
                xyz_bg_t, bg_z_colors, alpha_bg_t, scales_bg_t, rot_bg_t,
                src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
                rW, rH, device,
            )
        bg_depth_color_map = bg_depth_render[0].detach()       # (H, W), no grad
        bg_occlusion_mask  = (bg_depth_color_map > 0).float()  # (H, W)
        print(f"  BG depth: shape={bg_depth_color_map.shape}, "
              f"non-zero={bg_occlusion_mask.sum().int().item()} px")

    # -----------------------------------------------------------------------
    # DeformMLP + optimiser
    # -----------------------------------------------------------------------
    print(f"\n--- Initialising DeformMLP (hidden={args.hidden}, n_layers={args.n_layers}) ---")
    deform_mlp = DeformMLP(hidden=args.hidden, n_layers=args.n_layers).to(device)
    optimizer = torch.optim.Adam(deform_mlp.parameters(), lr=args.lr)

    # -----------------------------------------------------------------------
    # Optimisation loop
    # -----------------------------------------------------------------------
    print(f"\n--- Optimising {args.n_steps} steps ---")
    depth_composite = None  # kept for depth debug viz
    pbar = tqdm(range(args.n_steps), desc="optimising")
    for step in pbar:
        optimizer.zero_grad()

        # 1. Deformation MLP → Δxyz offsets (grad flows through MLP params)
        # Wrap in autocast(enabled=False): LangSAM leaves bfloat16 autocast active,
        # which degrades MLP gradient flow if left unchecked.
        with torch.amp.autocast("cuda", enabled=False):
            delta_xyz = deform_mlp(xyz_norm_t.float()).float()  # (N_fg, 3)

        # 2. Deformed world-space fg positions
        xyz_world = xyz_world_init_t + delta_xyz    # (N_fg, 3)

        # 3. All Gaussians: bg (frozen) + deformed fg
        means3D = torch.cat([xyz_bg_t, xyz_world], dim=0)

        # 4. Frozen attribute tensors
        colors_t = torch.cat([colors_bg_t, colors_fg_t], dim=0)  # (N_total, 3)
        alpha_t  = torch.cat([alpha_bg_t,  alpha_fg_t],  dim=0)  # (N_total, 1)
        rot_t    = torch.cat([rot_bg_t,    rot_fg_t],    dim=0)  # (N_total, 4)
        scales_t = torch.cat([scales_bg_t, scales_fg_t], dim=0)  # (N_total, 3)

        # ---- Render from source camera ----
        color_src, depth_src, _ = render_diff(
            means3D, colors_t, alpha_t, scales_t, rot_t,
            src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
            rW, rH, device,
        )
        rgb_src = color_src.clamp(0, 1)  # (3, H, W)

        # ---- Photo loss (global L1) ----
        rgb_loss = (rgb_src - ref_t).abs().mean()
        total_loss = args.rgb_weight * rgb_loss

        # ---- Silhouette loss (occlusion-aware via alpha compositing) ----
        sil_loss = torch.tensor(0.0, device=device)
        if args.silhouette_weight > 0 and mask_src_t is not None:
            sil_colors = torch.zeros_like(colors_t)
            sil_colors[N_bg:] = 1.0  # fg = white, bg = black
            sil_render, _, _ = render_diff(
                means3D, sil_colors, alpha_t, scales_t, rot_t,
                src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
                rW, rH, device,
            )
            sil_pred = sil_render.mean(0)  # (H, W), in [0, 1]
            with torch.amp.autocast("cuda", enabled=False):
                sil_loss = F.binary_cross_entropy(
                    sil_pred.clamp(1e-6, 1 - 1e-6).float(), mask_src_t.float()
                )
            total_loss = total_loss + args.silhouette_weight * sil_loss

        # ---- Depth loss (Marigold SDS prior) ----
        depth_loss = torch.tensor(0.0, device=device)
        _depth_pred_px = None
        if args.depth_weight > 0 and marigold_sds_pipe is not None and mask_src_t is not None:
            all_hom = torch.cat([means3D, torch.ones(means3D.shape[0], 1, device=device)], dim=1)
            all_z = (all_hom @ src_viewmat)[:, 2]  # (N_total,); fg part has grad
            all_z_colors = all_z.unsqueeze(1).expand(-1, 3).float()
            depth_render, _, _ = render_diff(
                means3D, all_z_colors, alpha_t, scales_t, rot_t,
                src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
                rW, rH, device,
            )
            depth_composite = depth_render[0]  # (H, W), fg part has grad
            if args.sds_type == "pseudo_sds":
                depth_loss, _depth_pred_px = compute_depth_pseudo_sds(
                    depth_composite, sds_img_lat, sds_text_emb, marigold_sds_pipe,
                    device, args.sds_size, args.sds_size, args.sds_t_min, args.sds_t_max,
                )
            else:
                depth_loss = compute_depth_sds(
                    depth_composite, sds_img_lat, sds_text_emb, marigold_sds_pipe,
                    device, args.sds_size, args.sds_size, args.sds_t_min, args.sds_t_max,
                )
            total_loss = total_loss + args.depth_weight * depth_loss

        # ---- Occlusion loss: penalise fg Gaussians rendered behind bg ----
        # Pre-render fg depth map using depth-as-color (camera-space z).
        # Loss = mean of (relu(fg_depth - bg_depth))^2 over pixels where bg exists.
        # Gradient path: loss → fg_depth_map → fg_z → xyz_world → delta_xyz → MLP params.
        occlusion_loss = torch.tensor(0.0, device=device)
        if args.occlusion_weight > 0 and bg_depth_color_map is not None:
            fg_hom      = torch.cat([xyz_world, torch.ones(N_fg, 1, device=device)], dim=1)
            fg_z        = (fg_hom @ src_viewmat)[:, 2]               # (N_fg,)
            fg_z_colors = fg_z.unsqueeze(1).expand(-1, 3).float()    # (N_fg, 3)
            fg_depth_render, _, _ = render_diff(
                xyz_world, fg_z_colors, alpha_fg_t, scales_fg_t, rot_fg_t,
                src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
                rW, rH, device,
            )
            fg_depth_map   = fg_depth_render[0]  # (H, W), has grad via xyz_world
            occlusion_loss = (
                F.relu(fg_depth_map - bg_depth_color_map) ** 2 * bg_occlusion_mask
            ).sum() / (bg_occlusion_mask.sum() + 1e-6)
            total_loss = total_loss + args.occlusion_weight * occlusion_loss

        # ---- Deformation regularisation: L2 penalty on Δxyz magnitude ----
        deform_loss = (delta_xyz ** 2).mean()
        total_loss = total_loss + args.deform_reg * deform_loss

        # ---- Backprop + step ----
        total_loss.backward()
        optimizer.step()

        pbar.set_postfix(
            total=f"{total_loss.item():.4f}",
            photo=f"{rgb_loss.item():.4f}",
            sil=f"{sil_loss.item():.4f}",
            depth=f"{depth_loss.item():.4f}",
            occl=f"{occlusion_loss.item():.4f}",
            dxyz=f"{delta_xyz.detach().abs().max().item():.2e}",
        )

        if step % args.val_interval == 0 or step == args.n_steps - 1:
            with torch.no_grad():
                rendered_np = (rgb_src.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                # Silhouette render
                sil_colors_v = torch.zeros_like(colors_t)
                sil_colors_v[N_bg:] = 1.0
                sil_render_v, _, _ = render_diff(
                    means3D, sil_colors_v, alpha_t, scales_t, rot_t,
                    src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
                    rW, rH, device,
                )
                sil_np = (sil_render_v.mean(0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

                # FG-only render
                fg_only_v, _, _ = render_diff(
                    xyz_world, colors_fg_t, alpha_fg_t, scales_fg_t, rot_fg_t,
                    src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
                    rW, rH, device,
                )
                fg_only_np = (fg_only_v.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            sil_rendered = np.stack([sil_np] * 3, axis=-1)
            if mask_src_t is not None:
                gt_sil = (mask_src_t.cpu().numpy() > 0.5).astype(np.uint8) * 255
                gt_sil_rgb = np.stack([gt_sil] * 3, axis=-1)
            else:
                gt_sil_rgb = np.zeros_like(sil_rendered)
            panels = [rendered_np, ref_img_r, sil_rendered, gt_sil_rgb, fg_only_np]
            val_img = np.concatenate(panels, axis=1)
            imageio.imwrite(os.path.join(val_dir, f"val_{step:04d}.png"), val_img)

            # Depth debug (pseudo_sds only)
            if _depth_pred_px is not None and depth_composite is not None:
                import matplotlib.cm as cm
                rendered_depth_np = depth_composite.detach().cpu().numpy()
                rendered_depth_col = colorize_depth(rendered_depth_np)
                pred_np = _depth_pred_px.squeeze(0).mean(0).cpu().numpy()
                pred_01 = (pred_np.clip(-1, 1) + 1.0) / 2.0
                pred_col = (cm.get_cmap("turbo")(pred_01)[..., :3] * 255).astype(np.uint8)
                pred_col_resized = np.array(
                    Image.fromarray(pred_col).resize(
                        (rendered_depth_col.shape[1], rendered_depth_col.shape[0]),
                        Image.BILINEAR,
                    )
                )
                depth_dbg = np.concatenate([rendered_depth_col, pred_col_resized], axis=1)
                imageio.imwrite(os.path.join(depth_debug_dir, f"depth_{step:04d}.png"), depth_dbg)

    # -----------------------------------------------------------------------
    # Save trained MLP
    # -----------------------------------------------------------------------
    mlp_path = os.path.join(args.output_path, "deform_mlp.pt")
    torch.save(deform_mlp.state_dict(), mlp_path)
    print(f"\n  Saved MLP: {mlp_path}")

    # -----------------------------------------------------------------------
    # Save debug renders (final step)
    # -----------------------------------------------------------------------
    print("\n--- Saving debug renders ---")
    with torch.no_grad():
        delta_xyz_final = deform_mlp(xyz_norm_t)
        xyz_world_final = xyz_world_init_t + delta_xyz_final
        means3D_final   = torch.cat([xyz_bg_t, xyz_world_final], dim=0)
        colors_final = torch.cat([colors_bg_t, colors_fg_t], dim=0)
        alpha_final  = torch.cat([alpha_bg_t,  alpha_fg_t],  dim=0)
        rot_final    = torch.cat([rot_bg_t,    rot_fg_t],    dim=0)
        scales_final = torch.cat([scales_bg_t, scales_fg_t], dim=0)

        color_final, _, _ = render_diff(
            means3D_final, colors_final, alpha_final, scales_final, rot_final,
            src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
            rW, rH, device,
        )
        debug_render = (color_final.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(debug_dir, "refined_render_src.png"), debug_render)
        imageio.imwrite(os.path.join(debug_dir, "reference_frame.png"), ref_img_r)

        color_fg_only, _, _ = render_diff(
            xyz_world_final, colors_fg_t, alpha_fg_t, scales_fg_t, rot_fg_t,
            src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
            rW, rH, device,
        )
        fg_only_final = (color_fg_only.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(debug_dir, "fg_only_render_src.png"), fg_only_final)

    print(f"  Debug renders: {debug_dir}/")

    # -----------------------------------------------------------------------
    # Apply learned deformation to all T frames → fg_positions_world_deformed.npy
    # -----------------------------------------------------------------------
    print(f"\n--- Applying deformation to all frames in {fg_pos_world_path} ---")
    if not os.path.exists(fg_pos_world_path):
        print(f"  Warning: {fg_pos_world_path} not found — skipping deformed positions output")
    else:
        all_fg_pos_world = np.load(fg_pos_world_path)  # (T, N_fg, 3)
        T = all_fg_pos_world.shape[0]
        print(f"  Loaded fg_positions_world.npy: shape {all_fg_pos_world.shape}")

        with torch.no_grad():
            delta_xyz_np = deform_mlp(xyz_norm_t).float().cpu().numpy()  # (N_fg, 3)

        # Anchor frame 0 to refined+deformed position; add animation motion on top.
        # motion_raw is in R_orig frame (from fg_positions_world.npy built with original placement).
        # Re-express it in the refined placement frame via the correction matrix.
        motion_raw = all_fg_pos_world - all_fg_pos_world[0:1]   # (T, N_fg, 3), R_orig frame
        T_frames, N_pts = motion_raw.shape[:2]
        motion_refined = (motion_raw.reshape(-1, 3) @ correction.T).reshape(T_frames, N_pts, 3)
        fg_pos_deformed = xyz_world_init_np[None] + motion_refined + delta_xyz_np[None]

        deformed_path = os.path.join(gaussians_dir, "fg_positions_world_deformed.npy")
        np.save(deformed_path, fg_pos_deformed)
        print(f"  Saved fg_positions_world_deformed.npy: {T} frames, shape {fg_pos_deformed.shape}")
        print(f"  Max |Δxyz|: {np.abs(delta_xyz_np).max():.6f}")
        print(f"  Mean |Δxyz|: {np.abs(delta_xyz_np).mean():.6f}")

    print("\nDone.")
    print(f"  MLP:         {mlp_path}")
    print(f"  Val images:  {val_dir}/")
    if os.path.exists(fg_pos_world_path):
        print(f"  Deformed:    {deformed_path}")


if __name__ == "__main__":
    main()
