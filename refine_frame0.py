"""
refine_frame0.py — Stage 1: rigid refinement of foreground placement in frame 0.

Optimises 7 parameters: unit-quaternion Δq (4), translation Δt (3), log-scale Δs (1)
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
    --silhouette_weight 1.0
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
# Differentiable rigid transform helpers
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


def apply_rigid(xyz_canon_t, R_base_t, t_base_t, s_base, delta_q, delta_t, delta_s):
    """
    Apply refined rigid transform to canonical fg positions.
    xyz_canon_t: (N_fg, 3) float32, no grad
    Returns: means3D_fg (N_fg, 3), grad flows through delta_q/delta_t/delta_s.
    """
    R_delta = quat_to_matrix(delta_q)                     # (3,3)
    R = R_delta @ R_base_t                                # (3,3)
    s = s_base * torch.exp(delta_s.squeeze())             # scalar
    # p_world = R @ (s * p_canon) + t_refined
    means3D_fg = (R @ (xyz_canon_t * s).T).T + t_base_t + delta_t   # (N_fg, 3)
    return means3D_fg


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


def compute_crop_bbox(mask_hw, margin=0.3):
    """
    Square crop bbox around the fg mask region with fractional margin.
    Returns (x0, y0, x1, y1) in pixel coordinates, or None if mask is empty.
    """
    ys, xs = np.where(mask_hw > 0.5)
    if len(ys) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    H, W = mask_hw.shape
    h, w = y1 - y0, x1 - x0
    my = max(1, int(h * margin))
    mx = max(1, int(w * margin))
    y0, y1 = max(0, y0 - my), min(H, y1 + my)
    x0, x1 = max(0, x0 - mx), min(W, x1 + mx)
    # Expand shorter side to make square
    h, w = y1 - y0, x1 - x0
    if h < w:
        pad = (w - h) // 2
        y0, y1 = max(0, y0 - pad), min(H, y1 + pad)
    elif w < h:
        pad = (h - w) // 2
        x0, x1 = max(0, x0 - pad), min(W, x1 + pad)
    return x0, y0, x1, y1


def compute_depth_target(ref_img_np, fg_mask_hw, bg_depth_hw, marigold_pipe, debug_dir=None):
    """
    Pre-compute a metric depth target for fg pixels using Marigold anchored to
    exact bg depths.

    Strategy:
      1. Run Marigold on the FULL reference image — the full image contains bg
         pixels at varied depths (table, walls, chairs) which properly constrains
         the affine fit. Running on the full image at native resolution avoids
         distortion (no fixed-size resize).
      2. Fit affine (scale + offset) using ALL bg pixels in the full image where
         the exact metric depth is known from the rasterizer.
      3. Apply affine → metric depth for fg pixels.

    Returns:
      depth_target_hw: (H, W) float32 — metric depth at fg pixels, 0 elsewhere.
    """
    H, W = fg_mask_hw.shape

    # Run Marigold on the full reference image (native resolution, no distortion)
    print(f"  Running Marigold on full reference image ({W}×{H})")
    marigold_pred = predict_marigold(marigold_pipe, ref_img_np)  # (H, W) in [0,1]

    # Fit affine using ALL bg pixels (exact metric depth known, fg excluded)
    # Full-image bg gives varied depths → well-constrained scale+offset
    bg_valid = (bg_depth_hw > 0) & (fg_mask_hw < 0.5)
    if bg_valid.sum() < 10:
        print("  Warning: too few bg pixels — skipping depth loss")
        return None

    pred_bg = marigold_pred[bg_valid].astype(np.float64)
    rend_bg = bg_depth_hw[bg_valid].astype(np.float64)
    # Subsample if too many pixels (regression is slow for millions of points)
    if len(pred_bg) > 50000:
        idx = np.random.choice(len(pred_bg), 50000, replace=False)
        pred_bg, rend_bg = pred_bg[idx], rend_bg[idx]
    A = np.stack([pred_bg, np.ones_like(pred_bg)], axis=1)
    (s, b), _, _, _ = np.linalg.lstsq(A, rend_bg, rcond=None)
    print(f"  Depth affine: scale={s:.4f}  offset={b:.4f}"
          f"  (fit on {int(bg_valid.sum())} bg pixels)")

    # Apply affine → metric depth map; extract fg pixels only
    metric_full = (s * marigold_pred + b).astype(np.float32)
    depth_target_hw = np.where(fg_mask_hw > 0.5, metric_full, 0.0).astype(np.float32)

    fg_vals = depth_target_hw[fg_mask_hw > 0.5]
    print(f"  Depth target fg: min={fg_vals.min():.3f}  mean={fg_vals.mean():.3f}  max={fg_vals.max():.3f}")
    bg_vals = bg_depth_hw[bg_valid]
    print(f"  BG metric depth: min={bg_vals.min():.3f}  mean={bg_vals.mean():.3f}  max={bg_vals.max():.3f}")

    if debug_dir is not None:
        imageio.imwrite(os.path.join(debug_dir, "depth_target.png"),
                        colorize_depth(depth_target_hw))
        imageio.imwrite(os.path.join(debug_dir, "depth_marigold_full.png"),
                        (np.clip(marigold_pred, 0, 1) * 255).astype(np.uint8))

    return depth_target_hw


# ---------------------------------------------------------------------------
# Marigold SDS depth loss
# ---------------------------------------------------------------------------

def prepare_sds_components(marigold_pipe, ref_img_np, sds_size, device):
    """
    Pre-compute fixed components for Marigold SDS: image latent + text embedding.
    Called once before the optimisation loop.
    Returns (img_lat, text_emb) — no-grad tensors on device.
    """
    sds_H = sds_W = sds_size
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=False):
            # Preprocess reference image → [-1, 1], (1, 3, H', W')
            ref_pil = Image.fromarray(ref_img_np).resize((sds_W, sds_H), Image.BILINEAR)
            img_np = np.array(ref_pil).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            img_lat = marigold_pipe.vae.encode(img_tensor).latent_dist.mode()
            img_lat = img_lat * marigold_pipe.vae.config.scaling_factor

        # Empty-prompt text embedding (Marigold is image-conditioned, no text)
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
    """
    One SDS step using Marigold as a depth prior.

    depth_composite_hw: (H, W) float tensor WITH GRAD
        Composite depth: bg pixels have fixed bg depth, fg pixels have rendered fg depth.
    img_lat, text_emb: pre-computed, no grad.

    Gradient flows: SDS grad → depth_lat → VAE encoder → depth_composite_hw
                    → fg_z_rendered → means3D_fg → delta_t / delta_q / delta_s
    """
    import random

    H, W = depth_composite_hw.shape

    # 1. Resize composite depth to SDS resolution (differentiable via interpolate)
    d = depth_composite_hw.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    d_r = F.interpolate(d, size=(sds_H, sds_W), mode='bilinear', align_corners=False)  # (1,1,H',W')

    # 2. Normalise to [-1, 1] (affine-invariant — Marigold only cares about relative depth)
    d_min = d_r.detach().min()
    d_max = d_r.detach().max()
    d_norm = 2.0 * (d_r - d_min) / (d_max - d_min + 1e-6) - 1.0  # [-1, 1], grad preserved
    d_vae = d_norm.expand(1, 3, sds_H, sds_W)  # (1, 3, H', W')

    # 3. Encode with VAE (WITH grad — needed so SDS gradient flows back to depth_composite)
    with torch.amp.autocast("cuda", enabled=False):
        depth_lat = marigold_pipe.vae.encode(d_vae.float()).latent_dist.mode()
        depth_lat = depth_lat * marigold_pipe.vae.config.scaling_factor  # (1, 4, H'/8, W'/8)

    # 4. Sample noise level t and noise
    t_val = random.randint(t_min, t_max)
    t_b = torch.tensor([t_val], device=device, dtype=torch.long)
    eps = torch.randn_like(depth_lat)

    # 5. Noisy depth latent
    noisy_depth = marigold_pipe.scheduler.add_noise(depth_lat.detach(), eps, t_b)

    # 6. U-Net input: [image_latent (4ch) | noisy_depth (4ch)] = 8 channels
    unet_in = torch.cat([img_lat, noisy_depth], dim=1)

    # 7. Predict noise — no grad through U-Net params
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=False):
            noise_pred = marigold_pipe.unet(
                unet_in.float(), t_b, encoder_hidden_states=text_emb.float(),
            ).sample  # (1, 4, H'/8, W'/8)

    # 8. Handle v-prediction parameterisation (SD 2.1 uses v_prediction)
    pred_type = getattr(marigold_pipe.scheduler.config, 'prediction_type', 'epsilon')
    if pred_type == 'v_prediction':
        alpha_bar = marigold_pipe.scheduler.alphas_cumprod[t_val].to(device=device, dtype=torch.float32)
        alpha_t = alpha_bar ** 0.5
        sigma_t = (1.0 - alpha_bar) ** 0.5
        # Convert v → epsilon prediction
        noise_pred = sigma_t * noisy_depth.detach() + alpha_t * noise_pred

    # 9. SDS gradient: push depth_lat toward the data manifold
    #    grad = eps_hat - eps  (no grad through U-Net)
    sds_grad = (noise_pred - eps).detach()

    # 10. Inject gradient: d(loss)/d(depth_lat) = sds_grad / N
    #     .mean() keeps loss O(1) and comparable to other loss terms.
    loss = (depth_lat * sds_grad).mean()

    return loss


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
    parser.add_argument("--seg_prompt_path", default=None,
                        help="Text prompt file for LangSAM auto-segmentation of reference frame "
                             "(used if --mask_path is not provided)")
    parser.add_argument("--prompt_path", default=None,
                        help="Text prompt file (.txt) for Flux Fill")
    parser.add_argument("--output_path", required=True,
                        help="Output directory for placement_refined.json and debug images")
    # Optimisation
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for all parameters (default for --lr_rot/--lr_t/--lr_s)")
    parser.add_argument("--lr_rot", type=float, default=1e-3,
                        help="Learning rate for rotation quaternion (default 1e-3)")
    parser.add_argument("--lr_t", type=float, default=1e-2,
                        help="Learning rate for translation (default 1e-2, 10x lr_rot)")
    parser.add_argument("--lr_s", type=float, default=1e-3,
                        help="Learning rate for log-scale (default 1e-3)")
    parser.add_argument("--render_scale", type=float, default=1.0,
                        help="Render scale for source camera (default 1.0)")
    # Loss weights
    parser.add_argument("--rgb_weight",         type=float, default=1.0)
    parser.add_argument("--silhouette_weight", type=float, default=1.0)
    parser.add_argument("--depth_weight",      type=float, default=0.0,
                        help="Marigold depth loss weight (default 0, loads Marigold if > 0)")
    parser.add_argument("--depth_mode", default="sds", choices=["sds", "affine"],
                        help="'sds': Marigold SDS prior (default); 'affine': L1 to bg-anchored affine target")
    # SDS settings
    parser.add_argument("--sds_t_min", type=int, default=50,
                        help="Min noise timestep for SDS (default 50)")
    parser.add_argument("--sds_t_max", type=int, default=500,
                        help="Max noise timestep for SDS (default 500)")
    parser.add_argument("--sds_size", type=int, default=512,
                        help="Spatial size for SDS U-Net (square, must be 64-multiple, default 512)")
    # Affine-mode settings
    parser.add_argument("--depth_one_sided", action="store_true",
                        help="(affine mode) One-sided hinge: only penalise when fg is too deep")
    parser.add_argument("--flux_weight",       type=float, default=0.0,
                        help="Flux pseudo-GT loss weight (default 0, loads Flux if > 0)")
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
    # Reference frame — load first so render resolution matches it
    # -----------------------------------------------------------------------
    # The reference was produced by PIL-resizing the 3DGS render to (W_ref, H_ref)
    # (e.g. Wan's 832×480 target).  To match that, we render at the same aspect
    # ratio by using rW = W_ref * render_scale, rH = H_ref * render_scale while
    # keeping the camera's tanfovx / tanfovy unchanged.  The GaussianRasterizer
    # produces identical content to a render-then-PIL-resize pipeline.
    print("\n--- Loading reference frame ---")
    ref_img = np.array(Image.open(args.reference_frame).convert("RGB"))
    W_ref, H_ref = ref_img.shape[1], ref_img.shape[0]
    rW = max(1, round(W_ref * args.render_scale))
    rH = max(1, round(H_ref * args.render_scale))
    ref_img_r = np.array(Image.fromarray(ref_img).resize((rW, rH), Image.BILINEAR))
    ref_t = torch.from_numpy(ref_img_r.astype(np.float32) / 255.0).to(device).permute(2, 0, 1)
    print(f"  Reference: {W_ref}×{H_ref} → render {rW}×{rH}")

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
    # GT silhouette mask (source camera)
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
            delta_q0 = torch.tensor([1., 0., 0., 0.], device=device)
            delta_t0 = torch.zeros(3, device=device)
            delta_s0 = torch.zeros(1, device=device)
            means3D_fg_init = apply_rigid(
                xyz_fg_t, R_base_t, t_base_t, s_base, delta_q0, delta_t0, delta_s0
            )
            means3D_init = torch.cat([xyz_bg_t, means3D_fg_init], dim=0)
            scales_init  = torch.cat([scales_bg_t, scales_fg_base_t * s_base], dim=0)

        for cam_i in selected:
            pose_w2c_i, FoVx_i, FoVy_i, W_i, H_i = all_cams[cam_i]
            W_f, H_f = scale_to_flux_res(W_i, H_i, args.flux_res)
            viewmat_i, fullproj_i, campos_i, tfovx_i, tfovy_i = \
                make_raster_camera_at_resolution(pose_w2c_i, FoVx_i, W_f, H_f, device)

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
    # Depth supervision pre-computation
    # -----------------------------------------------------------------------
    bg_depth_src_t  = None   # (H, W) fixed bg depth at source cam — used by both modes
    marigold_sds_pipe = None  # kept alive for SDS mode
    sds_img_lat     = None
    sds_text_emb    = None
    depth_fg_target_t = None  # only for affine mode

    if args.depth_weight > 0:
        if mask_np is None:
            print("  Warning: --depth_weight > 0 but no mask available — skipping depth loss")
        else:
            print("\n--- Pre-computing background depth at source camera ---")
            with torch.no_grad():
                _, bg_depth_raw, _ = render_diff(
                    xyz_bg_t, colors_t[:N_bg], alpha_t[:N_bg], scales_bg_t, rot_t[:N_bg],
                    src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
                    rW, rH, device,
                )
            bg_depth_src_t = bg_depth_raw.squeeze(0)  # (H, W) tensor, no grad

            if args.depth_mode == 'sds':
                print(f"\n--- Loading Marigold for SDS (size={args.sds_size}, t=[{args.sds_t_min},{args.sds_t_max}]) ---")
                marigold_sds_pipe = load_marigold(device)
                sds_img_lat, sds_text_emb = prepare_sds_components(
                    marigold_sds_pipe, ref_img_r, args.sds_size, device
                )
            else:  # affine
                print("\n--- Computing Marigold depth target (bg-anchored affine) ---")
                marigold_pipe = load_marigold(device)
                depth_target_hw = compute_depth_target(
                    ref_img_r, fg_mask_gt, bg_depth_src_t.cpu().numpy(),
                    marigold_pipe, debug_dir=debug_dir,
                )
                del marigold_pipe
                torch.cuda.empty_cache()
                if depth_target_hw is not None:
                    depth_fg_target_t = torch.from_numpy(depth_target_hw).float().to(device)
                    print(f"  Depth target: {int((depth_target_hw > 0).sum())} fg pixels")

    # -----------------------------------------------------------------------
    # Rigid parameters + optimiser
    # -----------------------------------------------------------------------
    print("\n--- Initialising optimisation ---")
    delta_q = nn.Parameter(torch.tensor([1., 0., 0., 0.], device=device))
    delta_t = nn.Parameter(torch.zeros(3, device=device))
    delta_s = nn.Parameter(torch.zeros(1, device=device))
    lr_rot = args.lr_rot
    lr_t   = args.lr_t
    lr_s   = args.lr_s
    optimizer = torch.optim.Adam([
        {"params": [delta_q], "lr": lr_rot},
        {"params": [delta_t], "lr": lr_t},
        {"params": [delta_s], "lr": lr_s},
    ])

    best_total = float("inf")
    best_state = {"delta_q": delta_q.data.clone(),
                  "delta_t": delta_t.data.clone(),
                  "delta_s": delta_s.data.clone()}

    # -----------------------------------------------------------------------
    # Optimisation loop
    # -----------------------------------------------------------------------
    print(f"\n--- Optimising {args.n_steps} steps ---")
    pbar = tqdm(range(args.n_steps), desc="optimising")
    for step in pbar:
        optimizer.zero_grad()

        # Compute refined fg positions and scales (both depend on delta_s)
        means3D_fg = apply_rigid(
            xyz_fg_t, R_base_t, t_base_t, s_base, delta_q, delta_t, delta_s
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
        rgb_loss = (rgb_src - ref_t).abs()  # (3, H, W)
        if alpha_src is not None:
            fg_weight = alpha_src.detach().clamp(0, 1)  # (1, H, W), no grad
            rgb_loss = (rgb_loss * fg_weight).mean()
        else:
            rgb_loss = rgb_loss.mean()

        total_loss = args.rgb_weight * rgb_loss

        # ---- Silhouette loss (occlusion-aware via alpha compositing) ----
        # Render combined scene: fg=white, bg=black. Alpha compositing naturally
        # dims fg contribution where bg Gaussians are in front (occlusion).
        # Differentiable through the standard color backward pass.
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

        # ---- Depth loss (Marigold SDS prior or affine target) ----
        # Common: render fg depth as colour so gradient flows differentiably to delta_t.
        # SDS mode:   composite with fixed bg depth → Marigold scores the full depth map.
        # Affine mode: compare fg depth against bg-anchored Marigold affine target (L1/hinge).
        depth_loss = torch.tensor(0.0, device=device)
        if args.depth_weight > 0 and bg_depth_src_t is not None and mask_src_t is not None:
            # Camera-space z for each fg Gaussian (differentiable via means3D_fg)
            fg_hom = torch.cat([means3D_fg,
                                 torch.ones(means3D_fg.shape[0], 1, device=device)], dim=1)
            fg_z = (fg_hom @ src_viewmat)[:, 2]  # (N_fg,) metric depth, has grad
            fg_z_colors = fg_z.unsqueeze(1).expand(-1, 3).float()  # (N_fg, 3) depth-as-color
            fg_depth_render, _, _ = render_diff(
                means3D_fg, fg_z_colors, alpha_t[N_bg:], scales_t[N_bg:], rot_t[N_bg:],
                src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
                rW, rH, device,
            )
            fg_z_rendered = fg_depth_render[0]  # (H, W) alpha-weighted fg depth, has grad

            if args.depth_mode == 'sds' and marigold_sds_pipe is not None:
                # Composite: fg pixels → fg_z_rendered (grad), bg pixels → bg_depth_src_t (no grad)
                fg_mask_f = (mask_src_t > 0.5).float()
                depth_composite = bg_depth_src_t * (1.0 - fg_mask_f) + fg_z_rendered * fg_mask_f
                depth_loss = compute_depth_sds(
                    depth_composite, sds_img_lat, sds_text_emb, marigold_sds_pipe,
                    device, args.sds_size, args.sds_size, args.sds_t_min, args.sds_t_max,
                )
                total_loss = total_loss + args.depth_weight * depth_loss

            elif args.depth_mode == 'affine' and depth_fg_target_t is not None:
                if step == 0:
                    tgt = depth_fg_target_t[depth_fg_target_t > 0]
                    print(f"\n  [Depth diag step 0]"
                          f"  fg_z mean={fg_z.detach().mean():.3f}"
                          f"  target mean={tgt.mean():.3f}"
                          f"  one_sided={args.depth_one_sided}")
                fg_valid = (mask_src_t > 0.5) & (depth_fg_target_t > 0)
                if fg_valid.sum() > 0:
                    if args.depth_one_sided:
                        depth_loss = F.relu(fg_z_rendered[fg_valid] - depth_fg_target_t[fg_valid]).mean()
                    else:
                        depth_loss = (fg_z_rendered[fg_valid] - depth_fg_target_t[fg_valid]).abs().mean()
                    total_loss = total_loss + args.depth_weight * depth_loss

        # ---- Flux pseudo-GT loss (cycle through training cameras) ----
        flux_loss = torch.tensor(0.0, device=device)
        if args.flux_weight > 0 and flux_pseudogt_targets:
            cam_info = flux_pseudogt_targets[step % len(flux_pseudogt_targets)]
            viewmat_f, fullproj_f, campos_f, tfovx_f, tfovy_f = \
                make_raster_camera_at_resolution(
                    cam_info["pose_w2c"], cam_info["FoVx"],
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
                "delta_q": delta_q.data.clone(),
                "delta_t": delta_t.data.clone(),
                "delta_s": delta_s.data.clone(),
            }

        dq_dev = (delta_q.data / delta_q.data.norm() - delta_q.data.new_tensor([1., 0., 0., 0.])).norm()
        pbar.set_postfix(
            total=f"{total_loss.item():.4f}",
            photo=f"{rgb_loss.item():.4f}",
            sil=f"{sil_loss.item():.4f}",
            depth=f"{depth_loss.item():.4f}",
            flux=f"{flux_loss.item():.4f}",
            dq=f"{dq_dev.item():.4f}",
        )

        if step % args.val_interval == 0 or step == args.n_steps - 1:
            # Save rendered | reference | depth side-by-side
            with torch.no_grad():
                rendered_np = (rgb_src.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                depth_np    = depth_src.detach().cpu().squeeze().numpy()   # (H, W)
                alpha_np    = (alpha_src.detach().cpu().squeeze().numpy()
                               if alpha_src is not None else None)
                # Occlusion-aware silhouette via alpha compositing
                sil_colors_v = torch.zeros_like(colors_t)
                sil_colors_v[N_bg:] = 1.0
                sil_render_v, _, _ = render_diff(
                    means3D, sil_colors_v, alpha_t, scales_t, rot_t,
                    src_viewmat, src_fullproj, src_campos, src_tfovx, src_tfovy,
                    rW, rH, device,
                )
                sil_np = (sil_render_v.mean(0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            depth_colored = colorize_depth(depth_np, alpha_hw=alpha_np)
            # Silhouette panels: rendered fg silhouette and GT mask (always shown)
            sil_rendered = np.stack([sil_np] * 3, axis=-1)
            if mask_src_t is not None:
                gt_sil = (mask_src_t.cpu().numpy() > 0.5).astype(np.uint8) * 255
                gt_sil_rgb = np.stack([gt_sil] * 3, axis=-1)
            else:
                gt_sil_rgb = np.zeros_like(sil_rendered)
            panels = [rendered_np, ref_img_r, depth_colored, sil_rendered, gt_sil_rgb]
            val_img = np.concatenate(panels, axis=1)
            imageio.imwrite(os.path.join(val_dir, f"val_{step:04d}.png"), val_img)

    # -----------------------------------------------------------------------
    # Apply best delta to base transform → refined placement
    # -----------------------------------------------------------------------
    print("\n--- Computing refined placement ---")
    with torch.no_grad():
        R_delta = quat_to_matrix(best_state["delta_q"])
        R_refined = (R_delta @ R_base_t).float().cpu().numpy().tolist()
        t_refined = (t_base_t + best_state["delta_t"]).float().cpu().numpy().tolist()
        s_refined = float(s_base * torch.exp(best_state["delta_s"].squeeze()).item())
        dq_norm = best_state["delta_q"] / best_state["delta_q"].norm()
        dq_dev  = (dq_norm - dq_norm.new_tensor([1., 0., 0., 0.])).norm().item()

    print(f"  s_refined:  {s_refined:.4f}  (base {s_base:.4f})")
    print(f"  t_refined:  {[f'{x:.4f}' for x in t_refined]}")
    print(f"  |Δq - I|:   {dq_dev:.4f}")
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
        dq = best_state["delta_q"].to(device)
        dt = best_state["delta_t"].to(device)
        ds = best_state["delta_s"].to(device)
        means3D_fg_ref = apply_rigid(xyz_fg_t, R_base_t, t_base_t, s_base, dq, dt, ds)
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
