"""
run_nvs_refine.py

Refine a 4DGS render video using Wan I2V with Langevin guidance.

The fg region is guided toward the reference render with strength controlled
by --lambda_val.  The bg region is hard-replaced at each denoising step.

Algorithm (Euler-Maruyama Langevin in flow-matching space):
  At each outer step t:
    1. Model call (2 CFG passes) → v_pred; compute x0_pred = x_t - sigma_t * v_pred
    2. N inner Langevin steps (NO extra model calls, uses cached x0_pred):
         score_x = -(x_t - x0_pred)               [standard FM denoising]
         score_y = -(1+λ)*(x_t - y) + λ*(x_t - x0_pred)  [guided toward render]
         score   = score_x*(1-fg_mask) + score_y*fg_mask
         x_t ← x_t + eta*score + sqrt(2*eta)*noise_inj
    3. Scheduler step from post-Langevin x_t
    4. Hard-replace bg with noisy render at next timestep level

lambda_val=0   → fg loosely guided toward render (more diffusion freedom)
lambda_val=4   → fg guided toward render, bg still locked
lambda_val=100 → fg nearly identical to render (quality enhancement only)
"""

import sys
import os
import glob
import math

# Fix sklearn TLS issue: LD_PRELOAD must be set before the dynamic linker loads
# scikit-learn's bundled libgomp.  Re-exec the process with it preloaded.
if "LIBGOMP_PRELOADED" not in os.environ:
    libgomp_files = glob.glob(
        "/home/ubuntu/miniconda3/envs/mvadapter/lib/python*/site-packages/scikit_learn.libs/libgomp*.so*"
    )
    if libgomp_files:
        os.environ["LD_PRELOAD"] = libgomp_files[0]
        os.environ["LIBGOMP_PRELOADED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from huggingface_hub import snapshot_download
import torchvision.transforms.functional as TF
import argparse
import gc
import cv2

sys.path.insert(0, os.path.abspath("official_wan_repo"))

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import cache_video
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def load_video_frames(path, width, height, frame_num):
    """Load video → [3, F, H, W] tensor, values in [-1, 1]."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame).resize((width, height))
        frames.append(TF.to_tensor(frame_pil).sub_(0.5).div_(0.5))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    # Pad or truncate to frame_num
    while len(frames) < frame_num:
        frames.extend(frames[:frame_num - len(frames)])
    frames = frames[:frame_num]
    return torch.stack(frames, dim=1)  # [3, F, H, W]


def load_mask_video(path, width, height, frame_num):
    """Load mask video → [F, H, W] binary float (1=fg, 0=bg)."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST)
        frames.append(torch.from_numpy((gray > 128).astype(np.float32)))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    while len(frames) < frame_num:
        frames.extend(frames[:frame_num - len(frames)])
    frames = frames[:frame_num]
    return torch.stack(frames)  # [F, H, W]


def main():
    parser = argparse.ArgumentParser(description="NVS Refinement via Wan I2V + Langevin guidance")
    parser.add_argument("--front_video",     type=str, required=True,
                        help="front.mp4 — full scene RGB render (fg + bg)")
    parser.add_argument("--alpha_video",     type=str, required=True,
                        help="front_alpha.mp4 — fg silhouette (white=fg)")
    parser.add_argument("--frame0",          type=str, required=True,
                        help="front_frame0.png — I2V conditioning image (frame 0)")
    parser.add_argument("--prompt_path",     type=str, required=True,
                        help="Text prompt .txt file")
    parser.add_argument("--output_path",     type=str, required=True,
                        help="Output .mp4 path")
    parser.add_argument("--lambda_val",      type=float, default=4.0,
                        help="Fg guidance strength (0=loose, inf=locked to render)")
    parser.add_argument("--langevin_steps",  type=int,   default=5,
                        help="Inner Langevin iterations per denoising step (0=off)")
    parser.add_argument("--langevin_step_size", type=float, default=0.2,
                        help="Langevin base step size (eta ∝ step_size * sigma_t)")
    parser.add_argument("--guide_scale",     type=float, default=5.0,
                        help="CFG scale")
    parser.add_argument("--sampling_steps",  type=int,   default=40,
                        help="Number of denoising steps")
    parser.add_argument("--seed",            type=int,   default=42,
                        help="RNG seed")
    parser.add_argument("--frame_num",       type=int,   default=81,
                        help="Number of frames")
    parser.add_argument("--width",           type=int,   default=832,
                        help="Video width")
    parser.add_argument("--height",          type=int,   default=480,
                        help="Video height")
    parser.add_argument("--checkpoint_dir",  type=str,   default=None,
                        help="WanI2V 14B checkpoint dir (auto-downloads if None)")
    parser.add_argument("--mask_dilation",   type=int,   default=15,
                        help="Spatial mask dilation (pixels) before latent downsampling")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(out_dir, exist_ok=True)

    # Save args for reference
    with open(os.path.join(out_dir, "nvs_refine_args.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    device_id = 0
    rank = 0
    device = torch.device(f"cuda:{device_id}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    task = "i2v-14B"
    repo_id = "Wan-AI/Wan2.1-I2V-14B-720P"

    if args.checkpoint_dir is None:
        print(f"Downloading/loading {repo_id}...")
        checkpoint_dir = snapshot_download(repo_id=repo_id)
    else:
        checkpoint_dir = args.checkpoint_dir
    print(f"Checkpoint: {checkpoint_dir}")

    cfg = WAN_CONFIGS[task]
    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        device_id=device_id,
        rank=rank,
        t5_cpu=True,
    )

    # ------------------------------------------------------------------
    # 1. Load and encode reference video (y_latents)
    # ------------------------------------------------------------------
    frame_num = args.frame_num
    print(f"Loading reference video: {args.front_video}")
    ref_video = load_video_frames(args.front_video, args.width, args.height, frame_num).to(device)
    # ref_video: [3, F, H, W]

    print("Encoding reference video to latents...")
    with torch.no_grad():
        y_latents = wan_i2v.vae.encode([ref_video])[0]       # [16, F_lat, H_lat, W_lat]
        y_latents_batched = y_latents.unsqueeze(0)            # [1, 16, F_lat, H_lat, W_lat]
    del ref_video

    f_lat  = y_latents.shape[1]
    lat_h  = y_latents.shape[2]
    lat_w  = y_latents.shape[3]
    print(f"Latent dims: f_lat={f_lat} lat_h={lat_h} lat_w={lat_w}")

    # ------------------------------------------------------------------
    # 2. Load and prepare dynamic fg mask
    # ------------------------------------------------------------------
    print(f"Loading mask video: {args.alpha_video}")
    mask_raw = load_mask_video(args.alpha_video, args.width, args.height, frame_num)
    # mask_raw: [F, H, W] binary float

    # [1, 1, F, H, W] for 3-D pooling ops
    mask_5d = mask_raw.unsqueeze(0).unsqueeze(0)  # [1, 1, F, H, W]

    # Spatial dilation before downsampling (preserves fg boundaries)
    if args.mask_dilation > 0:
        k = args.mask_dilation
        if k % 2 == 0:
            k += 1
        mask_5d = torch.nn.functional.max_pool3d(
            mask_5d,
            kernel_size=(1, k, k),
            stride=1,
            padding=(0, k // 2, k // 2),
        )

    # Temporal dilation: fill tracking gaps (any valid frame within window counts)
    tk = 9
    mask_5d = torch.nn.functional.max_pool3d(
        mask_5d,
        kernel_size=(tk, 1, 1),
        stride=1,
        padding=(tk // 2, 0, 0),
    )

    # Downsample to latent spatial/temporal resolution using max-pool
    fg_mask_latent = torch.nn.functional.adaptive_max_pool3d(
        mask_5d, output_size=(f_lat, lat_h, lat_w)
    )  # [1, 1, F_lat, H_lat, W_lat]
    fg_mask_latent = (fg_mask_latent > 0.5).float()
    # Expand to all 16 latent channels
    fg_mask_latent = fg_mask_latent.repeat(1, 16, 1, 1, 1).to(device)  # [1, 16, F_lat, H_lat, W_lat]
    fg_mask = fg_mask_latent.squeeze(0)                                  # [16, F_lat, H_lat, W_lat]
    print(f"Fg mask: shape={fg_mask.shape}, fg_fraction={fg_mask.mean():.3f}")

    # ------------------------------------------------------------------
    # 3. I2V conditioning
    # ------------------------------------------------------------------
    input_img = Image.open(args.frame0).convert("RGB").resize((args.width, args.height))
    input_tensor = TF.to_tensor(input_img).sub_(0.5).div_(0.5).to(device)  # [3, H, W]

    with open(args.prompt_path, "r") as f:
        prompt = f.read().strip()
    print(f"Prompt: {prompt}")

    print("Encoding text...")
    wan_i2v.text_encoder.model.to(device)
    context      = wan_i2v.text_encoder([prompt], device)
    context_null = wan_i2v.text_encoder([wan_i2v.sample_neg_prompt], device)
    wan_i2v.text_encoder.model.cpu()

    print("Encoding CLIP image...")
    wan_i2v.clip.model.to(device)
    clip_context = wan_i2v.clip.visual([input_tensor.unsqueeze(1)])  # expects [C, 1, H, W] inside list
    wan_i2v.clip.model.cpu()

    # VAE-encode frame0 + zero-padding for I2V conditioning (y)
    vae_stride = wan_i2v.vae_stride   # (4, 8, 8)
    patch_size = wan_i2v.patch_size   # (1, 2, 2)

    y_cond = wan_i2v.vae.encode([
        torch.cat([
            torch.nn.functional.interpolate(
                input_tensor.unsqueeze(0).cpu(),
                size=(lat_h * vae_stride[1], lat_w * vae_stride[2]),
                mode='bicubic',
            ).transpose(0, 1),                                                    # [3, 1, H_px, W_px]
            torch.zeros(3, frame_num - 1, lat_h * vae_stride[1], lat_w * vae_stride[2]),
        ], dim=1).to(device)
    ])[0]  # [16, F_lat, H_lat, W_lat]

    # I2V conditioning mask (first temporal frame valid; rest zero)
    msk = torch.ones(1, frame_num, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]
    y_cond = torch.concat([msk, y_cond])  # prepend mask channels

    # Offload VAE to free GPU memory before loading the 14B DiT
    wan_i2v.vae.model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Initialise noise, scheduler, latent
    # ------------------------------------------------------------------
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
    )
    scheduler.set_timesteps(args.sampling_steps, device=device, shift=5.0)
    timesteps = scheduler.timesteps

    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(args.seed)

    noise = torch.randn(
        16, f_lat, lat_h, lat_w,
        dtype=torch.float32, generator=seed_g, device=device,
    )
    latent = noise.clone()

    max_seq_len = f_lat * lat_h * lat_w // (patch_size[1] * patch_size[2])
    arg_c    = {'context': [context[0]],  'clip_fea': clip_context, 'seq_len': max_seq_len, 'y': [y_cond]}
    arg_null = {'context': context_null, 'clip_fea': clip_context, 'seq_len': max_seq_len, 'y': [y_cond]}

    wan_i2v.model = wan_i2v.model.to(torch.bfloat16).to(device)

    # ------------------------------------------------------------------
    # 5. Denoising loop with Langevin guidance
    # ------------------------------------------------------------------
    print(f"\nStarting denoising: {args.sampling_steps} steps, "
          f"{args.langevin_steps} Langevin iters/step, "
          f"lambda={args.lambda_val}, guide_scale={args.guide_scale}")

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=wan_i2v.param_dtype):
        for i, t in enumerate(tqdm(timesteps)):
            # sigma_t ≈ t/1000 ≈ t_norm (flow-matching noise level at this step)
            sigma_t = scheduler.sigmas[i].to(device)

            # --- 1. Standard CFG model call ---
            v_cond   = wan_i2v.model([latent], t=torch.stack([t]), **arg_c)[0]
            v_uncond = wan_i2v.model([latent], t=torch.stack([t]), **arg_null)[0]
            v_pred   = v_uncond + args.guide_scale * (v_cond - v_uncond)

            # x0 prediction: flow-matching identity x_0 = x_t - t_norm * v
            x0_pred = latent - sigma_t * v_pred

            # --- 2. Inner Langevin iterations (no model calls; uses cached x0_pred) ---
            # Implements the LanPaint IS_FLOW score in FM space:
            #   score_x = -(x_t - x0_pred)         [standard diffusion score]
            #   score_y = -(1+λ)*(x_t-y) + λ*(x_t-x0_pred)  [guided toward render]
            #   step:  x_t ← x_t + eta*score + sqrt(2*eta)*noise
            x_t = latent.clone()
            if args.langevin_steps > 0:
                eta = args.langevin_step_size * max(sigma_t.item(), 1e-4)
                noise_scale = math.sqrt(2.0 * eta)
                for _ in range(args.langevin_steps):
                    score_x = -(x_t - x0_pred)
                    score_y = (-(1.0 + args.lambda_val) * (x_t - y_latents)
                               + args.lambda_val * (x_t - x0_pred))
                    score = score_x * (1.0 - fg_mask) + score_y * fg_mask
                    x_t = x_t + eta * score + noise_scale * torch.randn_like(x_t)

            # --- 3. Scheduler step from post-Langevin x_t ---
            # Uses v_pred from the pre-Langevin latent (cached, no extra model call).
            latent_new = scheduler.step(
                v_pred.unsqueeze(0), t, x_t.unsqueeze(0),
                return_dict=False, generator=seed_g,
            )[0].squeeze(0)

            # --- 4. Hard bg replace ---
            # bg_noisy = noisy version of the clean render at the *next* noise level
            if i < len(timesteps) - 1:
                next_t = timesteps[i + 1]
                bg_noisy = scheduler.add_noise(
                    y_latents_batched,
                    noise.unsqueeze(0),
                    torch.stack([next_t]),
                ).squeeze(0)
            else:
                bg_noisy = y_latents  # final step: use clean render for bg

            latent = fg_mask * latent_new + (1.0 - fg_mask) * bg_noisy

    # ------------------------------------------------------------------
    # 6. Decode and save
    # ------------------------------------------------------------------
    print("Decoding...")
    wan_i2v.model.cpu()
    wan_i2v.vae.model.to(device)
    with torch.no_grad():
        videos = wan_i2v.vae.decode([latent.to(device)])

    cache_video(
        tensor=videos[0][None],
        save_file=args.output_path,
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    print(f"Saved → {args.output_path}")


if __name__ == "__main__":
    main()
