"""
run_nvs_refine.py

Refine a 4DGS render video using Wan I2V with Langevin guidance.

Algorithm: LanPaint IS_FLOW=True (closely following LanPaint/src/LanPaint/lanpaint.py).

At each outer denoising step (sigma_t = t_norm):
  1. Replace step: set fg to VE-noisy render
       VE_sigma = sigma_t / (1-sigma_t)
       x_replaced = x_bg*(1-fg_mask) + (y_latents + VE_sigma*noise)*fg_mask
  2. VP conversion:  f = sqrt(1-sigma_t) + sqrt(sigma_t)
                    x_t_vp = x_replaced * f
  3. N inner Langevin steps (each makes a model call):
       x_fm = x_t_vp / f
       x0_fm = x_fm - sigma_t * CFG_velocity(x_fm, t)
       score_x = -(x_t_vp - x0_fm)                            [bg]
       score_y = -(1+λ)*(x_t_vp - y_latents) + λ*(x_t_vp-x0_fm)  [fg, CLEAN y_latents]
       score   = score_x*(1-fg_mask) + score_y*fg_mask
       Overdamped OU step (exact):
         x0_eff = x_t_vp + score
         mean   = e^{-A*η}*x_t_vp + (1-e^{-A*η})*sqrt(abt)*x0_eff
         var    = sigma_t*(1-e^{-2*A*η}),  A = 1/sigma_t,  η = step_size
         x_t_vp ~ N(mean, var)
  4. VP→FM:  x_post = x_t_vp / f
  5. Final model call → x0_final; pin fg: x0_final_fg ← y_latents
  6. Scheduler step with v_eff = (x_post - x0_final) / sigma_t
  7. Hard bg replace: latent_bg = (1-sigma_next)*y_latents + sigma_next*noise

NOTE: With n_inner inner steps, each outer denoising step makes 2*(n_inner+1) model
passes (CFG = 2 per call).  Default 5 inner steps → 480 passes for 40 outer steps.

lambda_val=0   → fg freely generated (standard I2V, only bg locked)
lambda_val=4   → fg guided toward render (default)
lambda_val=100 → fg nearly identical to render
"""

import sys
import os
import glob

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
from taehv import WanCompatibleTAEHV
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
    parser.add_argument("--preview_interval", type=int,  default=0,
                        help="Save a TAEHV preview video every N denoising steps (0=disabled). "
                             "Requires checkpoints/taew2_1.pth.")
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
    # Optional: TAEHV fast decoder for step previews
    # ------------------------------------------------------------------
    taehv = None
    if args.preview_interval > 0:
        taehv_ckpt = "checkpoints/taew2_1.pth"
        if os.path.exists(taehv_ckpt):
            print(f"Loading TAEHV for previews from {taehv_ckpt}")
            taehv = WanCompatibleTAEHV(checkpoint_path=taehv_ckpt).to(device).eval()
        else:
            print(f"WARNING: --preview_interval set but {taehv_ckpt} not found; previews disabled.")

    preview_dir = os.path.join(out_dir, "previews")
    if taehv is not None:
        os.makedirs(preview_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 5. Denoising loop — LanPaint IS_FLOW=True
    # ------------------------------------------------------------------
    print(f"\nStarting denoising: {args.sampling_steps} steps, "
          f"{args.langevin_steps} Langevin iters/step, "
          f"lambda={args.lambda_val}, guide_scale={args.guide_scale}")

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=wan_i2v.param_dtype):
        for i, t in enumerate(tqdm(timesteps)):
            sigma_t = scheduler.sigmas[i].to(device)       # ≈ t_norm ∈ (0,1)
            abt     = (1.0 - sigma_t).clamp(min=0.0)       # alpha_bar_t = 1 - t_norm
            sigma_t_safe = sigma_t.clamp(min=1e-4)

            # VP conversion factor:  f = sqrt(abt) + sqrt(sigma_t)
            f = abt.sqrt() + sigma_t_safe.sqrt()

            if args.langevin_steps > 0:
                # ---- 1. Replace step (LanPaint __call__ line 60) ----
                # Force fg to match the VE-noisy reference render.
                # VE_sigma = sigma_FM / (1 - sigma_FM)  [flow-matching → VE parameterisation]
                abt_safe = abt.clamp(min=1e-4)
                VE_sigma = sigma_t / abt_safe
                x_replaced = latent * (1 - fg_mask) + (y_latents + VE_sigma * noise) * fg_mask

                # ---- 2. VP conversion (LanPaint line 63) ----
                x_t_vp = x_replaced * f

                # ---- 3. Inner Langevin steps ----
                # Overdamped OU (exact): A = 1/sigma_t, D = sqrt(2).
                # Over one step of size eta:
                #   mean = e^{-A*eta}*x + (1-e^{-A*eta})*sqrt(abt)*x0_eff
                #   var  = sigma_t * (1 - e^{-2*A*eta})
                A_eta   = args.langevin_step_size / sigma_t_safe   # = A * eta
                exp_neg  = torch.exp(-A_eta)
                exp_neg2 = torch.exp(-2.0 * A_eta)

                for _ in range(args.langevin_steps):
                    # Convert VP → FM, call model
                    x_fm = x_t_vp / f
                    v_c  = wan_i2v.model([x_fm], t=torch.stack([t]), **arg_c)[0]
                    v_u  = wan_i2v.model([x_fm], t=torch.stack([t]), **arg_null)[0]
                    v_inner = v_u + args.guide_scale * (v_c - v_u)
                    x0_fm   = x_fm - sigma_t * v_inner

                    # Score (LanPaint score_model IS_FLOW, lines 139-141).
                    # x_t_vp is VP; x0_fm and y_latents are FM — LanPaint mixes spaces.
                    score_x = -(x_t_vp - x0_fm)
                    score_y = (-(1.0 + args.lambda_val) * (x_t_vp - y_latents)
                               + args.lambda_val * (x_t_vp - x0_fm))
                    score   = score_x * (1 - fg_mask) + score_y * fg_mask

                    # Effective attractor: x0_eff = x_t_vp + score
                    #   bg: x0_eff = x0_fm
                    #   fg: x0_eff = (1+λ)*y_latents - λ*x0_fm
                    x0_eff = x_t_vp + score

                    # Exact OU step
                    mean   = exp_neg * x_t_vp + (1.0 - exp_neg) * abt.sqrt() * x0_eff
                    var    = sigma_t_safe * (1.0 - exp_neg2)
                    x_t_vp = mean + var.clamp(min=0.0).sqrt() * torch.randn_like(x_t_vp)

                # ---- 4. VP → FM ----
                x_post = x_t_vp / f

            else:
                # No Langevin: standard denoising from scheduler-updated latent
                x_post = latent

            # ---- 5. Final model call (LanPaint lines 117-120) ----
            v_c    = wan_i2v.model([x_post], t=torch.stack([t]), **arg_c)[0]
            v_u    = wan_i2v.model([x_post], t=torch.stack([t]), **arg_null)[0]
            v_pred = v_u + args.guide_scale * (v_c - v_u)
            x0_final = x_post - sigma_t * v_pred

            # Pin fg to clean reference
            x0_final = x0_final * (1 - fg_mask) + y_latents * fg_mask

            # Effective velocity consistent with pinned x0
            v_eff = (x_post - x0_final) / sigma_t_safe

            # ---- 6. Scheduler step ----
            latent_new = scheduler.step(
                v_eff.unsqueeze(0), t, x_post.unsqueeze(0),
                return_dict=False, generator=seed_g,
            )[0].squeeze(0)

            # ---- 7. Hard bg replace ----
            if i < len(timesteps) - 1:
                next_t   = timesteps[i + 1]
                bg_noisy = scheduler.add_noise(
                    y_latents_batched, noise.unsqueeze(0), torch.stack([next_t]),
                ).squeeze(0)
            else:
                bg_noisy = y_latents  # final step: use clean render for bg

            latent = fg_mask * latent_new + (1.0 - fg_mask) * bg_noisy

            # ---- Optional preview using x0_final via TAEHV ----
            if taehv is not None and (i + 1) % args.preview_interval == 0:
                with torch.no_grad():
                    preview_frames = taehv.decode([x0_final.float()])[0]  # [3, T, H, W] in [-1,1]
                cache_video(
                    tensor=preview_frames[None],
                    save_file=os.path.join(preview_dir, f"step_{i+1:03d}.mp4"),
                    fps=16, nrow=1, normalize=True, value_range=(-1, 1),
                )

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
