import sys
import os
import glob

# Fix sklearn TLS issue: LD_PRELOAD must be set before the dynamic linker loads
# scikit-learn's bundled libgomp. We do this by re-execing the process.
if "LIBGOMP_PRELOADED" not in os.environ:
    libgomp_files = glob.glob(
        "/home/ubuntu/miniconda3/envs/mvadapter/lib/python*/site-packages/scikit_learn.libs/libgomp*.so*"
    )
    if libgomp_files:
        os.environ["LD_PRELOAD"] = libgomp_files[0]
        os.environ["LIBGOMP_PRELOADED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

import torch
import torch.nn as nn
import math
import gc
import argparse
import shutil
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import Namespace
from huggingface_hub import snapshot_download
import torchvision.transforms.functional as TF

# Add Inpaint360GS FIRST and import its modules before wan,
# so that the Inpaint360GS 'utils' namespace package is registered in
# sys.modules before wan (or its deps) can shadow it with a plain module.
sys.path.insert(0, os.path.abspath("Inpaint360GS"))

from scene.colmap_loader import (
    read_extrinsics_binary, read_intrinsics_binary,
    read_extrinsics_text, read_intrinsics_text,
    qvec2rotmat,
)
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from scene.cameras import MiniCam
from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration
from gaussian_renderer import render

# Add official_wan_repo and import wan AFTER Inpaint360GS modules are loaded
sys.path.insert(0, os.path.abspath("official_wan_repo"))

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import cache_video
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.modules.attention import ATTENTION_STORE, capture_attention
from taehv import WanCompatibleTAEHV
from sam2.build_sam import build_sam2_video_predictor


def parse_cfg_args(cfg_path):
    """Parse Inpaint360GS cfg_args file (eval'd Namespace string) to dict."""
    with open(cfg_path) as f:
        ns = eval(f.read())
    return vars(ns)


def make_wan_minicam(cam_info, width=832, height=480):
    """
    Create a MiniCam at (width x height) from a CameraInfo.
    Keeps the original fovx and adjusts fovy to maintain square pixels at the target resolution.
    """
    fovx = cam_info["fovx"]
    fovy = 2 * math.atan(math.tan(fovx / 2) * height / width)
    znear, zfar = 0.01, 100.0

    world_view_transform = torch.tensor(
        getWorld2View2(cam_info["R"], cam_info["T"])
    ).transpose(0, 1).cuda()

    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    full_proj = world_view_transform.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)

    return MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj)


def load_cameras_from_colmap(source_path):
    """
    Load COLMAP camera extrinsics and intrinsics from source_path/sparse/0/.
    Returns list of dicts with R, T, fovx, fovy, image_name.
    Sorted by image_name, test images (starting with 'test') excluded.
    """
    sparse_dir = os.path.join(source_path, "sparse", "0")
    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
    except Exception:
        cam_extrinsics = read_extrinsics_text(os.path.join(sparse_dir, "images.txt"))
        cam_intrinsics = read_intrinsics_text(os.path.join(sparse_dir, "cameras.txt"))

    cams = []
    for key in cam_extrinsics:
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        R = np.transpose(qvec2rotmat(extr.qvec))  # camera-to-world rotation
        T = np.array(extr.tvec)                    # world-to-camera translation

        if intr.model == "SIMPLE_PINHOLE":
            fovx = focal2fov(intr.params[0], intr.width)
            fovy = focal2fov(intr.params[0], intr.height)
        elif intr.model == "PINHOLE":
            fovx = focal2fov(intr.params[0], intr.width)
            fovy = focal2fov(intr.params[1], intr.height)
        else:
            raise ValueError(f"Unsupported COLMAP camera model: {intr.model}")

        image_name = os.path.splitext(os.path.basename(extr.name))[0]
        cams.append({"R": R, "T": T, "fovx": fovx, "fovy": fovy,
                     "width": intr.width, "height": intr.height,
                     "image_name": image_name})

    # Sort by image_name, filter out test images
    cams = sorted(cams, key=lambda c: c["image_name"])
    cams = [c for c in cams if "test" not in c["image_name"].lower()]
    return cams


def find_ply_path(base_dir, subdir, fallback_subdir=None):
    """
    Find the point_cloud.ply with the maximum iteration number in base_dir/subdir/.
    Falls back to fallback_subdir if subdir doesn't exist.
    Returns (ply_path, iteration, resolved_subdir).
    """
    pc_dir = os.path.join(base_dir, subdir)
    resolved_subdir = subdir
    if not os.path.isdir(pc_dir) and fallback_subdir is not None:
        pc_dir = os.path.join(base_dir, fallback_subdir)
        resolved_subdir = fallback_subdir
    if not os.path.isdir(pc_dir):
        raise FileNotFoundError(f"Point cloud directory not found: {pc_dir}")

    max_iter = searchForMaxIteration(pc_dir)
    ply_path = os.path.join(pc_dir, f"iteration_{max_iter}", "point_cloud.ply")
    return ply_path, max_iter, resolved_subdir


class SimplePipeline:
    """Minimal pipeline params for gaussian_renderer.render()."""
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False


def render_background(bg_ply_path, minicam, width, height):
    """Load bg Gaussians and render RGB at minicam. Returns PIL Image (RGB)."""
    print(f"Loading background PLY: {bg_ply_path}")
    bg_gaussians = GaussianModel(sh_degree=3)
    bg_gaussians.load_ply(bg_ply_path)
    bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    pipeline = SimplePipeline()

    with torch.no_grad():
        result = render(minicam, bg_gaussians, pipeline, bg_color)
    rendered_rgb = result["render"]  # (3, H, W), values in [0, 1]

    del bg_gaussians
    torch.cuda.empty_cache()

    img_np = (rendered_rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, 3)
    return Image.fromarray(img_np)


def render_object_mask(seg_ply_path, classifier_path, minicam, target_obj_id, num_classes, mask_dilation):
    """
    Load seg Gaussians + classifier, render object feature map,
    classify per-pixel, threshold at target_obj_id. Returns binary PIL mask (L mode).
    """
    print(f"Loading seg PLY: {seg_ply_path}")
    seg_gaussians = GaussianModel(sh_degree=3)
    seg_gaussians.load_ply(seg_ply_path)

    ckpt = torch.load(classifier_path, map_location="cuda")
    weight = ckpt["field_head"]["weight"] if isinstance(ckpt, dict) and "field_head" in ckpt else ckpt["weight"]
    num_classes = weight.shape[0]  # infer from checkpoint
    classifier = nn.Conv2d(seg_gaussians.num_objects, num_classes, kernel_size=1).cuda()
    if isinstance(ckpt, dict) and "field_head" in ckpt:
        classifier.load_state_dict(ckpt["field_head"])
    else:
        classifier.load_state_dict(ckpt)
    classifier.eval()

    bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    pipeline = SimplePipeline()

    with torch.no_grad():
        result = render(minicam, seg_gaussians, pipeline, bg_color)
        render_obj = result["render_object"]              # (16, H, W)
        logits = classifier(render_obj.unsqueeze(0))      # (1, num_classes, H, W)
        probs = torch.softmax(logits, dim=1)[0]           # (num_classes, H, W)
        mask = (probs[target_obj_id] > 0.5).float()       # (H, W)

    del seg_gaussians, classifier
    torch.cuda.empty_cache()

    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)  # (H, W)
    mask_img = Image.fromarray(mask_np, mode="L")

    if mask_dilation > 0:
        from PIL import ImageFilter
        filter_size = 2 * mask_dilation + 1
        mask_img = mask_img.filter(ImageFilter.MaxFilter(filter_size))

    return mask_img


def run_flux_inpainting(bg_image, mask_image, prompt, guidance_scale, num_steps, seed):
    """Load FluxFillPipeline, run inpainting, unload. Returns PIL Image."""
    from diffusers import FluxFillPipeline

    print("Loading FluxFillPipeline...")
    pipe = FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()

    width, height = bg_image.size
    result = pipe(
        prompt=prompt,
        image=bg_image,
        mask_image=mask_image,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_wan_i2v(
    bg_image, input_image, initial_mask_pil, prompt,
    save_dir, mask_dir,
    width, height, frame_num,
    sampling_steps, guide_scale, wan_mask_dilation,
    log_frequency, seed,
):
    """
    Wan I2V animation with SAM2 mask blending (SAM2 mode from run_wan_fg_anim.py).
    bg_image: PIL Image (background)
    input_image: PIL Image (CLIP visual condition, i.e. Flux output)
    initial_mask_pil: PIL Image mode 'L' (binary mask for SAM2 initialisation)
    """
    device_id = 0
    rank = 0
    device = torch.device(f"cuda:{device_id}")

    task = "i2v-14B"
    repo_id = "Wan-AI/Wan2.1-I2V-14B-720P"
    print(f"Downloading/Loading model {repo_id}...")
    checkpoint_dir = snapshot_download(repo_id=repo_id)

    cfg = WAN_CONFIGS[task]
    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        device_id=device_id,
        rank=rank,
        t5_cpu=True,
    )

    # Resize images to target resolution
    bg_img = bg_image.resize((width, height))
    input_img = input_image.resize((width, height))

    # ------------------------------------------------------------------
    # Encode Background to Latents (Static Video)
    # ------------------------------------------------------------------
    bg_tensor = TF.to_tensor(bg_img).sub_(0.5).div_(0.5).to(device)
    bg_video = bg_tensor.unsqueeze(1).repeat(1, frame_num, 1, 1).unsqueeze(0)

    print("Encoding background video...")
    with torch.no_grad():
        bg_latents = wan_i2v.vae.encode([bg_video.squeeze(0)])[0]
        bg_latents = bg_latents.unsqueeze(0)
    lat_h = bg_latents.shape[-2]
    lat_w = bg_latents.shape[-1]

    # ------------------------------------------------------------------
    # Prepare I2V Condition (Input Image)
    # ------------------------------------------------------------------
    input_tensor = TF.to_tensor(input_img).sub_(0.5).div_(0.5).to(device)

    # ------------------------------------------------------------------
    # Encoding Context
    # ------------------------------------------------------------------
    print("Encoding text...")
    wan_i2v.text_encoder.model.to(device)
    context = wan_i2v.text_encoder([prompt], device)
    context_null = wan_i2v.text_encoder([wan_i2v.sample_neg_prompt], device)
    wan_i2v.text_encoder.model.cpu()

    print("Encoding CLIP image...")
    wan_i2v.clip.model.to(device)
    clip_context = wan_i2v.clip.visual([input_tensor.unsqueeze(1)])
    wan_i2v.clip.model.cpu()

    # Prepare Y (Visual Condition)
    vae_stride = wan_i2v.vae_stride
    patch_size = wan_i2v.patch_size

    y = wan_i2v.vae.encode([
        torch.cat([
            torch.nn.functional.interpolate(
                input_tensor.unsqueeze(0).cpu(),
                size=(lat_h * vae_stride[1], lat_w * vae_stride[2]),
                mode='bicubic'
            ).transpose(0, 1),
            torch.zeros(3, frame_num - 1, lat_h * vae_stride[1], lat_w * vae_stride[2])
        ], dim=1).to(device)
    ])[0]

    msk = torch.ones(1, frame_num, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]
    y = torch.concat([msk, y])

    wan_i2v.vae.model.cpu()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Generation Loop Setup
    # ------------------------------------------------------------------
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
    )
    scheduler.set_timesteps(sampling_steps, device=device, shift=5.0)
    timesteps = scheduler.timesteps

    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    noise = torch.randn(
        16, (frame_num - 1) // 4 + 1, lat_h, lat_w,
        dtype=torch.float32, generator=seed_g, device=device
    )
    latent = noise

    max_seq_len = ((frame_num - 1) // vae_stride[0] + 1) * lat_h * lat_w // (patch_size[1] * patch_size[2])
    arg_c = {'context': [context[0]], 'clip_fea': clip_context, 'seq_len': max_seq_len, 'y': [y]}
    arg_null = {'context': context_null, 'clip_fea': clip_context, 'seq_len': max_seq_len, 'y': [y]}

    # ------------------------------------------------------------------
    # Initialize SAM2 and TinyVAE
    # ------------------------------------------------------------------
    tiny_vae_path = "checkpoints/taew2_1.pth"
    tiny_vae = None
    if os.path.exists(tiny_vae_path):
        print(f"Loading TinyVAE from {tiny_vae_path}...")
        tiny_vae = WanCompatibleTAEHV(checkpoint_path=tiny_vae_path).to(device).eval()
    else:
        print(f"WARNING: TinyVAE checkpoint not found at {tiny_vae_path}.")

    print("Initializing SAM2...")
    sam2_checkpoint = "checkpoints/sam2_hiera_tiny.pt"
    sam2_config = "sam2_hiera_t.yaml"
    if not os.path.exists(sam2_checkpoint):
        raise FileNotFoundError(f"SAM2 checkpoint not found at {sam2_checkpoint}")
    sam2_predictor = build_sam2_video_predictor(sam2_config, sam2_checkpoint)

    # Prepare initial mask for SAM2 (resize to video resolution)
    mask_np = np.array(initial_mask_pil.resize((512, 512), Image.NEAREST))
    initial_mask_binary = (mask_np > 128).astype(np.float32)
    print("Initial mask ready for SAM2.")

    wan_i2v.model.to(device)

    # ------------------------------------------------------------------
    # Generation Loop
    # ------------------------------------------------------------------
    print("Starting generation...")
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=wan_i2v.param_dtype):
        for i, t in enumerate(tqdm(timesteps)):
            annealing_weight = 1.0 - i / (len(timesteps) - 1)

            if "weights" in ATTENTION_STORE:
                del ATTENTION_STORE["weights"]

            wan_i2v.model.to(device)
            noise_pred_cond = wan_i2v.model(
                [latent.to(device)], t=torch.stack([t]).to(device), **arg_c
            )[0]
            noise_pred_uncond = wan_i2v.model(
                [latent.to(device)], t=torch.stack([t]).to(device), **arg_null
            )[0]

            # CFG
            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

            # Scheduler step
            temp_x0 = scheduler.step(
                noise_pred.unsqueeze(0), t, latent.unsqueeze(0),
                return_dict=False, generator=seed_g
            )[0]
            latent_next = temp_x0.squeeze(0)

            # ------------------------------------------------------------------
            # SAM2 mask extraction
            # ------------------------------------------------------------------
            with torch.no_grad():
                sigma_t = scheduler.sigmas[i].to(device)
                pred_x0 = latent - sigma_t * noise_pred
                rec_video = tiny_vae.decode([pred_x0])[0]

                frame_dir = "/tmp/ram4d_temp_frames"
                if os.path.exists(frame_dir):
                    shutil.rmtree(frame_dir)
                os.makedirs(frame_dir, exist_ok=True)

                rec_video_np = (rec_video * 0.5 + 0.5).clamp(0, 1)
                rec_video_np = (rec_video_np * 255).to(torch.uint8).cpu().permute(1, 2, 3, 0).numpy()

                for f_idx in range(rec_video_np.shape[0]):
                    f_path = os.path.join(frame_dir, f"{f_idx:05d}.jpg")
                    cv2.imwrite(f_path, cv2.cvtColor(rec_video_np[f_idx], cv2.COLOR_RGB2BGR))

                inference_state = sam2_predictor.init_state(video_path=frame_dir)
                sam2_predictor.reset_state(inference_state)

                _, _, _ = sam2_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,
                    mask=initial_mask_binary
                )

                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(inference_state):
                    if 1 in out_obj_ids:
                        idx = out_obj_ids.index(1)
                        mask_seg = (out_mask_logits[idx] > 0.0).float().cpu()
                        video_segments[out_frame_idx] = mask_seg
                    else:
                        video_segments[out_frame_idx] = torch.zeros(1, 512, 512)

                sorted_frames = sorted(video_segments.keys())
                mask_stack = torch.stack([video_segments[f] for f in sorted_frames], dim=0)

                if wan_mask_dilation > 0:
                    dilation_kernel = wan_mask_dilation
                    if dilation_kernel % 2 == 0:
                        dilation_kernel += 1
                    padding = dilation_kernel // 2
                    mask_stack = torch.nn.functional.max_pool2d(
                        mask_stack, kernel_size=dilation_kernel, stride=1, padding=padding
                    )

                mask_stack = mask_stack.permute(1, 0, 2, 3).unsqueeze(0)

                temporal_kernel = 9
                mask_stack = torch.nn.functional.max_pool3d(
                    mask_stack,
                    kernel_size=(temporal_kernel, 1, 1),
                    stride=1,
                    padding=(temporal_kernel // 2, 0, 0),
                )

                last_seen_mask = None
                for f in range(mask_stack.shape[2]):
                    frame = mask_stack[0, 0, f]
                    if frame.sum() > 0:
                        last_seen_mask = frame.clone()
                    elif last_seen_mask is not None:
                        mask_stack[0, 0, f] = last_seen_mask

                mask_lat = torch.nn.functional.adaptive_max_pool3d(
                    mask_stack,
                    output_size=(latent.shape[1], latent.shape[2], latent.shape[3])
                )
                mask_lat = (mask_lat > 0.5).float()
                final_mask = mask_lat.to(device).repeat(1, 16, 1, 1, 1)
                binary_mask = mask_lat.squeeze(0).to(device)

            # Save debug mask video
            if i % log_frequency == 0:
                bin_video_tensor = binary_mask.unsqueeze(1)
                bin_video_tensor = torch.nn.functional.interpolate(
                    bin_video_tensor, size=(frame_num, height, width), mode='nearest'
                )
                try:
                    cache_video(
                        tensor=bin_video_tensor,
                        save_file=os.path.join(mask_dir, f"step_{i:03d}_mask.mp4"),
                        fps=16, nrow=1, normalize=False, value_range=(0, 1)
                    )
                except Exception as e:
                    print(f"Error saving mask video: {e}")

            # x0 debug visualization
            if i % log_frequency == 0:
                with torch.no_grad():
                    sigma_t = scheduler.sigmas[i].to(device)
                    pred_x0_vis = latent - sigma_t * noise_pred
                    annealed_bg_debug = (1 - final_mask.squeeze(0)) * annealing_weight
                    bg_clean = bg_latents.squeeze(0)
                    mixed_x0 = pred_x0_vis + annealed_bg_debug * (bg_clean - pred_x0_vis)

                    print(f"Decoding debug videos at step {i}...")
                    wan_i2v.model.cpu()
                    torch.cuda.empty_cache()
                    vae = tiny_vae if tiny_vae is not None else wan_i2v.vae
                    if tiny_vae is None:
                        wan_i2v.vae.model.to(device)

                    decoded_pred = vae.decode([pred_x0_vis.to(device)])[0]
                    torch.cuda.empty_cache()
                    cache_video(
                        tensor=decoded_pred[None],
                        save_file=os.path.join(mask_dir, f"step_{i:03d}_pred_x0.mp4"),
                        fps=16, nrow=1, normalize=True, value_range=(-1, 1)
                    )
                    del decoded_pred

                    decoded_mixed = vae.decode([mixed_x0.to(device)])[0]
                    torch.cuda.empty_cache()
                    if tiny_vae is None:
                        wan_i2v.vae.model.cpu()
                    cache_video(
                        tensor=decoded_mixed[None],
                        save_file=os.path.join(mask_dir, f"step_{i:03d}_mixed_x0.mp4"),
                        fps=16, nrow=1, normalize=True, value_range=(-1, 1)
                    )
                    del decoded_mixed
                    wan_i2v.model.to(device)
                    torch.cuda.empty_cache()

            # ------------------------------------------------------------------
            # Background blending
            # ------------------------------------------------------------------
            if i < len(timesteps) - 1:
                next_t = timesteps[i + 1]
                bg_noisy = scheduler.add_noise(
                    bg_latents, noise, torch.stack([next_t])
                ).squeeze(0)
            else:
                bg_noisy = bg_latents.squeeze(0)

            annealed_bg = (1 - final_mask.squeeze(0)) * annealing_weight
            latent = latent_next + annealed_bg * (bg_noisy - latent_next)

            gc.collect()
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Decode and Save
    # ------------------------------------------------------------------
    print("Decoding final video...")
    wan_i2v.model.cpu()
    wan_i2v.vae.model.to(device)
    with torch.no_grad():
        videos = wan_i2v.vae.decode([latent.to(device)])

    cache_video(
        tensor=videos[0][None],
        save_file=os.path.join(save_dir, "final_output.mp4"),
        fps=16, nrow=1, normalize=True, value_range=(-1, 1)
    )
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Replace 3DGS object with animated foreground")

    # 3DGS scene inputs
    parser.add_argument("--inpainted_3dgs_path", type=str, required=True,
                        help="Path to Inpaint360GS output directory (e.g. Inpaint360GS/output/inpaint360/bag)")
    parser.add_argument("--target_obj_id", type=int, required=True,
                        help="Object ID to extract as mask from classifier output")
    parser.add_argument("--camera_idx", type=int, default=0,
                        help="Index into sorted training camera list")
    parser.add_argument("--num_classes", type=int, default=256,
                        help="Number of object classes in classifier")

    # New-object inputs
    parser.add_argument("--prompt_path", type=str, required=True,
                        help="Path to text prompt file")
    parser.add_argument("--output_name", type=str, required=True,
                        help="Output subdirectory name under output/")

    # Mask
    parser.add_argument("--mask_dilation", type=int, default=60,
                        help="Dilation on the rendered object mask (pixels)")

    # Flux stage
    parser.add_argument("--flux_guidance", type=float, default=30.0)
    parser.add_argument("--flux_steps", type=int, default=50)
    parser.add_argument("--flux_seed", type=int, default=0)

    # Wan stage
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--wan_mask_dilation", type=int, default=15)
    parser.add_argument("--log_frequency", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_video", action="store_true",
                        help="Exit after saving bg_render.png, mask_render.png, flux_output.png")

    args = parser.parse_args()

    save_dir = os.path.join("output", args.output_name)
    mask_dir = os.path.join(save_dir, "debug_masks")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Save args
    with open(os.path.join(save_dir, "args.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # Load prompt
    with open(args.prompt_path) as f:
        prompt = f.read().strip()
    print(f"Prompt: {prompt}")

    # ======================================================================
    # Stage 0: 3DGS Rendering
    # ======================================================================
    print("\n=== Stage 0: 3DGS Rendering ===")

    cfg_path = os.path.join(args.inpainted_3dgs_path, "cfg_args")
    cfg = parse_cfg_args(cfg_path)
    source_path = cfg["source_path"]
    print(f"Source path (from cfg_args): {source_path}")

    # Load cameras
    print("Loading COLMAP cameras...")
    train_cams = load_cameras_from_colmap(source_path)
    print(f"Found {len(train_cams)} training cameras")
    if args.camera_idx >= len(train_cams):
        raise IndexError(f"camera_idx {args.camera_idx} out of range (0–{len(train_cams)-1})")
    cam_info = train_cams[args.camera_idx]
    print(f"Using camera {args.camera_idx}: {cam_info['image_name']}")

    minicam = make_wan_minicam(cam_info, width=args.width, height=args.height)

    # Find seg PLY (segmentation-aware model with object features).
    # Try point_cloud_distilled first (Inpaint360GS stage 3 output), then point_cloud.
    seg_ply_path, seg_iter, seg_subdir = find_ply_path(
        args.inpainted_3dgs_path, "point_cloud_distilled", "point_cloud"
    )
    classifier_path = os.path.join(
        args.inpainted_3dgs_path, seg_subdir,
        f"iteration_{seg_iter}", "classifier.pth"
    )
    print(f"Seg PLY: {seg_ply_path} (iter {seg_iter})")
    print(f"Classifier: {classifier_path}")

    # Render object mask
    mask_image = render_object_mask(
        seg_ply_path, classifier_path, minicam,
        args.target_obj_id, args.num_classes, args.mask_dilation
    )
    mask_path = os.path.join(save_dir, "mask_render.png")
    mask_image.save(mask_path)
    print(f"Saved mask render: {mask_path}")

    # Find background PLY (object-removal / inpainting model, no object)
    bg_ply_path, bg_iter, _ = find_ply_path(
        args.inpainted_3dgs_path,
        "point_cloud_object_inpaint_virtual",
        fallback_subdir="point_cloud_object_removal",
    )
    print(f"Background PLY: {bg_ply_path} (iter {bg_iter})")

    bg_image = render_background(bg_ply_path, minicam, args.width, args.height)
    bg_path = os.path.join(save_dir, "bg_render.png")
    bg_image.save(bg_path)
    print(f"Saved background render: {bg_path}")

    gc.collect()
    torch.cuda.empty_cache()

    # ======================================================================
    # Stage 1: Flux Fill Inpainting
    # ======================================================================
    print("\n=== Stage 1: Flux Fill Inpainting ===")

    flux_output = run_flux_inpainting(
        bg_image=bg_image,
        mask_image=mask_image,
        prompt=prompt,
        guidance_scale=args.flux_guidance,
        num_steps=args.flux_steps,
        seed=args.flux_seed,
    )
    flux_path = os.path.join(save_dir, "flux_output.png")
    flux_output.save(flux_path)
    print(f"Saved Flux output: {flux_path}")

    if args.skip_video:
        print("\nImages saved. Exiting (--skip_video).")
        return

    # ======================================================================
    # Stage 2: Wan I2V Animation
    # ======================================================================
    print("\n=== Stage 2: Wan I2V Animation ===")

    run_wan_i2v(
        bg_image=bg_image,
        input_image=flux_output,
        initial_mask_pil=mask_image,
        prompt=prompt,
        save_dir=save_dir,
        mask_dir=mask_dir,
        width=args.width,
        height=args.height,
        frame_num=args.frame_num,
        sampling_steps=args.sampling_steps,
        guide_scale=args.guide_scale,
        wan_mask_dilation=args.wan_mask_dilation,
        log_frequency=args.log_frequency,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
