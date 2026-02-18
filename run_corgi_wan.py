
import sys
import os
import torch
import math
import random
import logging
import gc
from PIL import Image
from tqdm import tqdm
from huggingface_hub import snapshot_download
import numpy as np
import torchvision.transforms.functional as TF
import argparse

# Add Wan repo to path
sys.path.insert(0, os.path.abspath("official_wan_repo"))

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import cache_video
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
# Import from the attention module where we added the hook
from wan.modules.attention import ATTENTION_STORE, capture_attention

def main():
    parser = argparse.ArgumentParser(description="Wan Corgi Attention Experiment")
    parser.add_argument("--bg_image", type=str, default="data/images/plain_bg.png", help="Path to background image")
    parser.add_argument("--input_image", type=str, default="data/images/corgi_on_plain_bg.png", help="Path to input image (with corgi)")
    parser.add_argument("--prompt_path", type=str, default="data/captions/corgi_video.txt", help="Path to text prompt")
    parser.add_argument("--frame_num", type=int, default=81, help="Number of frames")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_file", type=str, default="corgi_experiment.mp4", help="Output filename")
    parser.add_argument("--output_mask_dir", type=str, default="output/debug_masks", help="Directory to save debug masks")
    args = parser.parse_args()

    os.makedirs(args.output_mask_dir, exist_ok=True)

    device_id = 0
    rank = 0
    device = torch.device(f"cuda:{device_id}")
    
    # Model parameters
    task = "i2v-14B"
    repo_id = "Wan-AI/Wan2.1-I2V-14B-720P"
    
    print(f"Downloading/Loading model {repo_id}...")
    checkpoint_dir = snapshot_download(repo_id=repo_id)
    
    # Initialize WanI2V
    cfg = WAN_CONFIGS[task]
    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        device_id=device_id,
        rank=rank,
        t5_cpu=True,
    )
    
    # ------------------------------------------------------------------
    # 1. Prepare Data
    # ------------------------------------------------------------------
    
    # Load images
    print(f"Loading background: {args.bg_image}")
    bg_img = Image.open(args.bg_image).convert("RGB").resize((512, 512))
    
    print(f"Loading input: {args.input_image}")
    input_img = Image.open(args.input_image).convert("RGB").resize((512, 512))
    
    with open(args.prompt_path, "r") as f:
        prompt = f.read().strip()
    print(f"Prompt: {prompt}")

    # ------------------------------------------------------------------
    # 2. Encode Background to Latents (Static Video)
    # ------------------------------------------------------------------
    frame_num = args.frame_num
    
    # Create static video tensor from BG
    bg_tensor = TF.to_tensor(bg_img).sub_(0.5).div_(0.5).to(device) # [3, H, W]
    bg_video = bg_tensor.unsqueeze(1).repeat(1, frame_num, 1, 1).unsqueeze(0) # [1, 3, F, H, W]
    
    print("Encoding background video...")
    with torch.no_grad():
        # vae.encode expects list of [3, F, H, W]
        # returns list of [C, F_lat, H_lat, W_lat]
        bg_latents = wan_i2v.vae.encode([bg_video.squeeze(0)])[0]
        bg_latents = bg_latents.unsqueeze(0) # [1, C, F, H, W]
    
    # ------------------------------------------------------------------
    # 3. Prepare I2V Condition (Input Image)
    # ------------------------------------------------------------------
    # The input to the I2V model (condition) is the `corgi_on_plain_bg.png`
    input_tensor = TF.to_tensor(input_img).sub_(0.5).div_(0.5).to(device) # [3, H, W]
    
    # ------------------------------------------------------------------
    # 4. Find 'corgi' token index
    # ------------------------------------------------------------------
    
    print("Analyzing prompt tokens...")
    # T5 tokenizer wrapper
    tokenizer_wrapper = wan_i2v.text_encoder.tokenizer
    # The wrapper's __call__ returns input_ids tensor directly
    input_ids = tokenizer_wrapper(prompt, return_tensors="pt")
    input_ids = input_ids[0] # [L]
    
    # Access underlying HF tokenizer for decoding
    tokenizer = tokenizer_wrapper.tokenizer
    
    print(f"Token IDs: {input_ids}")
    
    # Decode each token to find "corgi"
    corgi_indices = []
    print("Token Debugging:")
    for idx, t_id in enumerate(input_ids):
        decoded = tokenizer.decode([t_id])
        # Only print non-pad tokens for brevity
        if hasattr(tokenizer, 'pad_token_id') and t_id == tokenizer.pad_token_id:
            continue
        if decoded.strip() == "<pad>":
             continue
             
        print(f"Idx {idx}: {t_id} -> '{decoded}'")
        # Check for 'corgi' in various forms (case insensitive, stripped)
        # T5 often uses ' ' (SPIECE_UNDERLINE) for spaces.
        clean_decoded = decoded.replace(' ', '').strip().lower()
        if "corgi" in clean_decoded:
             corgi_indices.append(idx)
        elif "cor" in clean_decoded or "gi" in clean_decoded:
             # Heuristic for split tokens often seen for 'corgi' -> ' cor' + 'gi'
             # We might want to be more specific but collecting parts is okay for attention map.
             if clean_decoded in ["cor", "gi", "corg"]:
                 corgi_indices.append(idx)

    if not corgi_indices:
        print("WARNING: 'corgi' not found in individual tokens. Trying to find sequence...")
        # Fallback: check full decoding
        full_decoded = tokenizer.decode(input_ids)
        print(f"Full decoded: '{full_decoded}'")
        # If we can't find it token-wise, defaulting to a heuristic range might be safer if we knew where it was.
        # But let's stick to the warning.
        raise ValueError("Could not locate 'corgi' or related tokens in the prompt tokens. Aborting.")
    
    print(f"Targeting token indices: {corgi_indices}")

    # ------------------------------------------------------------------
    # 5. Encoding Context
    # ------------------------------------------------------------------
    print("Encoding text...")
    wan_i2v.text_encoder.model.to(device)
    context = wan_i2v.text_encoder([prompt], device)
    context_null = wan_i2v.text_encoder([wan_i2v.sample_neg_prompt], device)
    wan_i2v.text_encoder.model.cpu()
    
    print("Encoding CLIP image...")
    wan_i2v.clip.model.to(device)
    clip_context = wan_i2v.clip.visual([input_tensor.unsqueeze(1)]) # [1, 1, 3, H, W]
    wan_i2v.clip.model.cpu()
    
    # Prepare Y (Visual Condition)
    aspect_ratio = 1.0
    max_area = 512 * 512
    vae_stride = wan_i2v.vae_stride
    patch_size = wan_i2v.patch_size
    
    lat_h = round(np.sqrt(max_area * aspect_ratio) // vae_stride[1] // patch_size[1] * patch_size[1])
    lat_w = round(np.sqrt(max_area / aspect_ratio) // vae_stride[2] // patch_size[2] * patch_size[2])
    
    y = wan_i2v.vae.encode([
        torch.cat([
            torch.nn.functional.interpolate(
                input_tensor.unsqueeze(0).cpu(), size=(lat_h*vae_stride[1], lat_w*vae_stride[2]), mode='bicubic'
            ).transpose(0, 1), 
            torch.zeros(3, frame_num - 1, lat_h*vae_stride[1], lat_w*vae_stride[2])
        ], dim=1).to(device)
    ])[0]
    
    # Mask for Y (First frame valid)
    msk = torch.ones(1, frame_num, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]
    y = torch.concat([msk, y])
    
    # ------------------------------------------------------------------
    # 6. Generation Loop
    # ------------------------------------------------------------------
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(50, device=device, shift=5.0) 
    timesteps = scheduler.timesteps
    
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(args.seed)
    
    noise = torch.randn(
        16, (frame_num - 1) // 4 + 1, lat_h, lat_w,
        dtype=torch.float32, generator=seed_g, device=device
    )
    latent = noise
    
    max_seq_len = ((frame_num - 1) // vae_stride[0] + 1) * lat_h * lat_w // (patch_size[1] * patch_size[2])
    arg_c = {'context': [context[0]], 'clip_fea': clip_context, 'seq_len': max_seq_len, 'y': [y]}
    arg_null = {'context': context_null, 'clip_fea': clip_context, 'seq_len': max_seq_len, 'y': [y]}
    
    wan_i2v.model.to(device)
    
    print("Starting generation...")
    # Initial Latents (Noise)
    # Note: 'latent' is already initialized to 'noise'.
    
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=wan_i2v.param_dtype):
        for i, t in enumerate(tqdm(timesteps)):
            
            # Clear attention store
            if "weights" in ATTENTION_STORE:
                del ATTENTION_STORE["weights"]
            
            # Predict with Attention Capture on Cond path
            wan_i2v.model.to(device)
            with capture_attention():
                noise_pred_cond = wan_i2v.model([latent.to(device)], t=torch.stack([t]).to(device), **arg_c)[0]
            
            # Get Uncond (no need to capture)
            # We can offload here if needed, but switching back and forth is slow.
            # Let's keep model on GPU for both passes, then offload if we are really tight.
            noise_pred_uncond = wan_i2v.model([latent.to(device)], t=torch.stack([t]).to(device), **arg_null)[0]
            
            # Offload model to CPU to save memory for the hefty attention processing / vae / etc?
            # Actually, standard Wan generation offloads model at the end of loop or keeps it?
            # The 'generate' function in text2video.py has an 'offload_model' flag.
            # Here we are inside the loop. Moving 14B params to CPU and back every step is TOO slow.
            # We should check if we are accumulating gradients (we are under no_grad).
            # We should check if ATTENTION_STORE is growing too large.
            # We clear ATTENTION_STORE["weights"] at start of loop.
            # But inside one step, we store 32 layers * heads * seq_len^2 matrix?
            # 14B model has huge context?
            # 512x512 = 262k pixels? No.
            # 512x512 image -> VAE encoded -> 64x64 latent (stride 8).
            # Frame num 81 -> Video Latent [F, H, W] = [21, 64, 64] (stride 4 temporal).
            # Token count = 21 * 64 * 64 / (patch_size) ?
            # Patch size (1, 2, 2).
            # Tokens = 21 * 32 * 32 = 21,504 tokens.
            # Attention Matrix = Tokens^2 = 462M entries.
            # Float16 = 2 bytes. ~1GB per head per layer?
            # 14B model has many heads (16? 32?).
            # 32 layers.
            # This is MASSIVE. Storing full attention weights is impossible for video this size.

            # SOLUTION: We MUST NOT store the full attention matrix in `attention.py`.
            # We only need the Cross-Attention to Text (Query=Image, Key=Text).
            # Text length is small (512).
            # Image tokens = 21,504.
            # Matrix = 21,504 * 512 ~= 11M entries. Very manageable.
            # Self-attention (Image-Image) is the killer (21k * 21k).
            
            # We need to modify `attention.py` to ONLY store Cross-Attention weights.
            # How to distinguish?
            # In `attention(q, k, v...)`, we can check shapes.
            # If K length == 512 (context length) or similar small number, it's cross attn.
            # If K length == Q length (large), it's self attn.

            
            # CFG
            noise_pred = noise_pred_uncond + 5.0 * (noise_pred_cond - noise_pred_uncond)
            
            # Standard Scheduler Step
            temp_x0 = scheduler.step(noise_pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False, generator=seed_g)[0]
            latent_next = temp_x0.squeeze(0)
            
            # --------------------------------------------------------------
            # Extract Attention Mask
            # --------------------------------------------------------------
            if "weights" in ATTENTION_STORE:
                all_maps = ATTENTION_STORE["weights"]
                # Structure: Self, CrossImg, CrossText per block.
                # Total blocks: 32. Total maps expected: 96.
                
                # Average maps from semantic blocks
                block_start = 8
                block_end = 24
                
                selected_maps = []
                image_maps_count = 0
                text_maps_count = 0
                
                for b_idx in range(len(all_maps)):
                    attn_map = all_maps[b_idx] # [B, Heads, Lq, Lk]
                    # Check if this is the T5 Cross-Attention (Lk == 512)
                    if attn_map.shape[-1] == 512:
                         selected_maps.append(attn_map)
                         text_maps_count += 1
                    else:
                         image_maps_count += 1
                
                if i % 10 == 0:
                     print(f"Step {i}: Used {text_maps_count} text maps, filtered {image_maps_count} other maps (image/self).")

                if selected_maps:
                    # [N_layers, B, Heads, Lq, Lk] -> mean over layers
                    stacked_maps = torch.stack(selected_maps).mean(dim=0)
                    
                    # Mean over heads -> [B, Lq, Lk]
                    combined_map = stacked_maps.mean(dim=1)
                    
                    # Extract corgi
                    combined_map = combined_map[:, :, corgi_indices].mean(dim=-1) # [B, Lq]
                    
                    corgi_map = combined_map[0] # [Lq]
                    
                    # Reshape Lq -> (F_lat, H_lat, W_lat)
                    # NOTE: Lq flattening order in Wan?
                    # x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
                    # x = [u.flatten(2).transpose(1, 2) for u in x]
                    # flatten(2) flattens D, H, W (where D=F_patches).
                    # So Lq = F_p * H_p * W_p.
                    
                    f_lat_dim = latent.shape[1] # F_lat_encoded
                    h_lat_dim = latent.shape[2]
                    w_lat_dim = latent.shape[3]
                    
                    grid_f = f_lat_dim
                    grid_h = h_lat_dim // patch_size[1]
                    grid_w = w_lat_dim // patch_size[2]
                    # Actually grid sizes are computed in model.py:
                    # grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
                    # where u is after patch_embedding.
                    # patch_embedding stride = patch_size.
                    # so grid_f = F / P_t
                    # grid_h = H / P_h
                    # grid_w = W / P_w
                    
                    # Let's trust logic
                    try:
                        corgi_map_3d = corgi_map.view(grid_f, grid_h, grid_w)
                        
                        # Interpolate to Latent Size [1, 1, F, H, W]
                        # 3D interpolate needs [N, C, D, H, W]
                        corgi_map_3d = corgi_map_3d.unsqueeze(0).unsqueeze(0)
                        
                        corgi_mask = torch.nn.functional.interpolate(
                            corgi_map_3d,
                            size=(f_lat_dim, h_lat_dim, w_lat_dim),
                            mode='nearest' # Use nearest or linear?
                        )
                        corgi_mask = corgi_mask.squeeze(1) # [1, F, H, W]
                        
                        # Normalize 0-1
                        vis_mask = (corgi_mask - corgi_mask.min()) / (corgi_mask.max() - corgi_mask.min() + 1e-6)
                        
                        # Save visualization
                        if i % 10 == 0:
                            f_idx = f_lat_dim // 2
                            slice_np = vis_mask[0, f_idx].cpu().numpy()
                            slice_img = Image.fromarray((slice_np * 255).astype(np.uint8))
                            slice_img.save(os.path.join(args.output_mask_dir, f"step_{i:03d}_attn.png"))
                        
                        # Create Binary Mask
                        # Thresholding
                        binary_mask = (vis_mask > 0.15).float() # Lower threshold to catch more corgi?
                        
                        # Enhance mask (fill holes, dilate) if possible, but difficult in tensor.
                        # Simple Dilation via MaxPool
                        binary_mask = torch.nn.functional.max_pool3d(
                            binary_mask, kernel_size=3, stride=1, padding=1
                        )
                        
                        final_mask = binary_mask.unsqueeze(1).repeat(1, 16, 1, 1, 1).to(device)
                        
                        # Save binary mask
                        if i % 10 == 0:
                            f_idx = f_lat_dim // 2
                            slice_np = binary_mask[0, f_idx].cpu().numpy()
                            slice_img = Image.fromarray((slice_np * 255).astype(np.uint8))
                            slice_img.save(os.path.join(args.output_mask_dir, f"step_{i:03d}_mask.png"))
                            
                    except Exception as e:
                        print(f"Reshape Error: {e}")
                        final_mask = torch.zeros_like(latent).to(device)
                else:
                    final_mask = torch.zeros_like(latent).to(device)
            else:
                 final_mask = torch.zeros_like(latent).to(device)

            # --------------------------------------------------------------
            # Blend
            # --------------------------------------------------------------
            # Get background noisy sample for t_next (or t?)
            # Flow matching moves from t=1 (noise) to t=0 (clean) usually?
            # Or t=0 (noise) to t=1? Check scheduler.
            # Unipc scheduler: timesteps go from 1000 down to 0 usually.
            # timesteps[i] is current t.
            # timesteps[i+1] is next t (t_next).
            
            # We want latent_next to match bg_noisy at t_next in the background region.
            if i < len(timesteps) - 1:
                next_t = timesteps[i+1]
                # Add noise to clean background latents to match 'next_t' level
                # scheduler.add_noise
                # Need to check scheduler sigmas.
                # If t=0 (last step), we want fully clean.
                
                # Careful: Scheduler might use different conventions.
                # FlowUniPCMultistepScheduler
                # timesteps are usually decreasing.
                
                bg_noisy = scheduler.add_noise(
                    bg_latents,
                    noise, # Is this correct noise?
                    # Ideally we use the *same* noise used for initialization to ensure consistency?
                    # Or random noise?
                    # If we use random noise, the background will "shake" every step.
                    # We MUST use fixed noise.
                    # Is 'noise' variable still holding the initial noise? Yes.
                    torch.stack([next_t])
                ).squeeze(0)
            else:
                # Final step, t=0
                bg_noisy = bg_latents.squeeze(0)
                
            # Blend
            # Mask is 1 for Corgi, 0 for BG.
            # latent_next = Mask * latent_next + (1-Mask) * bg_noisy
            latent = final_mask.squeeze(0) * latent_next + (1 - final_mask.squeeze(0)) * bg_noisy

    # ------------------------------------------------------------------
    # 7. Decode and Save
    # ------------------------------------------------------------------
    print("Decoding...")
    wan_i2v.model.cpu()
    with torch.no_grad():
        x0 = [latent.to(device)]
        videos = wan_i2v.vae.decode(x0)
    
            
    cache_video(
        tensor=videos[0][None],
        save_file=os.path.join("output", args.output_file),
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )
    print("Done!")

if __name__ == "__main__":
    main()
