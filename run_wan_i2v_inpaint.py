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
from contextlib import contextmanager

# Add official_wan_repo to path so we can import wan
sys.path.insert(0, os.path.abspath("official_wan_repo"))

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import cache_video
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

def prepare_mask_latents(mask, vae, device, param_dtype):
    # mask: [1, F, 1, H, W]
    mask = mask.to(device=device, dtype=param_dtype)
    
    # 1. Resize mask to video resolution if needed
    # We assume mask input is already resized or we resize here. 
    # But mask coming in might be Latent sized? No, usually pixel sized.
    
    bs, frames, channels, height, width = mask.shape
    vae_stride = vae.stride if hasattr(vae, 'stride') else (4, 8, 8) # Fallback to default config
    
    # Resize to latent dimensions
    # Latent H = H // 8, Latent W = W // 8
    target_h = height // vae_stride[1]
    target_w = width // vae_stride[2]
    
    frame_masks = []
    # Temporal stride is 4
    for f in range(0, frames, vae_stride[0]):
        if f < frames:
            frame_mask = mask[0, f] # [1, H, W]
            
            # Resize
            resized_mask = torch.nn.functional.interpolate(
                frame_mask.unsqueeze(0), 
                size=(target_h, target_w), 
                mode='nearest'
            ).squeeze(0)
            frame_masks.append(resized_mask)
            
    final_masks = torch.stack(frame_masks).unsqueeze(0) # [1, F_lat, 1, H_lat, W_lat]
    final_masks = final_masks.permute(0, 2, 1, 3, 4) # [1, 1, F_lat, H_lat, W_lat]
    # Repeat for latent channels (16 for Wan)
    final_masks = final_masks.repeat(1, 16, 1, 1, 1) # [1, 16, F_lat, H_lat, W_lat]
    
    return final_masks

def main():
    device_id = 0
    rank = 0
    device = torch.device(f"cuda:{device_id}")
    
    # Model parameters
    task = "i2v-14B"
    repo_id = "Wan-AI/Wan2.1-I2V-14B-720P"
    
    print(f"Downloading model {repo_id}...")
    checkpoint_dir = snapshot_download(repo_id=repo_id)
    print(f"Model downloaded to {checkpoint_dir}")
    
    # Inputs
    image_path = "output/flux_inpainting_output.png"
    mask_path = "data/images/statue_mask.png" 
    
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    
    print(f"Loading mask: {mask_path}")
    mask_img = Image.open(mask_path).convert("L")
    
    # Dilate mask as before (MaxFilter 21)
    from PIL import ImageFilter
    print("Applying 10px dilation to mask (MaxFilter 21)...")
    mask_img = mask_img.filter(ImageFilter.MaxFilter(21))
    
    prompt = "a cute corgi, jumping and running, barking, 8k quality, detailed"
    
    # Initialize WanI2V
    cfg = WAN_CONFIGS[task]
    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        device_id=device_id,
        rank=rank,
        t5_cpu=True,
    )
    
    # Parameters
    # User asked for 512x512.
    # Wan 14B I2V supported sizes: 720*1280, 1280*720, 480*832, 832*480.
    # 512*512 is not standard. Closest valid area logic?
    # WanI2V.generate takes `max_area`.
    # Let's set max_area roughly to 512*512 = 262144.
    # 480*832 = 399360.
    # If we pass 512*512 area, the internal logic will calculate H/W.
    # If input image is 1024x1024 (Flux output?), and we want 512x512 video.
    # We should resize input image to 512x512 first?
    img_resized = img.resize((512, 512))
    mask_resized = mask_img.resize((512, 512))
    
    # Custom Generation Logic with Mask
    # Based on WanI2V.generate but with inpainting loop
    
    # 1. Preprocess Image and Mask
    # WanI2V.generate normalizes image to [-1, 1]
    
    first_frame = TF.to_tensor(img_resized).sub_(0.5).div_(0.5).to(device)
    # Mask: 1 = Inpaint (Subject), 0 = Keep (Background)
    mask_tensor = TF.to_tensor(mask_resized).to(device) # [1, H, W], 0..1
    # WanT2VInpaint uses inverted mask: 1=Keep, 0=Inpaint.
    # Lets assume input mask is White=Subject(Inpaint).
    # So Keep Mask = 1 - Mask.
    keep_mask = 1.0 - mask_tensor # 1=Keep (Background), 0=Change (Subject) (if mask was white for subject)
    
    # Create video tensors (repeated)
    frame_num = 81
    init_video_tensor = first_frame.unsqueeze(1).repeat(1, frame_num, 1, 1).unsqueeze(0) # [1, 3, F, H, W]
    
    # Mask tensor shape for prepare_mask_latents matches WanT2VInpaint [B, F, C, H, W]
    # keep_mask is [1, H, W]
    keep_mask_tensor = keep_mask.unsqueeze(0).repeat(frame_num, 1, 1, 1).unsqueeze(0) # [1, F, 1, H, W]
    
    # Encode Init Video (Static) to Latents
    # We need to use VAE encode. WanI2V has self.vae.
    # WanVAE encode expects list of tensors?
    # wan_i2v.vae.encode([video_tensor]) returns list of latents.
    # The input to VAE encode is typically [3, F, H, W] (unbatched list)
    
    print("Encoding init video...")
    with torch.no_grad():
        init_latents = wan_i2v.vae.encode([init_video_tensor.squeeze(0)])[0] # [C, F_lat, H_lat, W_lat]
        init_latents = init_latents.unsqueeze(0) # [1, C, F, H, W]

    print("Preparing mask latents...")
    # Resize mask to latent space
    # WanT2VInpaint logic:
    mask_latents = prepare_mask_latents(keep_mask_tensor, wan_i2v.vae, device, wan_i2v.param_dtype)
    
    # Generate calls...
    # We need to re-implement the loop.
    # Copying essentials from WanI2V.generate
    
    # Calculate Latent Dimensions from Image Size
    aspect_ratio = 1.0 # 512x512
    max_area = 512 * 512
    patch_size = wan_i2v.patch_size
    vae_stride = wan_i2v.vae_stride
    
    # This logic matches WanI2V.generate
    lat_h = round(np.sqrt(max_area * aspect_ratio) // vae_stride[1] // patch_size[1] * patch_size[1])
    lat_w = round(np.sqrt(max_area / aspect_ratio) // vae_stride[2] // patch_size[2] * patch_size[2])
    
    # ... (Seed, Noise, etc.) ...
    seed = 42
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)
    
    noise = torch.randn(
        16, (frame_num - 1) // 4 + 1, lat_h, lat_w,
        dtype=torch.float32, generator=seed_g, device=device
    )
    
    # I2V Conditioning (Image Embedding + Text)
    # Using WanI2V helper methods if possible? No, we have to call components.
    
    # Text
    print("Encoding text...")
    wan_i2v.text_encoder.model.to(device)
    context = wan_i2v.text_encoder([prompt], device)
    context_null = wan_i2v.text_encoder([wan_i2v.sample_neg_prompt], device)
    wan_i2v.text_encoder.model.cpu()
    
    # CLIP Image
    print("Encoding image...")
    wan_i2v.clip.model.to(device)
    clip_context = wan_i2v.clip.visual([first_frame.unsqueeze(1)])
    wan_i2v.clip.model.cpu()
    
    # VAE Encode Image for Y (Conditioning)
    # WanI2V logic: y = vae.encode(concat([img_padded, zeros]))
    # For 14B I2V, y is constructed from the input image placed at the beginning?
    # Inspecting image2video.py again:
    # y = self.vae.encode([ torch.concat([ img_interp, zeros ], dim=1) ])
    # We need to replicate this exactly.
    
    y = wan_i2v.vae.encode([
        torch.cat([
            torch.nn.functional.interpolate(
                first_frame.unsqueeze(0).cpu(), size=(lat_h*vae_stride[1], lat_w*vae_stride[2]), mode='bicubic'
            ).transpose(0, 1), # [C, 1, H, W]
            torch.zeros(3, frame_num - 1, lat_h*vae_stride[1], lat_w*vae_stride[2])
        ], dim=1).to(device)
    ])[0]
    
    # Msk for I2V (Conditioning Mask)
    # WanI2V.generate logic
    msk = torch.ones(1, frame_num, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]
    y = torch.concat([msk, y]) # Attach mask channel to y
    
    # Scheduler
    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(50, device=device, shift=16.0) # Using 16.0 shift as in generic FLF/I2V for high res? 
    # generate.py says shift=5.0 for I2V generally, 3.0 for 480p. 
    # Let's use 5.0 for 512x512.
    
    timesteps = scheduler.timesteps
    
    # Loop
    latent = noise
    
    # Arguments
    # Calculate seq_len
    max_seq_len = ((frame_num - 1) // vae_stride[0] + 1) * lat_h * lat_w // (patch_size[1] * patch_size[2])
    arg_c = {'context': [context[0]], 'clip_fea': clip_context, 'seq_len': max_seq_len, 'y': [y]}
    arg_null = {'context': context_null, 'clip_fea': clip_context, 'seq_len': max_seq_len, 'y': [y]}
    
    wan_i2v.model.to(device)
    
    print("Sampling with mask constraint...")
    
    # Use autocast and no_grad
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=wan_i2v.param_dtype):
        for i, t in enumerate( tqdm(timesteps) ):
            # Predict
            latent_model_input = [latent.to(device)]
            timestep_t = torch.stack([t]).to(device)
            
            noise_pred_cond = wan_i2v.model(latent_model_input, t=timestep_t, **arg_c)[0]
            noise_pred_uncond = wan_i2v.model(latent_model_input, t=timestep_t, **arg_null)[0]
            noise_pred = noise_pred_uncond + 5.0 * (noise_pred_cond - noise_pred_uncond)
            
            # Step
            temp_x0 = scheduler.step(noise_pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False, generator=seed_g)[0]
            latent = temp_x0.squeeze(0)
            
            # Mask Blending (The "Inpainting" Part)
            # We need init_latents_proper (noisy version of init_latents at *next* step? No, at *current* step target?
            # In Flow Matching, we move from t (noisy) to t_next (less noisy).
            # We want to force the 'keep' areas to match the noisy encoding of the original video at the *new* timestep (next step).
            # Wait, scheduler.step returns x_prev (t_next).
            # So we need to add noise to init_latents corresponding to t_next.
            
            # Get next timestep
            if i < len(timesteps) - 1:
                next_t = timesteps[i+1]
            else:
                next_t = torch.tensor(0.0).to(device) # Final step
                
            # Add noise to init_latents to match next_t
            if next_t > 0:
                # Need a fresh noise generator for this or reuse? 
                # Usually independent noise for blending? Or same noise as initial?
                # WanT2VInpaint uses `noise` (initial noise) + add_noise.
                # scheduler.add_noise(original_samples, noise, timesteps)
                init_latents_proper = scheduler.add_noise(
                    init_latents, 
                    noise, # The original noise used for `latent` initialization?
                    # In WanT2VInpaint, `prepare_latents` creates `noise` and `latents = add_noise(encoded, noise, t)`.
                    # Here `latent` (start) is pure noise.
                    # So we should probably use the SAME noise tensor to corrupt init_latents?
                    # Yes, consistency.
                    torch.stack([next_t])
                ).squeeze(0)
            else:
                init_latents_proper = init_latents.squeeze(0)
                
            # Blend
            # mask_latents: 1=Keep, 0=Change
            # latents = (1 - mask_latents) * latents + mask_latents * init_latents_proper
            # My mask_latents definition in main: keep_mask (1=Keep).
            # So: latents = (1 - keep_mask) * latents + keep_mask * init_latents_proper
            # Wait, check shapes. mask_latents [1, 16, F, H, W]
            # latent [16, F, H, W]
            
            latent = (1 - mask_latents.squeeze(0)) * latent + mask_latents.squeeze(0) * init_latents_proper
        
    # Decode
    wan_i2v.model.cpu()
    x0 = [latent.to(device)]
    videos = wan_i2v.vae.decode(x0)
    
    # Save
    cache_video(
        tensor=videos[0][None],
        save_file="output/wan_i2v_inpaint_14b.mp4",
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )
    print("Done!")

if __name__ == "__main__":
    main()
