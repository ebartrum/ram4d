import sys
import os
import torch
from huggingface_hub import snapshot_download
from PIL import Image
import numpy as np

# Add official_wan_repo to path so we can import wan
sys.path.insert(0, os.path.abspath("official_wan_repo"))

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.utils.utils import cache_video

def main():
    device_id = 0
    rank = 0
    
    # Model parameters
    task = "flf2v-14B"
    repo_id = "Wan-AI/Wan2.1-FLF2V-14B-720P"
    
    # Download model
    print(f"Downloading model {repo_id}...")
    checkpoint_dir = snapshot_download(repo_id=repo_id)
    print(f"Model downloaded to {checkpoint_dir}")
    
    # Load inputs
    first_frame_path = "output/flux_inpainting_output.png"
    last_frame_path = "output/flux_inpainting_output.png" 
    
    if not os.path.exists(first_frame_path):
        print(f"Error: Input frame {first_frame_path} not found.")
        return

    print(f"Loading input frames: {first_frame_path}")
    first_frame = Image.open(first_frame_path).convert("RGB")
    last_frame = Image.open(last_frame_path).convert("RGB")
    
    prompt = "a cute corgi, jumping and running, barking, 8k quality, detailed"
    print(f"Prompt: {prompt}")

    # Initialize model
    cfg = WAN_CONFIGS[task]
    print(f"Initializing WanFLF2V with config: {task}")
    
    wan_flf = wan.WanFLF2V(
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        device_id=device_id,
        rank=rank,
        t5_cpu=True, # Offload T5 to CPU to save VRAM
    )
    
    # Generate
    print("Generating video... (this may take a while for 14B model)")
    video = wan_flf.generate(
        input_prompt=prompt,
        first_frame=first_frame,
        last_frame=last_frame,
        max_area=SIZE_CONFIGS['1280*720'][0] * SIZE_CONFIGS['1280*720'][1],
        frame_num=81,
        shift=16.0, # Default for flf
        sampling_steps=40, # Reduce slightly from 50 for speed/compat
        guide_scale=5.0,
        seed=42,
        offload_model=True # Offload model to CPU
    )
    
    # Save output
    output_filename = "output/wan_flf_14b_output.mp4"
    print(f"Saving video to {output_filename}")
    
    cache_video(
        tensor=video[None],
        save_file=output_filename,
        fps=16, # Default fps
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )
    print("Done!")

if __name__ == "__main__":
    main()
