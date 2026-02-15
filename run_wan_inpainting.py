
import sys
import os
import torch
import numpy as np
import cv2
from PIL import Image

# Add Wan2.1 to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "Wan2.1"))

from wan.configs import wan_t2v_1_3B
from wan.text2video_inpaint import WanT2VInpaint
from huggingface_hub import snapshot_download

def download_model(local_dir):
    if os.path.exists(local_dir) and os.path.isdir(local_dir):
        # Basic check to see if files exist
        if os.path.exists(os.path.join(local_dir, "config.json")):
            print(f"Model seems to be present at {local_dir}")
            return

    print(f"Downloading Wan2.1-T2V-1.3B to {local_dir}...")
    snapshot_download(repo_id="Wan-AI/Wan2.1-T2V-1.3B", local_dir=local_dir)
    print("Download complete.")

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    print(f"Loaded {len(frames)} frames from {video_path}")
    return frames

def save_video(tensor, output_path, fps=30):
    # Tensor: [C, F, H, W] -> [F, H, W, C]
    video = tensor.detach().cpu().numpy()
    video = np.transpose(video, (1, 2, 3, 0))
    video = (video * 255).astype(np.uint8)
    
    height, width, _ = video[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in video:
        # RGB to BGR
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()
    print(f"Saved video to {output_path}")

def main():
    device_id = 0
    checkpoint_dir = "Wan2.1-T2V-1.3B"
    
    # Download model if needed
    download_model(checkpoint_dir)
    
    # Input paths
    input_video_path = "data/video_input/wan_input.mp4"
    mask_video_path = "data/video_input/wan_mask.mp4"
    output_path = "output/wan_inpainting_output.mp4"
    
    # Load videos
    input_frames = load_video_frames(input_video_path)
    mask_frames = load_video_frames(mask_video_path)
    
    # Ensure same length
    min_len = min(len(input_frames), len(mask_frames))
    input_frames = input_frames[:min_len]
    mask_frames = mask_frames[:min_len]
    
    # Config
    cfg = wan_t2v_1_3B.t2v_1_3B
    
    # Initialize model
    print("Initializing WanT2VInpaint...")
    wan_inpaint = WanT2VInpaint(config=cfg, checkpoint_dir=checkpoint_dir, device_id=device_id)
    
    # Run inpainting
    prompt = "a cute corgi, jumping and running, barking, 8k quality, detailed"
    width, height = input_frames[0].size
    
    print("Running inpainting...")
    generated_video = wan_inpaint.generate_inpaint(
        input_prompt=prompt,
        init_video=input_frames,
        mask=mask_frames,
        size=(width, height),
        strength=1.0, # Full inpainting in masked areas
        sampling_steps=50,
        guide_scale=6.0,
        seed=42,
        offload_model=True
    )
    
    # Save output
    save_video(generated_video, output_path)

if __name__ == "__main__":
    main()
