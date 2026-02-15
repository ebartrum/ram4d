
import cv2
import numpy as np
import os
from PIL import Image

def load_image(path):
    return Image.open(path).convert("RGB")

def load_mask(path):
    return Image.open(path).convert("L")

def create_video_from_frames(frames, output_path, fps=30):
    if isinstance(frames[0], np.ndarray):
        height, width, layers = frames[0].shape
    else:
         width, height = frames[0].size
         
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        # Convert RGB (PIL) to BGR (OpenCV)
        bgr_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        video.write(bgr_frame)

    video.release()
    print(f"Saved video to {output_path}")

def main():
    statue_path = "data/images/statue_IMG_2730.jpg"
    flux_output_path = "output/flux_inpainting_output.png"
    mask_path = "data/images/statue_mask.png"
    
    output_dir = "data/video_input"
    os.makedirs(output_dir, exist_ok=True)
    
    statue_img = load_image(statue_path)
    flux_img = load_image(flux_output_path)
    mask_img = load_mask(mask_path)
    
    # Resize flux image to match statue image if needed (they should match)
    if flux_img.size != statue_img.size:
        print(f"Resizing flux output {flux_img.size} to match statue {statue_img.size}")
        flux_img = flux_img.resize(statue_img.size, Image.LANCZOS)
        
    if mask_img.size != statue_img.size:
        print(f"Resizing mask {mask_img.size} to match statue {statue_img.size}")
        mask_img = mask_img.resize(statue_img.size, Image.NEAREST)

    # Dilate mask by ~10 pixels (MaxFilter 21)
    from PIL import ImageFilter
    print("Applying 10px dilation to mask...")
    mask_img = mask_img.filter(ImageFilter.MaxFilter(21))

    width, height = statue_img.size
    
    # Create 81 frames
    video_frames = []
    mask_frames = []
    
    total_frames = 81
    
    # Black mask (no inpainting)
    black_mask = Image.new("L", (width, height), 0)
    
    for i in range(total_frames):
        if i == 0 or i == total_frames - 1:
            # Frame 0 and 80: Flux output, unmasked (keep content)
            video_frames.append(flux_img)
            mask_frames.append(black_mask)
        else:
            # Frames 1-79: Statue image, masked (inpaint this)
            video_frames.append(statue_img)
            mask_frames.append(mask_img)

    # Save input video
    create_video_from_frames(video_frames, os.path.join(output_dir, "wan_input.mp4"))
    
    # Save mask video
    # Convert mask frames to RGB for video saving (grayscale is fine but opencv handles 3ch usually)
    mask_rgb_frames = [f.convert("RGB") for f in mask_frames]
    create_video_from_frames(mask_rgb_frames, os.path.join(output_dir, "wan_mask.mp4"))

if __name__ == "__main__":
    main()
