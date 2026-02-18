
import os
import cv2
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor


# Define paths
VIDEO_PATH = "data/videos/corgi_wan.mp4"
MASK_PATH = "data/images/corgi_mask.png"
FRAMES_DIR = "data/cache/corgi_wan_frames"
OUTPUT_DIR = "output/corgi_mask_propagated" # Changed to output/
CHECKPOINT = "checkpoints/sam2_hiera_tiny.pt"
CONFIG = "sam2_hiera_t.yaml"

def extract_frames(video_path, output_dir):
    """Extracts frames from a video to a directory."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Save as JPEG with 0-filled index
        frame_path = os.path.join(output_dir, f"{frame_count:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()
    print(f"Extracted {frame_count} frames to {output_dir}")

def resize_mask(mask_path, size=(512, 512)):
    """Resizes a mask to a target size using nearest neighbor interpolation."""
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(size, Image.NEAREST)
    return np.array(mask)

def main():
    # 1. Extract Frames
    if not os.path.exists(FRAMES_DIR) or not os.listdir(FRAMES_DIR):
         print(f"Extracting frames to {FRAMES_DIR}...")
         extract_frames(VIDEO_PATH, FRAMES_DIR)
    else:
         print(f"Frames already found in {FRAMES_DIR}, skipping extraction.")

    # 2. Initialize SAM2
    print("Initializing SAM2 model...")
    predictor = build_sam2_video_predictor(CONFIG, CHECKPOINT)
    inference_state = predictor.init_state(video_path=FRAMES_DIR)

    # 3. Load and Resize Mask
    print("Loading and resizing mask...")
    mask = resize_mask(MASK_PATH, size=(512, 512))
    
    unique_labels = np.unique(mask)
    print(f"Unique mask values: {unique_labels}")
    
    binary_mask = (mask > 128).astype(np.float32)
    
    # Add object id 1
    ann_frame_idx = 0
    ann_obj_id = 1
    
    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        mask=binary_mask
    )
    
    # 4. Propagate
    print(f"Propagating mask and saving to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize Video Writer
    video_output_path = os.path.join(OUTPUT_DIR, "propagated_masks.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_output_path, fourcc, 16.0, (512, 512), isColor=False) # isColor=False for grayscale
    
    video_segments = {} 
    
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        
        if 1 in video_segments[out_frame_idx]:
             mask_array = video_segments[out_frame_idx][1] # [1, H, W]
             mask_array = mask_array.squeeze() # [H, W]
             
             # Save as PNG
             save_path = os.path.join(OUTPUT_DIR, f"{out_frame_idx:05d}.png")
             mask_img = (mask_array * 255).astype(np.uint8)
             Image.fromarray(mask_img).save(save_path)
             
             # Write to Video
             out_video.write(mask_img)
             
    out_video.release()
    print(f"Propagation complete. Masks saved to {OUTPUT_DIR}")
    print(f"Video saved to {video_output_path}")

if __name__ == "__main__":
    main()
