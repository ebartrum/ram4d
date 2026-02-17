import argparse
import torch
from PIL import Image
import numpy as np
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os

def run_inference(image_path, prompt, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. OWLv2 for Box Detection (Compatible with newer transformers)
    print("Loading OWLv2...")
    model_id = "google/owlvit-base-patch32"
    processor = OwlViTProcessor.from_pretrained(model_id)
    model = OwlViTForObjectDetection.from_pretrained(model_id).to(device)

    image = Image.open(image_path).convert("RGB")
    data = processor(images=image, text=[[prompt]], return_tensors="pt").to(device) # Text format is nested list
    
    print("Running OWLv2 inference...")
    with torch.no_grad():
        outputs = model(**data)

    # Post-process to get boxes
    target_sizes = torch.Tensor([image.size[::-1]]).to(device)
    # Access post_process_object_detection via image_processor
    results = processor.image_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)[0]
    
    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"] # Not strictly needed as we only check one class

    if len(boxes) == 0:
        print(f"No objects found for prompt: '{prompt}'")
        return

    # Filter by top score if too many (optional, but OWLv2 can be noisy)
    # Let's take the top 3 boxes max to avoid cluttering if threshold is low
    if len(boxes) > 3:
         top_indices = scores.argsort(descending=True)[:3]
         boxes = boxes[top_indices]
         scores = scores[top_indices]

    print(f"Found {len(boxes)} boxes.")
    
    # 2. SAM2 for Segmentation
    print("Loading SAM2...")
    checkpoint_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/sam2.1_hiera_small.pt")
    
    # Use the relative config path expected by Hydra (relative to sam2 package)
    config_path = "configs/sam2.1/sam2.1_hiera_s.yaml"
    
    try:
        sam2_model = build_sam2(config_path, checkpoint_path, device=device)
    except Exception as e:
        print(f"Error loading SAM2: {e}")
        # Try finding absolute path fallback relative to package
        try:
             import sam2
             package_dir = os.path.dirname(sam2.__file__)
             abs_config_path = os.path.join(package_dir, "configs", "sam2.1", "sam2.1_hiera_s.yaml")
             if os.path.exists(abs_config_path):
                 print(f"Retrying with absolute config path: {abs_config_path}")
                 sam2_model = build_sam2(abs_config_path, checkpoint_path, device=device)
             else:
                 raise e
        except Exception as e2:
             print(f"Failed to load SAM2: {e2}")
             return

    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(np.array(image))

    print("Running SAM2 inference...")
    input_boxes = boxes.cpu().numpy()
    
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    # Combine masks
    final_mask = np.zeros((image.height, image.width), dtype=bool)
    for mask in masks:
        if mask.ndim == 3:
            mask = mask[0]
        final_mask = np.logical_or(final_mask, mask)
    
    # Post-process: Keep only the largest connected component
    try:
        import cv2
        mask_uint8 = (final_mask * 255).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        
        if num_labels > 1:
            # Find the label with the largest area (ignoring background label 0)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            final_mask = (labels == largest_label)
            print("Filtered mask to keep only the largest connected component.")
    except ImportError:
        print("cv2 not found, skipping connected component filtering.")

    final_mask_img = Image.fromarray((final_mask * 255).astype(np.uint8))
    final_mask_img.save(output_path)
    print(f"Mask saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OWLv2 + SAM2 segmentation")
    parser.add_argument("--image", type=str, default="data/images/statue_IMG_2730.jpg", help="Path to input image")
    parser.add_argument("--prompt", type=str, default="a photo of a statue", help="Text prompt for OWLv2")
    parser.add_argument("--output", type=str, default="data/images/statue_mask.png", help="Path to save output mask")
    args = parser.parse_args()
    
    image_path = args.image
    prompt = args.prompt
    output_path = args.output
    
    try:
        run_inference(image_path, prompt, output_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
