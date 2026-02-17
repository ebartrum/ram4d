from PIL import Image
from lang_sam import LangSAM
import numpy as np

import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run LangSAM segmentation on an image.")
    parser.add_argument("--image_path", type=str, default="data/images/flux_inpainted_corgi.png", help="Path to the input image.")
    parser.add_argument("--prompt_path", type=str, default="data/captions/corgi_segmentation.txt", help="Path to the text prompt file.")
    args = parser.parse_args()

    print("Loading LangSAM...")
    model = LangSAM()

    image_path = args.image_path
    print(f"Loading image: {image_path}")
    image_pil = Image.open(image_path).convert("RGB")

    prompt_path = args.prompt_path
    print(f"Reading prompt from: {prompt_path}")
    with open(prompt_path, "r") as f:
        text_prompt = f.read().strip()

    print(f"Running prediction for prompt: '{text_prompt}'")
    results = model.predict([image_pil], [text_prompt])

    # Results is a list of results.
    if len(results) == 0:
        print("No results returned.")
    else:
        result = results[0]
        # Check what result contains. Usually it is a dict or object with 'masks', 'boxes', etc.
        # If LangSAM from luca-medeiros: returns (masks, boxes, phrases, logits) tuple per image?
        # Or list of such tuples?
        # Let's inspect the type of result.
        
        masks = result['masks'] if isinstance(result, dict) else getattr(result, 'masks', None)
        
        if masks is None:
            # Fallback if structure is different
            # Standard LangSAM usage: masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
            # But here we passed LISTs.
            pass

        # If results is a list of objects with .masks
        if hasattr(result, 'masks'):
            masks = result.masks
        elif isinstance(result, dict) and 'masks' in result:
            masks = result['masks']
            
        if masks is None:
            print(f"Could not extract masks from result: {result}")
        elif len(masks) == 0:
            print("No objects found.")
        else:
            print(f"Found {len(masks)} masks.")
            # Combine masks if multiple
            final_mask = np.zeros((image_pil.height, image_pil.width), dtype=bool)
            
            # masks is likely tensor or numpy array [N, H, W]
            if hasattr(masks, 'cpu'):
                masks_np = masks.cpu().numpy()
            else:
                masks_np = np.array(masks)
                
            for mask in masks_np:
                if mask.ndim == 3:
                     mask = mask[0]
                final_mask = np.logical_or(final_mask, mask)
                
            final_mask_img = Image.fromarray((final_mask * 255).astype(np.uint8))
            
            # Create output path based on input path
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = os.path.dirname(image_path)
            output_path = os.path.join(output_dir, f"{base_name}_mask.png")
            
            final_mask_img.save(output_path)
            print(f"Mask saved to {output_path}")

if __name__ == "__main__":
    main()