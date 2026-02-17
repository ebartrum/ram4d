from PIL import Image
from lang_sam import LangSAM
import numpy as np

print("Loading LangSAM...")
model = LangSAM()
image_path = "data/images/flux_inpainted_corgi.png"
print(f"Loading image: {image_path}")
image_pil = Image.open(image_path).convert("RGB")
text_prompt = "corgi"

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
        output_path = "data/images/flux_inpainted_corgi_mask.png"
        final_mask_img.save(output_path)
        print(f"Mask saved to {output_path}")