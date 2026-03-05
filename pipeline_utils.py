from PIL import Image, ImageFilter
import numpy as np
import cv2


def run_langsam(image_pil, prompt_path):
    """Run LangSAM on image_pil using the text prompt in prompt_path.
    Returns a binary uint8 numpy mask (H, W), values 0 or 255."""
    from lang_sam import LangSAM
    with open(prompt_path) as f:
        text_prompt = f.read().strip()
    print(f"  Running LangSAM with prompt: '{text_prompt}'")
    model = LangSAM()
    results = model.predict([image_pil], [text_prompt])
    if not results:
        raise RuntimeError("LangSAM returned no results")
    result = results[0]
    if hasattr(result, "masks"):
        masks = result.masks
    elif isinstance(result, dict):
        masks = result["masks"]
    else:
        raise RuntimeError(f"Unexpected LangSAM result type: {type(result)}")
    if hasattr(masks, "cpu"):
        masks_np = masks.cpu().numpy()
    else:
        masks_np = np.array(masks)
    combined = np.zeros((image_pil.height, image_pil.width), dtype=bool)
    for mask in masks_np:
        if mask.ndim == 3:
            mask = mask[0]
        combined = np.logical_or(combined, mask)
    return (combined.astype(np.uint8) * 255)

def preprocess_mask_image(mask_path, dilation_pixels=30):
    """
    Loads a mask image, fills holes using contour filling, and applies dilation.
    """
    print(f"Loading mask: {mask_path}")
    mask_img = Image.open(mask_path).convert("L")
    
    # Hole filling
    # Convert to numpy
    mask_np = np.array(mask_img)
    # Binarize (assuming mask is 0 or 255, but use threshold to be safe)
    _, mask_thresh = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    # Find external contours and fill them
    contours, _ = cv2.findContours(mask_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Filled {len(contours)} holes/contours in mask.")
    cv2.drawContours(mask_thresh, contours, -1, 255, -1)
    # Convert back to PIL
    mask_img = Image.fromarray(mask_thresh)
    
    # Dilation
    # Optimization for large dilation (no-mask condition)
    if dilation_pixels >= 256:
        print(f"Large dilation ({dilation_pixels}px) requested. Checking if mask is non-empty...")
        if np.max(mask_thresh) > 0:
            print("Mask is non-empty. Setting mask to full white (full generation).")
            mask_img = Image.new("L", mask_img.size, 255)
            return mask_img
            
    # MaxFilter size approx 2 * radius + 1
    filter_size = 2 * dilation_pixels + 1
    
    print(f"Applying {dilation_pixels}px dilation to mask (MaxFilter {filter_size})...")
    mask_img = mask_img.filter(ImageFilter.MaxFilter(filter_size))
    return mask_img
