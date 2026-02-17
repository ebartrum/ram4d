from PIL import Image, ImageFilter
import numpy as np
import cv2

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
    # MaxFilter size approx 2 * radius + 1
    filter_size = 2 * dilation_pixels + 1
    
    print(f"Applying {dilation_pixels}px dilation to mask (MaxFilter {filter_size})...")
    mask_img = mask_img.filter(ImageFilter.MaxFilter(filter_size))
    return mask_img
