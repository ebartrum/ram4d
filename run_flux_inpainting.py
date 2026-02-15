
import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from PIL import Image, ImageDraw

def generate_mask(image, box_coords):
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(box_coords, fill=255)
    return mask

def main():
    # Load the pipeline
    pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    # Load image
    image_path = "data/images/statue_IMG_2730.jpg"
    image = load_image(image_path)
    width, height = image.size
    
    # Load mask
    mask_path = "data/images/statue_mask.png"
    try:
        mask = load_image(mask_path)
        print(f"Loaded mask from {mask_path}")
        
        # Dilate mask by ~10 pixels
        from PIL import ImageFilter
        # MaxFilter with size 21 corresponds to a radius of 10 pixels (center + 10 neighbors on each side)
        mask = mask.filter(ImageFilter.MaxFilter(21)) 
        print("Applied 10px dilation to mask.")
        
    except Exception:
        print(f"Mask not found at {mask_path}, generating fallback box mask...")
        # Generate mask (center box) as fallback
        box_size = min(width, height) // 4
        center_x, center_y = width // 2, height // 2
        box_coords = (center_x - box_size, center_y - box_size, center_x + box_size, center_y + box_size)
        mask = generate_mask(image, box_coords)

    # Prompt
    prompt = "a cute corgi"

    # Run inference
    image = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        height=height,
        width=width,
        guidance_scale=30,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]


    # Save output
    output_path = "output/flux_inpainting_output.png"
    image.save(output_path)
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    main()
