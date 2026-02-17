
import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from PIL import Image, ImageDraw

def generate_mask(image, box_coords):
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(box_coords, fill=255)
    return mask

import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run Flux Inpainting.")
    parser.add_argument("--image_path", type=str, default="data/images/statue_IMG_2730.jpg", help="Path to input image")
    parser.add_argument("--mask_path", type=str, default="data/images/statue_mask.png", help="Path to mask image")
    parser.add_argument("--prompt_path", type=str, default="data/captions/plinth.txt", help="Path to prompt text file")
    parser.add_argument("--output_path", type=str, default="output/flux_inpainting_output.png", help="Path to save output image")
    parser.add_argument("--dilation", type=int, default=10, help="Mask dilation radius in pixels")
    parser.add_argument("--guidance_scale", type=int, default=30, help="Guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    # Load the pipeline
    pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    # Load image
    print(f"Loading image from {args.image_path}")
    image = load_image(args.image_path)
    width, height = image.size
    
    # Load mask
    try:
        print(f"Loading mask from {args.mask_path}")
        mask = load_image(args.mask_path)
        
        # Dilate mask
        from PIL import ImageFilter
        # MaxFilter with size 2*r + 1
        filter_size = 2 * args.dilation + 1
        mask = mask.filter(ImageFilter.MaxFilter(filter_size)) 
        print(f"Applied {args.dilation}px dilation to mask.")
        
    except Exception as e:
        print(f"Mask not found or error loading at {args.mask_path}: {e}")
        print("Generating fallback box mask...")
        # Generate mask (center box) as fallback
        box_size = min(width, height) // 4
        center_x, center_y = width // 2, height // 2
        box_coords = (center_x - box_size, center_y - box_size, center_x + box_size, center_y + box_size)
        mask = generate_mask(image, box_coords)

    # Prompt
    with open(args.prompt_path, 'r') as f:
        prompt = f.read().strip()
    print(f"Loaded prompt from {args.prompt_path}: '{prompt}'")

    # Run inference
    image = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        height=height,
        width=width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        max_sequence_length=args.max_sequence_length,
        generator=torch.Generator("cpu").manual_seed(args.seed)
    ).images[0]


    # Save output
    image.save(args.output_path)
    print(f"Output saved to {args.output_path}")

if __name__ == "__main__":
    main()
