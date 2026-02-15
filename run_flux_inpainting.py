
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
    image_path = "data/images/table.jpg"
    image = load_image(image_path)
    
    # Generate mask (center box)
    width, height = image.size
    box_size = min(width, height) // 4
    center_x, center_y = width // 2, height // 2
    box_coords = (center_x - box_size, center_y - box_size, center_x + box_size, center_y + box_size)
    mask = generate_mask(image, box_coords)

    # Prompt
    prompt = "a vase of flowers on the table"

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
