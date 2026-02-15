
import PIL.Image
import PIL.ImageDraw
import numpy as np

def create_test_data():
    # Create a simple RGB image (white background with a red circle)
    img = PIL.Image.new('RGB', (512, 512), color='white')
    draw = PIL.ImageDraw.Draw(img)
    draw.ellipse((156, 156, 356, 356), fill='red')
    img.save('data/images/test_image.png')
    
    # Create a mask (black background with white circle matching the red circle position)
    mask = PIL.Image.new('L', (512, 512), color='black')
    mask_draw = PIL.ImageDraw.Draw(mask)
    mask_draw.ellipse((156, 156, 356, 356), fill='white')
    mask.save('data/images/test_mask.png')
    
    print("Created data/images/test_image.png and data/images/test_mask.png")

if __name__ == "__main__":
    create_test_data()
