
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image

def test_grounding_dino():
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    image = Image.new("RGB", (512, 512), "white")
    text = "a cat."

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    print("GroundingDINO run successfully via transformers!")

if __name__ == "__main__":
    try:
        test_grounding_dino()
    except Exception as e:
        print(f"Error: {e}")
