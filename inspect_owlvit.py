
from transformers import OwlViTProcessor
model_id = "google/owlvit-base-patch32"
processor = OwlViTProcessor.from_pretrained(model_id)
print("Attributes of OwlViTProcessor:")
print(dir(processor))
if hasattr(processor, "image_processor"):
    print("\nAttributes of processor.image_processor:")
    print(dir(processor.image_processor))
