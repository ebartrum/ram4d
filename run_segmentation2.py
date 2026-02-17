from PIL import Image
from lang_sam import LangSAM

model = LangSAM()
image_pil = Image.open("data/images/flux_inpainted_corgi.png").convert("RGB")
text_prompt = "corgi"
results = model.predict([image_pil], [text_prompt])