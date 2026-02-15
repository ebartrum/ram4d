
import PIL.Image
img = PIL.Image.open('data/images/table.jpg')
print(f"Format: {img.format}")
print(f"Mode: {img.mode}")
print(f"Size: {img.size}")
