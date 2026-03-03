"""Extract frame 0 from a video file. Usage: python extract_frame0.py <video_path> <output_png>"""
import sys
import imageio
from PIL import Image

video_path, output_path = sys.argv[1], sys.argv[2]
reader = imageio.get_reader(video_path)
frame0 = next(iter(reader))
reader.close()
Image.fromarray(frame0).save(output_path)
print(f"Saved frame 0 ({frame0.shape[1]}x{frame0.shape[0]}) to {output_path}")
