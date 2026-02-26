"""
Generate a color legend image for object_pred segmentation outputs.
Reads a grayscale objects_pred PNG (pixel value = object ID) and the
corresponding objects_pred_color PNG, then outputs a legend image showing
each ID with its color swatch.

Usage:
    python tools/make_seg_legend.py \
        --pred output/inpaint360/bag/train/ours_2000/objects_pred/IMG_0088.png \
        --color output/inpaint360/bag/train/ours_2000/objects_pred_color/IMG_0088.png \
        --out output/inpaint360/bag/seg_legend.png
"""
import argparse
import colorsys
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        return np.zeros(3, dtype=np.uint8)
    golden_ratio = 2.7182818284
    h = (id * golden_ratio) % 1
    s = 0.5 + (id % 2) * 0.5
    l = 0.5
    rgb = np.zeros(3, dtype=np.uint8)
    if id == 0:
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r * 255), int(g * 255), int(b * 255)
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Grayscale objects_pred PNG")
    parser.add_argument("--color", required=True, help="Color objects_pred_color PNG")
    parser.add_argument("--out", required=True, help="Output legend PNG path")
    args = parser.parse_args()

    gray = np.array(Image.open(args.pred).convert("L"))
    color_img = np.array(Image.open(args.color).convert("RGB"))

    unique_ids = sorted(int(v) for v in np.unique(gray))
    print(f"Unique object IDs: {unique_ids}")

    # For each ID, find average color from the color image
    swatch_size = 40
    row_h = swatch_size + 4
    margin = 8
    text_w = 60
    legend_w = margin + swatch_size + 8 + text_w + margin
    legend_h = margin + row_h * len(unique_ids) + margin

    legend = Image.new("RGB", (legend_w, legend_h), (240, 240, 240))
    draw = ImageDraw.Draw(legend)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for i, obj_id in enumerate(unique_ids):
        y = margin + i * row_h
        # Compute color from id2rgb
        rgb = id2rgb(obj_id)
        color_tuple = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        draw.rectangle([margin, y, margin + swatch_size, y + swatch_size], fill=color_tuple, outline=(0, 0, 0))
        draw.text((margin + swatch_size + 8, y + swatch_size // 4), f"ID {obj_id}", fill=(0, 0, 0), font=font)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    legend.save(args.out)
    print(f"Legend saved to {args.out}")

    # Also overlay IDs on the color image
    overlay = Image.fromarray(color_img).convert("RGBA")
    overlay_draw = ImageDraw.Draw(overlay)
    for obj_id in unique_ids:
        if obj_id == 0:
            continue
        mask = gray == obj_id
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        overlay_draw.text((cx - 10, cy - 10), str(obj_id), fill=(255, 255, 255, 255), font=font)
        overlay_draw.text((cx - 11, cy - 11), str(obj_id), fill=(0, 0, 0, 255), font=font)

    annotated_path = args.out.replace(".png", "_annotated.png")
    overlay.convert("RGB").save(annotated_path)
    print(f"Annotated color image saved to {annotated_path}")


if __name__ == "__main__":
    main()
