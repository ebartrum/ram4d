import sys
import os
import glob

# Fix sklearn TLS issue: LD_PRELOAD must be set before the dynamic linker loads
# scikit-learn's bundled libgomp. We do this by re-execing the process.
if "LIBGOMP_PRELOADED" not in os.environ:
    libgomp_files = glob.glob(
        "/home/ubuntu/miniconda3/envs/mvadapter/lib/python*/site-packages/scikit_learn.libs/libgomp*.so*"
    )
    if libgomp_files:
        os.environ["LD_PRELOAD"] = libgomp_files[0]
        os.environ["LIBGOMP_PRELOADED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

from PIL import Image
import numpy as np
import argparse
from pipeline_utils import run_langsam

def main():
    parser = argparse.ArgumentParser(description="Run LangSAM segmentation on an image.")
    parser.add_argument("--image_path", type=str, default="data/images/flux_inpainted_corgi.png")
    parser.add_argument("--prompt_path", type=str, default="data/captions/corgi_segmentation.txt")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Defaults to <image_dir>/<image_stem>_mask.png")
    args = parser.parse_args()

    image_pil = Image.open(args.image_path).convert("RGB")
    mask_np = run_langsam(image_pil, args.prompt_path)

    if args.output_path is not None:
        output_path = args.output_path
    else:
        base_name = os.path.splitext(os.path.basename(args.image_path))[0]
        output_path = os.path.join(os.path.dirname(args.image_path), f"{base_name}_mask.png")

    Image.fromarray(mask_np).save(output_path)
    print(f"Mask saved to {output_path}")

if __name__ == "__main__":
    main()