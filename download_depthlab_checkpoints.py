"""
One-time download of DepthLab checkpoints to submodules/DepthLab/checkpoints/.

Requires: pip install huggingface_hub
"""
from huggingface_hub import snapshot_download, hf_hub_download
import os

BASE = "submodules/DepthLab/checkpoints"
os.makedirs(f"{BASE}/DepthLab", exist_ok=True)

print("Downloading marigold-depth-v1-0 ...")
snapshot_download("prs-eth/marigold-depth-v1-0",
    local_dir=f"{BASE}/marigold-depth-v1-0")

print("Downloading CLIP-ViT-H-14-laion2B-s32B-b79K ...")
snapshot_download("laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    local_dir=f"{BASE}/CLIP-ViT-H-14-laion2B-s32B-b79K")

print("Downloading DepthLab weights ...")
for fname in ["denoising_unet.pth", "reference_unet.pth", "mapping_layer.pth"]:
    hf_hub_download("Johanan0528/DepthLab", filename=fname,
        local_dir=f"{BASE}/DepthLab")

print("Done.")
