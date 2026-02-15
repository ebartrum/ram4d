
import diffusers
print(f"Diffusers version: {diffusers.__version__}")
try:
    from diffusers import FluxFillPipeline
    print("FluxFillPipeline found")
except ImportError:
    print("FluxFillPipeline NOT found")
