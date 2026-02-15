
import sam2
import os
import pkgutil

package_path = os.path.dirname(sam2.__file__)
print(f"SAM2 path: {package_path}")

print("\nSubmodules:")
for _, name, _ in pkgutil.iter_modules([package_path]):
    print(name)

print("\nChecking specific imports:")
try:
    from sam2.build_sam import build_sam2
    print("from sam2.build_sam import build_sam2 - SUCCESS")
except ImportError as e:
    print(f"from sam2.build_sam import build_sam2 - FAILED: {e}")

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print("from sam2.sam2_image_predictor import SAM2ImagePredictor - SUCCESS")
except ImportError as e:
    print(f"from sam2.sam2_image_predictor import SAM2ImagePredictor - FAILED: {e}")
