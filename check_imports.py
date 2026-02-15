
try:
    import segment_anything
    print("segment_anything found")
except ImportError:
    print("segment_anything NOT found")

try:
    import sam2
    print("sam2 found")
except ImportError:
    print("sam2 NOT found")

try:
    import transformers
    print(f"transformers version: {transformers.__version__}")
except ImportError:
    print("transformers NOT found")
