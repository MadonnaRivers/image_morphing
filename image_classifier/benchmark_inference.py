"""
Benchmark inference speed: time per image (CPU, local model).
Reports average ms per image for preprocessing + inference.
"""
import os
os.environ.setdefault("TRANSFORMERS_USE_FAST_IMAGE_PROCESSOR", "0")

import time
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR
TEST_FOLDER = SCRIPT_DIR / "test_folder"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

def get_image_paths(folder: Path, limit: int = 50):
    paths = []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(f)
            if len(paths) >= limit:
                break
        elif f.is_dir():
            paths.extend(get_image_paths(f, limit - len(paths)))
            if len(paths) >= limit:
                break
    return paths[:limit]


def main():
    device = torch.device("cpu")
    from transformers import AutoImageProcessor, SiglipForImageClassification
    from PIL import Image

    print("Loading model (once)...")
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model = SiglipForImageClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    # Gather images from test_folder
    all_paths = []
    for sub in TEST_FOLDER.iterdir():
        if sub.is_dir():
            all_paths.extend(get_image_paths(sub, limit=20))
    all_paths = all_paths[:50]
    if not all_paths:
        print("No images in test_folder.")
        return

    # Warmup (2 images)
    for path in all_paths[:2]:
        img = Image.open(path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)

    # Timed runs: preprocessing + inference per image
    n = len(all_paths)
    start = time.perf_counter()
    for path in all_paths:
        img = Image.open(path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
    elapsed = time.perf_counter() - start
    ms_per_image = (elapsed / n) * 1000
    images_per_sec = n / elapsed

    print(f"\nInference benchmark (CPU): {n} images")
    print(f"  Total time:     {elapsed:.2f} s")
    print(f"  Per image:      {ms_per_image:.1f} ms")
    print(f"  Throughput:     {images_per_sec:.2f} images/sec")


if __name__ == "__main__":
    main()
