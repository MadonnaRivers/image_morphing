"""
Evaluate the AI vs Human classifier on test_folder (ai, canva, real, forged).
Reports per-section: samples, correctly identified, misidentified (count + %).
Runs on CPU using local weights.
"""
import os
os.environ.setdefault("TRANSFORMERS_USE_FAST_IMAGE_PROCESSOR", "0")

import torch
from pathlib import Path
from collections import defaultdict

# Folder name -> expected model label (model has "ai" and "hum" only)
# ai, canva, forged = non-real -> expect "ai"; real -> expect "hum"
FOLDER_TO_EXPECTED = {
    "ai": "ai",
    "canva": "ai",
    "real": "hum",
    "forged": "ai",
}

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR
TEST_FOLDER = SCRIPT_DIR / "test_folder"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def get_image_paths(folder: Path) -> list[Path]:
    paths = []
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(f)
        elif f.is_dir():
            paths.extend(get_image_paths(f))
    return paths


def main():
    device = torch.device("cpu")
    print(f"Using device: {device}\n")

    from transformers import AutoImageProcessor, SiglipForImageClassification
    from PIL import Image

    print(f"Loading model from: {MODEL_PATH}")
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model = SiglipForImageClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    if not TEST_FOLDER.exists():
        print(f"Error: test_folder not found at {TEST_FOLDER}")
        return

    # Collect all section paths
    sections = {}
    for sub in TEST_FOLDER.iterdir():
        if sub.is_dir() and sub.name in FOLDER_TO_EXPECTED:
            paths = get_image_paths(sub)
            sections[sub.name] = paths

    if not sections:
        print("No valid sections (ai, canva, real, forged) found in test_folder.")
        return

    # Run inference per section and aggregate
    results = []
    for section_name in ["ai", "canva", "real", "forged"]:
        if section_name not in sections:
            results.append((section_name, 0, 0, 0))
            continue
        paths = sections[section_name]
        expected = FOLDER_TO_EXPECTED[section_name]
        correct = 0
        misidentified = 0
        for i, path in enumerate(paths):
            try:
                image = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"  Skip (unreadable): {path} - {e}")
                misidentified += 1  # count as error for totals
                continue
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            pred_idx = logits.argmax(-1).item()
            pred_label = model.config.id2label[pred_idx]
            if pred_label == expected:
                correct += 1
            else:
                misidentified += 1
            if (i + 1) % 20 == 0 or (i + 1) == len(paths):
                print(f"  {section_name}: {i + 1}/{len(paths)}")
        total = correct + misidentified
        results.append((section_name, total, correct, misidentified))

    # Print report
    print("\n" + "=" * 80)
    print("EVALUATION REPORT (test_folder)")
    print("=" * 80)
    print(f"{'Section':<10} {'Samples':>8} {'Correct':>10} {'Correct %':>10} {'Misidentified':>14} {'Misidentified %':>16}")
    print("-" * 80)
    grand_total = 0
    grand_correct = 0
    grand_mis = 0
    for section_name, total, correct, mis in results:
        grand_total += total
        grand_correct += correct
        grand_mis += mis
        pct_correct = (100.0 * correct / total) if total else 0.0
        pct_mis = (100.0 * mis / total) if total else 0.0
        print(f"{section_name:<10} {total:>8} {correct:>10} {pct_correct:>9.2f}% {mis:>14} {pct_mis:>15.2f}%")
    print("-" * 80)
    overall_correct_pct = (100.0 * grand_correct / grand_total) if grand_total else 0.0
    overall_mis_pct = (100.0 * grand_mis / grand_total) if grand_total else 0.0
    print(f"{'TOTAL':<10} {grand_total:>8} {grand_correct:>10} {overall_correct_pct:>9.2f}% {grand_mis:>14} {overall_mis_pct:>15.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
