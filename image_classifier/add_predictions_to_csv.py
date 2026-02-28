"""
Read test_folder_images.csv, run the vision transformer model on each image,
add column 'this model (vision transformer)' with prediction: 'ai' or 'real'.
Overwrites the CSV with the new column. Runs on CPU.
"""
import os
os.environ.setdefault("TRANSFORMERS_USE_FAST_IMAGE_PROCESSOR", "0")

import csv
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR
TEST_FOLDER = SCRIPT_DIR / "test_folder"
OUT_CSV = SCRIPT_DIR / "test_folder_images.csv"
PREDICTION_COL = "this model (vision transformer)"


def main():
    device = torch.device("cpu")
    from transformers import AutoImageProcessor, SiglipForImageClassification
    from PIL import Image

    print("Loading model...")
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model = SiglipForImageClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    # Read existing CSV
    if not OUT_CSV.exists():
        print(f"Run export_test_folder_csv.py first to create {OUT_CSV}")
        return
    rows = []
    with open(OUT_CSV, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        rows = [list(row) for row in r]
    if not rows:
        print("CSV has no data rows.")
        return

    # Add new column to header
    if PREDICTION_COL not in header:
        header.append(PREDICTION_COL)
    n = len(rows)
    for i, row in enumerate(rows):
        rel_path = row[0] if row else ""
        full_path = TEST_FOLDER / rel_path.replace("/", os.sep)
        pred = ""
        try:
            image = Image.open(full_path).convert("RGB")
        except Exception as e:
            pred = "error"
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{n}] skip {rel_path}: {e}")
        if pred != "error":
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            pred_idx = logits.argmax(-1).item()
            label = model.config.id2label[pred_idx]
            pred = "real" if label == "hum" else "ai"
        # Ensure row has enough columns
        while len(row) < len(header):
            row.append("")
        row[-1] = pred
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  {i+1}/{n}")
    # Write back
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Updated {OUT_CSV} with predictions ({PREDICTION_COL}).")


if __name__ == "__main__":
    main()
