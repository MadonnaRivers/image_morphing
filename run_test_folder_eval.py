"""
Run ForensicsSAM on test_folder images (4 classes: ai, canva, forged, real).
Read test_folder_images_new.csv, run model on each image, add column 'forensicsam',
and save back to test_folder_images_new.csv.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import cv2
import torch
from torchvision import transforms
from tqdm import tqdm

# Project imports (run from repo root)
from segment_anything import sam_model_registry
from forensics_sam import ForensicsSAM
from adversary_detector import AdversaryDetector

# Default paths — 1024 original/full quality; use --size 512 or 768 for speed
TEST_FOLDER = "test_folder"
CSV_PATH = "test_folder_images_new.csv"
IMAGE_SIZE = 1024
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

model_type = ["vit_b", "vit_l", "vit_h"]
checkpoint = {
    "vit_b": "./weight/sam_vit_b_01ec64.pth",
    "vit_l": "./weight/sam_vit_l_0b3195.pth",
    "vit_h": "./weight/sam_vit_h_4b8939.pth",
}


def load_and_preprocess(image_path, image_size=IMAGE_SIZE):
    """Load image and preprocess like BasicDataloader (val, no augment)."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, (image_size, image_size), cv2.INTER_AREA)
    img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, [0,1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    return transform(img).unsqueeze(0)  # (1, 3, H, W)


def main():
    parser = argparse.ArgumentParser(description="Run ForensicsSAM on test_folder and write forensicsam column to CSV")
    parser.add_argument("--test-folder", default=TEST_FOLDER, help=f"Root folder containing class subdirs (default: {TEST_FOLDER})")
    parser.add_argument("--csv", default=CSV_PATH, help=f"Input/output CSV (default: {CSV_PATH})")
    parser.add_argument("--size", type=int, default=IMAGE_SIZE, help=f"Inference size (default: {IMAGE_SIZE}; use 1024 for max quality)")
    parser.add_argument("--device", default=None, help="Device: cuda or cpu (default: auto)")
    args = parser.parse_args()

    test_folder = args.test_folder
    csv_path = args.csv
    image_size = args.size if args.size in (512, 768, 1024) else IMAGE_SIZE
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    if not os.path.isdir(test_folder):
        print(f"Error: test folder not found: {os.path.abspath(test_folder)}")
        sys.exit(1)
    if not os.path.isfile(csv_path):
        print(f"Error: CSV not found: {os.path.abspath(csv_path)}")
        sys.exit(1)

    # Load CSV (handle quoted paths with commas)
    df = pd.read_csv(csv_path)
    if "image" not in df.columns:
        print("Error: CSV must have an 'image' column")
        sys.exit(1)

    # Load models (same size as preprocessing)
    sam_type = model_type[2]  # vit_h
    r = 8
    print("Loading SAM and ForensicsSAM...")
    sam, _ = sam_model_registry[sam_type](image_size=image_size, checkpoint=checkpoint.get(sam_type))
    forensics_sam = ForensicsSAM(sam, r).to(device).eval()
    adv_detector = AdversaryDetector().to(device).eval()
    adv_detector.load_detector("./weight/adversary_detector.pth")

    results = []
    missing = []
    errors = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="ForensicsSAM"):
        rel_path = row["image"]
        if pd.isna(rel_path) or not str(rel_path).strip():
            results.append("")
            continue
        rel_path = str(rel_path).strip()
        full_path = os.path.join(test_folder, rel_path)
        if not os.path.isfile(full_path):
            missing.append(rel_path)
            results.append("")
            continue
        try:
            x = load_and_preprocess(full_path, image_size)
            x = x.to(device)
            with torch.no_grad():
                logits, _ = adv_detector(x)
                preds = torch.argmax(logits, dim=1)
                _, cls_prediction = forensics_sam(x, preds)
            prob = torch.sigmoid(cls_prediction).squeeze().item()
            label = "forged" if prob > 0.5 else "real"
            results.append(f"{label} ({prob:.4f})")
        except Exception as e:
            errors.append((rel_path, str(e)))
            results.append("")

    if missing:
        print(f"Warning: {len(missing)} images not found (e.g. {missing[:3]})")
    if errors:
        print(f"Warning: {len(errors)} errors (e.g. {errors[:3]})")

    df["forensicsam"] = results
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} with column 'forensicsam' ({len(results)} rows).")


if __name__ == "__main__":
    main()
