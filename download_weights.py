"""
Download ForensicsSAM weights: Google Drive bundle (official) + Hugging Face SAM fallback.
Run once before inference: python download_weights.py
"""
from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHT_DIR = ROOT / "weight"
CLASSIFIER_DIR = ROOT / "image_classifier"
CLASSIFIER_HF_REPO = "Organika/sdxl-detector"
GD_FILE_ID = "1stLg8bJ1W2E7dVAHC8TYj917REO4sttt"
# Hugging Face SAM checkpoints (fallback if not in GD bundle)
SAM_HF_REPOS = {
    "sam_vit_h_4b8939.pth": "segments-arnaud/sam_vit_h",
    "sam_vit_b_01ec64.pth": "segments-arnaud/sam_vit_b",
}


def ensure_weight_dir() -> None:
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Weight directory: {WEIGHT_DIR}")


def download_from_google_drive() -> bool:
    try:
        import gdown
    except ImportError:
        print("Install gdown: pip install gdown")
        return False

    out_zip = WEIGHT_DIR / "forensics_sam_weights.zip"
    if out_zip.exists():
        print(f"Already downloaded: {out_zip}")
        return True
    print("Downloading pre-trained weights from Google Drive...")
    try:
        url = f"https://drive.google.com/uc?id={GD_FILE_ID}"
        gdown.download(url, str(out_zip), quiet=False, fuzzy=True)
        if not out_zip.exists():
            # gdown sometimes saves with a different name
            for f in WEIGHT_DIR.glob("*.zip"):
                out_zip = f
                break
        if out_zip.exists() and out_zip.stat().st_size > 0:
            return True
    except Exception as e:
        print(f"Google Drive download failed: {e}")
    return False


def unzip_if_needed() -> None:
    for z in WEIGHT_DIR.glob("*.zip"):
        print(f"Extracting {z.name}...")
        try:
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(WEIGHT_DIR)
            # If zip had a top-level "weight" folder, move contents up
            nested = WEIGHT_DIR / "weight"
            if nested.is_dir():
                for f in nested.iterdir():
                    dest = WEIGHT_DIR / f.name
                    if dest.exists() and dest.is_file() and f.stat().st_size == dest.stat().st_size:
                        continue
                    import shutil
                    shutil.move(str(f), str(WEIGHT_DIR / f.name))
                nested.rmdir()
            print("Done.")
        except Exception as e:
            print(f"Unzip failed: {e}")


def download_sam_from_huggingface(filename: str) -> bool:
    repo = SAM_HF_REPOS.get(filename)
    if not repo:
        return False
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo, filename=filename, local_dir=str(WEIGHT_DIR))
        target = WEIGHT_DIR / filename
        if path and os.path.isfile(path):
            if os.path.realpath(path) != os.path.realpath(target):
                import shutil
                shutil.copy2(path, target)
            return True
        if target.exists():
            return True
    except Exception as e:
        print(f"Hugging Face download failed for {filename}: {e}")
    return False


def download_classifier_from_huggingface() -> bool:
    """Pull the AI-vs-Human classifier (Organika/sdxl-detector, ~347 MB) into image_classifier/."""
    target = CLASSIFIER_DIR / "model.safetensors"
    if target.exists() and target.stat().st_size > 0:
        print(f"OK: classifier weights already present ({target})")
        return True
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        return False
    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {CLASSIFIER_HF_REPO} into {CLASSIFIER_DIR}...")
    try:
        snapshot_download(
            repo_id=CLASSIFIER_HF_REPO,
            local_dir=str(CLASSIFIER_DIR),
            allow_patterns=["*.json", "*.safetensors", "*.txt"],
        )
        return target.exists()
    except Exception as e:
        print(f"Classifier download failed: {e}")
        return False


def main() -> int:
    ensure_weight_dir()

    # 1) Try Google Drive bundle
    ok = download_from_google_drive()
    if ok:
        unzip_if_needed()

    # 2) Ensure SAM checkpoints exist (vit_h + vit_b for optional ViT-B speed test)
    sam_files = ["sam_vit_h_4b8939.pth", "sam_vit_b_01ec64.pth"]
    for name in sam_files:
        path = WEIGHT_DIR / name
        if path.exists() and path.stat().st_size > 0:
            print(f"OK: {path.name}")
            continue
        print(f"Missing {name}, trying Hugging Face...")
        if download_sam_from_huggingface(name):
            print(f"Downloaded: {name}")
        else:
            print(f"Could not get {name}")

    # 3) AI-vs-Human classifier (Swin-T, Organika/sdxl-detector)
    download_classifier_from_huggingface()

    # 4) List what we have
    required = [
        "sam_vit_h_4b8939.pth",
        "adversary_detector.pth",
        "adversary_experts.pth",
        "forgery_experts.pth",
    ]
    missing = [r for r in required if not (WEIGHT_DIR / r).exists()]
    if not (CLASSIFIER_DIR / "model.safetensors").exists():
        missing.append("image_classifier/model.safetensors")
    if missing:
        print("\nStill missing (required for inference):", missing)
        print("Download the bundle from: https://drive.google.com/file/d/1stLg8bJ1W2E7dVAHC8TYj917REO4sttt/view")
        return 1
    print("\nAll required weights are present. Run: python -m uvicorn api:app")
    return 0


if __name__ == "__main__":
    sys.exit(main())
