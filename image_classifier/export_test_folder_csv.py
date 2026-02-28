"""
Scan test_folder (ai, canva, real, forged) and export a CSV with all images
and their real type. CSV opens in Excel.
"""
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_FOLDER = SCRIPT_DIR / "test_folder"
OUT_CSV = SCRIPT_DIR / "test_folder_images.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

SECTION_NAMES = ("ai", "canva", "real", "forged")


def collect_images_by_section():
    """Returns list of (relative_path, real_type)."""
    rows = []
    for section in SECTION_NAMES:
        section_path = TEST_FOLDER / section
        if not section_path.is_dir():
            continue
        for f in section_path.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                rel = f.relative_to(TEST_FOLDER)
                rows.append((str(rel).replace("\\", "/"), section))
    return rows


def main():
    if not TEST_FOLDER.exists():
        print(f"test_folder not found: {TEST_FOLDER}")
        return
    rows = collect_images_by_section()
    rows.sort(key=lambda x: (x[1], x[0]))
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "real_type"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} records to {OUT_CSV}")


if __name__ == "__main__":
    main()
