# Image Morphing

Image forgery detection and AI-vs-human classification API. Upload an image to get real/forged prediction plus a forgery heatmap, and an AI vs human label.

## What it does

- **Forensics pipeline**: Detects whether an image is real or forged and outputs a pixel-level forgery mask (heatmap).
- **AI vs human**: Classifies the image as AI-generated or human (e.g. photo).
- **REST API**: Single endpoint accepts an image and returns both results plus a 3-panel visualization.

## Quick start

1. **Clone and install**
   ```bash
   git clone https://github.com/MadonnaRivers/image_morphing.git
   cd image_morphing
   pip install -r requirements.txt
   ```

2. **Download weights**
   ```bash
   python download_weights.py
   ```
   Puts SAM and ForensicsSAM weights under `weight/`. Get the full bundle from the link printed if needed.

3. **Run the API**
   ```bash
   python -m uvicorn api:app --host 127.0.0.1 --port 8080
   ```
   Then `POST` an image to `http://127.0.0.1:8080/analyze` (form field `file`).

4. **Single image (no API)**
   ```bash
   python run_single_image.py --image path/to/image.jpg
   ```
   Prints real/forged and saves a visualization PNG.

5. **Batch CSV**
   ```bash
   python run_test_folder_eval.py --csv test_folder_images_new.csv --test-folder test_folder
   ```
   Runs on all images in the CSV and writes a `forensicsam` column.

## Project layout

- `api.py` — FastAPI app: `/analyze` runs ForensicsSAM + AI-vs-human in parallel.
- `run_single_image.py` — One-image inference and visualization.
- `run_test_folder_eval.py` — Batch evaluation over a CSV of image paths.
- `download_weights.py` — Fetches SAM and ForensicsSAM weights.
- `forensics_sam/` — Forgery detection model (SAM backbone + experts).
- `adversary_detector/` — Adversary detection module.
- `segment_anything/` — SAM backbone.
- `image_classifier/` — SigLIP-based AI vs human classifier (run as subprocess from API).

## Config

- **Inference resolution**: Set env `FORENSICSAM_IMAGE_SIZE` to `512`, `768`, or `1024` (default 1024). Lower = faster, less detail.
- **Port**: Use `--port 5000` (or another) if 8080 is in use.

## Requirements

- Python 3.8–3.11
- See `requirements.txt` for dependencies (PyTorch, FastAPI, OpenCV, etc.). CPU-only install is supported.

## License

See [LICENSE](LICENSE) in this repository.
