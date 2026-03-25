"""
FastAPI app: upload an image, run ForensicsSAM and AI-vs-Human detector in parallel, return both results.

Run server (from project root):
  .\venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8080
  (If 8080 fails, try --port 5000 or 3000)

Then POST an image:
  curl -X POST "http://localhost:8000/analyze" -F "file=@your_image.jpg"

Uploads are normalized to FORENSICSAM_IMAGE_SIZE x FORENSICSAM_IMAGE_SIZE (default 512) before
inference. If the image is already that size, it is left unchanged. Env FORENSICSAM_IMAGE_SIZE
can be 512, 768, or 1024.
"""
import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Project root
ROOT = Path(__file__).resolve().parent
DETECTOR_SCRIPT = ROOT / "image_classifier" / "predict.py"
_DETECTOR_VENV_WIN = ROOT / "image_classifier" / "venv" / "Scripts" / "python.exe"
_DETECTOR_VENV_UNIX = ROOT / "image_classifier" / "venv" / "bin" / "python"
if _DETECTOR_VENV_WIN.exists():
    DETECTOR_PYTHON = _DETECTOR_VENV_WIN
elif _DETECTOR_VENV_UNIX.exists():
    DETECTOR_PYTHON = _DETECTOR_VENV_UNIX
else:
    import sys
    DETECTOR_PYTHON = Path(sys.executable)  # use current Python (same venv)

app = FastAPI(title="Image Forensics API", description="ForensicsSAM + AI vs Human detector")

# ForensicsSAM inference size; must match normalized upload dimensions.
_IMAGE_SIZE = int(os.environ.get("FORENSICSAM_IMAGE_SIZE", "512"))
if _IMAGE_SIZE not in (512, 768, 1024):
    _IMAGE_SIZE = 512

# Load ForensicsSAM once at startup
_forensics_sam = None
_adv_detector = None
_device = None
_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
_NORMALIZE_STD = (0.229, 0.224, 0.225)
_checkpoint = {
    "vit_b": str(ROOT / "weight" / "sam_vit_b_01ec64.pth"),
    "vit_l": str(ROOT / "weight" / "sam_vit_l_0b3195.pth"),
    "vit_h": str(ROOT / "weight" / "sam_vit_h_4b8939.pth"),
}


def _load_forensics_models():
    global _forensics_sam, _adv_detector, _device
    if _forensics_sam is not None:
        return
    import torch
    from torchvision import transforms
    from segment_anything import sam_model_registry
    from forensics_sam import ForensicsSAM
    from adversary_detector import AdversaryDetector

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_type = "vit_h"
    sam, _ = sam_model_registry[sam_type](image_size=_IMAGE_SIZE, checkpoint=_checkpoint.get(sam_type))
    _forensics_sam = ForensicsSAM(sam, 8).to(_device).eval()
    _adv_detector = AdversaryDetector().to(_device).eval()
    _adv_detector.load_detector(str(ROOT / "weight" / "adversary_detector.pth"))


def _normalize_upload_to_model_size(path: str) -> dict:
    """Resize image on disk to _IMAGE_SIZE x _IMAGE_SIZE if needed; overwrites path. Returns metadata."""
    import cv2

    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError("Could not read or decode image file")
    h, w = bgr.shape[:2]
    info = {
        "original_width": w,
        "original_height": h,
        "input_resized": False,
    }
    if w != _IMAGE_SIZE or h != _IMAGE_SIZE:
        interp = cv2.INTER_AREA if w >= _IMAGE_SIZE and h >= _IMAGE_SIZE else cv2.INTER_LINEAR
        bgr = cv2.resize(bgr, (_IMAGE_SIZE, _IMAGE_SIZE), interp)
        info["input_resized"] = True
    cv2.imwrite(path, bgr)
    info["processing_width"] = _IMAGE_SIZE
    info["processing_height"] = _IMAGE_SIZE
    return info


def _run_forensics_sam_sync(image_path: str) -> dict:
    """Blocking ForensicsSAM inference + visualization (OpenCV only, no matplotlib)."""
    import base64
    import numpy as np
    import cv2
    import torch
    from torchvision import transforms

    _load_forensics_models()
    orig_bgr = cv2.imread(image_path)
    if orig_bgr is None:
        return {"error": f"Could not read image: {image_path}", "image_base64": None, "image_data_url": None}
    h_orig, w_orig = orig_bgr.shape[:2]
    img = cv2.resize(orig_bgr, (_IMAGE_SIZE, _IMAGE_SIZE), cv2.INTER_AREA)
    img = img[:, :, ::-1].astype("float32") / 255.0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_NORMALIZE_MEAN, _NORMALIZE_STD),
    ])
    x = transform(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits, _ = _adv_detector(x)
        preds_adv = torch.argmax(logits, dim=1)
        mask_prediction, cls_prediction = _forensics_sam(x, preds_adv)
    prob = torch.sigmoid(cls_prediction).squeeze().item()
    label = "forged" if prob > 0.5 else "real"

    # 3-panel image using OpenCV only (no matplotlib)
    mask = torch.sigmoid(mask_prediction).squeeze().cpu().numpy()
    mask_resized = cv2.resize(mask, (w_orig, h_orig), cv2.INTER_LINEAR)
    mask_uint8 = (np.clip(mask_resized, 0, 1) * 255).astype(np.uint8)
    mask_heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_HOT)
    mask_heatmap = cv2.cvtColor(mask_heatmap, cv2.COLOR_BGR2RGB)

    overlay = orig_bgr.copy()
    mask_binary = (mask_resized > 0.5).astype(np.uint8)
    overlay[mask_binary == 1] = [0, 0, 255]  # BGR red
    blended = cv2.addWeighted(orig_bgr, 0.7, overlay, 0.3, 0)

    # mask_heatmap from applyColorMap is BGR; orig_bgr, blended are BGR
    # Scale panels to max 400px height for smaller response
    max_h = 400
    if h_orig > max_h:
        scale = max_h / h_orig
        w_s, h_s = int(w_orig * scale), int(h_orig * scale)
        orig_panel = cv2.resize(orig_bgr, (w_s, h_s))
        heat_panel = cv2.resize(mask_heatmap, (w_s, h_s))
        blend_panel = cv2.resize(blended, (w_s, h_s))
    else:
        orig_panel = orig_bgr
        heat_panel = mask_heatmap
        blend_panel = blended
        w_s, h_s = w_orig, h_orig

    three_panel = np.hstack([orig_panel, heat_panel, blend_panel])

    # Add title bar
    title_h = 30
    title_bar = np.ones((title_h, three_panel.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(title_bar, f"ForensicsSAM: {label.upper()} ({prob:.4f})", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    out_bgr = np.vstack([title_bar, three_panel])
    _, png_bytes = cv2.imencode(".png", out_bgr)
    b64 = base64.b64encode(png_bytes.tobytes()).decode("utf-8")

    return {
        "predicted_label": label,
        "confidence": round(prob, 4),
        "probability_forged": prob,
        "image_base64": b64,
        "image_data_url": f"data:image/png;base64,{b64}",
    }


def _run_detector_sync(image_path: str) -> dict:
    """Run AI-vs-Human detector subprocess (sync, for use in executor on Windows)."""
    result = subprocess.run(
        [str(DETECTOR_PYTHON), str(DETECTOR_SCRIPT), image_path, "--json"],
        capture_output=True,
        cwd=str(ROOT),
        timeout=120,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout).decode().strip() or "Detector failed"
        return {"error": err}
    try:
        return json.loads(result.stdout.decode().strip())
    except Exception as e:
        return {"error": f"Parse error: {e}", "raw": result.stdout.decode()[:200]}


@app.on_event("startup")
async def startup():
    _load_forensics_models()


@app.get("/")
async def root():
    return {
        "message": "POST an image (form field file) to /analyze or / to run ForensicsSAM + AI-vs-Human detector.",
        "forensicsam_image_size": _IMAGE_SIZE,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (e.g. image/jpeg, image/png)")
    suffix = Path(file.filename or "upload").suffix or ".jpg"
    if suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        suffix = ".jpg"
    try:
        body = await file.read()
    except Exception as e:
        raise HTTPException(400, f"Failed to read file: {e}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=ROOT) as tmp:
        tmp.write(body)
        tmp_path = tmp.name

    try:
        normalize_info = _normalize_upload_to_model_size(tmp_path)
    except ValueError as e:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise HTTPException(400, str(e)) from e

    metadata = {
        "filename": file.filename,
        "content_type": file.content_type,
        "file_size_bytes": len(body),
        "format": suffix.lstrip(".").lower() or "jpg",
        "forensicsam_inference_size": _IMAGE_SIZE,
        **normalize_info,
    }

    try:
        loop = asyncio.get_event_loop()
        forensics_task = loop.run_in_executor(None, _run_forensics_sam_sync, tmp_path)
        detector_task = loop.run_in_executor(None, _run_detector_sync, tmp_path)
        forensics_result, detector_result = await asyncio.gather(forensics_task, detector_task)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return JSONResponse(content={
        "forensicsam": forensics_result,
        "ai_vs_human": detector_result,
        "metadata": metadata,
    })


@app.post("/")
async def analyze_at_root(file: UploadFile = File(...)):
    """Same as POST /analyze (avoids 405 when clients POST to /)."""
    return await analyze(file)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
