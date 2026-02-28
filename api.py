"""
FastAPI app: upload an image, run ForensicsSAM and AI-vs-Human detector in parallel, return both results.

Run server (from project root):
  .\venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8080
  (If 8080 fails, try --port 5000 or 3000)

Then POST an image:
  curl -X POST "http://localhost:8000/analyze" -F "file=@your_image.jpg"

Speed vs quality: set env FORENSICSAM_IMAGE_SIZE to control inference resolution (default 1024).
  - 1024 = best quality (original); 768 or 512 = faster.
  Example: set FORENSICSAM_IMAGE_SIZE=512 before starting the server for faster inference.
"""
import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Project root
ROOT = Path(__file__).resolve().parent
DETECTOR_SCRIPT = ROOT / "image_classifier" / "predict.py"
DETECTOR_PYTHON = ROOT / "image_classifier" / "venv" / "Scripts" / "python.exe"
if not DETECTOR_PYTHON.exists():
    DETECTOR_PYTHON = ROOT / "image_classifier" / "venv" / "bin" / "python"

app = FastAPI(title="Image Forensics API", description="ForensicsSAM + AI vs Human detector")

# ForensicsSAM inference size: 1024 = best quality (original). Env FORENSICSAM_IMAGE_SIZE can set 512 or 768 for speed.
_IMAGE_SIZE = int(os.environ.get("FORENSICSAM_IMAGE_SIZE", "1024"))
if _IMAGE_SIZE not in (512, 768, 1024):
    _IMAGE_SIZE = 1024  # fallback to full resolution

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


async def _run_detector_async(image_path: str) -> dict:
    """Run AI-vs-Human detector subprocess, parse JSON from stdout."""
    proc = await asyncio.create_subprocess_exec(
        str(DETECTOR_PYTHON),
        str(DETECTOR_SCRIPT),
        image_path,
        "--json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(ROOT),
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        return {"error": (stderr or stdout).decode().strip() or "Detector failed"}
    try:
        import json
        return json.loads(stdout.decode().strip())
    except Exception as e:
        return {"error": f"Parse error: {e}", "raw": stdout.decode()[:200]}


@app.on_event("startup")
async def startup():
    _load_forensics_models()


@app.get("/")
async def root():
    return {
        "message": "Upload an image to /analyze to run ForensicsSAM + AI-vs-Human detector in parallel.",
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

    # Image metadata (dimensions from file)
    import cv2
    meta_img = cv2.imread(tmp_path)
    if meta_img is not None:
        h, w = meta_img.shape[:2]
        image_width, image_height = w, h
    else:
        image_width, image_height = None, None
    metadata = {
        "filename": file.filename,
        "content_type": file.content_type,
        "file_size_bytes": len(body),
        "width": image_width,
        "height": image_height,
        "format": suffix.lstrip(".").lower() or "jpg",
        "forensicsam_inference_size": _IMAGE_SIZE,
    }

    try:
        loop = asyncio.get_event_loop()
        forensics_task = loop.run_in_executor(None, _run_forensics_sam_sync, tmp_path)
        detector_task = _run_detector_async(tmp_path)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
