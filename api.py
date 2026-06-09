"""
FastAPI app: upload an image, run ForensicsSAM and AI-vs-Human detector in parallel, return both results.

Run server (from project root):
  .\venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8080
  (If 8080 fails, try --port 5000 or 3000)

Then POST an image:
  curl -X POST "http://localhost:8000/analyze" -F "file=@your_image.jpg"

Env FORENSICSAM_IMAGE_SIZE controls model inference size (default 1024).
Accepted values: 512, 768, 1024.
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Project root
ROOT = Path(__file__).resolve().parent
DETECTOR_MODEL_DIR = ROOT / "image_classifier"

app = FastAPI(title="Image Forensics API", description="ForensicsSAM + AI vs Human detector")

# ForensicsSAM inference size; must match normalized upload dimensions.
_IMAGE_SIZE = int(os.environ.get("FORENSICSAM_IMAGE_SIZE", "1024"))
if _IMAGE_SIZE not in (512, 768, 1024):
    _IMAGE_SIZE = 1024

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


_detector_model = None
_detector_processor = None


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


def _load_detector_model():
    """Load the AI-vs-Human classifier (Organika/sdxl-detector, Swin-T) once at startup."""
    global _detector_model, _detector_processor
    if _detector_model is not None:
        return
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    _detector_processor = AutoImageProcessor.from_pretrained(str(DETECTOR_MODEL_DIR))
    _detector_model = AutoModelForImageClassification.from_pretrained(str(DETECTOR_MODEL_DIR)).eval()


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

    # Per-panel caption strip (formal labels under each panel)
    cap_h = 26
    cap_bar = np.ones((cap_h, three_panel.shape[1], 3), dtype=np.uint8) * 240
    panel_w = three_panel.shape[1] // 3
    for i, name in enumerate(("Original", "Forgery Heatmap", "Mask Overlay")):
        (tw, _th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x = i * panel_w + (panel_w - tw) // 2
        cv2.putText(cap_bar, name, (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)

    panel_with_caption = np.vstack([three_panel, cap_bar])

    return {
        "predicted_label": label,
        "confidence": round(prob, 4),
        "probability_forged": prob,
        "_panel_bgr": panel_with_caption,  # consumed by handler to compose final visualization
    }


def _compose_visualization(forensics: dict, detector: dict) -> tuple:
    """Add a formal two-line header (ForensicsSAM + AI-vs-Human) above the panel strip."""
    import base64
    import numpy as np
    import cv2

    panel = forensics.pop("_panel_bgr", None)
    if panel is None:
        return None, None

    header_h = 60
    width = panel.shape[1]
    header = np.ones((header_h, width, 3), dtype=np.uint8) * 255

    f_label = str(forensics.get("predicted_label", "?")).upper()
    f_conf = forensics.get("confidence", 0.0)
    d_label = str(detector.get("predicted_label", "?")).upper() if isinstance(detector, dict) else "?"
    d_conf = float(detector.get("confidence", 0.0)) if isinstance(detector, dict) else 0.0

    line1 = f"ForensicsSAM     : {f_label}  ({f_conf:.4f})"
    line2 = f"AI-vs-Human      : {d_label}  ({d_conf:.4f})"
    cv2.putText(header, line1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(header, line2, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(header, (0, header_h - 1), (width, header_h - 1), (180, 180, 180), 1)

    out_bgr = np.vstack([header, panel])
    _, png_bytes = cv2.imencode(".png", out_bgr)
    b64 = base64.b64encode(png_bytes.tobytes()).decode("utf-8")
    return b64, f"data:image/png;base64,{b64}"


def _run_detector_sync(image_path: str) -> dict:
    """Run AI-vs-Human classifier inline (model loaded once at startup, no subprocess)."""
    import torch
    from PIL import Image

    _load_detector_model()
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": f"Could not open image: {e}"}
    inputs = _detector_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = _detector_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    idx2label = _detector_model.config.id2label
    scores = {idx2label[i]: float(probs[i].item()) for i in idx2label}
    top_idx = int(torch.argmax(logits, dim=-1).item())
    return {
        "predicted_label": idx2label[top_idx],
        "confidence": float(probs[top_idx].item()),
        "scores": scores,
    }


@app.on_event("startup")
async def startup():
    _load_forensics_models()
    _load_detector_model()


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
    loop = asyncio.get_running_loop()
    import cv2
    import numpy as np

    def _write_and_normalize():
        """Decode upload, capture original dims, resize to model size, save normalized file."""
        arr = np.frombuffer(body, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None, None, None
        orig_h, orig_w = img.shape[:2]
        # Single upfront resize to model inference size so both pipelines see identical input.
        resized = cv2.resize(img, (_IMAGE_SIZE, _IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=ROOT) as tmp:
            cv2.imwrite(tmp.name, resized)
            return tmp.name, orig_w, orig_h

    tmp_path, image_width, image_height = await loop.run_in_executor(None, _write_and_normalize)
    if tmp_path is None:
        raise HTTPException(400, "Could not decode image")

    forensics_task = loop.run_in_executor(None, _run_forensics_sam_sync, tmp_path)
    detector_task = loop.run_in_executor(None, _run_detector_sync, tmp_path)

    try:
        forensics_result, detector_result = await asyncio.gather(forensics_task, detector_task)
    finally:
        await loop.run_in_executor(None, lambda: os.unlink(tmp_path) if os.path.exists(tmp_path) else None)

    b64, data_url = _compose_visualization(forensics_result, detector_result)
    if b64 is not None:
        forensics_result["image_base64"] = b64
        forensics_result["image_data_url"] = data_url

    metadata = {
        "filename": file.filename,
        "content_type": file.content_type,
        "file_size_bytes": len(body),
        "width": image_width,
        "height": image_height,
        "format": suffix.lstrip(".").lower() or "jpg",
        "forensicsam_inference_size": _IMAGE_SIZE,
    }
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
