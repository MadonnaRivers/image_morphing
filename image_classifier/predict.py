"""
AI vs Human image classifier - runs on CPU using local weights.
Usage: python predict.py <image_path> [--json]
       python predict.py  (uses default sample or pass path via env)
       --json  print single-line JSON to stdout (for API)
"""
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ.setdefault("TRANSFORMERS_USE_FAST_IMAGE_PROCESSOR", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import sys
import json
import torch
from pathlib import Path

# Project root = folder containing config.json and model.safetensors
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR

def main():
    argv = [a for a in sys.argv[1:] if a != "--json"]
    json_output = "--json" in sys.argv[1:]
    if json_output:
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    # Force CPU
    device = torch.device("cpu")
    if not json_output:
        print(f"Using device: {device}")

    # Resolve image path
    if len(argv) >= 1:
        image_path = Path(argv[0])
    else:
        image_path = os.environ.get("IMAGE_PATH")
        if image_path:
            image_path = Path(image_path)
        else:
            # Default: look for a test image in project or current dir
            for name in ("test_image.jpg", "sample.jpg", "image.jpg", "results.jpg"):
                p = SCRIPT_DIR / name
                if p.exists():
                    image_path = p
                    break
            else:
                print("Usage: python predict.py <image_path>")
                print("  Or set IMAGE_PATH and run: python predict.py")
                sys.exit(1)

    image_path = image_path.resolve()
    if not image_path.exists():
        if json_output:
            print(json.dumps({"error": f"Image not found: {image_path}"}), file=sys.stderr)
        else:
            print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Load model and processor from local folder (no HuggingFace download)
    from transformers import AutoImageProcessor, SiglipForImageClassification
    from PIL import Image

    if not json_output:
        print(f"Loading model from: {MODEL_PATH}")
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model = SiglipForImageClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    if not json_output:
        print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    _label_map = {"artificial": "ai", "human": "real"}

    def _norm(lbl):
        return _label_map.get(lbl.lower(), lbl.lower())

    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = _norm(model.config.id2label[predicted_class_idx])
    probabilities = torch.softmax(logits, dim=-1)
    predicted_prob = probabilities[0, predicted_class_idx].item()

    if json_output:
        scores = {_norm(model.config.id2label[i]): float(probabilities[0, i].item()) for i in model.config.id2label}
        out = {"predicted_label": predicted_label, "confidence": predicted_prob, "scores": scores}
        print(json.dumps(out))
        return

    print("-" * 40)
    print(f"Image: {image_path}")
    print(f"Predicted: {predicted_label.upper()} (confidence: {predicted_prob:.4f})")
    print("-" * 40)
    print("Scores:")
    for i, label in model.config.id2label.items():
        print(f"  {_norm(label)}: {probabilities[0, i].item():.4f}")

if __name__ == "__main__":
    main()
