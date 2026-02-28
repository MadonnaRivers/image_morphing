"""
Run ForensicsSAM on a single image: print result (real/forged + probability)
and save visualization with forgery mask overlay.
"""
import os
import sys
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

from segment_anything import sam_model_registry
from forensics_sam import ForensicsSAM
from adversary_detector import AdversaryDetector

IMAGE_SIZE = 1024  # original full resolution; use --size 512 or 768 for speed
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

model_type = ["vit_b", "vit_l", "vit_h"]
checkpoint = {
    "vit_b": "./weight/sam_vit_b_01ec64.pth",
    "vit_l": "./weight/sam_vit_l_0b3195.pth",
    "vit_h": "./weight/sam_vit_h_4b8939.pth",
}


def load_and_preprocess(image_path, image_size=IMAGE_SIZE):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    orig = img.copy()
    img = cv2.resize(img, (image_size, image_size), cv2.INTER_AREA)
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    return transform(img).unsqueeze(0), orig


def main():
    parser = argparse.ArgumentParser(description="Run ForensicsSAM on one image and visualize")
    parser.add_argument("--image", default="example.jpg", help="Path to input image")
    parser.add_argument("--output", default=None, help="Path for visualization (default: <image>_forensicsam_result.png)")
    parser.add_argument("--size", type=int, default=IMAGE_SIZE, help=f"Inference size (default: {IMAGE_SIZE}; use 1024 for max quality)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    args = parser.parse_args()

    image_path = args.image
    if not os.path.isfile(image_path):
        print(f"Error: image not found: {os.path.abspath(image_path)}")
        sys.exit(1)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_path = args.output or (os.path.splitext(image_path)[0] + "_forensicsam_result.png")
    image_size = args.size if args.size in (512, 768, 1024) else IMAGE_SIZE

    print("Loading image and preprocessing...")
    x, orig_bgr = load_and_preprocess(image_path, image_size)
    x = x.to(device)
    h_orig, w_orig = orig_bgr.shape[:2]

    print("Loading models...")
    sam_type = model_type[2]  # vit_h
    sam, _ = sam_model_registry[sam_type](image_size=image_size, checkpoint=checkpoint.get(sam_type))
    forensics_sam = ForensicsSAM(sam, 8).to(device).eval()
    adv_detector = AdversaryDetector().to(device).eval()
    adv_detector.load_detector("./weight/adversary_detector.pth")

    print("Running ForensicsSAM...")
    with torch.no_grad():
        logits, _ = adv_detector(x)
        preds_adv = torch.argmax(logits, dim=1)
        mask_prediction, cls_prediction = forensics_sam(x, preds_adv)

    prob = torch.sigmoid(cls_prediction).squeeze().item()
    label = "forged" if prob > 0.5 else "real"

    print("\n" + "=" * 50)
    print("ForensicsSAM result")
    print("=" * 50)
    print(f"  Image-level: {label.upper()} (probability: {prob:.4f})")
    print("=" * 50 + "\n")

    # Mask: (1, 1, H/4, W/4) -> resize to original size for overlay
    mask = torch.sigmoid(mask_prediction).squeeze().cpu().numpy()
    mask = cv2.resize(mask, (w_orig, h_orig), cv2.INTER_LINEAR)
    mask_binary = (mask > 0.5).astype(np.uint8)

    # Overlay: red where forged (mask=1)
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    overlay = orig_rgb.copy()
    overlay[mask_binary == 1] = [255, 0, 0]  # red for forged
    blended = cv2.addWeighted(orig_rgb, 0.7, overlay, 0.3, 0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(mask, cmap="hot")
    axes[1].set_title("Forgery mask (heatmap)")
    axes[1].axis("off")
    axes[2].imshow(blended)
    axes[2].set_title("Overlay (red = predicted forged)")
    axes[2].axis("off")
    plt.suptitle(f"ForensicsSAM: {label.upper()} ({prob:.4f})", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
