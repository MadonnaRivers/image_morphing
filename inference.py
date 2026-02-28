import torch
import random
from segment_anything import sam_model_registry
from mini_dataloader import BasicDataloader
from torch.utils.data import DataLoader
from utils import AverageMeter, cal_f1, cal_auc, cal_permute_f1, cal_image_level_acc
import numpy as np
from forensics_sam import ForensicsSAM
from adversary_detector import AdversaryDetector


seed = 42
# Python 内置随机模块
random.seed(seed)
# NumPy 随机数
np.random.seed(seed)
# PyTorch 随机数（CPU）
torch.manual_seed(seed)
# PyTorch 随机数（GPU，仅当可用时）
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

model_type = ['vit_b', 'vit_l', 'vit_h']
checkpoint = {
    'vit_b': './weight/sam_vit_b_01ec64.pth',
    'vit_l': './weight/sam_vit_l_0b3195.pth',
    'vit_h': './weight/sam_vit_h_4b8939.pth'
}

def _ensure_weights():
    """Download weights if SAM checkpoint is missing."""
    import os
    ckpt = checkpoint.get(model_type[2])
    if not ckpt or os.path.isfile(ckpt):
        return
    os.makedirs("./weight", exist_ok=True)
    try:
        from download_weights import main as download_main
        print("SAM checkpoint missing. Running download_weights.py ...")
        download_main()
    except ImportError:
        pass
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Run: python download_weights.py\n"
            "Or download from: https://drive.google.com/file/d/1stLg8bJ1W2E7dVAHC8TYj917REO4sttt/view"
        )


if __name__ == '__main__':
    from tqdm import tqdm

    _ensure_weights()

    # CPU-only: use 'cpu'; set to 'cuda' if GPU available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sam_type = model_type[2]
    r, image_size = 8, 1024
    normalize_type = 2

    sam, image_embedding_size = sam_model_registry[sam_type](image_size=image_size, checkpoint=checkpoint.get(sam_type))
    forensics_sam = ForensicsSAM(sam, r).to(device).eval()

    adv_detector = AdversaryDetector().to(device).eval()
    save_path = "./weight/adversary_detector.pth"
    adv_detector.load_detector(save_path)

    val_dataset = BasicDataloader(
        dataset_list=[
            # ''' au '''
            # ["./data", "casia1_au.txt", 1, 1, 0, 0],
            # ["./data", "columbia_au.txt", 1, 1, 0, 0],
            # ["./data", "misd_au.txt", 1, 1, 0, 0],
            # ["./data", "dso_au.txt", 1, 1, 0, 0],
            # ["./data", "nist16_au.txt", 1, 1, 0, 0],
            # ["./data", "coverage_au.txt", 1, 1, 0, 0],
            # ["./data", "cocoglide_au.txt", 1, 1, 0, 0],
            # ["./data", "acdsee_au.txt", 1, 1, 0, 0],

            # ''' forged '''
            # ["./data", "casia.txt", 1, 1, 1, 0],
            # ["./data", "columbia.txt", 1, 1, 1, 0],
            # ["./data", "misd.txt", 1, 1, 1, 0],
            # ["./data", "dso.txt", 1, 1, 1, 0],
            # ["./data", "nist16.txt", 1, 1, 1, 0],
            # ["./data", "coverage.txt", 1, 1, 1, 0],
            # ["./data", "CocoGlide.txt", 1, 1, 1, 0],
            # ["./data", "acdsee.txt", 1, 1, 1, 0],
            # ["./data", "wild.txt", 1, 1, 1, 0],
            # ["./data", "ipm15k.txt", 1, 1, 1, 0],
        ],
        input_size=image_size,
        normalize_type=normalize_type,
        augment_prob=0.0,
        enable_aug_types=[6],
        intensity={"rates": [0.8], "qfs": [75], "sds": [9], "ksizes": [9]},
        mode="val"
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=1)

    t = tqdm(val_loader)
    auc, f1, p_f1 = AverageMeter(), AverageMeter(), AverageMeter()
    F_ACC, A_ACC = AverageMeter(), AverageMeter()

    with torch.no_grad():
        for idx, (images, gt_masks, forged_label, adv_label) in enumerate(t):
            images, gt_masks = images.to(device), gt_masks.to(device)
            forged_label, adv_label = forged_label.to(device), adv_label.to(device)

            logits, feats = adv_detector(images)
            preds = torch.argmax(logits, dim=1)

            mask_prediction, cls_prediction = forensics_sam(images, preds)

            mask_prediction_logits = torch.sigmoid(mask_prediction)
            mask_prediction = torch.where(mask_prediction_logits > 0.5, 1, 0)
            cls_prediction_logits = torch.sigmoid(cls_prediction)
            cls_prediction = torch.where(cls_prediction_logits > 0.5, 1, 0)

            ''' F1, P-F1, ACC '''
            if forged_label.squeeze() == 1:
                auc.update(cal_auc(mask_prediction_logits.cpu().numpy(), gt_masks.cpu().numpy()))
                f1.update(cal_f1(mask_prediction.cpu().numpy(), gt_masks.cpu().numpy()))
                p_f1.update(cal_permute_f1(mask_prediction.cpu().numpy(), gt_masks.cpu().numpy()))
            F_ACC.update(cal_image_level_acc(cls_prediction_logits.cpu().numpy(), forged_label.cpu().numpy(), 0.5))
            A_ACC.update(cal_image_level_acc(preds.cpu().numpy(), adv_label.cpu().numpy(), 0.5))

            t.set_description(f'AUC:{auc.avg:.3f}, F1:{f1.avg:.3f}, P-F1:{p_f1.avg:.3f}, F_ACC:{F_ACC.avg:.3f}, A_Acc:{A_ACC.avg:.3f}')