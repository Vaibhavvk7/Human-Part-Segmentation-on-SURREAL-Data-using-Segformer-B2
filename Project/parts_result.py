import torch, os, numpy as np
from transformers import SegformerForSemanticSegmentation
from torch.utils.data import DataLoader
from torch.nn import functional as F
from surreal_dataset import SurrealSegmentationDataset  # from your existing code
from train_segformer_v3 import SurrealSegmentationDataset

NUM_CLASSES = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "/home/channagiri.b/SmallData_Project/Output_Sanity/segformer_b2_epoch_6.pth"
DATA_ROOT = "/home/channagiri.b/SmallData_Project/Dataset/SURREAL/data"
IMG_SIZE = (160,160)
BATCH_SIZE = 4

# Define class groupings (example — adjust to your label mapping)
HEAD_CLASSES = [0,1,2,3]
TORSO_CLASSES = [4,5,6,7,8,9]
LEGS_CLASSES = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

def compute_iou(preds, targets, classes):
    ious = []
    for c in classes:
        pred_c = (preds == c)
        mask_c = (targets == c)
        inter = (pred_c & mask_c).sum().item()
        union = (pred_c | mask_c).sum().item()
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious) if ious else 0.0

def main():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=NUM_CLASSES, ignore_mismatched_sizes=True).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = SurrealSegmentationDataset(DATA_ROOT, "val", img_size=IMG_SIZE, max_samples=10000)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    head_iou, torso_iou, legs_iou, count = 0, 0, 0, 0
    with torch.no_grad():
        for imgs, masks in dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = model(pixel_values=imgs).logits
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            preds = logits.argmax(1)
            head_iou += compute_iou(preds, masks, HEAD_CLASSES)
            torso_iou += compute_iou(preds, masks, TORSO_CLASSES)
            legs_iou += compute_iou(preds, masks, LEGS_CLASSES)
            count += 1

    print(f"Head IoU:  {head_iou / count:.3f}")
    print(f"Torso IoU: {torso_iou / count:.3f}")
    print(f"Legs IoU:  {legs_iou / count:.3f}")

if __name__ == "__main__":
    main()
