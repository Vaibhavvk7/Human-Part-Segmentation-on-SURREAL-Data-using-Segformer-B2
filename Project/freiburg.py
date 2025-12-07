# import os
# import cv2
# import torch
# import numpy as np
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from scipy.io import loadmat
# from transformers import SegformerForSemanticSegmentation
# from tqdm import tqdm

# # --------------------------
# # Dataset Loader
# # --------------------------
# class FreiburgMatDataset(Dataset):
#     def __init__(self, img_dir, mask_dir, img_size=(160, 160)):
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.img_size = img_size
#         self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img_name = self.images[idx]
#         base = os.path.splitext(img_name)[0]

#         # --- RGB image ---
#         img_path = os.path.join(self.img_dir, img_name)
#         img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, self.img_size)
#         img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

#         # --- Segmentation mask (.mat) ---
#         mat_path = os.path.join(self.mask_dir, base + ".mat")
#         mat_data = loadmat(mat_path)
#         # Try common key names
#         possible_keys = [k for k in mat_data.keys() if not k.startswith("__")]
#         seg = mat_data[possible_keys[0]]
#         seg = cv2.resize(seg.astype(np.uint8), self.img_size, interpolation=cv2.INTER_NEAREST)
#         mask_t = torch.from_numpy(seg).long()

#         return img_t, mask_t


# # --------------------------
# # Metrics
# # --------------------------
# def mean_iou(preds, masks, num_classes=25):
#     ious = []
#     for c in range(num_classes):
#         p = (preds == c)
#         m = (masks == c)
#         inter = (p & m).sum().item()
#         union = (p | m).sum().item()
#         if union > 0:
#             ious.append(inter / union)
#     return np.mean(ious) if ious else 0.0


# # --------------------------
# # Evaluation Function
# # --------------------------
# @torch.no_grad()
# def evaluate(model_path, freiburg_root, num_classes=25, img_size=(320, 320)):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"[Device] Using {device}")

#     # --- Load pretrained SegFormer backbone ---
#     model = SegformerForSemanticSegmentation.from_pretrained(
#         "nvidia/segformer-b2-finetuned-ade-512-512",
#         num_labels=num_classes,
#         ignore_mismatched_sizes=True,
#     ).to(device)

#     # --- Load trained weights (.pth checkpoint) ---
#     ckpt = torch.load(model_path, map_location=device)
#     model.load_state_dict(ckpt["model_state"], strict=False)
#     model.eval()

#     # --- Freiburg dataset ---
#     dataset = FreiburgMatDataset(
#         img_dir=os.path.join(freiburg_root, "img"),
#         mask_dir=os.path.join(freiburg_root, "masks"),
#         img_size=img_size
#     )
#     loader = DataLoader(dataset, batch_size=2, shuffle=False)

#     total_pix, correct_pix, miou_sum, batches = 0, 0, 0, 0
#     for imgs, masks in tqdm(loader):
#         imgs, masks = imgs.to(device), masks.to(device)
#         out = model(pixel_values=imgs)
#         logits = F.interpolate(out.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
#         preds = logits.argmax(1)
#         correct_pix += (preds == masks).sum().item()
#         total_pix += masks.numel()
#         miou_sum += mean_iou(preds, masks, num_classes)
#         batches += 1

#     print(f"\nPixel Accuracy: {correct_pix / total_pix:.4f}")
#     print(f"Mean IoU: {miou_sum / batches:.4f}")


# # --------------------------
# # Main Entry
# # --------------------------
# if __name__ == "__main__":
#     evaluate(
#         model_path="/home/channagiri.b/SmallData_Project/Output_FineTune/segformer_b2_weighted_epoch_7.pth",
#         freiburg_root="/home/channagiri.b/SmallData_Project/Sitting",
#         num_classes=25,
#         img_size=(160, 160)
#     )


import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm

# ============================================================
# Dataset Loader
# ============================================================
class FreiburgMatDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(160, 160)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base = os.path.splitext(img_name)[0]

        # --- RGB image ---
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # --- Segmentation mask (.mat) ---
        mat_path = os.path.join(self.mask_dir, base + ".mat")
        mat_data = loadmat(mat_path)
        possible_keys = [k for k in mat_data.keys() if not k.startswith("__")]
        seg = mat_data[possible_keys[0]]  # e.g. 'Segmentation' or 'PartMask'
        seg = cv2.resize(seg.astype(np.uint8), self.img_size, interpolation=cv2.INTER_NEAREST)
        mask_t = torch.from_numpy(seg).long()

        return img_t, mask_t


# ============================================================
# Metrics
# ============================================================
def mean_iou(preds, masks, num_classes=25):
    """Compute mean IoU over all classes."""
    ious = []
    for c in range(num_classes):
        p = (preds == c)
        m = (masks == c)
        inter = (p & m).sum().item()
        union = (p | m).sum().item()
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious) if ious else 0.0


def class_iou(preds, masks, class_ids):
    """Compute IoU for specific class IDs."""
    inter, union = 0, 0
    for c in class_ids:
        p = (preds == c)
        m = (masks == c)
        inter += (p & m).sum().item()
        union += (p | m).sum().item()
    if union == 0:
        return 0.0
    return inter / union


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate(model_path, freiburg_root, num_classes=25, img_size=(160, 160)):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using {device}")

    # --- Load SegFormer backbone + fine-tuned weights ---
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    # --- Freiburg dataset ---
    dataset = FreiburgMatDataset(
        img_dir=os.path.join(freiburg_root, "img"),
        mask_dir=os.path.join(freiburg_root, "masks"),
        img_size=img_size
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    total_pix, correct_pix = 0, 0
    mean_iou_sum, head_iou_sum, torso_iou_sum, legs_iou_sum = 0, 0, 0, 0
    batches = 0

    # === Define part groupings (based on SURREAL IDs) ===
    head_ids = [1, 2]                   # example: head, face
    torso_ids = [3, 4, 5]               # chest, back, abdomen
    legs_ids = [6, 7, 8, 9, 10, 11]     # upper/lower left+right legs

    for imgs, masks in tqdm(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        out = model(pixel_values=imgs)
        logits = F.interpolate(out.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        preds = logits.argmax(1)

        correct_pix += (preds == masks).sum().item()
        total_pix += masks.numel()

        mean_iou_sum += mean_iou(preds, masks, num_classes)
        head_iou_sum += class_iou(preds, masks, head_ids)
        torso_iou_sum += class_iou(preds, masks, torso_ids)
        legs_iou_sum += class_iou(preds, masks, legs_ids)
        batches += 1

    # === Final averaged metrics ===
    pixel_acc = correct_pix / total_pix
    mean_iou_val = mean_iou_sum / batches
    head_iou_val = head_iou_sum / batches
    torso_iou_val = torso_iou_sum / batches
    legs_iou_val = legs_iou_sum / batches

    # === Print Summary ===
    print("\n================ Evaluation Summary ================")
    print(f"Pixel Accuracy : {pixel_acc:.4f}")
    print(f"Mean IoU        : {mean_iou_val:.4f}")
    print("---------------------------------------------------")
    print(f"Head IoU        : {head_iou_val:.4f}")
    print(f"Torso IoU       : {torso_iou_val:.4f}")
    print(f"Legs IoU        : {legs_iou_val:.4f}")
    print("===================================================\n")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    evaluate(
        model_path="/home/channagiri.b/SmallData_Project/Output_Sanity/segformer_b2_epoch_6.pth",
        freiburg_root="/home/channagiri.b/SmallData_Project/Sitting",
        num_classes=25,
        img_size=(160, 160)
    )
