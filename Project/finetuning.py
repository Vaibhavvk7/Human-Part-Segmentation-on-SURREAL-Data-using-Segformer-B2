import os, math, time, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import SegformerForSemanticSegmentation
from collections import deque
from train_segformer_v3 import (  # <-- import your existing helper functions
    SurrealSegmentationDataset,
    create_dataloaders,
    mean_iou,
    save_checkpoint,
    load_checkpoint,
    evaluate,
    train_one_epoch,
    cfg,
    get_device,
)

# ============================================================
# Fine-tune Config
# ============================================================
CKPT_PATH = "/home/channagiri.b/SmallData_Project/Output_FineTune/segformer_b2_weighted_epoch_8.pth"
DATA_ROOT = "/home/channagiri.b/SmallData_Project/Dataset/SURREAL/data"
OUTPUT_DIR = "/home/channagiri.b/SmallData_Project/Output_FineTune"
IMG_SIZE = (160, 160)
EPOCHS = 1
LR = 1e-4
WD = 1e-4
BATCH_SIZE = 8
NUM_WORKERS = 8
NUM_CLASSES = 25

# ============================================================
# Main fine-tuning procedure
# ============================================================
def main():
    device = get_device()
    print(f"[Device] Using {device}")

    # 1. Load class frequencies and compute weights
    freqs = np.load("class_freqs.npy")
    w = 1.0 / np.sqrt(freqs + 1e-8)
    w = w / w.mean()
    weight_t = torch.tensor(w, dtype=torch.float32, device=device)
    print(f"[Class Weights] Loaded from class_freqs.npy")

    # 2. Load dataloaders
    train_loader, val_loader = create_dataloaders(
        DATA_ROOT, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, max_train_samples=700000, max_val_samples=14000
    )

    # 3. Create model + optimizer
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    # 4. Resume from best checkpoint
    start_epoch = load_checkpoint(model, optimizer, CKPT_PATH, device) + 1

    # 5. Define weighted + smoothed loss (after loading weights)
    criterion = nn.CrossEntropyLoss(
        weight=weight_t,         # <--- your computed tensor
        ignore_index=255,
        label_smoothing=0.05
    )

    # 6. TensorBoard writer for logging
    writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, "tb"))

    print(f"[Fine-tune] Resuming from epoch {start_epoch}")

    # 5. Fine-tune loop
    for epoch in range(start_epoch, start_epoch + EPOCHS):
        t0 = time.time()
        train_one_epoch(model, train_loader, optimizer, device, epoch, writer)
        val_loss, val_pix, val_miou = evaluate(model, val_loader, device, writer, epoch)
        dt = (time.time() - t0) / 60
        print(f"[Fine-tune] Ep{epoch}: ValLoss={val_loss:.4f} | Acc={val_pix:.3f} | mIoU={val_miou:.3f} | {dt:.1f} min")

        save_checkpoint(model, optimizer, epoch, os.path.join(OUTPUT_DIR, f"segformer_b2_weighted_epoch_{epoch}.pth"))

    writer.close()
    print("[Fine-tune Complete]")

if __name__ == "__main__":
    main()
