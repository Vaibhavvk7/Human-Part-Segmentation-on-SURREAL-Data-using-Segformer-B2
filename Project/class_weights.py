import numpy as np
import torch

# ====== Config ======
FREQS_PATH = "class_freqs.npy"   # path to your saved frequencies
NUM_CLASSES = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====== Load & compute weights ======
freqs = np.load(FREQS_PATH)
if len(freqs) != NUM_CLASSES:
    print(f"[Warning] Expected {NUM_CLASSES} classes but found {len(freqs)} frequencies")

weights = 1.0 / np.sqrt(freqs + 1e-8)
weights = weights / weights.mean()
weight_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

# ====== Summary statistics ======
sorted_idx = np.argsort(weights)[::-1]  # descending
top_classes = sorted_idx[:3]
low_classes = sorted_idx[-3:]

print("\n=== Class Frequency & Weight Summary ===")
print(f"Device: {DEVICE}")
print(f"Total classes: {NUM_CLASSES}")
print(f"Mean weight:  {weights.mean():.3f}")
print(f"Min weight:   {weights.min():.3f}")
print(f"Max weight:   {weights.max():.3f}\n")

print("Top-3 highest weighted (rarest) classes:")
for i in top_classes:
    print(f"  Class {i:02d} | freq={freqs[i]:.6f} | weight={weights[i]:.3f}")

print("\nBottom-3 lowest weighted (most frequent) classes:")
for i in low_classes:
    print(f"  Class {i:02d} | freq={freqs[i]:.6f} | weight={weights[i]:.3f}")

print("\nFull weight tensor shape:", weight_t.shape)
print("Preview:", weight_t[:10].cpu().numpy())
print("\n[Info] Use these weights directly in your fine-tuning loss:")
print("criterion = nn.CrossEntropyLoss(weight=weight_t, ignore_index=255, label_smoothing=0.05)")
