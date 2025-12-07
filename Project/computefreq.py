import numpy as np, torch
from torch.utils.data import DataLoader
from train_segformer_v3 import SurrealSegmentationDataset  # adjust import
from tqdm import tqdm

NUM_CLASSES = 25
DATA_ROOT = "/home/channagiri.b/SmallData_Project/Dataset/SURREAL/data"
BATCH = 8
IMG_SIZE = (160,160)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ds = SurrealSegmentationDataset(DATA_ROOT, "train", img_size=IMG_SIZE, max_samples=100000)
dl = DataLoader(ds, batch_size=BATCH, num_workers=8)
counts = np.zeros(NUM_CLASSES, dtype=np.int64)

for _, masks in tqdm(dl):
    for c in range(NUM_CLASSES):
        counts[c] += (masks == c).sum().item()

freqs = counts / counts.sum()
np.save("class_freqs.npy", freqs)
print("Saved class_freqs.npy", freqs)
