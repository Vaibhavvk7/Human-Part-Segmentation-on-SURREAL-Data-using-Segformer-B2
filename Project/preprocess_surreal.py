#!/usr/bin/env python3
import os
import cv2
import sys
import math
import glob
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.io import loadmat

def iter_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

def load_mask_for_frame(mat, frame_idx):
    key = f"segm_{frame_idx + 1}"  # 1-indexed in SURREAL .mat
    if key not in mat:
        return None
    return mat[key]

def resize_image_and_mask(img_rgb, mask, out_h, out_w):
    img_resized = cv2.resize(img_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return img_resized, mask_resized

def find_pairs(root):
    # SURREAL under: <root>/cmu/{train,val,test}/**/*.mp4
    mp4s = glob.glob(os.path.join(root, "cmu", "**", "*.mp4"), recursive=True)
    pairs = []
    for v in mp4s:
        seg = v.replace(".mp4", "_segm.mat")
        if os.path.exists(seg):
            pairs.append((v, seg))
    return pairs

def save_shard(out_dir, shard_id, imgs, masks):
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"shard_{shard_id:06d}.npz"
    # Store as uint8 (images) + int16 (masks) to keep shards compact
    np.savez_compressed(
        shard_path,
        images=np.asarray(imgs, dtype=np.uint8),     # [N, H, W, 3]
        masks=np.asarray(masks, dtype=np.int16),     # [N, H, W]
    )
    return shard_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="SURREAL root: .../SURREAL/data")
    ap.add_argument("--out", required=True, help="Output directory for preprocessed shards")
    ap.add_argument("--img_size", type=int, nargs=2, default=[160, 160], help="H W")
    ap.add_argument("--split", type=str, default="train", choices=["train","val","test"])
    ap.add_argument("--shard_size", type=int, default=1000)
    ap.add_argument("--max_samples", type=int, default=None, help="Cap total samples per split")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out) / args.split
    H, W = args.img_size

    pairs = find_pairs(root)
    # Filter by split
    pairs = [p for p in pairs if f"/{args.split}/" in p[0] or f"\\{args.split}\\" in p[0]]
    if not pairs:
        print(f"No pairs found for split={args.split} under {root}", file=sys.stderr)
        sys.exit(1)

    shard_id = 0
    imgs, masks = [], []
    total_written = 0
    index_meta = {"split": args.split, "shards": []}

    for vid_path, seg_path in pairs:
        mat = loadmat(seg_path)
        for frame_idx, bgr in enumerate(iter_video_frames(vid_path)):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask = load_mask_for_frame(mat, frame_idx)
            if mask is None:
                continue
            rgb_r, mask_r = resize_image_and_mask(rgb, mask, H, W)
            imgs.append(rgb_r)
            masks.append(mask_r)

            if len(imgs) >= args.shard_size:
                shard_path = save_shard(out_root, shard_id, imgs, masks)
                index_meta["shards"].append({"path": str(shard_path), "count": len(imgs)})
                total_written += len(imgs)
                imgs, masks = [], []
                shard_id += 1
                if args.max_samples and total_written >= args.max_samples:
                    break

        if args.max_samples and total_written >= args.max_samples:
            break

    if imgs:
        shard_path = save_shard(out_root, shard_id, imgs, masks)
        index_meta["shards"].append({"path": str(shard_path), "count": len(imgs)})
        total_written += len(imgs)

    # Write an index.json for the split
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "index.json", "w") as f:
        json.dump(index_meta, f, indent=2)

    print(f"[Preprocess] Done. Wrote {total_written} samples in {len(index_meta['shards'])} shards to {out_root}")

if __name__ == "__main__":
    main()
