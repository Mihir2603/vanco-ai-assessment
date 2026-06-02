"""
prepare_dataset.py — Split annotated raw images into train / val / test
and copy them into the YOLO dataset directory structure.

Expected input layout (produced by auto_annotate.py):
    data/raw/<LABEL>/<LABEL>_XXXX.jpg   ← image
    data/raw/<LABEL>/<LABEL>_XXXX.txt   ← YOLO annotation sidecar

Output layout (required by dataset.yaml / YOLOv8):
    data/annotated/images/{train,val,test}/
    data/annotated/labels/{train,val,test}/

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --train 0.70 --val 0.20 --test 0.10
    python scripts/prepare_dataset.py --seed 99
"""

import argparse
import os
import random
import shutil
from collections import defaultdict

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "annotated")


def parse_args():
    p = argparse.ArgumentParser(description="Split ASL dataset into train/val/test")
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val",   type=float, default=0.20)
    p.add_argument("--test",  type=float, default=0.10)
    p.add_argument("--seed",  type=int,   default=42)
    p.add_argument("--max_per_class", type=int, default=0,
                   help="Cap images per class (0 = no limit)")
    return p.parse_args()


def copy_pair(img_src, lbl_src, split):
    """Copy image and label to the correct split subfolder."""
    img_dst_dir = os.path.join(OUT_DIR, "images", split)
    lbl_dst_dir = os.path.join(OUT_DIR, "labels", split)
    os.makedirs(img_dst_dir, exist_ok=True)
    os.makedirs(lbl_dst_dir, exist_ok=True)
    shutil.copy2(img_src, os.path.join(img_dst_dir, os.path.basename(img_src)))
    shutil.copy2(lbl_src, os.path.join(lbl_dst_dir, os.path.basename(lbl_src)))


def main():
    args = parse_args()
    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, \
        "Train + val + test must sum to 1.0"

    random.seed(args.seed)
    stats = defaultdict(lambda: defaultdict(int))   # label → split → count

    for label in CLASS_NAMES:
        folder = os.path.join(RAW_DIR, label)
        if not os.path.isdir(folder):
            print(f"[SKIP] No folder: {folder}")
            continue

        pairs = []
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(folder, fname)
            lbl_path = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(lbl_path):
                pairs.append((img_path, lbl_path))
            else:
                print(f"  [WARN] No annotation for {fname}, skipping.")

        if not pairs:
            print(f"[SKIP] No annotated images for '{label}'")
            continue

        random.shuffle(pairs)
        if args.max_per_class and len(pairs) > args.max_per_class:
            pairs = pairs[: args.max_per_class]
        n = len(pairs)
        n_train = max(1, int(n * args.train))
        n_val   = max(1, int(n * args.val))
        # Remainder goes to test
        splits = (["train"] * n_train
                + ["val"]   * n_val
                + ["test"]  * max(0, n - n_train - n_val))

        for (img, lbl), split in zip(pairs, splits):
            copy_pair(img, lbl, split)
            stats[label][split] += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'Label':>6}  {'Train':>6}  {'Val':>5}  {'Test':>5}  {'Total':>6}")
    print("-" * 38)
    grand = defaultdict(int)
    for label in CLASS_NAMES:
        if label in stats:
            tr = stats[label]["train"]
            va = stats[label]["val"]
            te = stats[label]["test"]
            tot = tr + va + te
            print(f"  {label:>4}  {tr:>6}  {va:>5}  {te:>5}  {tot:>6}")
            grand["train"] += tr
            grand["val"]   += va
            grand["test"]  += te

    print("-" * 38)
    tot = grand["train"] + grand["val"] + grand["test"]
    print(f"  {'ALL':>4}  {grand['train']:>6}  {grand['val']:>5}  {grand['test']:>5}  {tot:>6}")
    print(f"\n✓ Dataset written to: {os.path.abspath(OUT_DIR)}")
    print("Next step → python scripts/train.py")


if __name__ == "__main__":
    main()
