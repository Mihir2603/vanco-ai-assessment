"""
download_dataset.py — Download and organize the Kaggle ASL Alphabet dataset.

Dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
  87,000 images (200×200), 29 classes:
  A-Z (26 letters) + del + nothing + space
  3,000 images per class (train) + 29 test images

This script:
  1. Downloads via Kaggle API (requires ~/.kaggle/kaggle.json)
  2. Extracts both train and test zips
  3. Copies images into data/raw/<CLASS>/ (our pipeline's expected format)
  4. Optionally subsamples to --max_per_class images per class

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --max_per_class 300
    python scripts/download_dataset.py --skip_download   # if already downloaded

Kaggle API credentials setup:
    1. Go to https://www.kaggle.com/settings → API → "Create New Token"
    2. Place the downloaded kaggle.json at ~/.kaggle/kaggle.json
    3. chmod 600 ~/.kaggle/kaggle.json
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "kaggle_download"

DATASET_SLUG = "grassknoted/asl-alphabet"

# 29 Kaggle classes — folder names inside the zip
KAGGLE_CLASSES = (
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
)


def parse_args():
    p = argparse.ArgumentParser(description="Download Kaggle ASL Alphabet dataset")
    p.add_argument("--max_per_class", type=int, default=300,
                   help="Max images to keep per class (default 300; use 0 for all 3000)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for subsampling (default 42)")
    p.add_argument("--skip_download", action="store_true",
                   help="Skip download/extract if zip already exists in data/kaggle_download/")
    p.add_argument("--classes", nargs="+", default=None,
                   help="Only process these classes (default: all 29)")
    return p.parse_args()


def check_kaggle_credentials():
    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    if not cred_path.exists():
        print("\n[ERROR] Kaggle credentials not found!")
        print(f"  Expected: {cred_path}")
        print("\n  To fix:")
        print("  1. Go to https://www.kaggle.com/settings → API → 'Create New Token'")
        print("  2. mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json")
        print("  3. chmod 600 ~/.kaggle/kaggle.json")
        print("  4. Re-run this script")
        sys.exit(1)
    os.chmod(cred_path, 0o600)
    print(f"[INFO] Kaggle credentials: {cred_path} ✓")


def download_dataset(download_dir: Path):
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[STEP 1] Downloading dataset '{DATASET_SLUG}' …")
    print(f"  Destination: {download_dir}")
    print("  This may take a few minutes (~1 GB) …\n")

    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", DATASET_SLUG,
        "-p", str(download_dir),
        "--unzip",
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        # Try the CLI directly
        cmd[0] = "kaggle"
        result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        sys.exit("[ERROR] Kaggle download failed. Check credentials and internet connection.")
    print("\n[INFO] Download complete.")


def find_class_folder(base_dir: Path, class_name: str) -> Path | None:
    """Search for class folder — Kaggle zip nests folders inconsistently."""
    # Try common layouts
    candidates = [
        base_dir / "asl_alphabet_train" / "asl_alphabet_train" / class_name,
        base_dir / "asl_alphabet_train" / class_name,
        base_dir / class_name,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    # Fallback: recursive search
    matches = list(base_dir.rglob(class_name))
    dirs = [m for m in matches if m.is_dir()]
    if dirs:
        # Return the one with the most images
        return max(dirs, key=lambda d: len(list(d.glob("*.jpg"))))
    return None


def find_test_folder(base_dir: Path) -> Path | None:
    """Locate the test images folder."""
    candidates = [
        base_dir / "asl_alphabet_test" / "asl_alphabet_test",
        base_dir / "asl_alphabet_test",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def organize_class(src_dir: Path, dst_dir: Path, class_name: str,
                   max_per_class: int, seed: int) -> int:
    """Copy (subsampled) images to data/raw/<CLASS>/."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.JPG")) + \
             sorted(src_dir.glob("*.jpeg")) + sorted(src_dir.glob("*.png"))

    if max_per_class > 0 and len(images) > max_per_class:
        random.seed(seed)
        images = random.sample(images, max_per_class)
        images = sorted(images)

    copied = 0
    for img in images:
        dst = dst_dir / f"{class_name}_{copied:04d}.jpg"
        shutil.copy2(img, dst)
        copied += 1
    return copied


def organize_test_images(test_dir: Path, raw_dir: Path) -> int:
    """Copy test images — Kaggle test has 1 image per class (e.g. A_test.jpg)."""
    copied = 0
    for img in test_dir.glob("*_test.jpg"):
        class_name = img.stem.replace("_test", "")
        dst_dir = raw_dir / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        # Append to existing images in that class folder
        existing = len(list(dst_dir.glob("*.jpg")))
        dst = dst_dir / f"{class_name}_{existing:04d}_test.jpg"
        shutil.copy2(img, dst)
        copied += 1
    return copied


def main():
    args = parse_args()
    target_classes = args.classes if args.classes else KAGGLE_CLASSES

    if not args.skip_download:
        check_kaggle_credentials()
        download_dataset(DOWNLOAD_DIR)
    else:
        print(f"[INFO] Skipping download — using existing files in {DOWNLOAD_DIR}")

    if not DOWNLOAD_DIR.exists():
        sys.exit(f"[ERROR] Download dir not found: {DOWNLOAD_DIR}\n"
                 "  Run without --skip_download first.")

    print(f"\n[STEP 2] Organizing into data/raw/ …")
    print(f"  Max per class : {args.max_per_class if args.max_per_class > 0 else 'all (3000)'}")
    print(f"  Classes       : {target_classes}\n")

    total = 0
    missing = []

    for class_name in target_classes:
        src = find_class_folder(DOWNLOAD_DIR, class_name)
        if src is None:
            missing.append(class_name)
            print(f"  [WARN] Could not find folder for class '{class_name}'")
            continue
        dst = RAW_DIR / class_name
        n = organize_class(src, dst, class_name, args.max_per_class, args.seed)
        print(f"  {class_name:>8}: {n} images → {dst}")
        total += n

    # Also grab the test images (1 per class)
    test_dir = find_test_folder(DOWNLOAD_DIR)
    if test_dir:
        print(f"\n[INFO] Copying test images from {test_dir} …")
        tc = organize_test_images(test_dir, RAW_DIR)
        print(f"  Copied {tc} test images (1 per class)")
    else:
        print("\n[WARN] Test folder not found — skipping test image copy.")

    if missing:
        print(f"\n⚠  Could not locate: {missing}")
        print("   Check the extracted folder structure in:", DOWNLOAD_DIR)

    print(f"\n✓ Total images organized: {total}")
    print(f"  Raw data root: {RAW_DIR}")
    print("\nNext step → python scripts/auto_annotate.py")


if __name__ == "__main__":
    main()
