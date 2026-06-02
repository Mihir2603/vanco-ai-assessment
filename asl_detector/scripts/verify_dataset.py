"""
verify_dataset.py — Sanity-check the dataset before training.

Checks:
  • Each class folder exists and has at least 20 images
  • Every image has a matching annotation file
  • YOLO label format is valid (5 fields, normalised values)
  • Class ID in labels matches expected range
  • Visualises a random sample with bounding boxes

Usage:
    python scripts/verify_dataset.py
    python scripts/verify_dataset.py --show    # display sample images
    python scripts/verify_dataset.py --split train
"""

import argparse
import os
import random

import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw")
ANN_DIR      = os.path.join(PROJECT_ROOT, "data", "annotated")

CLASS_NAMES  = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
MIN_IMAGES   = 20


def parse_args():
    p = argparse.ArgumentParser(description="Verify ASL dataset integrity")
    p.add_argument("--split", default="train", choices=["train", "val", "test", "raw"])
    p.add_argument("--show", action="store_true",
                   help="Display a random sample of annotated images (requires display)")
    p.add_argument("--n_samples", type=int, default=5,
                   help="Number of images to visualise (default 5)")
    return p.parse_args()


def check_raw_folders():
    print("\n[CHECK] Raw data folders …")
    issues = []
    for label in CLASS_NAMES:
        folder = os.path.join(RAW_DIR, label)
        if not os.path.isdir(folder):
            issues.append(f"  MISSING folder: {folder}")
            continue
        images  = [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png"))]
        labels  = [f for f in os.listdir(folder) if f.endswith(".txt")]
        missing = len(images) - len(labels)
        status  = "✓" if len(images) >= MIN_IMAGES else "⚠"
        print(f"  {status} {label}: {len(images)} images, {len(labels)} annotations"
              + (f"  [{missing} unannotated]" if missing else ""))
        if len(images) < MIN_IMAGES:
            issues.append(f"  Too few images for '{label}': {len(images)} < {MIN_IMAGES}")
    return issues


def validate_label_file(txt_path, nc):
    """Return list of error strings for a label file, or [] if OK."""
    errors = []
    with open(txt_path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{txt_path}:{lineno} — expected 5 fields, got {len(parts)}")
                continue
            try:
                cls_id = int(parts[0])
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                errors.append(f"{txt_path}:{lineno} — non-numeric values")
                continue
            if cls_id < 0 or cls_id >= nc:
                errors.append(f"{txt_path}:{lineno} — class_id {cls_id} out of range [0,{nc-1}]")
            for v in vals:
                if not (0.0 <= v <= 1.0):
                    errors.append(f"{txt_path}:{lineno} — value {v} outside [0,1]")
    return errors


def check_split(split, nc):
    print(f"\n[CHECK] Split '{split}' label validity …")
    img_dir = os.path.join(ANN_DIR, "images", split)
    lbl_dir = os.path.join(ANN_DIR, "labels", split)
    if not os.path.isdir(img_dir):
        print(f"  [SKIP] Not found: {img_dir}")
        return []

    images = sorted(f for f in os.listdir(img_dir)
                    if f.lower().endswith((".jpg",".jpeg",".png")))
    errors = []
    ok = 0
    for fname in images:
        lbl_name = os.path.splitext(fname)[0] + ".txt"
        lbl_path = os.path.join(lbl_dir, lbl_name)
        if not os.path.exists(lbl_path):
            errors.append(f"  MISSING label for {fname}")
        else:
            errs = validate_label_file(lbl_path, nc)
            errors.extend(errs)
            if not errs:
                ok += 1

    print(f"  {ok}/{len(images)} images with valid labels")
    return errors


def visualise_samples(split, n_samples):
    """Draw bbox overlays on random images and display them."""
    img_dir = os.path.join(ANN_DIR, "images", split)
    lbl_dir = os.path.join(ANN_DIR, "labels", split)
    if not os.path.isdir(img_dir):
        print(f"  [SKIP] Split dir not found: {img_dir}")
        return

    images = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
    random.shuffle(images)

    for fname in images[:n_samples]:
        img_path = os.path.join(img_dir, fname)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        h, w = frame.shape[:2]

        if os.path.exists(lbl_path):
            with open(lbl_path) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, label, (x1, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow(f"Verify: {fname}", frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def main():
    args = parse_args()
    nc   = len(CLASS_NAMES)
    all_errors = []

    all_errors += check_raw_folders()
    if args.split != "raw":
        for sp in ["train", "val", "test"]:
            all_errors += check_split(sp, nc)

    if all_errors:
        print(f"\n⚠  {len(all_errors)} issues found:")
        for e in all_errors[:30]:
            print(e)
        if len(all_errors) > 30:
            print(f"  … and {len(all_errors)-30} more.")
    else:
        print("\n✓ Dataset looks clean — no issues found.")

    if args.show:
        print(f"\n[VISUAL] Showing {args.n_samples} samples from '{args.split}' …")
        visualise_samples(args.split if args.split != "raw" else "train", args.n_samples)


if __name__ == "__main__":
    main()
