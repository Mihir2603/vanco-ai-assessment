"""
auto_annotate.py — Automatically generate YOLO bounding-box annotations
using MediaPipe hand detection (Tasks API, MediaPipe 0.10+).

Reads images from  data/raw/<LABEL>/
Writes YOLO .txt sidecar files to the same folder.

YOLO label format (one line per object):
    <class_id> <cx> <cy> <w> <h>   (all normalised to [0,1])

Usage:
    python scripts/auto_annotate.py
    python scripts/auto_annotate.py --padding 0.2 --fallback_full
    python scripts/auto_annotate.py --labels A B C
"""

import argparse
import os
import sys

# Allow importing mp_hands_helper from the same scripts/ directory
sys.path.insert(0, os.path.dirname(__file__))
from mp_hands_helper import HandDetector

from tqdm import tqdm

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
CLASS_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def parse_args():
    p = argparse.ArgumentParser(description="Auto-annotate ASL images with MediaPipe")
    p.add_argument("--padding", type=float, default=0.20,
                   help="Fractional padding around hand landmarks (default 0.20)")
    p.add_argument("--fallback_full", action="store_true",
                   help="Use full-frame bbox when MediaPipe detects no hand")
    p.add_argument("--confidence", type=float, default=0.5,
                   help="MediaPipe detection confidence threshold (default 0.5)")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Process only these labels (default: all)")
    return p.parse_args()


def annotate_folder(label, class_id, raw_dir, padding, fallback_full, confidence):
    folder = os.path.join(raw_dir, label)
    if not os.path.isdir(folder):
        print(f"[SKIP] No folder for label '{label}': {folder}")
        return 0, 0

    import cv2
    detector = HandDetector(mode="image",
                            detection_confidence=confidence,
                            presence_confidence=confidence)

    images = sorted(f for f in os.listdir(folder)
                    if f.lower().endswith((".jpg", ".jpeg", ".png")))

    annotated = 0
    fallbacks = 0

    for fname in tqdm(images, desc=f"  {label}", leave=False):
        img_path = os.path.join(folder, fname)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        yolo_bbox = detector.get_yolo_bbox(frame, padding=padding)

        if yolo_bbox is None:
            if fallback_full:
                yolo_bbox = (0.5, 0.5, 1.0, 1.0)
                fallbacks += 1
            else:
                continue   # skip — no hand found

        cx, cy, bw, bh = yolo_bbox
        label_path = os.path.splitext(img_path)[0] + ".txt"
        with open(label_path, "w") as f:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        annotated += 1

    detector.close()
    return annotated, fallbacks


def main():
    args = parse_args()
    target_labels = args.labels if args.labels else CLASS_NAMES

    print(f"[INFO] Raw data root : {os.path.abspath(RAW_DIR)}")
    print(f"[INFO] Padding        : {args.padding}")
    print(f"[INFO] Fallback full  : {args.fallback_full}")
    print(f"[INFO] Labels         : {target_labels}\n")

    total_annotated = 0
    total_fallbacks = 0

    for label in target_labels:
        if label not in CLASS_INDEX:
            print(f"[WARN] Unknown label '{label}', skipping.")
            continue
        class_id = CLASS_INDEX[label]
        ann, fb = annotate_folder(label, class_id, RAW_DIR,
                                  args.padding, args.fallback_full,
                                  args.confidence)
        print(f"  {label}: {ann} annotated  ({fb} fallbacks)")
        total_annotated += ann
        total_fallbacks += fb

    print(f"\n✓ Total annotated : {total_annotated}")
    print(f"  Total fallbacks  : {total_fallbacks}")
    print("\nNext step → python scripts/prepare_dataset.py")


if __name__ == "__main__":
    main()
