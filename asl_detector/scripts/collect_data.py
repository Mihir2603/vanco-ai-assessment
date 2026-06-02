"""
collect_data.py — Webcam-based ASL image collection.

Use this to supplement the Kaggle dataset with custom images from your own camera
(different backgrounds, lighting, signer) for improved real-world robustness.

Usage:
    python scripts/collect_data.py --label A --count 30
    python scripts/collect_data.py --label A --count 30 --burst
    python scripts/collect_data.py --label A --count 30 --camera 1

Controls:
    SPACE  : capture current frame
    Q      : quit
"""

import argparse
import os
import sys
import time

import cv2

# Allow importing mp_hands_helper from the same scripts/ directory
sys.path.insert(0, os.path.dirname(__file__))
from mp_hands_helper import HandDetector

# 29 classes from Kaggle ASL Alphabet dataset
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def parse_args():
    p = argparse.ArgumentParser(description="Collect ASL images from webcam")
    p.add_argument("--label", required=True, choices=CLASS_NAMES,
                   help="ASL sign to capture (e.g. A, del, space)")
    p.add_argument("--count", type=int, default=30,
                   help="Number of images to collect (default 30)")
    p.add_argument("--camera", type=int, default=0,
                   help="Camera index (default 0)")
    p.add_argument("--delay", type=float, default=0.5,
                   help="Minimum seconds between auto-captures in burst mode")
    p.add_argument("--burst", action="store_true",
                   help="Enable burst mode (auto-capture when hand detected)")
    return p.parse_args()


def draw_ui(frame, label, collected, target, hand_bbox=None):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, f"Label: {label}  [{collected}/{target}]",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2)
    cv2.putText(frame, "SPACE=capture  Q=quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    if hand_bbox is not None:
        x1, y1, x2, y2 = hand_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Hand detected", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    return frame


def main():
    args = parse_args()
    save_dir = os.path.join(RAW_DIR, args.label)
    os.makedirs(save_dir, exist_ok=True)

    existing = len([f for f in os.listdir(save_dir) if f.endswith(".jpg")])
    print(f"[INFO] Saving to    : {save_dir}")
    print(f"[INFO] Existing     : {existing}  —  collecting {args.count} more")

    detector = HandDetector(mode="video", detection_confidence=0.6)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        detector.close()
        sys.exit(f"[ERROR] Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    collected  = 0
    last_capture = 0.0

    while collected < args.count:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        bbox  = detector.get_bbox(frame, padding=0.20)
        display = draw_ui(frame.copy(), args.label, collected, args.count, bbox)
        cv2.imshow("ASL Data Collection", display)

        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        capture = False
        if key == ord(' '):
            capture = True
        elif args.burst and bbox is not None and (now - last_capture) >= args.delay:
            capture = True

        if capture:
            fname = os.path.join(save_dir, f"{args.label}_{existing + collected:04d}.jpg")
            cv2.imwrite(fname, frame)
            collected += 1
            last_capture = now
            print(f"  Saved [{collected}/{args.count}]: {fname}")
            flash = display.copy()
            h, w = flash.shape[:2]
            cv2.rectangle(flash, (0, 0), (w, h), (0, 255, 0), 6)
            cv2.imshow("ASL Data Collection", flash)
            cv2.waitKey(100)

        if key == ord('q'):
            print("[INFO] Quit by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print(f"\n✓ Done. Collected {collected} images for '{args.label}'.")
    print(f"  Total in folder: {existing + collected}")


if __name__ == "__main__":
    main()
