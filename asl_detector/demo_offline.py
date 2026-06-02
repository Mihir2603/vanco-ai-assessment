"""
demo_offline.py — Offline ASL detection demo for headless / no-webcam environments.

Runs the trained YOLOv8 model on test-set images and produces:
  results/demo/demo_video.mp4          — annotated slideshow (backup recorded demo)
  results/demo/sample_grid.jpg         — 5×6 grid of annotated predictions
  results/demo/predictions_log.csv     — per-image prediction log

Usage:
    python demo_offline.py
    python demo_offline.py --source data/annotated/images/test
    python demo_offline.py --conf 0.40 --fps 3
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
DEMO_OUT_DIR = os.path.join(PROJECT_ROOT, "results", "demo")
CLASS_NAMES  = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

np.random.seed(42)
PALETTE = {name: tuple(int(c) for c in np.random.randint(80, 230, 3))
           for name in CLASS_NAMES}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=None)
    p.add_argument("--source",  default=None,
                   help="Image dir (default: data/annotated/images/test)")
    p.add_argument("--conf",    type=float, default=0.40)
    p.add_argument("--iou",     type=float, default=0.45)
    p.add_argument("--imgsz",   type=int,   default=320)
    p.add_argument("--device",  default="cpu")
    p.add_argument("--fps",     type=int,   default=3,
                   help="Slideshow FPS in output video (default: 3)")
    p.add_argument("--max_imgs", type=int,  default=145)
    return p.parse_args()


def find_weights():
    candidates = [f for f in os.listdir(MODELS_DIR) if f.endswith("_best.pt")]
    if candidates:
        return os.path.join(MODELS_DIR, sorted(candidates)[-1])
    runs_best = os.path.join(PROJECT_ROOT, "runs", "detect",
                             "asl_yolov8", "weights", "best.pt")
    return runs_best if os.path.exists(runs_best) else None


def draw_box(frame, x1, y1, x2, y2, label, conf, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label}  {conf:.0%}"
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    by1 = max(0, y1 - th - bl - 4)
    cv2.rectangle(frame, (x1, by1), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - bl),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


def annotate_frame(frame, results, true_label=None):
    """Draw YOLO detections + ground-truth label banner."""
    h, w = frame.shape[:2]
    predictions = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = (r.names[cls_id] if r.names and cls_id in r.names
                      else CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES)
                      else str(cls_id))
            color  = PALETTE.get(label, (0, 200, 255))
            draw_box(frame, x1, y1, x2, y2, label, conf, color)
            predictions.append((label, conf))

    # Top banner: ground-truth vs predicted
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    if true_label:
        cv2.putText(frame, f"GT: {true_label}", (8, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 180), 2)
    if predictions:
        best_lbl, best_conf = max(predictions, key=lambda x: x[1])
        color = (0, 230, 0) if (true_label and best_lbl == true_label) else (0, 80, 255)
        cv2.putText(frame, f"PRED: {best_lbl}  {best_conf:.0%}",
                    (w - 260, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    else:
        cv2.putText(frame, "No detection", (w - 220, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 255), 2)

    return predictions


def build_sample_grid(frames, out_path, cols=5):
    """Tile frames into a grid image."""
    if not frames:
        return
    rows_needed = (len(frames) + cols - 1) // cols
    h, w = frames[0].shape[:2]
    grid = np.zeros((rows_needed * h, cols * w, 3), dtype=np.uint8)
    for idx, f in enumerate(frames):
        r, c = divmod(idx, cols)
        resized = cv2.resize(f, (w, h))
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = resized
    cv2.imwrite(out_path, grid)
    print(f"  Sample grid     : {out_path}")


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        import sys; sys.exit("[ERROR] ultralytics not installed.")

    weights = args.weights or find_weights()
    if not weights or not os.path.exists(weights):
        import sys; sys.exit("[ERROR] No weights found.")

    source = args.source or os.path.join(
        PROJECT_ROOT, "data", "annotated", "images", "test")
    if not os.path.isdir(source):
        import sys; sys.exit(f"[ERROR] Image dir not found: {source}")

    os.makedirs(DEMO_OUT_DIR, exist_ok=True)

    print(f"[INFO] Weights : {weights}")
    print(f"[INFO] Source  : {source}")
    print(f"[INFO] Output  : {DEMO_OUT_DIR}")

    model = YOLO(weights)

    # Collect image paths, grouped/sorted for logical flow
    exts = (".jpg", ".jpeg", ".png")
    img_paths = sorted([os.path.join(source, f) for f in os.listdir(source)
                        if f.lower().endswith(exts)])[:args.max_imgs]
    print(f"[INFO] Processing {len(img_paths)} images …")

    # ── Video writer setup ────────────────────────────────────────────────────
    frame_w, frame_h = 640, 480
    video_path = os.path.join(DEMO_OUT_DIR, "demo_video.mp4")
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    writer     = cv2.VideoWriter(video_path, fourcc, args.fps, (frame_w, frame_h))

    log_rows   = []
    grid_frames = []
    correct = total = 0
    t0 = time.perf_counter()

    for i, img_path in enumerate(img_paths):
        # Ground-truth label is encoded in the filename prefix (e.g. "A_0001.jpg")
        fname = os.path.basename(img_path)
        true_label = fname.split("_")[0] if "_" in fname else None

        frame = cv2.imread(img_path)
        if frame is None:
            continue
        frame = cv2.resize(frame, (frame_w, frame_h))

        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                iou=args.iou, device=args.device, verbose=False)

        predictions = annotate_frame(frame, results, true_label)

        writer.write(frame)

        if len(grid_frames) < 30:   # collect up to 30 for the grid
            grid_frames.append(frame.copy())

        # Log
        pred_lbl  = predictions[0][0] if predictions else "none"
        pred_conf = predictions[0][1] if predictions else 0.0
        correct  += int(true_label == pred_lbl) if true_label else 0
        total    += 1
        log_rows.append({
            "image": fname,
            "true": true_label or "",
            "predicted": pred_lbl,
            "confidence": f"{pred_conf:.4f}",
            "correct": int(true_label == pred_lbl) if true_label else "",
        })

        if (i + 1) % 20 == 0:
            elapsed = time.perf_counter() - t0
            fps_so_far = (i + 1) / elapsed
            print(f"  [{i+1:>3}/{len(img_paths)}]  "
                  f"inf FPS={fps_so_far:.1f}  "
                  f"acc={correct}/{total} ({100*correct/total:.0f}%)")

    writer.release()
    elapsed_total = time.perf_counter() - t0
    inf_fps = len(img_paths) / elapsed_total

    # ── Outputs ───────────────────────────────────────────────────────────────
    print(f"\n  Demo video      : {video_path}")

    grid_path = os.path.join(DEMO_OUT_DIR, "sample_grid.jpg")
    build_sample_grid(grid_frames, grid_path, cols=5)

    log_path = os.path.join(DEMO_OUT_DIR, "predictions_log.csv")
    with open(log_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image","true","predicted","confidence","correct"])
        w.writeheader(); w.writerows(log_rows)
    print(f"  Predictions log : {log_path}")

    print(f"\n{'='*46}")
    print(f"  Offline Demo Summary")
    print(f"{'='*46}")
    print(f"  Images processed  : {total}")
    print(f"  Top-1 accuracy    : {correct}/{total} = {100*correct/total:.1f}%")
    print(f"  Avg inference FPS : {inf_fps:.1f}")
    print(f"  Avg latency (ms)  : {1000/inf_fps:.1f}")
    print(f"{'='*46}")
    print(f"\n✓ Results saved to: {DEMO_OUT_DIR}")
    print("  To play the video:  vlc results/demo/demo_video.mp4")
    print("  Live webcam demo:   python demo.py   (requires webcam + display)")


if __name__ == "__main__":
    main()
