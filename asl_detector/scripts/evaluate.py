"""
evaluate.py — Comprehensive evaluation of the trained ASL detector.

Outputs:
    results/metrics/summary.json        — mAP, P, R, F1
    results/metrics/per_class.csv       — per-class breakdown
    results/plots/confusion_matrix.png  — normalised confusion matrix
    results/plots/pr_curve.png          — precision-recall curve
    results/metrics/fps_benchmark.txt   — FPS / latency benchmark

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --weights models/asl_yolov8_best.pt
    python scripts/evaluate.py --split val
"""

import argparse
import csv
import json
import os
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_YAML = os.path.join(PROJECT_ROOT, "dataset.yaml")
METRICS_DIR  = os.path.join(PROJECT_ROOT, "results", "metrics")
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "results", "plots")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained ASL YOLOv8 model")
    p.add_argument("--weights", default=None,
                   help="Path to .pt weights (default: auto-detect in models/)")
    p.add_argument("--split",   default="val", choices=["val", "test"],
                   help="Dataset split to evaluate (default: val)")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--iou",     type=float, default=0.45)
    p.add_argument("--device",  default="cpu",
                   help="cuda device or cpu (default: cpu)")
    p.add_argument("--fps_frames", type=int, default=200,
                   help="Number of frames for FPS benchmark (default 200)")
    return p.parse_args()


def find_best_weights():
    candidates = [f for f in os.listdir(MODELS_DIR) if f.endswith("_best.pt")]
    if candidates:
        return os.path.join(MODELS_DIR, sorted(candidates)[-1])
    # Fall back to runs directory
    runs_best = os.path.join(PROJECT_ROOT, "runs", "detect", "asl_yolov8", "weights", "best.pt")
    if os.path.exists(runs_best):
        return runs_best
    return None


def run_validation(model, args):
    """Run YOLOv8 val on the dataset and return metrics dict."""
    metrics = model.val(
        data=DATASET_YAML,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        workers=0,
        amp=False,
        plots=True,
        save_json=False,
        project=os.path.join(PROJECT_ROOT, "results"),
        name="eval",
        exist_ok=True,
        verbose=True,
    )
    return metrics


def fps_benchmark(model, test_img_dir, args):
    """Measure inference FPS on the test split images."""
    images = [os.path.join(test_img_dir, f)
              for f in os.listdir(test_img_dir)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not images:
        return None

    # Warm-up
    model.predict(images[0], imgsz=args.imgsz, conf=args.conf,
                  device=args.device, workers=0, verbose=False)

    n = min(args.fps_frames, len(images))
    t0 = time.perf_counter()
    for img_path in images[:n]:
        model.predict(img_path, imgsz=args.imgsz, conf=args.conf,
                      device=args.device, workers=0, verbose=False)
    elapsed = time.perf_counter() - t0
    fps = n / elapsed
    latency_ms = (elapsed / n) * 1000
    return fps, latency_ms, n


def save_per_class_csv(metrics):
    """Write per-class AP50, precision, recall to CSV."""
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = os.path.join(METRICS_DIR, "per_class.csv")

    try:
        ap50_per_class  = metrics.box.ap50          # shape (nc,)
        p_per_class     = metrics.box.p             # shape (nc,)
        r_per_class     = metrics.box.r             # shape (nc,)
        names           = list(metrics.names.values()) if hasattr(metrics, "names") else CLASS_NAMES
    except AttributeError:
        print("[WARN] Could not extract per-class metrics — skipping CSV.")
        return

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class", "AP@0.50", "Precision", "Recall", "F1"])
        for i, name in enumerate(names):
            if i >= len(ap50_per_class):
                break
            p = float(p_per_class[i])
            r = float(r_per_class[i])
            f1 = 2 * p * r / (p + r + 1e-9)
            writer.writerow([name,
                              f"{float(ap50_per_class[i]):.4f}",
                              f"{p:.4f}",
                              f"{r:.4f}",
                              f"{f1:.4f}"])
    print(f"  Per-class CSV  : {path}")


def save_summary_json(metrics, fps_result):
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = os.path.join(METRICS_DIR, "summary.json")
    summary = {
        "mAP@0.50":      round(float(metrics.box.map50),   4),
        "mAP@0.50:0.95": round(float(metrics.box.map),     4),
        "Precision":     round(float(metrics.box.mp),      4),
        "Recall":        round(float(metrics.box.mr),      4),
        "F1":            round(float(2 * metrics.box.mp * metrics.box.mr
                               / (metrics.box.mp + metrics.box.mr + 1e-9)), 4),
    }
    if fps_result:
        fps, lat, n = fps_result
        summary["FPS"]              = round(fps, 1)
        summary["Latency_ms"]       = round(lat, 2)
        summary["FPS_benchmark_n"]  = n

    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Summary JSON   : {path}")
    return summary


def print_summary_table(summary):
    print("\n" + "=" * 42)
    print("  ASL Detector — Evaluation Summary")
    print("=" * 42)
    for k, v in summary.items():
        print(f"  {k:<22}: {v}")
    print("=" * 42)


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        import sys
        sys.exit("[ERROR] ultralytics not installed. Run: pip install ultralytics")

    weights = args.weights or find_best_weights()
    if not weights or not os.path.exists(weights):
        import sys
        sys.exit(f"[ERROR] No weights found. Train first or pass --weights <path>")

    print(f"[INFO] Weights  : {weights}")
    print(f"[INFO] Split    : {args.split}")

    model = YOLO(weights)

    print("\n[STEP 1] Running YOLOv8 validation …")
    metrics = run_validation(model, args)

    print("\n[STEP 2] Per-class metrics …")
    save_per_class_csv(metrics)

    print("\n[STEP 3] FPS benchmark …")
    test_img_dir = os.path.join(PROJECT_ROOT, "data", "annotated",
                                "images", args.split)
    fps_result = None
    if os.path.isdir(test_img_dir):
        fps_result = fps_benchmark(model, test_img_dir, args)
        if fps_result:
            fps, lat, n = fps_result
            bench_path = os.path.join(METRICS_DIR, "fps_benchmark.txt")
            os.makedirs(METRICS_DIR, exist_ok=True)
            with open(bench_path, "w") as fh:
                fh.write(f"Benchmark on {n} images\n")
                fh.write(f"FPS          : {fps:.1f}\n")
                fh.write(f"Latency (ms) : {lat:.2f}\n")
            print(f"  FPS: {fps:.1f}  Latency: {lat:.2f} ms  (n={n})")
    else:
        print(f"  [WARN] Test images dir not found: {test_img_dir}")

    summary = save_summary_json(metrics, fps_result)
    print_summary_table(summary)

    print(f"\n✓ All results saved to: results/")
    print("Next step → python demo.py")


if __name__ == "__main__":
    main()
