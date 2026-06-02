"""
plot_results.py — Generate evaluation plots from training/validation outputs.

Produces:
    results/plots/confusion_matrix_normalized.png
    results/plots/pr_curve.png
    results/plots/per_class_ap.png
    results/plots/per_class_f1.png

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --weights models/asl_yolov8_best.pt
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")   # headless-safe
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METRICS_DIR  = os.path.join(PROJECT_ROOT, "results", "metrics")
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "results", "plots")
CLASS_NAMES  = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]


def parse_args():
    p = argparse.ArgumentParser(description="Generate evaluation plots")
    p.add_argument("--weights", default=None)
    p.add_argument("--split",   default="val", choices=["val", "test"])
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--iou",     type=float, default=0.45)
    p.add_argument("--device",  default="cpu")
    return p.parse_args()


def find_weights(models_dir):
    candidates = [f for f in os.listdir(models_dir) if f.endswith("_best.pt")]
    if candidates:
        return os.path.join(models_dir, sorted(candidates)[-1])
    runs_best = os.path.join(PROJECT_ROOT, "runs", "detect",
                             "asl_yolov8", "weights", "best.pt")
    return runs_best if os.path.exists(runs_best) else None


def plot_confusion_matrix(cm, class_names, out_path):
    """Plot and save a normalised confusion matrix heatmap."""
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums != 0, cm_norm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix (normalised)", fontsize=14, fontweight="bold")

    thresh = 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if val > thresh else "black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix : {out_path}")


def plot_per_class_bar(names, values, title, ylabel, out_path, color="steelblue"):
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, values, color=color, edgecolor="white", linewidth=0.5)
    ax.axhline(np.mean(values), color="red", linestyle="--",
               linewidth=1.2, label=f"Mean = {np.mean(values):.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("ASL Class", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  {title} : {out_path}")


def main():
    args = parse_args()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError:
        import sys
        sys.exit("[ERROR] ultralytics not installed.")

    models_dir = os.path.join(PROJECT_ROOT, "models")
    weights = args.weights or find_weights(models_dir)
    if not weights or not os.path.exists(weights):
        import sys
        sys.exit("[ERROR] No weights found.")

    print(f"[INFO] Weights: {weights}")
    model  = YOLO(weights)

    # ── Run validation to get metrics object ─────────────────────────────────
    metrics = model.val(
        data=os.path.join(PROJECT_ROOT, "dataset.yaml"),
        split=args.split,
        imgsz=320,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        workers=0,
        amp=False,
        plots=False,
        verbose=False,
        project=os.path.join(PROJECT_ROOT, "results"),
        name="plots_eval",
        exist_ok=True,
    )

    names = list(metrics.names.values()) if hasattr(metrics, "names") else CLASS_NAMES
    nc    = len(names)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    try:
        cm = metrics.confusion_matrix.matrix[:nc, :nc].astype(int)
        plot_confusion_matrix(cm, names,
                              os.path.join(PLOTS_DIR, "confusion_matrix_normalized.png"))
    except Exception as e:
        print(f"  [WARN] Could not plot confusion matrix: {e}")

    # ── Per-class AP@0.50 ─────────────────────────────────────────────────────
    try:
        ap50 = metrics.box.ap50[:nc]
        plot_per_class_bar(names, ap50,
                           "Per-class AP@0.50", "AP@0.50",
                           os.path.join(PLOTS_DIR, "per_class_ap50.png"),
                           color="#4C72B0")
    except Exception as e:
        print(f"  [WARN] Could not plot AP50: {e}")

    # ── Per-class F1 ──────────────────────────────────────────────────────────
    try:
        p = metrics.box.p[:nc]
        r = metrics.box.r[:nc]
        f1 = 2 * p * r / (p + r + 1e-9)
        plot_per_class_bar(names, f1,
                           "Per-class F1 Score", "F1",
                           os.path.join(PLOTS_DIR, "per_class_f1.png"),
                           color="#55A868")
    except Exception as e:
        print(f"  [WARN] Could not plot F1: {e}")

    # ── Summary to JSON ───────────────────────────────────────────────────────
    summary = {
        "mAP@0.50":      round(float(metrics.box.map50), 4),
        "mAP@0.50:0.95": round(float(metrics.box.map),   4),
        "Precision":     round(float(metrics.box.mp),    4),
        "Recall":        round(float(metrics.box.mr),    4),
    }
    summary_path = os.path.join(METRICS_DIR, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Summary JSON     : {summary_path}")

    print("\n  Results:")
    for k, v in summary.items():
        print(f"    {k:<22}: {v}")
    print(f"\n✓ All plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
