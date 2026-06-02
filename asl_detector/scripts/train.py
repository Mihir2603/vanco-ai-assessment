"""
train.py — Train a YOLOv8 object-detection model on the ASL dataset.

Usage:
    python scripts/train.py
    python scripts/train.py --model yolov8s --epochs 100 --batch 16
    python scripts/train.py --resume           # resume last run
"""

import argparse
import os
import sys

# ── Resolve project root regardless of where we call the script ──────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_YAML = os.path.join(PROJECT_ROOT, "dataset.yaml")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 ASL detector")
    p.add_argument("--model",   default="yolov8n",
                   choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l"],
                   help="YOLOv8 variant (default: yolov8n — fastest)")
    p.add_argument("--epochs",  type=int,   default=80)
    p.add_argument("--batch",   type=int,   default=16)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--workers", type=int,   default=4)
    p.add_argument("--device",  default="cpu",
                   help="cuda device (0, 0,1,…) or cpu (default: cpu)")
    p.add_argument("--resume",  action="store_true",
                   help="Resume training from last checkpoint")
    p.add_argument("--name",    default="asl_yolov8",
                   help="Run name under runs/detect/")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[ERROR] ultralytics not installed. Run: pip install ultralytics")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Augmentation overrides (passed via train kwargs) ─────────────────────
    # YOLOv8 already applies sensible defaults; we tune for hand signs:
    #   - moderate rotation (signs are orientation-sensitive)
    #   - brightness/contrast variation
    #   - mosaic on (helps with background diversity)
    #   - no shear / perspective (distorts finger positions too much)
    aug_kwargs = dict(
        hsv_h=0.015,     # hue shift ±1.5 %
        hsv_s=0.50,      # saturation ±50 %
        hsv_v=0.40,      # value (brightness) ±40 %
        degrees=15.0,    # rotation ±15°
        translate=0.10,  # translation ±10 %
        scale=0.30,      # scale ±30 %
        shear=0.0,       # no shear
        perspective=0.0, # no perspective
        flipud=0.0,      # never flip vertically (hand orientation matters)
        fliplr=0.5,      # horizontal flip OK
        mosaic=1.0,      # mosaic augmentation on
        mixup=0.1,       # light mixup
        copy_paste=0.0,
    )

    if args.resume:
        # Find last weights
        last_ckpt = os.path.join("runs", "detect", args.name, "weights", "last.pt")
        if not os.path.exists(last_ckpt):
            sys.exit(f"[ERROR] Cannot find checkpoint to resume: {last_ckpt}")
        model = YOLO(last_ckpt)
        print(f"[INFO] Resuming from {last_ckpt}")
    else:
        model = YOLO(f"{args.model}.pt")   # downloads pretrained weights if needed
        print(f"[INFO] Starting fresh training with {args.model}.pt")

    print(f"[INFO] Dataset : {DATASET_YAML}")
    print(f"[INFO] Epochs  : {args.epochs}  Batch: {args.batch}  ImgSz: {args.imgsz}")

    results = model.train(
        data=DATASET_YAML,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        project=os.path.join(PROJECT_ROOT, "runs", "detect"),
        name=args.name,
        save=True,
        save_period=10,       # checkpoint every 10 epochs
        plots=True,           # save training plots
        val=True,
        patience=20,          # early stopping
        exist_ok=True,
        verbose=True,
        amp=False,            # disable AMP — avoids CPU hang with torch CUDA build
        **aug_kwargs,
    )

    # Copy best weights to models/ for convenience
    best_src = os.path.join(PROJECT_ROOT, "runs", "detect", args.name, "weights", "best.pt")
    best_dst = os.path.join(MODELS_DIR, f"{args.name}_best.pt")
    if os.path.exists(best_src):
        import shutil
        shutil.copy2(best_src, best_dst)
        print(f"\n✓ Best weights saved to: {best_dst}")

    print("\n✓ Training complete.")
    print("Next step → python scripts/evaluate.py")


if __name__ == "__main__":
    main()
