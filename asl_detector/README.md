# ASL Alphabet Detector — VANCO Assessment

Real-time American Sign Language hand sign detection using **YOLOv8** and **MediaPipe**.  
Dataset: [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) — 87,000 images, 29 classes (A-Z + del + nothing + space).

---

## Project structure

```
asl_detector/
├── data/
│   ├── raw/                       # Images organised by class (A/, B/, …, del/, nothing/, space/)
│   ├── kaggle_download/           # Raw Kaggle download (auto-created)
│   └── annotated/                 # YOLO-format split dataset
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/
├── scripts/
│   ├── download_dataset.py        # Step 1 — download from Kaggle + organise
│   ├── auto_annotate.py           # Step 2 — MediaPipe bbox → YOLO .txt labels
│   ├── verify_dataset.py          # Step 3 — sanity check
│   ├── prepare_dataset.py         # Step 4 — train/val/test split
│   ├── train.py                   # Step 5 — YOLOv8 training
│   ├── evaluate.py                # Step 6 — mAP, P/R, FPS benchmark
│   ├── plot_results.py            # Step 7 — confusion matrix & per-class plots
│   └── collect_data.py            # Optional — augment with custom webcam images
├── models/                        # Best weights saved here after training
├── results/
│   ├── metrics/                   # summary.json, per_class.csv, fps_benchmark.txt
│   └── plots/                     # confusion_matrix.png, per-class AP/F1 charts
├── demo.py                        # Live webcam demo
├── dataset.yaml                   # YOLOv8 dataset config (29 classes)
├── classes.txt                    # Class names list
└── requirements.txt
```

---

## Setup

```bash
cd asl_detector
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU note**: For CUDA support replace the torch install:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Step-by-step workflow

### 1 — Kaggle credentials

1. Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**
2. Move the downloaded file:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

---

### 2 — Download dataset

```bash
python scripts/download_dataset.py
# Downloads ~1 GB, extracts, copies 300 images/class into data/raw/
```

Control the number of images per class:
```bash
python scripts/download_dataset.py --max_per_class 500   # more images
python scripts/download_dataset.py --max_per_class 0     # use all 3000
```

If the download already ran:
```bash
python scripts/download_dataset.py --skip_download
```

**Dataset stats (default: 300/class × 29 classes = 8,700 images)**

| Split    | ~Images |
|----------|---------|
| Train 70%| 6,090   |
| Val 20%  | 1,740   |
| Test 10% | 870     |

---

### 3 — Auto-annotate with MediaPipe

The Kaggle dataset is classification-only (no bounding boxes). MediaPipe detects 21 hand landmarks and computes a padded bounding box, saved as a `.txt` sidecar in YOLO format:

```bash
python scripts/auto_annotate.py --padding 0.20
```

Include fallback full-frame boxes for images where MediaPipe finds no hand:
```bash
python scripts/auto_annotate.py --padding 0.20 --fallback_full
```

---

### 4 — Verify dataset (recommended)

```bash
python scripts/verify_dataset.py
python scripts/verify_dataset.py --show     # visual check with bbox overlays
```

---

### 5 — Prepare train/val/test split

```bash
python scripts/prepare_dataset.py
# 70% train / 20% val / 10% test (stratified per class)
```

---

### 6 — Train

```bash
# CPU-optimised (used in this run — ~30 min on Xeon CPU):
python scripts/train.py --model yolov8n --epochs 30 --batch 32 --imgsz 320 --workers 0

# GPU / higher-accuracy run:
python scripts/train.py --model yolov8s --epochs 80 --batch 16 --imgsz 640
python scripts/train.py --resume    # resume from last checkpoint
```

> **Note**: On CPU set `--workers 0` to avoid dataloader hang and `--imgsz 320` for practical speed.

| Model   | Size  | CPU speed     | Recommended for         |
|---------|-------|---------------|-------------------------|
| yolov8n | ~6 MB | ~25 ms/frame  | Live demo, CPU-only     |
| yolov8s | ~22 MB| ~40 ms/frame  | Better accuracy, GPU    |
| yolov8m | ~52 MB| ~80 ms/frame  | High accuracy, GPU      |

Training logs and plots → `runs/detect/asl_yolov8/`  
Best weights copied → `models/asl_yolov8_best.pt`

---

### 7 — Evaluate

```bash
python scripts/evaluate.py
python scripts/evaluate.py --split test
```

Outputs `results/metrics/summary.json`, `per_class.csv`, `fps_benchmark.txt`.

---

### 8 — Generate plots

```bash
python scripts/plot_results.py
```

Outputs to `results/plots/`:
- `confusion_matrix_normalized.png`
- `per_class_ap50.png`
- `per_class_f1.png`

---

### 9 — Live webcam demo

```bash
python demo.py
python demo.py --weights models/asl_yolov8_best.pt --conf 0.45
python demo.py --camera 1    # second camera
```

| Key | Action                        |
|-----|-------------------------------|
| `Q` | Quit                          |
| `S` | Save screenshot               |
| `R` | Start / stop MP4 recording    |

The demo uses **MediaPipe pre-screening** to skip YOLO inference on frames with no hand detected, reducing CPU load on empty frames.

---

## Optional — Augment with custom webcam images

To supplement the Kaggle data with your own images (recommended for robustness):

```bash
python scripts/collect_data.py --label A --count 50
python scripts/collect_data.py --label B --count 50
# ... repeat per class
# Then re-run annotate → prepare → train
```

---

## Dataset details

| Property          | Value                                                        |
|-------------------|--------------------------------------------------------------|
| Source            | Kaggle: grassknoted/asl-alphabet                            |
| Classes           | 29 (A-Z + del + nothing + space)                            |
| Total images      | 87,000 (3,000/class) — subsampled to 50/class for this run   |
| Image size        | 200×200 px (resized to 320×320 during training)             |
| Annotation format | YOLO `.txt` sidecar (cx cy w h normalised), auto-generated  |
| Bounding box      | MediaPipe 21-landmark hand bbox + 20% padding               |
| Split             | 70% train / 20% val / 10% test                              |

---

## Model & training details

- **Architecture**: YOLOv8n — single-stage detector, CSPDarknet + C2f backbone
- **Input size**: 320×320 (CPU-optimised; 640 for GPU runs)
- **Images per class**: 50 (70/20/10 split → 1,015 train / 290 val / 145 test)
- **Augmentation**: HSV jitter, ±15° rotation, ±30% scale, horizontal flip, mosaic (ep 1-20), light mixup
- **Optimizer**: AdamW, cosine LR decay, warmup 3 epochs
- **Epochs**: 30, early stopping patience=20
- **Hardware**: Intel Xeon Platinum 8351N (CPU-only)

---

## Evaluation results (test split — 145 images, 29 classes)

| Metric            | Value    |
|-------------------|----------|
| **mAP@0.50**      | **90.6%** |
| **mAP@0.50:0.95** | **75.5%** |
| **Precision**     | **90.5%** |
| **Recall**        | **89.5%** |
| **F1**            | **90.0%** |
| **FPS (CPU)**     | **45.4 FPS** (22 ms/frame) |

Weakest classes: `space` (39.5% mAP50), `W` (59.5%) — both have high visual ambiguity.  
Strongest classes: `B`, `D`, `F`, `H`, `K`, `N`, `nothing`, `O`, `P`, `S`, `T` — all 99.5% mAP50.

---

## Evaluation metrics

| Metric          | Description                                      |
|-----------------|--------------------------------------------------|
| mAP@0.50        | Mean AP at IoU=0.50                              |
| mAP@0.50:0.95   | COCO-style averaged mAP                          |
| Precision       | Fraction of predictions correct                  |
| Recall          | Fraction of ground-truth objects detected        |
| F1              | Harmonic mean of P and R                         |
| FPS / Latency   | Per-frame inference time on test split           |

---

## Deployment considerations

| Constraint       | Notes                                                              |
|------------------|--------------------------------------------------------------------|
| CPU inference    | YOLOv8n: ~15–25 FPS on modern CPU                                 |
| GPU inference    | 50–100+ FPS on CUDA GPU                                            |
| Model size       | 6 MB (n), 22 MB (s)                                               |
| Webcam latency   | <50 ms with MediaPipe pre-screening                                |
| Robustness       | Kaggle images are mostly clean backgrounds — augment or collect custom data for real-world diversity |
| New signers      | Signer-independent split improves generalisation                   |

---

## Classes (29)

```
A  B  C  D  E  F  G  H  I  J  K  L  M  N  O
P  Q  R  S  T  U  V  W  X  Y  Z  del  nothing  space
```
