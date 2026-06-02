# Architecture Report — ASL Alphabet Detector

## 1. Problem Statement

Detect and classify 29 American Sign Language hand signs (A-Z + del + nothing + space) from webcam frames in real time. Each frame may contain zero or one hand; output: bounding box + class label + confidence.

---

## 2. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   DATA COLLECTION LAYER                         │
│                                                                 │
│  Source: Kaggle grassknoted/asl-alphabet (87,000 images)        │
│  • 29 classes: A-Z + del + nothing + space                      │
│  • 3,000 images/class, 200×200 px, white background             │
│  • Subsampled: 300/class = 8,700 images                         │
│  • download_dataset.py → data/raw/{A,B,...,space}/              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ANNOTATION LAYER                              │
│                                                                 │
│  auto_annotate.py                                               │
│  • MediaPipe Hands → 21 hand landmarks (x,y,z)                  │
│  • Bounding box = min/max of landmark coords + 20% padding       │
│  • Save YOLO format: cx cy w h (normalised 0-1)                 │
│  • Fallback: full-frame box when MediaPipe finds no hand        │
│                                                                 │
│  Output: data/raw/{class}/{image}.txt sidecar files             │
│                                                                 │
│  Sample annotation (YOLO format):                               │
│    class_id  cx      cy      w       h                          │
│    0         0.7505  0.5499  0.4859  0.6902                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DATASET SPLIT LAYER                           │
│                                                                 │
│  prepare_dataset.py                                             │
│  • Stratified split per class                                   │
│  • 70% train / 20% val / 10% test                               │
│  • Copy images + labels to annotated/{images,labels}/{split}/   │
│                                                                 │
│  Stats: 6,090 train / 1,740 val / 870 test                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TRAINING LAYER                                │
│                                                                 │
│  train.py → YOLOv8n (ultralytics)                               │
│                                                                 │
│  Architecture:                                                  │
│  Input (320×320) → CSPDarknet backbone (C2f blocks)             │
│                 → C2f neck (FPN + PAN)                          │
│                 → Detection head (3 scales)                     │
│                 → NMS → [bbox, class, conf]                     │
│                                                                 │
│  Training config:                                               │
│  • Epochs: 30 (early stop patience=20)                          │
│  • Batch: 32 (CPU-optimised)                                    │
│  • Optimizer: AdamW, cosine LR, warmup=3 epochs                 │
│  • Workers: 0 (CPU, avoids dataloader hang)                     │
│                                                                 │
│  Augmentations:                                                 │
│  • HSV jitter (hue±0.015, sat×0.7, val×0.4)                    │
│  • Rotation ±15°, scale ±30%                                    │
│  • Mosaic (epochs 1-20), light mixup                            │
│  • NO horizontal flip (ASL is handedness-specific!)             │
│                                                                 │
│  Output: models/asl_yolov8_best.pt (6 MB)                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EVALUATION LAYER                              │
│                                                                 │
│  evaluate.py + plot_results.py                                  │
│  • mAP@0.50 = 90.6%     mAP@0.50:0.95 = 75.5%                  │
│  • Precision = 90.5%    Recall = 89.5%    F1 = 90.0%           │
│  • FPS = 45.4 (22 ms/frame on Intel Xeon CPU)                   │
│  • Confusion matrix, per-class AP/F1 charts                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LIVE DEMO LAYER                               │
│                                                                 │
│  demo.py                                                        │
│                                                                 │
│  Frame loop:                                                    │
│  1. Capture frame (OpenCV VideoCapture)                         │
│  2. MediaPipe pre-screen → if no hand detected, skip YOLO       │
│  3. YOLO inference → detections (bbox + class + conf)           │
│  4. Draw bbox, class label, confidence, FPS overlay             │
│  5. Display (cv2.imshow)                                        │
│                                                                 │
│  Keys: Q=quit  S=screenshot  R=record MP4                       │
│  Typical CPU latency: 22-25 ms/frame (~40-45 FPS)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Model Choice — YOLOv8n

| Architecture | mAP@0.50 | FPS (CPU) | Size | Chosen? |
|---|---|---|---|---|
| YOLOv8n | 90.6% | **45 FPS** | 6 MB | ✅ |
| YOLOv8s | ~93% | ~25 FPS | 22 MB | GPU option |
| YOLOv8m | ~95% | ~12 FPS | 52 MB | GPU only |
| Faster R-CNN ResNet50 | ~94% | ~5 FPS | 160 MB | ❌ too slow |
| EfficientDet-D0 | ~91% | ~20 FPS | 15 MB | Alternative |

**Decision:** YOLOv8n provides 90.6% mAP at 45 FPS on CPU — sufficient for a smooth live demo without GPU. The single-stage architecture avoids the region proposal overhead of two-stage detectors.

---

## 4. Annotation Strategy

**Why auto-annotation with MediaPipe?**  
The Kaggle dataset has no bounding box labels. Manual annotation of 8,700 images is infeasible. MediaPipe Hands provides state-of-the-art 21-keypoint hand detection that is fast (< 5 ms/frame), accurate, and deterministic.

**Bounding box formula:**
```
x_min = min(landmark.x for all 21 landmarks) × width
x_max = max(landmark.x for all 21 landmarks) × width
y_min = min(landmark.y for all 21 landmarks) × height
y_max = max(landmark.y for all 21 landmarks) × height
padding = 20% of max(w, h)
```

**Annotation quality:** MediaPipe misses ~5% of images (hands with unusual angles, truncated images, or poor lighting). For these, a full-frame fallback box is used (class still correct).

---

## 5. Validation Design

- **70/20/10 stratified split** per class — ensures all 29 classes are represented in each split
- **No temporal ordering** — ASL images are independent (not a video stream)
- **Test set is held out** until final evaluation; val set used for early stopping

---

## 6. Deployment Constraints

| Constraint | Detail |
|---|---|
| CPU inference | YOLOv8n: 22 ms/frame on Intel Xeon = 45 FPS |
| GPU inference | 5-8 ms/frame on CUDA = 125-200 FPS |
| Model size | 6 MB (fits mobile/edge devices) |
| Webcam latency | <50 ms total (capture + prescreen + YOLO + render) |
| Memory | ~200 MB RAM for model + MediaPipe |
| New signers | Background/lighting variation reduces accuracy; collect custom data via `scripts/collect_data.py` |
| J & Z | Require motion; static frame detection only classifies the final position |

---

## 7. Per-Class Results

| Class | AP@0.50 | Precision | Recall | F1 | Notes |
|-------|---------|-----------|--------|-----|-------|
| A | 99.5% | 88.1% | 100% | 93.7% | |
| B | 99.5% | 100% | 100% | 100% | Perfect |
| C | 79.5% | 74.4% | 80.0% | 77.1% | Confused with O |
| D | 99.5% | 100% | 100% | 100% | Perfect |
| E | 79.5% | 80.0% | 80.0% | 80.0% | |
| F | 99.5% | 100% | 100% | 100% | Perfect |
| G | 99.5% | 100% | 91.1% | 95.4% | |
| H | 99.5% | 100% | 100% | 100% | Perfect |
| I | 79.5% | 85.5% | 80.0% | 82.7% | |
| J | 79.5% | 80.0% | 80.0% | 80.0% | Motion sign |
| K | 99.5% | 82.7% | 100% | 90.5% | |
| M | 75.5% | 66.7% | 80.0% | 72.7% | Similar to N |
| N | 99.5% | 91.7% | 100% | 95.7% | |
| O | 99.5% | 100% | 100% | 100% | Perfect |
| S | 99.5% | 100% | 100% | 100% | Perfect |
| T | 99.5% | 100% | 100% | 100% | Perfect |
| W | 59.5% | 100% | 60.0% | 75.0% | Weakest — similar to V |
| nothing | 99.5% | 100% | 100% | 100% | Perfect |
| space | 39.5% | 50.0% | 60.0% | 54.6% | Weakest — open palm |

---

## 8. Limitations & Improvement Roadmap

| Priority | Improvement | Expected Gain |
|---|---|---|
| High | Collect custom webcam data for W, space, C, M | +5-8% mAP on weak classes |
| High | YOLOv8s on GPU | +2-3% overall mAP |
| Medium | Signer-independent cross-validation | Better real-world accuracy estimate |
| Medium | Video-based J/Z detection (optical flow) | Enables full 26-letter alphabet |
| Medium | Background diversity augmentation | Improves real-world robustness |
| Low | ONNX export for mobile/edge deployment | 2× faster on mobile |
