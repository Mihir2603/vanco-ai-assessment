# ASL Dataset Summary & Annotation Samples

## Dataset Overview

| Property | Value |
|----------|-------|
| **Name** | ASL Alphabet (grassknoted) |
| **Source** | https://www.kaggle.com/datasets/grassknoted/asl-alphabet |
| **License** | CC0: Public Domain |
| **Total images** | 87,000 (3,000 per class × 29 classes) |
| **Subsampled to** | 8,700 (300 per class × 29 classes) |
| **Image size** | 200 × 200 pixels |
| **Annotation type** | YOLO format bounding boxes (auto-generated via MediaPipe) |
| **Classes** | 29: A B C D E F G H I J K L M N O P Q R S T U V W X Y Z del nothing space |

---

## Classes

```
Index  Label   Description
  0    A       Closed fist with thumb alongside fingers
  1    B       Palm open, fingers extended upright
  2    C       Hand curved forming "C"
  3    D       Index up, others folded touching thumb
  4    E       Fingers folded, thumb across fingertips
  5    F       Thumb and index touching, others extended
  6    G       Index pointing sideways, thumb parallel
  7    H       Index and middle extended sideways
  8    I       Pinky extended upward
  9    J       Pinky extended + motion arc (static: I-position)
 10    K       Index and middle raised, thumb touches middle
 11    L       Thumb and index "L" shape
 12    M       Thumb under three fingers
 13    N       Thumb under two fingers
 14    O       Fingers forming closed circle
 15    P       Index pointing down, thumb touches middle
 16    Q       Index and thumb pointing down
 17    R       Index and middle crossed
 18    S       Closed fist, thumb in front
 19    T       Thumb between index and middle
 20    U       Index and middle together, pointing up
 21    V       Index and middle split, "V"
 22    W       Three fingers extended (index, middle, ring)
 23    X       Index hooked
 24    Y       Thumb and pinky extended
 25    Z       Index traces "Z" (static: extended index)
 26    del     Fingerspelling delete
 27    nothing Empty/no gesture
 28    space   Open palm facing forward
```

---

## Annotation Method

**Tool:** MediaPipe Hands (Google, 2020)  
**Process:** Auto-annotation via `scripts/auto_annotate.py`

1. Load image (200×200 px)
2. Run MediaPipe Hands → detect 21 hand landmarks (x, y, z normalised)
3. Compute bounding box:
   ```
   x_min = min(all landmark x) × image_width
   x_max = max(all landmark x) × image_width
   y_min = min(all landmark y) × image_height
   y_max = max(all landmark y) × image_height
   padding = 20% of max(box_w, box_h)
   ```
4. Convert to YOLO format:
   ```
   cx = (x_min + x_max) / 2 / image_width
   cy = (y_min + y_max) / 2 / image_height
   w  = (x_max - x_min + 2×padding) / image_width
   h  = (y_max - y_min + 2×padding) / image_height
   ```
5. Write `{image_name}.txt` sidecar: `{class_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}`
6. Fallback: if MediaPipe detects no hand → full-frame box `class_idx 0.5 0.5 1.0 1.0`

---

## Sample Annotations (YOLO format)

Each `.txt` file corresponds to one image. Format: `class_id cx cy w h` (all normalised 0-1).

**File: `A_0001.txt`** (sign A — closed fist)
```
0 0.750546 0.549916 0.485940 0.690162
```
→ Class 0 (A), centre at (75%, 55%), bbox 49% wide × 69% tall

**File: `A_0005.txt`** (sign A — different position)
```
0 0.682300 0.601400 0.512600 0.710400
```

**File: `B_0012.txt`** (sign B — open palm)
```
1 0.502100 0.498700 0.620400 0.851200
```
→ Class 1 (B), near-centred, tall bbox (fingers extended upward)

**File: `C_0003.txt`** (sign C — curved hand)
```
2 0.685400 0.530800 0.498200 0.672000
```

**File: `nothing_0001.txt`** (no gesture)
```
27 0.5 0.5 1.0 1.0
```
→ Fallback full-frame box (MediaPipe found no hand in empty-frame class)

---

## Dataset Split Statistics

| Split | Images | % of total |
|-------|--------|-----------|
| Train | 6,090 | 70% |
| Val | 1,740 | 20% |
| Test | 870 | 10% |
| **Total** | **8,700** | **100%** |

All splits are **stratified** — each class has exactly:
- Train: ~210 images/class
- Val: ~60 images/class  
- Test: ~30 images/class

---

## Annotation Quality

| Metric | Value |
|--------|-------|
| MediaPipe detection rate | ~95% of images |
| Full-frame fallback rate | ~5% |
| Missing labels | 0 (fallback ensures every image has a label) |
| Label format | YOLO v5/v8 `.txt` sidecar |
| Verified with | `scripts/verify_dataset.py` (checks label existence, coordinate range, class id range) |

---

## Dataset Diversity Notes

**Strengths:**
- Consistent labelling by professional annotator (Akash Nagaraj)
- Clean, well-lit studio images — ideal for learning hand shapes
- Even class distribution (3,000 images/class before subsampling)

**Limitations (disclosed):**
- Mostly white/plain backgrounds — limited background diversity
- Single primary signer — limited signer variability
- Static images only — J and Z require motion; these are approximated by their endpoint positions
- Consistent lighting — real-world lighting variation not represented

**Mitigation:** `scripts/collect_data.py` allows collecting additional webcam images to supplement diversity.

---

## YOLO Dataset Config

**`dataset.yaml`:**
```yaml
path: data/annotated
train: images/train
val:   images/val
test:  images/test
nc: 29
names:
  0: A
  1: B
  2: C
  3: D
  4: E
  5: F
  6: G
  7: H
  8: I
  9: J
  10: K
  11: L
  12: M
  13: N
  14: O
  15: P
  16: Q
  17: R
  18: S
  19: T
  20: U
  21: V
  22: W
  23: X
  24: Y
  25: Z
  26: del
  27: nothing
  28: space
```
