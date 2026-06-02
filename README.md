# VANCO AI Assessment — Final Submission

**Candidate:** AI Solution Architect Assessment  
**Completed use cases:** 3 of 3

---

## Use Cases at a Glance

| # | Use Case | Folder | Key Metric | Live Demo |
|---|----------|--------|-----------|-----------|
| 1 | Grocery Sales Forecasting | `grocery_forecasting/` | Kaggle RMSLE **0.42234** | ❌ (Kaggle submission) |
| 2 | ASL Alphabet Detection | `asl_detector/` | mAP@0.50 **90.6%**, 45 FPS | ✅ Webcam |
| 3 | NCERT Physics RAG | `rag_physics/` | Hybrid retrieval, grounded answers | ✅ Web UI |

---

## Repository Structure

```
Vanco/
├── README.md                          ← This file
├── REPORT.md                          ← Summary report (approach, results, limitations)
├── grocery_forecasting/               ← Use Case 1
│   ├── notebooks/01_grocery_sales_forecasting.ipynb
│   ├── src/                           ← Modular pipeline (5 modules)
│   ├── train.py                       ← Standalone training script
│   ├── models/lgbm_final.txt          ← Pre-trained LightGBM (22 MB)
│   ├── submissions/submission_lgbm.csv← Kaggle submission (0.42234)
│   ├── reports/architecture_report.md ← Full architecture + results
│   ├── reports/feature_importance.png
│   ├── reports/rmsle_by_family.csv
│   ├── reports/rmsle_by_store.csv
│   └── README.md
├── asl_detector/                      ← Use Case 2
│   ├── demo.py                        ← Live webcam demo
│   ├── scripts/                       ← Full pipeline (7 scripts)
│   ├── models/asl_yolov8_best.pt      ← Trained YOLOv8n weights
│   ├── results/metrics/summary.json   ← mAP, FPS metrics
│   ├── results/plots/                 ← Confusion matrix, per-class charts
│   ├── reports/architecture_report.md ← Architecture diagram
│   ├── reports/dataset_summary.md     ← Dataset + annotation samples
│   └── README.md
└── rag_physics/                       ← Use Case 3
    ├── app/                           ← FastAPI backend (7 modules)
    ├── frontend/index.html            ← Chat UI
    ├── scripts/ingest.py              ← Build all indexes
    ├── indexes/                       ← Pre-built ChromaDB + BM25 + KG
    ├── reports/architecture_report.md ← Architecture diagram
    └── README.md
```

---

## Quick Start — All Three Use Cases

### Prerequisites

```bash
cd /home/intel/Vanco
source .venv/bin/activate
```

---

### Use Case 1 — Grocery Sales Forecasting

**Run the training notebook:**
```bash
cd grocery_forecasting/notebooks
jupyter notebook 01_grocery_sales_forecasting.ipynb
```

**Re-train the full model from scratch:**
```bash
cd grocery_forecasting
python train.py
```

**Regenerate Kaggle submission (uses pre-trained model):**
```bash
cd grocery_forecasting
python -c "
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, lightgbm as lgb
from src.data_loader import load_raw, build_master, combine_train_test
from src.feature_engineering import build_features

dfs = load_raw('data/raw')
train_merged, test_merged = build_master(dfs)
combined = combine_train_test(train_merged, test_merged)
featured = build_features(combined)
test_feat = featured[featured['split']=='test'].copy()

model = lgb.Booster(model_file='models/lgbm_final.txt')
feat_cols = model.feature_name()
X_test = test_feat[feat_cols].fillna(0)
preds = np.maximum(np.expm1(model.predict(X_test)), 0)
sub = test_feat[['id']].copy()
sub['sales'] = preds
test_raw = dfs['test'][['id']].copy()
test_raw = test_raw.merge(sub, on='id', how='left').fillna(0)
test_raw.to_csv('submissions/submission_lgbm.csv', index=False)
print('Saved submissions/submission_lgbm.csv')
"
```

**Submit to Kaggle:**
```bash
cd grocery_forecasting
kaggle competitions submit \
  -c store-sales-time-series-forecasting \
  -f submissions/submission_lgbm.csv \
  -m "LightGBM 1953 trees, 64 features"
```

**Kaggle Public Score: 0.42234**  
See `reports/architecture_report.md` for full pipeline diagram and results.

---

### Use Case 2 — ASL Alphabet Detection (Live Demo)

**Run the live webcam demo (pre-trained model):**
```bash
cd asl_detector
python demo.py
# Keys: Q=quit, S=screenshot, R=record
```

**Re-train from scratch:**
```bash
cd asl_detector

# Step 1: Download dataset (requires ~/.kaggle/kaggle.json)
python scripts/download_dataset.py --max_per_class 300

# Step 2: Auto-annotate with MediaPipe hand landmarks
python scripts/auto_annotate.py --padding 0.20

# Step 3: Build train/val/test split
python scripts/prepare_dataset.py

# Step 4: Train YOLOv8n
python scripts/train.py --model yolov8n --epochs 30 --batch 32 --imgsz 320 --workers 0

# Step 5: Evaluate
python scripts/evaluate.py --split test

# Step 6: Generate plots
python scripts/plot_results.py
```

**Results: mAP@0.50 = 90.6%, 45 FPS on CPU**  
See `reports/architecture_report.md` and `reports/dataset_summary.md`.

---

### Use Case 3 — NCERT Physics RAG (Live Demo)

**Start the server (indexes already built):**
```bash
cd rag_physics
uvicorn app.main:app --host 0.0.0.0 --port 8000
# Open browser: http://localhost:8000
```

**Ask a question via API:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Gauss law?", "show_evidence": true}'
```

**Rebuild all indexes from scratch:**
```bash
cd rag_physics
# Place NCERT PDF at: data/pdf/ncert_physics_12_part1.pdf
python scripts/ingest.py
```

**Run diagnostics:**
```bash
curl -X POST http://localhost:8000/diagnostics \
  -H "Content-Type: application/json" \
  -d '{"question": "explain Faraday law of induction"}'
```

See `reports/architecture_report.md` for full hybrid retrieval architecture.

---

## Architecture Diagrams

| Use Case | Diagram Location |
|----------|-----------------|
| Grocery Forecasting | `grocery_forecasting/reports/architecture_report.md` |
| ASL Detection | `asl_detector/reports/architecture_report.md` |
| Physics RAG | `rag_physics/reports/architecture_report.md` |

---

## Results Summary

### Use Case 1 — Grocery Forecasting

| Metric | Value |
|--------|-------|
| Baseline RMSLE (Seasonal Naïve, 5-fold CV) | 0.6558 |
| LightGBM Holdout RMSLE (last 16 days) | **0.3897** |
| Improvement over baseline | **40.6%** |
| **Kaggle Public Leaderboard** | **0.42234** |

### Use Case 2 — ASL Detection

| Metric | Value |
|--------|-------|
| mAP@0.50 | **90.6%** |
| mAP@0.50:0.95 | 75.5% |
| Precision | 90.5% |
| Recall | 89.5% |
| F1 | 90.0% |
| Inference FPS (CPU) | **45.4 FPS** |
| Latency | 22 ms/frame |

### Use Case 3 — RAG

| Property | Value |
|----------|-------|
| Retrieval strategy | Hybrid: Vector + BM25 + Knowledge Graph |
| Fusion | Reciprocal Rank Fusion (RRF) |
| LLM | Groq LLaMA-3 8B |
| Response latency | ~2-3 s (API round-trip) |
| Knowledge source | NCERT Class 12 Physics Part 1 |

---

## Dependencies

All three use cases share the same virtual environment:

```bash
source /home/intel/Vanco/.venv/bin/activate
```

Individual `requirements.txt` files are in each subfolder.  
Key packages:
- `lightgbm`, `pandas`, `scikit-learn` — Grocery Forecasting  
- `ultralytics` (YOLOv8), `mediapipe`, `opencv-python` — ASL Detection  
- `fastapi`, `chromadb`, `sentence-transformers`, `rank-bm25`, `networkx`, `groq` — RAG

---

## External Resources Disclosed

| Resource | Used in | Purpose |
|----------|---------|---------|
| [Kaggle ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) | ASL | Training images (87K images, 29 classes) |
| [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) | ASL | Pre-trained backbone (yolov8n.pt) |
| [MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) | ASL | Auto-annotation of bounding boxes |
| [Kaggle Store Sales dataset](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) | Grocery | Competition dataset |
| [LightGBM](https://lightgbm.readthedocs.io/) | Grocery | Gradient boosting model |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | RAG | Sentence embeddings |
| [Groq LLaMA-3 8B](https://console.groq.com) | RAG | Answer generation (free tier) |
| NCERT Class 12 Physics Part 1 PDF | RAG | Knowledge source |
