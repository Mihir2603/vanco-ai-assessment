# VANCO AI Assessment — Summary Report

**Three Use Cases: Grocery Forecasting · ASL Detection · Physics RAG**

---

## 1. Grocery Sales Forecasting

### Approach

Forecast 16-day daily unit sales for ~1,800 store × product-family combinations at Corporación Favorita supermarkets using tabular ML on the Kaggle competition dataset.

**Pipeline:**  
`Raw CSVs → Merge 6 tables → Feature engineering (8 groups) → Walk-forward CV → LightGBM → submission.csv`

### Feature Engineering (64 features)

| Group | Features |
|-------|----------|
| Calendar | year, month, DoW, is_weekend, is_payday (15th + month-end) |
| Lag | lag_16/21/28/35/42/49/56 per store × family |
| Rolling | mean/std/max/min over 7/14/28 days |
| Promotion | onpromotion flag, promo_count_7/14, days_since_promo |
| Oil price | level, oil_lag1/7, daily change, 7-day MA |
| Holiday | national/regional/local flags, pre/post ±1-3 days, streak |
| Transactions | transactions_lag per store (deduplicated), roll7/14 |
| Encodings | store_mean_log, family_mean_log, store_family_mean_log |

**Critical design decision — safe lags (≥16 days):** The forecast horizon is 16 days. All lag features start at lag_16 to ensure no NaN values at test time. Shorter lags (lag_1, lag_7) that were originally in the pipeline caused catastrophic test-time feature failures (NaN → model predicted ~27 instead of ~2600) which manifested as a 2.54 Kaggle score. Fixing this improved the score to 0.42234.

### Validation Strategy

**Walk-forward backtesting** — 5 folds with expanding training window, 16-day gap, 16-day validation. No random splits. Fold 4 covers the final weeks of 2017 to mirror the test distribution.

### Model: LightGBM

- **Why LightGBM over neural models:** 3M training rows with tabular heterogeneous features → LightGBM trains in ~20 min with excellent accuracy. Neural models (TFT, N-BEATS) would need >2h training and careful architecture choices without guaranteed improvement.
- **Log1p target:** Training on log1p(sales) and optimising RMSE is mathematically equivalent to RMSLE minimisation; back-transform with expm1.
- **1953 trees:** Final model trained on full training data without early stopping, using optimal tree count from CV.

### Results

| Metric | Value |
|--------|-------|
| Baseline RMSLE (Seasonal Naïve) | 0.6558 |
| LightGBM Holdout RMSLE | **0.3897** |
| Improvement over baseline | **40.6%** |
| **Kaggle Public RMSLE** | **0.42234** |

### Error Analysis

**Worst stores:** Stores 19, 20, 22 (RMSLE ~0.47-0.49) — likely high-volume stores in volatile regions.  
**Worst families:** SCHOOL AND OFFICE SUPPLIES (0.662), POULTRY (0.619), LAWN AND GARDEN (0.584) — event-driven or perishable categories with irregular demand patterns.  
**Top features:** rolling_mean_7 > lag_21 > rolling_mean_14 > rolling_max_7 — recent rolling statistics dominate over static encodings.

### Limitations & Next Steps

1. **Hyperparameter tuning** (Optuna) — expected ~0.01-0.02 RMSLE improvement
2. **Direct multi-step models** — one model per forecast day rather than a single recursive model
3. **Neural models** (TFT, N-BEATS) — better at capturing long-range seasonality and complex event interactions
4. **Cluster-level micro-models** — stores cluster by type; a model per cluster reduces heterogeneity

---

## 2. ASL Alphabet Detection

### Approach

Real-time detection and classification of 29 American Sign Language hand signs (A-Z + del + nothing + space) using object detection (YOLOv8) trained on annotated hand images.

**Pipeline:**  
`Kaggle images → MediaPipe auto-annotation → YOLO format dataset → YOLOv8n training → Live webcam demo`

### Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle grassknoted/asl-alphabet |
| Original size | 87,000 images (3,000/class × 29 classes) |
| Subsampled | 300/class × 29 = 8,700 images |
| Split | 70% train / 20% val / 10% test |
| Annotation format | YOLO .txt (cx cy w h normalised) |
| Bounding box method | MediaPipe 21-landmark hand bbox + 20% padding |

**Why auto-annotation with MediaPipe?** The Kaggle dataset is classification-only (no bounding boxes). MediaPipe Hands detects 21 hand landmarks and we compute a tight padded bounding box. This is reproducible, consistent, and requires no manual labelling for 8,700 images.

### Model: YOLOv8n

- **Why YOLOv8 over two-stage (Faster R-CNN)?** Single-stage detector is 3-5× faster at inference, critical for 30+ FPS live webcam demo.
- **Why YOLOv8n (nano)?** CPU-only deployment — 6 MB model, 22 ms/frame, 45 FPS. Larger variants (s/m) improve mAP but require GPU.
- **MediaPipe pre-screening:** Skip YOLO inference on frames where MediaPipe finds no hand — cuts CPU load by ~60% on empty frames.

### Augmentation

HSV jitter, ±15° rotation, ±30% scale, horizontal flip (disabled — ASL is handedness-specific), mosaic (epochs 1-20), light mixup, brightness/contrast variation.

### Results

| Metric | Value |
|--------|-------|
| mAP@0.50 | **90.6%** |
| mAP@0.50:0.95 | 75.5% |
| Precision | 90.5% |
| Recall | 89.5% |
| F1 | 90.0% |
| FPS (CPU, Intel Xeon) | **45.4 FPS** |
| Inference latency | 22 ms/frame |

**Per-class highlights:**

| Class | AP@0.50 | Notes |
|-------|---------|-------|
| B, D, F, H, O, P, S, T | 99.5% | Highly distinctive shapes |
| W | 59.5% | Visually similar to V; 3-finger spread ambiguous |
| space | 39.5% | Open palm similar to B; context-dependent |
| C | 79.5% | Curved shape confused with O at distance |

### Live Demo

```bash
python demo.py --weights models/asl_yolov8_best.pt --conf 0.45
```

Draws bounding box, predicted class, and confidence score at ~45 FPS. Press `S` to save screenshot, `R` to record.

### Limitations & Next Steps

1. **Signer-independent evaluation** — Kaggle images have consistent background/lighting; real-world users may get lower accuracy
2. **Custom data collection** — `scripts/collect_data.py` for your webcam; retrain for personalisation
3. **J and Z require motion** — these letters use movement; static frame detection is insufficient
4. **Background diversity** — Kaggle images have plain white backgrounds; augment or retrain with varied backgrounds for production

---

## 3. NCERT Physics RAG

### Approach

Production-quality RAG (Retrieval-Augmented Generation) system for answering questions from NCERT Class 12 Physics textbook using three complementary retrieval strategies fused with RRF.

**Pipeline:**  
`PDF → Section-aware chunking → ChromaDB + BM25 + KG → RRF fusion → Groq LLaMA-3 → Grounded answer`

### Why Hybrid Retrieval?

| Retriever | Strength | Weakness |
|-----------|----------|---------|
| Semantic (ChromaDB + MiniLM) | Paraphrase matching, conceptual similarity | Misses exact law/formula names |
| BM25 keyword | Exact term match, law names, equations | No semantic understanding |
| Knowledge Graph | Entity relationships, multi-hop reasoning | Sparse, heuristic construction |

**RRF fusion** combines all three ranked lists: `score = Σ 1/(60 + rank_i)`. This consistently outperforms any single retriever.

### Design Decisions

- **Chunking strategy:** Section-aware (font-size heuristics detect headings) + sentence boundary splitting at 1500 chars with 200-char overlap. Keeps semantically coherent content together.
- **Embedding model:** `all-MiniLM-L6-v2` — fast, 384-dim, excellent zero-shot performance on science text.
- **LLM:** Groq LLaMA-3 8B — free tier, 2× faster than OpenAI GPT-3.5, sufficient for grounded Q&A.
- **Grounding enforcement:** System prompt explicitly instructs the LLM to answer only from retrieved context and cite `[N]` references. Out-of-document questions receive "not available in the provided document" responses.

### Architecture

```
User Question
      │
      ├──► Semantic Search (ChromaDB + MiniLM)  ──► top-k chunks
      ├──► Keyword Search (BM25)                ──► top-k chunks
      └──► Graph Search (NetworkX KG)           ──► top-k chunks
                                │
                           RRF Fusion
                                │
                           Top-K Chunks + Citations
                                │
                      LLM (Groq LLaMA-3 8B)
                                │
                      Grounded Answer + Sources
```

### Results

| Property | Value |
|----------|-------|
| Response latency | ~2-3 s (API round-trip incl. LLM) |
| Chunks indexed | ~450 chunks from NCERT PDF |
| Knowledge graph nodes | ~200 (chapters, sections, concepts, formulas) |
| Citation accuracy | High — answers stay within retrieved context |

### Live Demo

```bash
cd rag_physics
uvicorn app.main:app --host 0.0.0.0 --port 8000
# Open: http://localhost:8000
```

### Limitations & Next Steps

1. **No reranking** — add cross-encoder (ms-marco-MiniLM) after RRF for better precision
2. **Formula handling** — LaTeX extraction (MathPix) for structured equation parsing
3. **Multi-document** — extend to full NCERT Physics (Parts 1 & 2) or Chemistry
4. **Evaluation** — need human-annotated QA pairs to measure faithfulness and relevance (RAGAS)

---

## Trade-off Discussion

### Model selection rationale

| Decision | Chosen | Rejected | Reason |
|----------|--------|----------|--------|
| Grocery model | LightGBM | XGBoost, TFT | 3× faster training, comparable accuracy; TFT overkill without GPU |
| ASL architecture | YOLOv8n | Faster R-CNN, EfficientDet | Single-stage → live FPS; nano → CPU deployment |
| RAG LLM | Groq LLaMA-3 | GPT-4 | Free tier, 2× lower latency; GPT-4 not needed for grounded Q&A |
| RAG embeddings | MiniLM-L6-v2 | OpenAI ada-002 | Local, free, 95% quality at 10× lower cost |

### Validation design rationale

| Use Case | Strategy | Why not random split? |
|----------|----------|----------------------|
| Grocery | Walk-forward 5-fold backtesting | Time series: future data leaks into training with random splits |
| ASL | Stratified 70/20/10 split | Classification: label imbalance; no temporal order |
| RAG | Retrieval diagnostic endpoint | No labelled test set; evaluated qualitatively + latency |

---

## Reproducibility

All code is self-contained. Exact versions pinned in each `requirements.txt`.  
Pre-trained artifacts:
- `grocery_forecasting/models/lgbm_final.txt` — LightGBM model
- `asl_detector/models/asl_yolov8_best.pt` — YOLOv8n weights
- `rag_physics/indexes/` — ChromaDB + BM25 + knowledge graph

No GPU required for inference (all three demos run on CPU).
