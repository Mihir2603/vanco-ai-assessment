# Grocery Sales Forecasting – Corporación Favorita

> **Kaggle Competition:** [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)  
> **Metric:** RMSLE | **Horizon:** 16 days

---

## Repository Structure

```
grocery_forecasting/
├── notebooks/
│   └── 01_grocery_sales_forecasting.ipynb   ← Main notebook (EDA + train + inference)
├── src/
│   ├── data_loader.py          ← Load & merge all 6 Kaggle tables
│   ├── feature_engineering.py  ← Full feature pipeline (8 feature groups)
│   ├── validation.py           ← Walk-forward backtesting, metrics
│   ├── models.py               ← Baseline, LightGBM, Ensemble
│   └── error_analysis.py       ← Breakdown & visualisation utilities
├── configs/
│   └── config.yaml             ← All hyperparameters & paths
├── reports/
│   └── architecture_report.md  ← Full architecture & design decisions
├── data/
│   └── raw/                    ← Place Kaggle CSVs here (see Setup)
├── submissions/                ← Generated submission files
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download data
```bash
# Requires Kaggle API credentials (~/.kaggle/kaggle.json)
kaggle competitions download -c store-sales-time-series-forecasting -p data/raw
cd data/raw && unzip store-sales-time-series-forecasting.zip
```

### 3. Run the notebook
```bash
cd notebooks
jupyter notebook 01_grocery_sales_forecasting.ipynb
```

---

## Pipeline Overview

```
Raw CSVs → Merge → Feature Engineering → Walk-Forward CV → LightGBM → submission.csv
```

See [`reports/architecture_report.md`](reports/architecture_report.md) for the full diagram.

---

## Feature Engineering (8 groups)

| Group | Features |
|---|---|
| Calendar | year, month, week, day_of_week, is_weekend, is_payday, Fourier terms |
| Lag | lag_1/7/14/21/28/35/42 (per store × family) |
| Rolling | mean/std/max/min over 7/14/28 days |
| Promotion | onpromotion, promo_count_7/14, days_since_promo |
| Oil price | level, lag_1/7, daily change, 7-day MA |
| Holiday | national/regional/local flags, pre/post ±1–3 days, holiday streak |
| Transactions | footfall lag + rolling per store |
| Encodings | store/family/store×family target-encoded mean log-sales |

---

## Validation Design

**Walk-forward backtesting** – 5 folds, each with:
- Expanding training window
- 16-day gap (= competition horizon) between train end and val start
- 16-day validation window (mirrors test set)

No random splits. Temporal order always respected.

---

## Models

| Model | Holdout RMSLE | Kaggle Public | Notes |
|---|---|---|---|
| Seasonal Naïve | 0.6558 (5-fold CV) | — | Baseline – same day last week |
| LightGBM (holdout) | 0.3897 | — | 16-day holdout, 2000 trees |
| LightGBM (full train) | — | **0.42234** | Trained on all data, submitted |

---

## Error Analysis

Breakdowns produced:
- By **store** – worst 20 stores ranked by RMSLE
- By **product family** – worst-performing categories
- By **holiday** – holiday vs non-holiday error
- By **promotion** – promo vs non-promo error
- By **day of week** – weekly error pattern
- **Residual plot** over time – bias detection
- **Feature importance** (Gain) + SHAP summary

---

## Results

| Metric | Value |
|---|---|
| Baseline CV RMSLE (Seasonal Naïve, 5-fold) | 0.6558 |
| LightGBM Holdout RMSLE (16-day holdout) | **0.3897** |
| Improvement over baseline | **40.6%** |
| LightGBM Holdout MAE | 59.72 |
| **Kaggle Public Leaderboard Score (RMSLE)** | **0.42234** |



- Lag features < 16 days are unavailable at test time for early forecast days
- LightGBM cannot extrapolate beyond training distribution
- No hyperparameter tuning (Optuna) in base version
- No prediction intervals

See [`reports/architecture_report.md`](reports/architecture_report.md) for full improvement roadmap.
