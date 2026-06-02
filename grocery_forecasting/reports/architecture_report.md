# Architecture Report – Store Sales Forecasting Pipeline

## 1. Problem Statement

Forecast daily unit sales for ~1,800 store × product-family combinations at
Corporación Favorita supermarkets (Ecuador) over a 16-day horizon.

**Dataset tables**

| Table | Rows | Description |
|---|---|---|
| train.csv | 3 million | Historical daily sales |
| test.csv | 28,512 | Target rows to forecast |
| stores.csv | 54 | Store metadata (type, cluster, city, state) |
| oil.csv | 1,218 | Daily WTI oil prices (Ecuador oil-dependent economy) |
| holidays_events.csv | 350 | National/regional/local holidays & events |
| transactions.csv | 83,488 | Daily store transaction counts |

---

## 2. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   DATA INGESTION LAYER                          │
│  train.csv  test.csv  stores.csv  oil.csv  holidays.csv  txn.csv│
│  • Parse dates          • Fill oil NaN (linear interp)          │
│  • Clip negative sales  • Resolve holiday locale hierarchy       │
└────────────────────────┬────────────────────────────────────────┘
                         │  Merge all tables on [date, store_nbr]
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING LAYER                      │
│                                                                 │
│  Calendar      year/month/week/DoW/DoY/quarter/is_weekend       │
│                is_payday (15th + month-end) · Fourier terms     │
│                                                                 │
│  Lag features  lag_1/7/14/21/28/35/42  (grouped by store×fam)  │
│                                                                 │
│  Rolling       mean/std/max/min over 7/14/28 days               │
│                (computed on shift(1) to prevent leakage)        │
│                                                                 │
│  Promotion     onpromotion flag · promo_count_7/14              │
│                days_since_promo                                 │
│                                                                 │
│  Oil           level · lag_1/7 · daily change · 7-day MA        │
│                                                                 │
│  Holiday       national / regional / local flags                │
│                pre_holiday ±1/2/3 · post_holiday ±1/2/3         │
│                holiday_streak (consecutive days)                │
│                                                                 │
│  Transactions  txn_lag1 · txn_roll7/14 per store                │
│                                                                 │
│  Encodings     store_mean_log_sales · family_mean_log_sales      │
│                store×family_mean_log_sales (train-only mean)     │
│                Label-encode: family, type, city, state           │
└────────────────────────┬────────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
┌─────────────────────┐  ┌──────────────────────────────────────┐
│  VALIDATION LAYER   │  │       TRAINING LAYER                  │
│                     │  │                                        │
│  Walk-Forward CV    │  │  Step 1: Seasonal Naïve Baseline       │
│  5 folds            │  │          (same day last week)          │
│  GAP  = 16 days     │  │                                        │
│  VAL  = 16 days     │  │  Step 2: LightGBM Forecaster           │
│  Expanding window   │  │          log1p(sales) target           │
│                     │  │          2000 trees · lr=0.05          │
│  Metrics:           │  │          num_leaves=127                │
│  • RMSLE (Kaggle)   │  │          feature_fraction=0.8          │
│  • MAE              │  │          early stopping=50 rounds      │
│  • SMAPE            │  │                                        │
│                     │  │  Step 3: (Optional) Ensemble           │
│  Leakage check:     │  │          blend LGB + Naïve             │
│  warn if lag < 16   │  └──────────────────────────────────────┘
└─────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   INFERENCE LAYER                               │
│  • Retrain final model on 100% of training data                 │
│  • No early stopping (use optimal n_estimators from CV)         │
│  • Predict test set → clip negatives → submission.csv           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ERROR ANALYSIS LAYER                          │
│  • By store_nbr    → RMSLE, MAE ranking                         │
│  • By family       → worst performing product lines             │
│  • By holiday      → holiday vs non-holiday performance         │
│  • By promotion    → promo vs non-promo performance             │
│  • By day-of-week  → weekly error pattern                       │
│  • Residuals over time → bias detection                         │
│  • Feature importance (Gain) + SHAP values                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Validation Strategy

### Why walk-forward (no random split)?

Grocery sales exhibit strong temporal autocorrelation: tomorrow's demand is
correlated with recent history, seasonal patterns, and external events.  
A random 80/20 split would allow the model to observe future data during
training, inflating metrics and producing overly optimistic results.

### Walk-forward configuration

```
Fold k:  [──────────── TRAIN (expanding) ────────────] [GAP=16d] [VAL=16d]
```

- **Expanding window**: each fold trains on all data up to the cut-off date
- **Gap = 16 days**: matches the competition horizon; ensures the model is
  never evaluated on data within its own prediction window
- **Validation window = 16 days**: mirrors the test set length exactly
- **5 folds**: provides stable mean RMSLE and fold-variance estimate

---

## 4. Feature Strategy – Trade-off Discussion

| Feature | Why included | Risk / mitigation |
|---|---|---|
| lag_7 | Strong weekly seasonality in grocery | Not available for days 1–6 of horizon; use lag_14+ for safer features |
| rolling_mean_28 | Smoothed recent trend | Shift(1) prevents leakage |
| is_payday | Ecuador payroll cycle drives spending spikes | Hardcoded for 15th + month-end |
| pre_holiday_3 | Demand anticipation effect | Uses shift(-3) which is future-looking – valid since holidays are known in advance |
| store_mean_log_sales | Store-level intercept | Computed from train only; applied to test |
| oil price | Ecuador economy is oil-dependent | Interpolated over weekends; carries 1-day lag |
| transactions | Footfall proxy | Missing from test set; use rolling average |

---

## 5. Modeling Choices

### LightGBM (primary)
- **Why**: Handles tabular data with heterogeneous feature types; built-in
  support for categorical features; fast training; SOTA on Kaggle tabular tasks
- **Log-transform target**: Training on log1p(sales) is mathematically
  equivalent to minimising RMSLE; back-transform with expm1
- **Trade-off vs neural models**: LightGBM trains faster and is more
  interpretable; neural models (TFT, N-BEATS) may capture long-range
  dependencies better but require more tuning and infrastructure

### Seasonal Naïve (baseline)
- Simple, fast, interpretable lower bound
- Competitive for stable weekly patterns; fails during holidays/promotions

### Ensemble (optional)
- Weighted blend of LightGBM + Naïve
- Weights proportional to inverse CV RMSLE

---

## 6. Kaggle Metric – RMSLE

```
RMSLE = sqrt( (1/n) Σ (log(1 + ŷᵢ) - log(1 + yᵢ))² )
```

**Properties:**
- Penalises relative errors (not absolute), good for skewed distributions
- Under-prediction penalised more than over-prediction (asymmetric)
- Clips negative predictions to 0 before evaluation
- Does not break for yᵢ = 0 (unlike MAPE)

**Business metrics to track in parallel:**
- Weighted MAPE by revenue share
- Inventory days-of-supply error
- Fill-rate at ≥95% service level

---

## 7. Results

| Metric | Value |
|---|---|
| Baseline CV RMSLE (Seasonal Naïve, 5-fold) | 0.6558 |
| LightGBM Holdout RMSLE (last 16 days of train) | **0.3897** |
| Improvement over baseline | **40.6%** |
| LightGBM Holdout MAE | 59.72 |
| **Kaggle Public Leaderboard RMSLE** | **0.42234** |

### Worst 5 Stores (Holdout RMSLE)

| Store | RMSLE | MAE |
|---|---|---|
| 19 | 0.487 | 34.2 |
| 20 | 0.476 | 103.6 |
| 22 | 0.465 | 49.5 |
| 14 | 0.453 | 50.7 |
| 26 | 0.451 | 29.6 |

### Worst 5 Families (Holdout RMSLE)

| Family Code | RMSLE | Notes |
|---|---|---|
| 31 (SCHOOL AND OFFICE SUPPLIES) | 0.662 | Highly seasonal/event-driven |
| 21 (POULTRY) | 0.619 | Volatile perishable |
| 13 (LAWN AND GARDEN) | 0.584 | Seasonal |
| 6 (BOOKS) | 0.542 | Low-volume, lumpy demand |
| 14 (LINGERIE) | 0.527 | Low-volume |

### Top 5 Features (by LightGBM Gain)

| Feature | Gain |
|---|---|
| rolling_mean_7 | 1.07 × 10⁸ |
| lag_21 | 2.32 × 10⁷ |
| rolling_mean_14 | 1.53 × 10⁷ |
| rolling_max_7 | 8.27 × 10⁶ |
| lag_56 | 3.77 × 10⁶ |

---

## 8. Limitations & Improvement Roadmap

| Priority | Improvement | Expected Gain |
|---|---|---|
| High | Optuna hyperparameter optimisation | ~0.01–0.02 RMSLE |
| High | Direct multi-step (one model per day 1–16) | Better horizon-specific accuracy |
| Medium | TFT / N-BEATS neural model | Long-range pattern capture |
| Medium | Store-cluster micro-models | Reduces heterogeneity |
| Medium | SHAP-driven feature pruning | Reduces overfitting |
| Low | Conformal prediction intervals | Uncertainty quantification |
| Low | Weather, sports events enrichment | Marginal external signal |
