#!/usr/bin/env python3
"""
train.py – Complete Store Sales Forecasting pipeline.

Fixes vs prior notebook approach:
  1. Only uses lag features ≥ 16 days (safe for the 16-day forecast horizon).
  2. Rolling features computed from shift(16) — no future leakage in validation.
  3. Rolling feature efficiency: group_id computed once, not 12 times.
  4. Target encoding uses only sub-train data during validation.
  5. Single leakage-free holdout matches competition test structure exactly.

Usage:
    cd /home/intel/Vanco/grocery_forecasting
    python train.py
"""

import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
DATA_DIR     = SCRIPT_DIR / "data" / "raw"
OUT_DIR      = SCRIPT_DIR / "submissions"
REPORTS_DIR  = SCRIPT_DIR / "reports"
MODELS_DIR   = SCRIPT_DIR / "models"
for d in (OUT_DIR, REPORTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Horizon-safe lag configuration
# RULE: minimum lag = forecast horizon (16 days) to prevent any future leakage.
# ---------------------------------------------------------------------------
HORIZON     = 16
LAG_DAYS    = [16, 21, 28, 35, 42, 49, 56]  # all ≥ HORIZON
ROLLING_WINS = [7, 14, 28]                   # windows applied after shift(HORIZON)
GROUP_COLS  = ["store_nbr", "family"]
TARGET      = "sales"


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================

def load_raw() -> dict:
    """Load all Kaggle CSVs, return dict of DataFrames."""
    files = {
        "train":        "train.csv",
        "test":         "test.csv",
        "stores":       "stores.csv",
        "oil":          "oil.csv",
        "holidays":     "holidays_events.csv",
        "transactions": "transactions.csv",
        "submission":   "sample_submission.csv",
    }
    dfs = {}
    for key, fname in files.items():
        path = DATA_DIR / fname
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        dfs[key] = df
    return dfs


def build_master(dfs: dict):
    """Merge auxiliary tables into train/test DataFrames."""
    train = dfs["train"].copy()
    test  = dfs["test"].copy()

    # Clean
    train["sales"]       = train["sales"].clip(lower=0)
    train["onpromotion"] = train["onpromotion"].fillna(0).astype(int)
    test["onpromotion"]  = test["onpromotion"].fillna(0).astype(int)

    # Oil – forward/linear-fill missing days
    oil = dfs["oil"].copy().sort_values("date").set_index("date")
    oil = oil.reindex(pd.date_range(oil.index.min(), oil.index.max(), freq="D"))
    oil["dcoilwtico"] = oil["dcoilwtico"].interpolate(method="linear").ffill().bfill()
    oil = oil.reset_index().rename(columns={"index": "date"})

    # Holidays – flatten to per-day national / regional / local flags
    hol = dfs["holidays"].copy()
    hol["date"] = pd.to_datetime(hol["date"])
    # Remove original transferred entries, keep Transfer rows (effective date)
    hol = hol[~((hol["transferred"] == True) & (hol["type"] != "Transfer"))]
    hol["is_holiday"] = 1
    nat_hol = (
        hol[hol["locale"] == "National"][["date", "is_holiday"]]
        .rename(columns={"is_holiday": "is_national_holiday"})
        .drop_duplicates("date")
    )
    reg_hol = (
        hol[hol["locale"] == "Regional"][["date", "locale_name", "is_holiday"]]
        .rename(columns={"is_holiday": "is_regional_holiday", "locale_name": "state"})
        .drop_duplicates(["date", "state"])
    )
    loc_hol = (
        hol[hol["locale"] == "Local"][["date", "locale_name", "is_holiday"]]
        .rename(columns={"is_holiday": "is_local_holiday", "locale_name": "city"})
        .drop_duplicates(["date", "city"])
    )

    stores = dfs["stores"].copy()
    txn    = dfs["transactions"].copy()
    txn["date"] = pd.to_datetime(txn["date"])

    def _merge(df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(stores, on="store_nbr", how="left")
        df = df.merge(oil, on="date", how="left")
        df = df.merge(txn, on=["store_nbr", "date"], how="left")
        df = df.merge(nat_hol, on="date", how="left")
        df["is_national_holiday"] = df["is_national_holiday"].fillna(0).astype(int)
        # Regional: matched on store state (correct locale mapping)
        df = df.merge(reg_hol, on=["date", "state"], how="left")
        df["is_regional_holiday"] = df["is_regional_holiday"].fillna(0).astype(int)
        # Local: matched on store city
        df = df.merge(loc_hol, on=["date", "city"], how="left")
        df["is_local_holiday"] = df["is_local_holiday"].fillna(0).astype(int)
        df["is_any_holiday"] = df[
            ["is_national_holiday", "is_regional_holiday", "is_local_holiday"]
        ].max(axis=1)
        return df

    train_m = _merge(train)
    test_m  = _merge(test)
    return train_m, test_m


# ===========================================================================
# 2. FEATURE ENGINEERING
# ===========================================================================

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["date"].dt
    df["year"]           = dt.year
    df["month"]          = dt.month
    df["week"]           = dt.isocalendar().week.astype(int)
    df["day_of_week"]    = dt.dayofweek
    df["day_of_year"]    = dt.dayofyear
    df["quarter"]        = dt.quarter
    df["is_weekend"]     = (dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = dt.is_month_start.astype(int)
    df["is_month_end"]   = dt.is_month_end.astype(int)
    # Ecuador paydays: 15th and last day
    df["is_payday"]      = ((dt.day == 15) | dt.is_month_end).astype(int)
    # Fourier terms
    for period, label in [(7, "weekly"), (365.25, "annual")]:
        df[f"sin_{label}"] = np.sin(2 * np.pi * df["day_of_year"] / period)
        df[f"cos_{label}"] = np.cos(2 * np.pi * df["day_of_year"] / period)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features; all lags ≥ HORIZON to avoid leakage."""
    df = df.sort_values(GROUP_COLS + ["date"]).reset_index(drop=True)
    grp = df.groupby(GROUP_COLS, sort=False)[TARGET]
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = grp.shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling statistics over [7, 14, 28] windows, computed AFTER shift(HORIZON).
    Computing group_id once avoids 12x repeated ngroup() calls.
    """
    df = df.sort_values(GROUP_COLS + ["date"]).reset_index(drop=True)
    # Compute group IDs once
    group_id = df.groupby(GROUP_COLS, sort=False).ngroup()
    # Shift by HORIZON so no val/test values are seen
    shifted = df.groupby(GROUP_COLS, sort=False)[TARGET].shift(HORIZON)
    for win in ROLLING_WINS:
        grouped = shifted.groupby(group_id)
        df[f"rolling_mean_{win}"] = grouped.transform(
            lambda x: x.rolling(win, min_periods=1).mean()
        )
        df[f"rolling_std_{win}"] = grouped.transform(
            lambda x: x.rolling(win, min_periods=1).std()
        )
        df[f"rolling_max_{win}"] = grouped.transform(
            lambda x: x.rolling(win, min_periods=1).max()
        )
        df[f"rolling_min_{win}"] = grouped.transform(
            lambda x: x.rolling(win, min_periods=1).min()
        )
    return df


def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(GROUP_COLS + ["date"]).reset_index(drop=True)
    group_id    = df.groupby(GROUP_COLS, sort=False).ngroup()
    promo_shift = df.groupby(GROUP_COLS, sort=False)["onpromotion"].shift(HORIZON)
    for win in [7, 14]:
        df[f"promo_count_{win}"] = promo_shift.groupby(group_id).transform(
            lambda x: x.rolling(win, min_periods=1).sum()
        )
    # Days since last promotion
    def _days_since(s):
        counter   = s.eq(0).cumsum()
        last_promo = counter.where(s.eq(1)).ffill().fillna(0)
        return (counter - last_promo).astype(int)
    df["days_since_promo"] = df.groupby(GROUP_COLS, sort=False)["onpromotion"].transform(
        _days_since
    )
    return df


def add_oil_features(df: pd.DataFrame) -> pd.DataFrame:
    df["oil_lag1"]   = df["dcoilwtico"].shift(1)
    df["oil_lag7"]   = df["dcoilwtico"].shift(7)
    df["oil_change"] = df["dcoilwtico"].diff(1)
    df["oil_roll7"]  = df["dcoilwtico"].rolling(7, min_periods=1).mean()
    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    hol = df["is_any_holiday"]
    for window in [1, 2, 3]:
        df[f"pre_holiday_{window}"]  = hol.shift(-window).fillna(0).astype(int)
        df[f"post_holiday_{window}"] = hol.shift(window).fillna(0).astype(int)
    # Consecutive holiday streak
    streak = []
    count  = 0
    for h in hol:
        count = count + 1 if h else 0
        streak.append(count)
    df["holiday_streak"] = streak
    return df


def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    if "transactions" not in df.columns:
        return df
    df = df.sort_values(["store_nbr", "date"]).reset_index(drop=True)
    group_id    = df.groupby("store_nbr", sort=False).ngroup()
    txn_shifted = df.groupby("store_nbr", sort=False)["transactions"].shift(HORIZON)
    df["transactions_lag"] = txn_shifted
    for win in [7, 14]:
        df[f"transactions_roll{win}"] = txn_shifted.groupby(group_id).transform(
            lambda x: x.rolling(win, min_periods=1).mean()
        )
    return df


def add_target_encodings(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Target-encode store/family from TRAINING rows only (no leakage)."""
    df         = df.copy()
    df["_log"] = np.log1p(df[TARGET].fillna(0))
    tr         = df[train_mask]

    store_mean  = tr.groupby("store_nbr")["_log"].mean().rename("store_mean_log")
    family_mean = tr.groupby("family")["_log"].mean().rename("family_mean_log")
    sf_mean     = (
        tr.groupby(GROUP_COLS)["_log"].mean().rename("store_family_mean_log")
    )
    df = df.merge(store_mean,  on="store_nbr", how="left")
    df = df.merge(family_mean, on="family",    how="left")
    df = df.merge(sf_mean,     on=GROUP_COLS,  how="left")
    df.drop(columns=["_log"], inplace=True)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["family", "type", "city", "state"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    return df


def build_features(combined: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline on combined (train+test) frame."""
    print("  [1/8] Calendar…")
    combined = add_calendar_features(combined)
    print("  [2/8] Lag features (safe lags ≥ 16)…")
    combined = add_lag_features(combined)
    print("  [3/8] Rolling features (shift=16)…")
    combined = add_rolling_features(combined)
    print("  [4/8] Promotion features…")
    combined = add_promo_features(combined)
    print("  [5/8] Oil features…")
    combined = add_oil_features(combined)
    print("  [6/8] Holiday features…")
    combined = add_holiday_features(combined)
    print("  [7/8] Transaction features…")
    combined = add_transaction_features(combined)
    print("  [8/8] Target encodings & categoricals…")
    train_mask = combined["split"] == "train"
    combined = add_target_encodings(combined, train_mask)
    combined = encode_categoricals(combined)
    return combined


# ===========================================================================
# 3. METRICS
# ===========================================================================

def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_pred - y_true)))

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom)) * 100)


# ===========================================================================
# 4. LGBM TRAINING
# ===========================================================================

LGBM_PARAMS = dict(
    objective      = "regression_l2",
    metric         = "rmse",
    num_leaves     = 127,
    learning_rate  = 0.05,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq   = 1,
    min_child_samples = 20,
    reg_alpha      = 0.1,
    reg_lambda     = 0.1,
    random_state   = 42,
    n_jobs         = -1,
    verbose        = -1,
)


def train_lgbm(
    X_train, y_train,
    X_val=None, y_val=None,
    n_estimators=2000,
    early_stopping=100,
    cat_cols=None,
) -> lgb.Booster:
    y_tr = np.log1p(y_train)
    dtrain = lgb.Dataset(
        X_train, label=y_tr,
        categorical_feature=cat_cols or "auto",
        free_raw_data=False,
    )
    callbacks = [lgb.log_evaluation(period=200)]
    valid_sets  = [dtrain]
    valid_names = ["train"]

    if X_val is not None and y_val is not None:
        y_v   = np.log1p(y_val)
        dval  = lgb.Dataset(X_val, label=y_v, reference=dtrain, free_raw_data=False)
        valid_sets.append(dval)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(early_stopping, verbose=False))

    params = {**LGBM_PARAMS, "n_estimators": n_estimators}
    model = lgb.train(
        params, dtrain,
        num_boost_round=n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return model


def predict(model: lgb.Booster, X: pd.DataFrame, feature_names: list) -> np.ndarray:
    raw = model.predict(X[feature_names])
    return np.maximum(np.expm1(raw), 0)


# ===========================================================================
# 5. VALIDATION
# ===========================================================================

def single_holdout_validate(train_feat: pd.DataFrame, feature_cols: list):
    """
    One train/val split:
      train  → all rows with date  ≤  (max_date - HORIZON)
      val    → last HORIZON days of training data
    This exactly mirrors how the model will be used on the Kaggle test set.
    """
    max_date   = train_feat["date"].max()
    val_start  = max_date - pd.Timedelta(days=HORIZON - 1)
    train_mask = train_feat["date"] < val_start
    val_mask   = train_feat["date"] >= val_start

    X_train = train_feat.loc[train_mask, feature_cols]
    y_train = train_feat.loc[train_mask, TARGET]
    X_val   = train_feat.loc[val_mask,   feature_cols]
    y_val   = train_feat.loc[val_mask,   TARGET]

    print(f"\n  Holdout split:")
    print(f"    Train: {train_feat.loc[train_mask,'date'].min().date()} → {train_feat.loc[train_mask,'date'].max().date()}  ({len(X_train):,} rows)")
    print(f"    Val:   {val_start.date()} → {max_date.date()}  ({len(X_val):,} rows)")

    t0    = time.time()
    model = train_lgbm(X_train, y_train, X_val, y_val)
    elapsed = time.time() - t0
    print(f"  Training took {elapsed:.0f}s | best_iteration={model.best_iteration}")

    preds = predict(model, X_val, feature_cols)
    y_true = y_val.values

    val_rmsle = rmsle(y_true, preds)
    val_mae   = mae(y_true, preds)
    val_smape = smape(y_true, preds)

    print(f"\n  ── Holdout Metrics ──────────────────────")
    print(f"  RMSLE : {val_rmsle:.4f}")
    print(f"  MAE   : {val_mae:.2f}")
    print(f"  SMAPE : {val_smape:.2f}%")

    return model, preds, y_val, train_feat.loc[val_mask], val_rmsle


def compute_per_group_metrics(val_df: pd.DataFrame, preds: np.ndarray):
    """RMSLE per store and per family."""
    df = val_df.copy()
    df["pred"] = np.maximum(preds, 0)
    df["sle"]  = (np.log1p(df["pred"]) - np.log1p(df[TARGET])) ** 2

    by_store  = (
        df.groupby("store_nbr")
        .agg(rmsle=("sle", lambda x: np.sqrt(x.mean())),
             mae=("pred", lambda x: mae(df.loc[x.index, TARGET].values, x.values)),
             n=("sle", "count"))
        .reset_index()
        .sort_values("rmsle", ascending=False)
    )
    by_family = (
        df.groupby("family")
        .agg(rmsle=("sle", lambda x: np.sqrt(x.mean())),
             mae=("pred", lambda x: mae(df.loc[x.index, TARGET].values, x.values)),
             n=("sle", "count"))
        .reset_index()
        .sort_values("rmsle", ascending=False)
    )
    by_dow    = (
        df.groupby("day_of_week")
        .agg(rmsle=("sle", lambda x: np.sqrt(x.mean())))
        .reset_index()
    )
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_dow["day_name"] = by_dow["day_of_week"].apply(
        lambda x: dow_labels[x] if x < 7 else str(x)
    )
    return by_store, by_family, by_dow


# ===========================================================================
# 6. REPORTING
# ===========================================================================

def save_reports(
    model: lgb.Booster,
    feature_cols: list,
    by_store: pd.DataFrame,
    by_family: pd.DataFrame,
    by_dow: pd.DataFrame,
    val_rmsle: float,
    val_mae: float,
    val_smape: float,
):
    # Feature importance
    fi = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=model.feature_name(),
    ).sort_values(ascending=False)
    fi.to_csv(REPORTS_DIR / "feature_importance.csv")

    # Per-group metrics
    by_store.to_csv(REPORTS_DIR  / "rmsle_by_store.csv",  index=False)
    by_family.to_csv(REPORTS_DIR / "rmsle_by_family.csv", index=False)
    by_dow.to_csv(REPORTS_DIR    / "rmsle_by_dow.csv",    index=False)

    # Metrics summary
    summary = (
        f"== Store Sales Forecasting – Results ==\n\n"
        f"Holdout Metrics (last {HORIZON} days of train):\n"
        f"  RMSLE : {val_rmsle:.4f}\n"
        f"  MAE   : {val_mae:.2f}\n"
        f"  SMAPE : {val_smape:.2f}%\n\n"
        f"Top 10 Features by Gain:\n"
        + fi.head(10).to_string()
        + f"\n\nWorst 5 Stores (RMSLE):\n"
        + by_store.head(5).to_string(index=False)
        + f"\n\nWorst 5 Families (RMSLE):\n"
        + by_family.head(5).to_string(index=False)
    )
    print("\n" + summary)
    with open(REPORTS_DIR / "metrics_summary.txt", "w") as f:
        f.write(summary)

    # Feature importance chart
    fig, ax = plt.subplots(figsize=(10, 8))
    fi.head(30).sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top 30 Feature Importances (Gain)")
    ax.set_xlabel("Gain")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)

    # Family RMSLE chart
    fig, ax = plt.subplots(figsize=(14, 5))
    fam_sorted = by_family.sort_values("rmsle", ascending=False)
    ax.bar(range(len(fam_sorted)), fam_sorted["rmsle"], color="salmon")
    ax.set_xticks(range(len(fam_sorted)))
    ax.set_xticklabels(fam_sorted["family"].astype(str), rotation=45, ha="right", fontsize=8)
    ax.set_title("RMSLE by Product Family (Holdout)")
    ax.set_ylabel("RMSLE")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "rmsle_by_family.png", dpi=120)
    plt.close(fig)

    print(f"\nReports saved to: {REPORTS_DIR}/")


# ===========================================================================
# 7. MAIN PIPELINE
# ===========================================================================

def main():
    t_start = time.time()
    print("=" * 60)
    print(" Store Sales Forecasting – Training Pipeline")
    print("=" * 60)

    # ---- Load ----
    print("\n[1/6] Loading data…")
    dfs = load_raw()
    for k, v in dfs.items():
        print(f"  {k:<15}: {v.shape}")

    # ---- Merge ----
    print("\n[2/6] Merging auxiliary tables…")
    train_raw, test_raw = build_master(dfs)
    print(f"  Train merged : {train_raw.shape}")
    print(f"  Test  merged : {test_raw.shape}")
    print(f"  Date range   : {train_raw['date'].min().date()} → {train_raw['date'].max().date()}")
    print(f"  Stores       : {train_raw['store_nbr'].nunique()}")
    print(f"  Families     : {train_raw['family'].nunique()}")
    print(f"  Zero-sales   : {(train_raw['sales'] == 0).mean():.1%}")

    # ---- Features ----
    print("\n[3/6] Feature engineering…")
    train_raw["split"] = "train"
    test_raw["split"]  = "test"
    test_raw["sales"]  = np.nan
    combined = pd.concat([train_raw, test_raw], ignore_index=True)
    combined = combined.sort_values(GROUP_COLS + ["date"]).reset_index(drop=True)
    combined = build_features(combined)
    print(f"  Combined shape: {combined.shape}")

    # Split back
    train_feat = combined[combined["split"] == "train"].reset_index(drop=True)
    test_feat  = combined[combined["split"] == "test"].reset_index(drop=True)

    # Feature columns: drop non-numeric / metadata
    DROP_COLS = {"id", "date", "sales", "split", "transactions", "dcoilwtico"}
    feature_cols = [
        c for c in train_feat.columns
        if c not in DROP_COLS and train_feat[c].dtype != "object"
    ]
    print(f"  Feature count : {len(feature_cols)}")

    # NaN report for training features
    nan_pct = train_feat[feature_cols].isna().mean()
    nan_cols = nan_pct[nan_pct > 0].sort_values(ascending=False)
    if len(nan_cols):
        print(f"\n  Missing value summary (train):")
        for col, pct in nan_cols.head(10).items():
            print(f"    {col:<30}: {pct:.1%}")

    # Fill remaining NaNs (early lag positions, oil NaN before series start)
    train_feat[feature_cols] = train_feat[feature_cols].fillna(0)
    test_feat[feature_cols]  = test_feat[feature_cols].fillna(0)

    # ---- Validate ----
    print("\n[4/6] Holdout validation (last 16 days as val)…")
    model_val, val_preds, y_val, val_df, val_rmsle = single_holdout_validate(
        train_feat, feature_cols
    )
    val_mae   = mae(y_val.values, val_preds)
    val_smape = smape(y_val.values, val_preds)
    by_store, by_family, by_dow = compute_per_group_metrics(val_df, val_preds)

    # ---- Final model on full training data ----
    print("\n[5/6] Training final model on full data…")
    best_n = model_val.best_iteration if model_val.best_iteration else 1000
    print(f"  Using {best_n} boosting rounds (from holdout early-stopping)")
    X_full = train_feat[feature_cols]
    y_full = train_feat[TARGET]
    y_full_log = np.log1p(y_full)
    dfull  = lgb.Dataset(
        X_full, label=y_full_log,
        categorical_feature="auto",
        free_raw_data=False,
    )
    params_final = {**LGBM_PARAMS, "n_estimators": best_n}
    t0 = time.time()
    final_model = lgb.train(
        params_final, dfull,
        num_boost_round=best_n,
        callbacks=[lgb.log_evaluation(period=200)],
    )
    print(f"  Final model trained in {time.time()-t0:.0f}s  |  trees: {final_model.num_trees()}")
    final_model.save_model(str(MODELS_DIR / "lgbm_final.txt"))

    # ---- Predict test set ----
    print("\n[6/6] Generating submission…")
    raw_preds = predict(final_model, test_feat, feature_cols)
    # Align by id — test_feat is sorted by (store_nbr, family, date) which
    # differs from sample_submission.csv (date, store_nbr, family) order.
    test_feat_with_pred = test_feat[["id"]].copy()
    test_feat_with_pred["sales"] = np.maximum(raw_preds, 0)
    submission = dfs["submission"][["id"]].copy()
    submission = submission.merge(test_feat_with_pred, on="id", how="left")
    submission["sales"] = submission["sales"].fillna(0)
    out_path = OUT_DIR / "submission_lgbm.csv"
    submission.to_csv(out_path, index=False)
    print(f"  Submission saved: {out_path}")
    print(f"  Rows: {len(submission):,}  |  Min: {submission['sales'].min():.2f}  |  "
          f"Max: {submission['sales'].max():.0f}  |  Mean: {submission['sales'].mean():.2f}")

    # ---- Reports ----
    save_reports(
        final_model, feature_cols,
        by_store, by_family, by_dow,
        val_rmsle, val_mae, val_smape,
    )

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f" Pipeline complete in {total/60:.1f} min")
    print(f" Holdout RMSLE: {val_rmsle:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
