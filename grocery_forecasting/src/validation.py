"""
Time-aware validation (walk-forward backtesting) for grocery sales forecasting.

Design principles
-----------------
* NO random splits – all validation respects temporal ordering.
* A fixed GAP equal to the competition horizon (16 days) is maintained between
  the end of each training fold and the start of each validation fold.
* Each fold validates on exactly TEST_DAYS (16) days – matching what Kaggle
  requires us to forecast.
* Metrics are collected per fold so we can compute stability across time.

        ┌──────────────────────────────────────────────────────────┐
        │  Fold k  │  train (expanding)  │ GAP │  val (16 days)   │
        └──────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd


GAP_DAYS  = 16   # match competition horizon
TEST_DAYS = 16   # validation window per fold


# ---------------------------------------------------------------------------
# Data class for one fold result
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: int
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    rmsle: float
    mae: float
    predictions: pd.Series = field(default_factory=pd.Series)
    actuals: pd.Series = field(default_factory=pd.Series)


# ---------------------------------------------------------------------------
# Walk-forward splitter
# ---------------------------------------------------------------------------

class WalkForwardSplitter:
    """
    Generates (train_idx, val_idx) index pairs using an expanding window.

    Parameters
    ----------
    n_splits    : number of folds
    gap_days    : days to skip between train end and val start (avoids leakage)
    test_days   : length of each validation window in days
    min_train_days : minimum days required in the training window
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap_days: int = GAP_DAYS,
        test_days: int = TEST_DAYS,
        min_train_days: int = 90,
    ):
        self.n_splits       = n_splits
        self.gap_days       = gap_days
        self.test_days      = test_days
        self.min_train_days = min_train_days

    def split(
        self, df: pd.DataFrame, date_col: str = "date"
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_indices, val_indices) tuples.
        df must be sorted by date_col before calling.
        """
        dates = df[date_col].sort_values().unique()
        total_days = len(dates)

        # Determine fold end-points (val windows), working backwards from
        # the last available date.
        val_ends = []
        cursor = len(dates) - 1
        for _ in range(self.n_splits):
            if cursor < self.test_days:
                break
            val_end   = dates[cursor]
            val_start = dates[cursor - self.test_days + 1]
            train_end = dates[cursor - self.test_days - self.gap_days]
            train_start_idx = 0
            train_end_idx   = np.searchsorted(dates, train_end, side="right") - 1
            if train_end_idx - train_start_idx + 1 < self.min_train_days:
                break
            val_ends.append((train_end, val_start, val_end))
            cursor -= self.test_days

        # Reverse so fold 0 is earliest
        val_ends = val_ends[::-1]

        for train_end, val_start, val_end in val_ends:
            train_idx = df.index[df[date_col] <= train_end].to_numpy()
            val_idx   = df.index[
                (df[date_col] >= val_start) & (df[date_col] <= val_end)
            ].to_numpy()
            yield train_idx, val_idx


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Logarithmic Error (Kaggle metric for this competition)."""
    y_pred = np.maximum(y_pred, 0)          # clip negatives
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_pred - y_true))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom)) * 100


# ---------------------------------------------------------------------------
# Cross-validation runner
# ---------------------------------------------------------------------------

def cross_validate(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_factory,          # callable() → fitted model with .fit(X, y) / .predict(X)
    n_splits: int = 5,
    date_col: str = "date",
    verbose: bool = True,
) -> Tuple[List[FoldResult], float]:
    """
    Run walk-forward cross-validation.

    Parameters
    ----------
    model_factory : zero-argument callable returning a fresh model instance.
                    The model must implement .fit(X, y) and .predict(X).

    Returns
    -------
    results  : list of FoldResult (one per fold)
    mean_rmsle : average RMSLE across folds
    """
    splitter = WalkForwardSplitter(n_splits=n_splits)
    results: List[FoldResult] = []

    df = df.copy().reset_index(drop=True)
    df = df.sort_values(date_col).reset_index(drop=True)

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(df, date_col)):
        X_train = df.loc[train_idx, feature_cols]
        y_train = df.loc[train_idx, target_col]
        X_val   = df.loc[val_idx, feature_cols]
        y_val   = df.loc[val_idx, target_col]

        model = model_factory()
        model.fit(X_train, y_train)
        preds = np.maximum(model.predict(X_val), 0)

        fold_rmsle = rmsle(y_val.values, preds)
        fold_mae   = mae(y_val.values, preds)

        train_end  = df.loc[train_idx, date_col].max()
        val_start  = df.loc[val_idx, date_col].min()
        val_end    = df.loc[val_idx, date_col].max()

        result = FoldResult(
            fold=fold_idx,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            rmsle=fold_rmsle,
            mae=fold_mae,
            predictions=pd.Series(preds, index=val_idx),
            actuals=y_val,
        )
        results.append(result)

        if verbose:
            print(
                f"  Fold {fold_idx}: train_end={train_end.date()}  "
                f"val={val_start.date()}→{val_end.date()}  "
                f"RMSLE={fold_rmsle:.4f}  MAE={fold_mae:.2f}"
            )

    mean_rmsle = np.mean([r.rmsle for r in results])
    if verbose:
        print(f"\n  Mean CV RMSLE: {mean_rmsle:.4f}")
    return results, mean_rmsle


# ---------------------------------------------------------------------------
# Leakage checker
# ---------------------------------------------------------------------------

def check_temporal_leakage(
    df: pd.DataFrame,
    feature_cols: List[str],
    date_col: str = "date",
    lag_threshold: int = 16,
) -> List[str]:
    """
    Heuristically flag feature columns that might encode future information.
    Checks for lag-feature names with lag < lag_threshold.
    Returns list of suspicious column names.
    """
    suspicious = []
    for col in feature_cols:
        if col.startswith("lag_"):
            try:
                lag_val = int(col.split("_")[1])
                if lag_val < lag_threshold:
                    suspicious.append(col)
            except ValueError:
                pass
        if col.startswith("rolling_") or col.startswith("pre_holiday"):
            pass  # rolling uses shift(1), pre_holiday uses shift(-N) intentionally
    return suspicious
