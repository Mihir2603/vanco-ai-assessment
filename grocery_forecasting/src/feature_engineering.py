"""
Feature engineering for Store Sales forecasting.

Strategy
--------
1. Calendar / time features        – year, month, week, day-of-week, day-of-year, quarter
2. Lag features                    – sales N days ago (aligned per store×family group)
3. Rolling window features         – mean/std/max/min over multiple windows
4. Promotional features            – promotion flag + rolling promotion counts
5. Oil price features              – level + lag + change
6. Holiday / event features        – pre/post holiday windows, consecutive holiday count
7. Transactions features           – daily footfall proxy
8. Store / product encodings       – type, cluster, family target-encoded mean
9. Trend / seasonality             – linear trend, Fourier terms
"""
import warnings
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Calendar features
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    dt = df[date_col].dt
    df["year"]        = dt.year
    df["month"]       = dt.month
    df["week"]        = dt.isocalendar().week.astype(int)
    df["day_of_week"] = dt.dayofweek          # 0=Monday
    df["day_of_year"] = dt.dayofyear
    df["quarter"]     = dt.quarter
    df["is_weekend"]  = (dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = dt.is_month_start.astype(int)
    df["is_month_end"]   = dt.is_month_end.astype(int)
    # Payday effect: 15th and last day of month are paydays in Ecuador
    df["is_payday"] = ((dt.day == 15) | dt.is_month_end).astype(int)
    # Fourier terms for weekly (7-day) and annual (365.25-day) seasonality
    for period, label in [(7, "weekly"), (365.25, "annual")]:
        df[f"sin_{label}"] = np.sin(2 * np.pi * df["day_of_year"] / period)
        df[f"cos_{label}"] = np.cos(2 * np.pi * df["day_of_year"] / period)
    return df


# ---------------------------------------------------------------------------
# 2 & 3. Lag and rolling features
# ---------------------------------------------------------------------------

HORIZON      = 16            # forecast horizon; all lags must be ≥ this value
LAG_DAYS     = [16, 21, 28, 35, 42, 49, 56]   # safe: min lag = HORIZON
ROLLING_WINS = [7, 14, 28]


def add_lag_features(
    df: pd.DataFrame,
    group_cols: List[str] = ["store_nbr", "family"],
    target: str = "sales",
    lags: List[int] = LAG_DAYS,
) -> pd.DataFrame:
    """Vectorised lag computation using groupby.shift — no Python loops per group."""
    df = df.copy()
    df = df.sort_values(group_cols + ["date"]).reset_index(drop=True)
    grp = df.groupby(group_cols, sort=False)[target]
    for lag in lags:
        df[f"lag_{lag}"] = grp.shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    group_cols: List[str] = ["store_nbr", "family"],
    target: str = "sales",
    windows: List[int] = ROLLING_WINS,
    min_periods: int = 1,
) -> pd.DataFrame:
    """Rolling statistics shifted by HORIZON to prevent any future leakage.
    group_id computed once (not once per window) for efficiency on 3M+ rows.
    """
    df = df.copy()
    df = df.sort_values(group_cols + ["date"]).reset_index(drop=True)
    # Compute group IDs once — avoids 12× repeated ngroup() calls
    group_id = df.groupby(group_cols, sort=False).ngroup()
    shifted = df.groupby(group_cols, sort=False)[target].shift(HORIZON)
    for win in windows:
        grouped = shifted.groupby(group_id)
        df[f"rolling_mean_{win}"] = grouped.transform(
            lambda x: x.rolling(win, min_periods=min_periods).mean()
        )
        df[f"rolling_std_{win}"] = grouped.transform(
            lambda x: x.rolling(win, min_periods=min_periods).std()
        )
        df[f"rolling_max_{win}"] = grouped.transform(
            lambda x: x.rolling(win, min_periods=min_periods).max()
        )
        df[f"rolling_min_{win}"] = grouped.transform(
            lambda x: x.rolling(win, min_periods=min_periods).min()
        )
    return df


# ---------------------------------------------------------------------------
# 4. Promotion features
# ---------------------------------------------------------------------------

def add_promo_features(
    df: pd.DataFrame,
    group_cols: List[str] = ["store_nbr", "family"],
    windows: List[int] = [7, 14],
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(group_cols + ["date"]).reset_index(drop=True)
    group_id    = df.groupby(group_cols, sort=False).ngroup()
    promo_shifted = df.groupby(group_cols, sort=False)["onpromotion"].shift(HORIZON)
    for win in windows:
        df[f"promo_count_{win}"] = promo_shifted.groupby(group_id).transform(
            lambda x: x.rolling(win, min_periods=1).sum()
        )
    # Days since last promotion (vectorised per group)
    def _days_since(s):
        s = s.copy()
        counter = s.eq(0).cumsum()
        last_promo = counter.where(s.eq(1)).ffill().fillna(0)
        return (counter - last_promo).astype(int)
    df["days_since_promo"] = df.groupby(group_cols, sort=False)["onpromotion"].transform(_days_since)
    return df


# ---------------------------------------------------------------------------
# 5. Oil price features
# ---------------------------------------------------------------------------

def add_oil_features(df: pd.DataFrame, oil_col: str = "dcoilwtico") -> pd.DataFrame:
    df = df.copy()
    df["oil_lag1"]   = df[oil_col].shift(1)
    df["oil_lag7"]   = df[oil_col].shift(7)
    df["oil_change"] = df[oil_col].diff(1)
    df["oil_roll7"]  = df[oil_col].rolling(7, min_periods=1).mean()
    return df


# ---------------------------------------------------------------------------
# 6. Holiday features
# ---------------------------------------------------------------------------

def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pre-holiday (N days before) and post-holiday (N days after) windows,
    plus a consecutive-holiday-streak counter.
    Holidays are per-date (same for all store-family rows on the same day),
    so we compute on a date-level frame and merge back.
    """
    df = df.copy()
    original_sort_cols = ["store_nbr", "family", "date"]

    # Date-level holiday flags (same for every row on a given date)
    hol_daily = (
        df[["date", "is_any_holiday"]]
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    hol = hol_daily["is_any_holiday"]
    for window in [1, 2, 3]:
        hol_daily[f"pre_holiday_{window}"]  = hol.shift(-window).fillna(0).astype(int)
        hol_daily[f"post_holiday_{window}"] = hol.shift(window).fillna(0).astype(int)
    # Consecutive holiday streak
    streak, count = [], 0
    for h in hol:
        count = count + 1 if h else 0
        streak.append(count)
    hol_daily["holiday_streak"] = streak

    hol_cols = [c for c in hol_daily.columns if c != "is_any_holiday"]
    df = df.merge(hol_daily[hol_cols], on="date", how="left")
    # Restore original sort order
    df = df.sort_values(original_sort_cols).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 7. Transactions features
# ---------------------------------------------------------------------------

def add_transaction_features(
    df: pd.DataFrame,
    group_col: str = "store_nbr",
    windows: List[int] = [7, 14],
) -> pd.DataFrame:
    df = df.copy()
    if "transactions" not in df.columns:
        return df
    # Transactions are per (store, date) — compute lags on that granularity,
    # then broadcast back to (store, family, date) rows.
    txn_daily = (
        df[[group_col, "date", "transactions"]]
        .drop_duplicates([group_col, "date"])
        .sort_values([group_col, "date"])
        .reset_index(drop=True)
    )
    grp = txn_daily.groupby(group_col, sort=False)["transactions"]
    txn_daily["transactions_lag"] = grp.shift(HORIZON)
    grp_id = txn_daily.groupby(group_col, sort=False).ngroup()
    shifted = txn_daily.groupby(group_col, sort=False)["transactions"].shift(HORIZON)
    for win in windows:
        txn_daily[f"transactions_roll{win}"] = shifted.groupby(grp_id).transform(
            lambda x: x.rolling(win, min_periods=1).mean()
        )
    lag_cols = ["transactions_lag"] + [f"transactions_roll{w}" for w in windows]
    df = df.merge(txn_daily[[group_col, "date"] + lag_cols], on=[group_col, "date"], how="left")
    return df


# ---------------------------------------------------------------------------
# 8. Store / product encoding
# ---------------------------------------------------------------------------

def add_store_family_encodings(
    df: pd.DataFrame,
    train_mask: pd.Series,
) -> pd.DataFrame:
    """
    Target-encode store×family mean log-sales from the training portion only.
    Applied to both train and test to avoid leakage.
    """
    df = df.copy()
    target = "sales"
    df["log_sales"] = np.log1p(df[target])
    train_df = df[train_mask]

    store_mean = (
        train_df.groupby("store_nbr")["log_sales"].mean().rename("store_mean_log")
    )
    family_mean = (
        train_df.groupby("family")["log_sales"].mean().rename("family_mean_log")
    )
    store_family_mean = (
        train_df.groupby(["store_nbr", "family"])["log_sales"]
        .mean()
        .rename("store_family_mean_log")
    )

    df = df.merge(store_mean, on="store_nbr", how="left")
    df = df.merge(family_mean, on="family", how="left")
    df = df.merge(store_family_mean, on=["store_nbr", "family"], how="left")
    df.drop(columns=["log_sales"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# 9. Categorical encoding
# ---------------------------------------------------------------------------

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode low-cardinality categoricals for LightGBM."""
    df = df.copy()
    cat_cols = ["family", "type", "city", "state"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    return df


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def build_features(
    combined: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline on the combined train+test frame.

    Parameters
    ----------
    combined : DataFrame with 'split' column ('train' / 'test')
    """
    df = combined.copy()
    train_mask = df["split"] == "train"

    print("  [1/8] Calendar features…")
    df = add_calendar_features(df)

    print("  [2/8] Lag features…")
    df = add_lag_features(df)

    print("  [3/8] Rolling features…")
    df = add_rolling_features(df)

    print("  [4/8] Promotion features…")
    df = add_promo_features(df)

    print("  [5/8] Oil features…")
    df = add_oil_features(df)

    print("  [6/8] Holiday features…")
    df = add_holiday_features(df)

    print("  [7/8] Transaction features…")
    df = add_transaction_features(df)

    print("  [8/8] Store/family encodings & categoricals…")
    df = add_store_family_encodings(df, train_mask)
    df = encode_categoricals(df)

    return df
