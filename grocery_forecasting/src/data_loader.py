"""
Data loading utilities for Store Sales - Time Series Forecasting.
Handles all Kaggle dataset tables and merges them into a modelling-ready frame.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_raw(data_dir: str = "data/raw") -> dict[str, pd.DataFrame]:
    """Load every CSV shipped with the competition and return a dict of DataFrames."""
    base = Path(data_dir)
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
        path = base / fname
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found: {path}")
        dfs[key] = pd.read_csv(path, parse_dates=["date"] if "date" in pd.read_csv(path, nrows=0).columns else [])
    return dfs


def _parse_dates(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col])
    return df


# ---------------------------------------------------------------------------
# Individual table cleaners
# ---------------------------------------------------------------------------

def clean_train(train: pd.DataFrame) -> pd.DataFrame:
    train = _parse_dates(train)
    # Sales are never negative in real data; clip to 0 just in case
    train["sales"] = train["sales"].clip(lower=0)
    train["onpromotion"] = train["onpromotion"].fillna(0).astype(int)
    return train.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)


def clean_test(test: pd.DataFrame) -> pd.DataFrame:
    test = _parse_dates(test)
    test["onpromotion"] = test["onpromotion"].fillna(0).astype(int)
    return test.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)


def clean_oil(oil: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill and backward-fill missing oil prices (weekends/holidays)."""
    oil = _parse_dates(oil)
    oil = oil.sort_values("date").set_index("date")
    oil = oil.reindex(pd.date_range(oil.index.min(), oil.index.max(), freq="D"))
    oil["dcoilwtico"] = oil["dcoilwtico"].interpolate(method="linear").ffill().bfill()
    return oil.reset_index().rename(columns={"index": "date"})


def clean_holidays(holidays: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten the holidays table into per-day flags.
    national > regional > local precedence.
    Transferred holidays are treated as effective on the transferred date.
    """
    holidays = _parse_dates(holidays)
    # Keep transferred holidays and drop the 'transferred' originals
    holidays = holidays[~((holidays["transferred"] == True) & (holidays["type"] != "Transfer"))]
    holidays["is_holiday"] = 1
    # Create locale-level columns
    nat = holidays[holidays["locale"] == "National"][["date", "is_holiday"]].rename(
        columns={"is_holiday": "is_national_holiday"}
    ).drop_duplicates("date")
    reg = holidays[holidays["locale"] == "Regional"][["date", "locale_name", "is_holiday"]].rename(
        columns={"is_holiday": "is_regional_holiday"}
    ).drop_duplicates(["date", "locale_name"])
    loc = holidays[holidays["locale"] == "Local"][["date", "locale_name", "is_holiday"]].rename(
        columns={"is_holiday": "is_local_holiday"}
    ).drop_duplicates(["date", "locale_name"])
    return nat, reg, loc


def clean_stores(stores: pd.DataFrame) -> pd.DataFrame:
    return stores.copy()


def clean_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    transactions = _parse_dates(transactions)
    return transactions.sort_values(["store_nbr", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Master merge
# ---------------------------------------------------------------------------

def build_master(dfs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge all auxiliary tables into train and test DataFrames.

    Returns
    -------
    train_df, test_df  – merged DataFrames ready for feature engineering
    """
    train = clean_train(dfs["train"].copy())
    test  = clean_test(dfs["test"].copy())
    oil   = clean_oil(dfs["oil"].copy())
    nat_hol, reg_hol, loc_hol = clean_holidays(dfs["holidays"].copy())
    stores = clean_stores(dfs["stores"].copy())
    transactions = clean_transactions(dfs["transactions"].copy())

    def _merge_all(df: pd.DataFrame) -> pd.DataFrame:
        # --- stores ---
        df = df.merge(stores, on="store_nbr", how="left")
        # --- oil ---
        df = df.merge(oil, on="date", how="left")
        # --- transactions (only in train, NaN for test) ---
        df = df.merge(transactions, on=["store_nbr", "date"], how="left")
        # --- national holidays ---
        df = df.merge(nat_hol, on="date", how="left")
        df["is_national_holiday"] = df["is_national_holiday"].fillna(0).astype(int)
        # --- regional holidays (matched on store state — correct locale) ---
        df = df.merge(
            reg_hol.rename(columns={"locale_name": "state"}),
            on=["date", "state"], how="left"
        )
        df["is_regional_holiday"] = df["is_regional_holiday"].fillna(0).astype(int)
        # --- local holidays (matched on store city as well) ---
        df = df.merge(
            loc_hol.rename(columns={"locale_name": "city"}),
            on=["date", "city"], how="left"
        )
        df["is_local_holiday"] = df["is_local_holiday"].fillna(0).astype(int)
        df["is_any_holiday"] = (
            df[["is_national_holiday", "is_regional_holiday", "is_local_holiday"]].max(axis=1)
        )
        return df

    train_m = _merge_all(train)
    test_m  = _merge_all(test)
    return train_m, test_m


# ---------------------------------------------------------------------------
# Convenience: combined frame for feature engineering
# ---------------------------------------------------------------------------

def combine_train_test(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Stack train and test with a 'split' column so features can be computed
    across the full date range before being re-split.
    """
    train = train.copy()
    test  = test.copy()
    train["split"] = "train"
    test["split"]  = "test"
    if "sales" not in test.columns:
        test["sales"] = np.nan
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    return combined.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
