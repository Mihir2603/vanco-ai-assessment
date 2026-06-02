"""
Error analysis utilities for Store Sales forecasting.

Produces breakdowns by:
- Store
- Product family
- Holiday / event type
- Promotion period
- Day of week / month

Also provides SHAP-based feature attribution helpers.
"""
from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")


# ---------------------------------------------------------------------------
# Helper: attach predictions to eval frame
# ---------------------------------------------------------------------------

def attach_predictions(
    df: pd.DataFrame,
    preds: np.ndarray,
    actuals_col: str = "sales",
    pred_col: str = "pred_sales",
) -> pd.DataFrame:
    df = df.copy()
    df[pred_col] = np.maximum(preds, 0)
    df["abs_error"]      = np.abs(df[actuals_col] - df[pred_col])
    df["pct_error"]      = df["abs_error"] / (df[actuals_col].clip(lower=1))
    df["log_error"]      = np.log1p(df[pred_col]) - np.log1p(df[actuals_col])
    df["squared_log_err"] = df["log_error"] ** 2
    return df


# ---------------------------------------------------------------------------
# Breakdown functions
# ---------------------------------------------------------------------------

def breakdown_by_store(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """RMSLE and MAE per store, sorted by worst performance."""
    grp = df.groupby("store_nbr").agg(
        rmsle=("squared_log_err", lambda x: np.sqrt(x.mean())),
        mae=("abs_error", "mean"),
        n_rows=("sales", "count"),
        mean_sales=("sales", "mean"),
    ).reset_index().sort_values("rmsle", ascending=False)
    return grp.head(top_n)


def breakdown_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """RMSLE and MAE per product family."""
    grp = df.groupby("family").agg(
        rmsle=("squared_log_err", lambda x: np.sqrt(x.mean())),
        mae=("abs_error", "mean"),
        n_rows=("sales", "count"),
        mean_sales=("sales", "mean"),
    ).reset_index().sort_values("rmsle", ascending=False)
    return grp


def breakdown_by_holiday(df: pd.DataFrame) -> pd.DataFrame:
    """Compare error on holiday vs non-holiday days."""
    if "is_any_holiday" not in df.columns:
        return pd.DataFrame()
    grp = df.groupby("is_any_holiday").agg(
        rmsle=("squared_log_err", lambda x: np.sqrt(x.mean())),
        mae=("abs_error", "mean"),
        n_rows=("sales", "count"),
    ).reset_index()
    grp["is_any_holiday"] = grp["is_any_holiday"].map({0: "Non-Holiday", 1: "Holiday"})
    return grp


def breakdown_by_promotion(df: pd.DataFrame) -> pd.DataFrame:
    """Compare error on promoted vs non-promoted items."""
    if "onpromotion" not in df.columns:
        return pd.DataFrame()
    grp = df.groupby("onpromotion").agg(
        rmsle=("squared_log_err", lambda x: np.sqrt(x.mean())),
        mae=("abs_error", "mean"),
        n_rows=("sales", "count"),
    ).reset_index()
    grp["onpromotion"] = grp["onpromotion"].map({0: "No Promo", 1: "On Promo"})
    return grp


def breakdown_by_dow(df: pd.DataFrame) -> pd.DataFrame:
    """Error by day of week."""
    if "day_of_week" not in df.columns:
        return pd.DataFrame()
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    grp = df.groupby("day_of_week").agg(
        rmsle=("squared_log_err", lambda x: np.sqrt(x.mean())),
        mae=("abs_error", "mean"),
    ).reset_index()
    grp["day_name"] = grp["day_of_week"].apply(lambda x: dow_labels[x] if x < 7 else x)
    return grp


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_error_by_family(df_errors: pd.DataFrame, figsize=(12, 6)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    data = df_errors.sort_values("rmsle", ascending=False)
    sns.barplot(data=data, x="family", y="rmsle", ax=ax, palette="rocket_r")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_title("RMSLE by Product Family")
    ax.set_xlabel("Family")
    ax.set_ylabel("RMSLE")
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(
    df: pd.DataFrame,
    store: int,
    family: str,
    actuals_col: str = "sales",
    pred_col: str = "pred_sales",
    figsize=(14, 4),
) -> plt.Figure:
    subset = df[(df["store_nbr"] == store) & (df["family"] == family)].sort_values("date")
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(subset["date"], subset[actuals_col], label="Actual", lw=1.5)
    ax.plot(subset["date"], subset[pred_col],   label="Predicted", lw=1.5, linestyle="--")
    ax.set_title(f"Store {store} | Family: {family}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_feature_importance(importance: pd.Series, top_n: int = 30, figsize=(10, 8)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    data = importance.head(top_n).sort_values()
    data.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"Top {top_n} Feature Importances (Gain)")
    ax.set_xlabel("Gain")
    fig.tight_layout()
    return fig


def plot_residuals_over_time(df: pd.DataFrame, figsize=(14, 4)) -> plt.Figure:
    daily = df.groupby("date").agg(mean_log_error=("log_error", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(daily["date"], daily["mean_log_error"], lw=1)
    ax.axhline(0, color="red", linestyle="--", lw=0.8)
    ax.set_title("Mean Log-Error over Time (positive = over-predicted)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Log Error")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# SHAP summary (optional – only if shap is installed)
# ---------------------------------------------------------------------------

def shap_summary(model, X_sample: pd.DataFrame, max_display: int = 20) -> None:
    try:
        import shap
        explainer = shap.TreeExplainer(model.model_)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample, max_display=max_display, show=True)
    except ImportError:
        print("shap not installed – skipping SHAP analysis.")
    except Exception as e:
        print(f"SHAP analysis failed: {e}")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def generate_error_report(df: pd.DataFrame) -> dict:
    """
    Return a dict of all breakdowns as DataFrames – easy to display in a notebook.
    """
    return {
        "by_store":     breakdown_by_store(df),
        "by_family":    breakdown_by_family(df),
        "by_holiday":   breakdown_by_holiday(df),
        "by_promotion": breakdown_by_promotion(df),
        "by_dow":       breakdown_by_dow(df),
    }
