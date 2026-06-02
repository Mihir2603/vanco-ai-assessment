"""
Models for Store Sales forecasting.

Hierarchy
---------
1. BaselineModel        – naive last-observed / seasonal naïve
2. LightGBMForecaster   – main production model
3. EnsembleForecaster   – weighted average of multiple models
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Optional


# ---------------------------------------------------------------------------
# 1. Baseline – Seasonal Naïve (same day last week)
# ---------------------------------------------------------------------------

class SeasonalNaiveForecaster:
    """
    Predicts the sales value from `seasonal_period` days ago.
    Used as a floor for model quality assessment.
    """
    def __init__(self, seasonal_period: int = 16):
        self.seasonal_period = seasonal_period
        self._lag_col = f"lag_{seasonal_period}"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalNaiveForecaster":
        # stateless
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._lag_col not in X.columns:
            raise ValueError(
                f"SeasonalNaiveForecaster needs '{self._lag_col}' in feature matrix."
            )
        preds = X[self._lag_col].fillna(0).values
        return np.maximum(preds, 0)


# ---------------------------------------------------------------------------
# 2. LightGBM Forecaster
# ---------------------------------------------------------------------------

LGBM_DEFAULT_PARAMS = dict(
    objective="regression_l2",   # MSE on log1p target → RMSLE optimisation
    metric="rmse",
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=127,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


class LightGBMForecaster:
    """
    LightGBM wrapper that trains on log1p(sales) and back-transforms predictions.

    Parameters
    ----------
    params              : dict of LightGBM parameters
    early_stopping      : early stopping rounds (set to None to disable)
    log_transform       : if True, fit on log1p(y) and expm1 predictions
    categorical_features: list of column names to declare as categorical to LGB
    """

    def __init__(
        self,
        params: Optional[dict] = None,
        early_stopping: int = 50,
        log_transform: bool = True,
        categorical_features: Optional[List[str]] = None,
    ):
        self.params = {**LGBM_DEFAULT_PARAMS, **(params or {})}
        self.early_stopping = early_stopping
        self.log_transform = log_transform
        self.categorical_features = categorical_features or []
        self.model_: Optional[lgb.Booster] = None
        self.feature_names_: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "LightGBMForecaster":
        self.feature_names_ = X_train.columns.tolist()
        y_tr = np.log1p(y_train) if self.log_transform else y_train.values

        cat_feats = [c for c in self.categorical_features if c in X_train.columns]
        dtrain = lgb.Dataset(
            X_train,
            label=y_tr,
            categorical_feature=cat_feats if cat_feats else "auto",
            free_raw_data=False,
        )

        callbacks = [lgb.log_evaluation(period=100)]
        valid_sets = [dtrain]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            y_v = np.log1p(y_val) if self.log_transform else y_val.values
            dval = lgb.Dataset(X_val, label=y_v, reference=dtrain, free_raw_data=False)
            valid_sets.append(dval)
            valid_names.append("valid")
            if self.early_stopping:
                callbacks.append(lgb.early_stopping(self.early_stopping, verbose=False))

        self.model_ = lgb.train(
            self.params,
            dtrain,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet.")
        raw = self.model_.predict(X[self.feature_names_])
        return np.maximum(np.expm1(raw) if self.log_transform else raw, 0)

    @property
    def feature_importance(self) -> pd.Series:
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet.")
        return pd.Series(
            self.model_.feature_importance(importance_type="gain"),
            index=self.model_.feature_name(),
        ).sort_values(ascending=False)

    def save(self, path: str) -> None:
        self.model_.save_model(path)

    def load(self, path: str) -> "LightGBMForecaster":
        self.model_ = lgb.Booster(model_file=path)
        self.feature_names_ = self.model_.feature_name()
        return self


# ---------------------------------------------------------------------------
# 3. Ensemble
# ---------------------------------------------------------------------------

class EnsembleForecaster:
    """
    Simple weighted average of multiple base forecasters.

    Parameters
    ----------
    models  : list of (name, model_instance) tuples
    weights : list of floats summing to 1. If None, uses uniform weights.
    """

    def __init__(self, models: list, weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        assert abs(sum(self.weights) - 1.0) < 1e-6, "Weights must sum to 1."

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleForecaster":
        for _, m in self.models:
            m.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(X))
        for w, (_, m) in zip(self.weights, self.models):
            preds += w * m.predict(X)
        return np.maximum(preds, 0)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_lgbm_factory(params: Optional[dict] = None, early_stopping: int = 50):
    """Return a zero-argument callable that creates a fresh LightGBMForecaster."""
    def factory():
        return LightGBMForecaster(params=params, early_stopping=early_stopping)
    return factory


def make_naive_factory(seasonal_period: int = 16):
    def factory():
        return SeasonalNaiveForecaster(seasonal_period=seasonal_period)
    return factory
