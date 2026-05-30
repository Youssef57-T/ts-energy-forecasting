"""Forecasting evaluation metrics and rolling cross-validation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class ForecastMetrics:
    mape: float
    rmse: float
    mase: float
    pinball_10: float
    pinball_50: float
    pinball_90: float

    def to_dict(self) -> dict[str, float]:
        return {
            "MAPE (%)": round(self.mape * 100, 3),
            "RMSE": round(self.rmse, 1),
            "MASE": round(self.mase, 4),
            "Pinball-10": round(self.pinball_10, 4),
            "Pinball-50": round(self.pinball_50, 4),
            "Pinball-90": round(self.pinball_90, 4),
        }


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase(
    y_true: np.ndarray, y_pred: np.ndarray, seasonal_period: int = 24
) -> float:
    """Mean Absolute Scaled Error, scaled by seasonal-naive in-sample MAE."""
    naive_err = np.abs(y_true[seasonal_period:] - y_true[:-seasonal_period])
    denom = np.mean(naive_err)
    if denom == 0:
        return np.inf
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def pinball_loss(
    y_true: np.ndarray, y_pred: np.ndarray, quantile: float
) -> float:
    errors = y_true - y_pred
    return float(
        np.mean(np.where(errors >= 0, quantile * errors, (quantile - 1) * errors))
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_low: np.ndarray | None = None,
    y_pred_high: np.ndarray | None = None,
    seasonal_period: int = 24,
) -> ForecastMetrics:
    low = y_pred_low if y_pred_low is not None else y_pred
    high = y_pred_high if y_pred_high is not None else y_pred
    return ForecastMetrics(
        mape=mape(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        mase=mase(y_true, y_pred, seasonal_period),
        pinball_10=pinball_loss(y_true, low, 0.1),
        pinball_50=pinball_loss(y_true, y_pred, 0.5),
        pinball_90=pinball_loss(y_true, high, 0.9),
    )


def rolling_cv_splits(
    n: int,
    initial_train_size: int,
    horizon: int,
    step_size: int,
    n_splits: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_indices, test_indices) for expanding-window CV.

    Each fold extends the training window by *step_size* hours. Temporal
    ordering is always respected — no future data leaks into training.
    """
    for i in range(n_splits):
        train_end = initial_train_size + i * step_size
        test_end = train_end + horizon
        if test_end > n:
            break
        yield np.arange(0, train_end), np.arange(train_end, test_end)
