"""Naive baseline forecasting models."""
from __future__ import annotations

import numpy as np


class NaiveLastValue:
    """Forecast = last observed value repeated for all horizon steps."""

    name = "Naive (last value)"

    def fit(self, y: np.ndarray) -> "NaiveLastValue":
        self._last = float(y[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._last)


class SeasonalNaive:
    """Forecast = value from *seasonal_period* hours ago, cyclically extended."""

    def __init__(self, seasonal_period: int = 24) -> None:
        self.seasonal_period = seasonal_period
        self.name = f"Seasonal Naive ({seasonal_period}h)"

    def fit(self, y: np.ndarray) -> "SeasonalNaive":
        self._season = y[-self.seasonal_period :].copy()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.array(
            [self._season[h % self.seasonal_period] for h in range(horizon)]
        )
