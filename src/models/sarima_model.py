"""SARIMA forecasting model — lightweight serialization via parameter extraction."""
from __future__ import annotations

import numpy as np
import pandas as pd


class SARIMAModel:
    """Seasonal ARIMA via statsmodels SARIMAX with slim serialization.

    Instead of pickling the full fitted result (which stores all Kalman filter
    state history, ~2 GB for 17k obs), we save only the estimated parameter
    vector and a short initialization window.  On predict(), we re-apply the
    Kalman filter to the initialization window using the frozen parameters —
    this reconstructs the end-state in milliseconds.

    Order (2,0,2)(1,0,1)[24] chosen from EDA:
      - ADF p=0 → d=0 (stationary, no differencing needed)
      - ACF/PACF strong at lags 1-2 → p=q=2
      - Daily seasonality → s=24; one seasonal AR/MA term each
    """

    name = "SARIMA"

    def __init__(
        self,
        order: tuple[int, int, int] = (2, 0, 2),
        seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 24),
        **_kwargs: object,  # absorb stale config keys gracefully
    ) -> None:
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self._params: np.ndarray | None = None
        self._init_window: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> "SARIMAModel":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        print(f"  Fitting SARIMAX{self.order}x{self.seasonal_order} on {len(y):,} obs")
        model = SARIMAX(
            y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=200)
        print(f"  AIC={result.aic:.1f}  BIC={result.bic:.1f}")

        # Save only params + enough history to reconstruct the end-state
        self._params = np.asarray(result.params).copy()
        # Keep 2×s + p observations so the filter has a proper warm-up window
        warmup = 2 * self.seasonal_order[3] + self.order[0]
        self._init_window = y[-warmup:].copy()
        return self

    def predict(
        self, horizon: int, return_conf_int: bool = True
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        if self._params is None:
            raise RuntimeError("Call fit() before predict().")

        from statsmodels.tsa.statespace.sarimax import SARIMAX

        # Re-apply the filter with frozen params to rebuild the end-state
        model = SARIMAX(
            self._init_window,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.filter(self._params)
        forecast = result.get_forecast(steps=horizon)
        # predicted_mean and conf_int may return ndarray or Series depending on
        # statsmodels version; use np.asarray() to normalise both cases
        pred = np.asarray(forecast.predicted_mean)

        if return_conf_int:
            ci = np.asarray(forecast.conf_int(alpha=0.2))
            return pred, ci[:, 0], ci[:, 1]

        return pred, None, None
