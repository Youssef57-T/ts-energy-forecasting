"""Facebook Prophet forecasting model."""
from __future__ import annotations

import numpy as np
import pandas as pd


class ProphetModel:
    """Wrapper around Facebook Prophet for hourly energy forecasting."""

    name = "Prophet"

    def __init__(
        self,
        seasonality_mode: str = "multiplicative",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
    ) -> None:
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self._model = None
        self._last_date: pd.Timestamp | None = None

    def fit(self, y: np.ndarray, index: pd.DatetimeIndex) -> "ProphetModel":
        from prophet import Prophet

        df = pd.DataFrame({"ds": index, "y": y})
        self._model = Prophet(
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            interval_width=0.8,
        )
        self._model.add_country_holidays(country_name="US")
        self._model.fit(df)
        self._last_date = index[-1]
        return self

    def predict(
        self, horizon: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")

        future = self._model.make_future_dataframe(
            periods=horizon, freq="h", include_history=False
        )
        forecast = self._model.predict(future)
        return (
            forecast["yhat"].values,
            forecast["yhat_lower"].values,
            forecast["yhat_upper"].values,
        )
