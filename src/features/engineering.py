"""Time-series feature engineering: lags, rolling statistics, calendar."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_lag_features(
    df: pd.DataFrame, target_col: str, lags: list[int]
) -> pd.DataFrame:
    """Append lag columns for each offset in *lags* (in hours)."""
    for lag in lags:
        df[f"lag_{lag}h"] = df[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, target_col: str, windows: list[int]
) -> pd.DataFrame:
    """Append rolling mean and std for each window size (in hours).

    Shifts by 1 to avoid leakage: each row's statistic uses only past values.
    """
    for w in windows:
        shifted = df[target_col].shift(1)
        df[f"roll_mean_{w}h"] = shifted.rolling(w).mean()
        df[f"roll_std_{w}h"] = shifted.rolling(w).std()
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day-of-week, month, weekend, US holiday flags + cyclical encodings."""
    import holidays as hol

    idx = df.index
    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["year"] = idx.year
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    us_holidays = hol.US()
    df["is_holiday"] = idx.normalize().map(lambda d: int(d in us_holidays))

    # Cyclical encoding avoids discontinuity at period boundaries
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)

    return df


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "AEP_MW",
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Full feature engineering pipeline. Drops rows with any NaN after lag creation."""
    if lags is None:
        lags = [1, 2, 3, 24, 25, 48, 168, 169]
    if rolling_windows is None:
        rolling_windows = [24, 168]

    df = df.copy()
    df = add_lag_features(df, target_col, lags)
    df = add_rolling_features(df, target_col, rolling_windows)
    df = add_calendar_features(df)
    df = df.dropna()
    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = "AEP_MW") -> list[str]:
    """Return all column names except the target."""
    return [c for c in df.columns if c != target_col]
