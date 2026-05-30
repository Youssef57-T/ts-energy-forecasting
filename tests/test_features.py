"""Tests for feature engineering module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_feature_matrix,
    get_feature_columns,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-01", periods=500, freq="h")
    return pd.DataFrame({"AEP_MW": rng.uniform(10_000, 20_000, 500)}, index=idx)


def test_lag_features_shape(sample_df: pd.DataFrame) -> None:
    result = add_lag_features(sample_df.copy(), "AEP_MW", lags=[1, 24])
    assert "lag_1h" in result.columns
    assert "lag_24h" in result.columns
    assert len(result) == len(sample_df)


def test_lag_features_values(sample_df: pd.DataFrame) -> None:
    result = add_lag_features(sample_df.copy(), "AEP_MW", lags=[1])
    assert result["lag_1h"].iloc[1] == sample_df["AEP_MW"].iloc[0]


def test_calendar_features_present(sample_df: pd.DataFrame) -> None:
    result = add_calendar_features(sample_df.copy())
    for col in ["hour", "dayofweek", "month", "is_weekend", "is_holiday"]:
        assert col in result.columns


def test_calendar_hour_range(sample_df: pd.DataFrame) -> None:
    result = add_calendar_features(sample_df.copy())
    assert result["hour"].min() == 0
    assert result["hour"].max() == 23


def test_rolling_features_present(sample_df: pd.DataFrame) -> None:
    result = add_rolling_features(sample_df.copy(), "AEP_MW", windows=[24])
    assert "roll_mean_24h" in result.columns
    assert "roll_std_24h" in result.columns


def test_build_feature_matrix_no_nans(sample_df: pd.DataFrame) -> None:
    result = build_feature_matrix(sample_df.copy(), "AEP_MW")
    assert result.isna().sum().sum() == 0


def test_get_feature_columns_excludes_target(sample_df: pd.DataFrame) -> None:
    feat_df = build_feature_matrix(sample_df.copy(), "AEP_MW")
    cols = get_feature_columns(feat_df, "AEP_MW")
    assert "AEP_MW" not in cols
    assert len(cols) > 0
