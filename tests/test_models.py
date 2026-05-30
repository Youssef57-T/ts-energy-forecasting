"""Smoke tests for model fit/predict contracts."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.baseline import NaiveLastValue, SeasonalNaive


@pytest.fixture
def y() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(10_000, 20_000, size=200)


def test_naive_last_value_output(y: np.ndarray) -> None:
    model = NaiveLastValue()
    model.fit(y)
    pred = model.predict(24)
    assert pred.shape == (24,)
    assert np.all(pred == y[-1])


def test_seasonal_naive_24h_output(y: np.ndarray) -> None:
    model = SeasonalNaive(24)
    model.fit(y)
    pred = model.predict(24)
    assert pred.shape == (24,)
    np.testing.assert_array_equal(pred, y[-24:])


@pytest.mark.parametrize("period", [24, 168])
def test_seasonal_naive_shape(y: np.ndarray, period: int) -> None:
    model = SeasonalNaive(period)
    model.fit(y)
    assert model.predict(24).shape == (24,)


def test_seasonal_naive_wraps_correctly(y: np.ndarray) -> None:
    model = SeasonalNaive(24)
    model.fit(y)
    pred_48 = model.predict(48)
    # First 24 and second 24 should be identical (same season repeats)
    np.testing.assert_array_equal(pred_48[:24], pred_48[24:])
