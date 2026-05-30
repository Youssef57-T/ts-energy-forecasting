"""Tests for evaluation metrics and CV split utilities."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_metrics,
    mape,
    mase,
    pinball_loss,
    rmse,
    rolling_cv_splits,
)


def test_mape_perfect() -> None:
    y = np.array([100.0, 200.0, 300.0])
    assert mape(y, y) == 0.0


def test_rmse_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


def test_mase_perfect() -> None:
    y = np.ones(48) * 100.0
    assert mase(y, y, seasonal_period=24) == 0.0


def test_pinball_perfect_median() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert pinball_loss(y, y, 0.5) == 0.0


def test_compute_metrics_keys() -> None:
    y = np.array([100.0, 200.0, 300.0, 400.0])
    m = compute_metrics(y, y + 5.0)
    d = m.to_dict()
    for key in ["MAPE (%)", "RMSE", "MASE", "Pinball-10", "Pinball-50", "Pinball-90"]:
        assert key in d


def test_rolling_cv_temporal_order() -> None:
    splits = list(
        rolling_cv_splits(n=5000, initial_train_size=3000, horizon=24, step_size=168, n_splits=5)
    )
    assert len(splits) > 0
    for train_idx, test_idx in splits:
        assert len(test_idx) == 24
        assert train_idx[-1] < test_idx[0]


def test_rolling_cv_expanding_window() -> None:
    splits = list(
        rolling_cv_splits(n=5000, initial_train_size=3000, horizon=24, step_size=168, n_splits=3)
    )
    train_sizes = [len(t) for t, _ in splits]
    assert train_sizes == sorted(train_sizes), "Training window must grow each fold"
