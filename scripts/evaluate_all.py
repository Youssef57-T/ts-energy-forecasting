"""Evaluate all models with rolling cross-validation and print a comparison table."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import compute_metrics, rolling_cv_splits
from src.models.baseline import NaiveLastValue, SeasonalNaive


def _predict_lgbm(model, context_df: pd.DataFrame, target: str, horizon: int):
    p, lo, hi = model.predict(context_df, target)
    return p[:horizon], lo, hi


def _predict_sarima(model, y_context: np.ndarray, horizon: int):
    # Slide the init window to the end of the fold's context
    warmup = len(model._init_window)
    model._init_window = y_context[-warmup:].copy()
    return model.predict(horizon)


def _predict_nbeats(model, context_df: pd.DataFrame, y_context: np.ndarray, horizon: int):
    from darts import TimeSeries

    n_ctx = model.input_chunk_length
    series = TimeSeries.from_times_and_values(
        context_df.index[-n_ctx:], y_context[-n_ctx:].astype(float), freq="h"
    )
    scaled = model._scaler.transform(series)
    pred_scaled = model._model.predict(n=horizon, series=scaled)
    pred = model._scaler.inverse_transform(pred_scaled)
    p = pred.values().flatten()
    return p, p * 0.95, p * 1.05


def _predict_prophet(model, test_df: pd.DataFrame, horizon: int):
    # Prophet uses its trained trend+seasonality to predict the test dates directly
    future = pd.DataFrame({"ds": test_df.index[:horizon]})
    forecast = model._model.predict(future)
    return (
        forecast["yhat"].values,
        forecast["yhat_lower"].values,
        forecast["yhat_upper"].values,
    )


def main() -> None:
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    processed_path = Path(cfg["data"]["processed_path"]) / "aep_hourly_clean.parquet"
    df = pd.read_parquet(processed_path)
    target = cfg["forecasting"]["target_col"]
    horizon = cfg["forecasting"]["horizon"]
    cv_cfg = cfg["cv"]

    n = len(df)
    test_n = int(n * cfg["split"]["test_size"])
    val_n = int(n * cfg["split"]["val_size"])
    eval_df = df.iloc[n - val_n - test_n :]
    y_eval = eval_df[target].values

    splits = list(
        rolling_cv_splits(
            n=len(y_eval),
            initial_train_size=cv_cfg["initial_train_size"],
            horizon=horizon,
            step_size=cv_cfg["step_size"],
            n_splits=cv_cfg["n_splits"],
        )
    )
    print(f"CV: {len(splits)} folds  horizon={horizon}h  step={cv_cfg['step_size']}h\n")

    results: list[dict] = []

    # ── Baselines (refit each fold, trivially fast) ──────────────────────────
    for ModelClass, kwargs, label in [
        (NaiveLastValue, {}, "Naive (last value)"),
        (SeasonalNaive, {"seasonal_period": 24}, "Seasonal Naive 24h"),
        (SeasonalNaive, {"seasonal_period": 168}, "Seasonal Naive 168h"),
    ]:
        fold_metrics = []
        for train_idx, test_idx in splits:
            m = ModelClass(**kwargs)
            m.fit(y_eval[train_idx])
            pred = m.predict(horizon)
            fold_metrics.append(compute_metrics(y_eval[test_idx], pred).to_dict())
        avg = {k: round(float(np.mean([fm[k] for fm in fold_metrics])), 4) for k in fold_metrics[0]}
        avg["Model"] = label
        results.append(avg)
        print(f"  {label:30s}  MAPE={avg['MAPE (%)']:.2f}%")

    print()

    # ── Saved model artifacts ────────────────────────────────────────────────
    artifacts_dir = Path(cfg["artifacts_path"])
    for fname in sorted(artifacts_dir.glob("*.joblib")):
        model = joblib.load(fname)
        fold_metrics = []
        stem = fname.stem

        print(f"  Evaluating {stem}...", end="", flush=True)

        for train_idx, test_idx in splits:
            y_context = y_eval[train_idx]
            context_df = eval_df.iloc[train_idx]
            test_slice = eval_df.iloc[test_idx]
            y_true = y_eval[test_idx]

            try:
                if stem == "lgbm":
                    p, lo, hi = _predict_lgbm(model, context_df, target, horizon)
                elif stem == "sarima":
                    p, lo, hi = _predict_sarima(model, y_context, horizon)
                elif stem == "nbeats":
                    p, lo, hi = _predict_nbeats(model, context_df, y_context, horizon)
                elif stem == "prophet":
                    p, lo, hi = _predict_prophet(model, test_slice, horizon)
                else:
                    p, lo, hi = model.predict(horizon)

                fold_metrics.append(
                    compute_metrics(y_true, p[:horizon], lo, hi).to_dict()
                )
            except Exception as exc:
                print(f"\n    Warning: fold failed — {exc}")

        if fold_metrics:
            avg = {k: round(float(np.mean([fm[k] for fm in fold_metrics])), 4) for k in fold_metrics[0]}
            avg["Model"] = model.name
            results.append(avg)
            print(f"  MAPE={avg['MAPE (%)']:.2f}%")
        else:
            print("  all folds failed — skipped")

    # ── Summary table ────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results).set_index("Model").sort_values("MAPE (%)")

    print("\n=== Model Comparison (5-fold rolling CV) ===")
    print(results_df.to_string())

    out = artifacts_dir / "model_comparison.csv"
    results_df.to_csv(out)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
