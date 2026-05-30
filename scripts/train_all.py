"""Train all forecasting models and save artifacts to disk."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.download import download_pjm_data
from src.data.preprocess import preprocess
from src.models.baseline import NaiveLastValue, SeasonalNaive
from src.models.lgbm_model import LGBMForecaster
from src.models.nbeats_model import NBEATSForecaster
from src.models.prophet_model import ProphetModel
from src.models.sarima_model import SARIMAModel


def main(data_only: bool = False) -> None:
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    artifacts_dir = Path(cfg["artifacts_path"])
    artifacts_dir.mkdir(exist_ok=True)

    processed_path = (
        Path(cfg["data"]["processed_path"]) / "aep_hourly_clean.parquet"
    )

    if not processed_path.exists():
        raw_path = download_pjm_data(
            dataset=cfg["data"]["kaggle_dataset"],
            region=cfg["data"]["region"],
            output_dir=Path(cfg["data"]["raw_path"]),
        )
        preprocess(raw_path, processed_path, target_col=cfg["forecasting"]["target_col"])

    if data_only:
        print("Data ready. Skipping training (--data-only).")
        return

    df = pd.read_parquet(processed_path)
    target = cfg["forecasting"]["target_col"]
    horizon = cfg["forecasting"]["horizon"]
    seed = cfg.get("random_seed", 42)
    np.random.seed(seed)

    n = len(df)
    test_n = int(n * cfg["split"]["test_size"])
    val_n = int(n * cfg["split"]["val_size"])
    train_df = df.iloc[: n - val_n - test_n]
    y_train = train_df[target].values

    print(f"\nTraining set: {len(train_df):,} hours ({len(train_df) / 8760:.1f} years)")

    jobs = {
        "sarima": lambda: _train_sarima(y_train, cfg),
        "prophet": lambda: _train_prophet(y_train, train_df.index, cfg),
        "lgbm": lambda: _train_lgbm(train_df, target, horizon, cfg),
        "nbeats": lambda: _train_nbeats(y_train, train_df.index, cfg),
    }

    for name, factory in jobs.items():
        out = artifacts_dir / f"{name}.joblib"
        if out.exists():
            print(f"\nSkipping {name} — artifact already exists at {out}")
            continue
        print(f"\n{'='*40}\nTraining {name}\n{'='*40}")
        try:
            model = factory()
            joblib.dump(model, out)
            print(f"Saved to {out}")
        except Exception as exc:
            print(f"ERROR training {name}: {exc}")

    print("\nAll models trained.")


def _train_sarima(y: np.ndarray, cfg: dict) -> SARIMAModel:
    # Use last 2 years for SARIMAX fitting — sufficient for order (2,0,2)(1,0,1)[24]
    subsample = y[-17_520:]
    sarima_cfg = cfg["models"]["sarima"].copy()
    # YAML loads lists; convert to tuples for statsmodels
    sarima_cfg["order"] = tuple(sarima_cfg["order"])
    sarima_cfg["seasonal_order"] = tuple(sarima_cfg["seasonal_order"])
    m = SARIMAModel(**sarima_cfg)
    m.fit(subsample)
    return m


def _train_prophet(
    y: np.ndarray, index: pd.DatetimeIndex, cfg: dict
) -> ProphetModel:
    m = ProphetModel(**cfg["models"]["prophet"])
    m.fit(y, index)
    return m


def _train_lgbm(
    df: pd.DataFrame, target: str, horizon: int, cfg: dict
) -> LGBMForecaster:
    m = LGBMForecaster(horizon=horizon, **cfg["models"]["lgbm"])
    m.fit(df, target)
    return m


def _train_nbeats(
    y: np.ndarray, index: pd.DatetimeIndex, cfg: dict
) -> NBEATSForecaster:
    # Subsample to last 3 years for manageable CPU training time
    subsample_hours = 26_280
    y_sub = y[-subsample_hours:]
    idx_sub = index[-subsample_hours:]
    print(f"  N-BEATS fitting on last {len(y_sub):,} hours (CPU constraint)")
    m = NBEATSForecaster(**cfg["models"]["nbeats"])
    m.fit(y_sub, idx_sub)
    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-only", action="store_true", help="Download and preprocess data only"
    )
    args = parser.parse_args()
    main(data_only=args.data_only)
