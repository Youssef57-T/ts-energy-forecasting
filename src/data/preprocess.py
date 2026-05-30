"""Clean and preprocess raw PJM energy data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_raw(filepath: Path | str, target_col: str = "AEP_MW") -> pd.DataFrame:
    """Load raw CSV, parse datetime index, standardise column name."""
    df = pd.read_csv(filepath, parse_dates=["Datetime"], index_col="Datetime")
    df.index = df.index.tz_localize(None)
    df.index.name = "datetime"
    # The CSV has a single value column; rename it to target_col
    df.columns = [target_col]
    return df.sort_index()


def resample_hourly(df: pd.DataFrame, target_col: str = "AEP_MW") -> pd.DataFrame:
    """Ensure a complete hourly DatetimeIndex by forward-filling gaps ≤ 3 h.

    DST "fall back" creates duplicate timestamps; average them before reindexing.
    """
    if df.index.duplicated().any():
        n_dups = df.index.duplicated().sum()
        print(f"Averaging {n_dups} duplicate DST timestamps.")
        df = df.groupby(df.index).mean()

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_idx)
    missing_before = int(df[target_col].isna().sum())
    df[target_col] = df[target_col].ffill(limit=3)
    missing_after = int(df[target_col].isna().sum())
    if missing_before:
        print(
            f"Filled {missing_before - missing_after} gaps via ffill "
            f"({missing_after} remaining > 3 h)"
        )
    return df


def remove_outliers(
    df: pd.DataFrame, target_col: str = "AEP_MW", z_thresh: float = 4.0
) -> pd.DataFrame:
    """Replace |z| > z_thresh values with the centred rolling median."""
    col = df[target_col]
    rolling_med = col.rolling(window=168, center=True, min_periods=24).median()
    z = (col - col.mean()) / col.std()
    mask = z.abs() > z_thresh
    n = int(mask.sum())
    if n:
        print(f"Replacing {n} outliers (|z| > {z_thresh}) with rolling median.")
        df.loc[mask, target_col] = rolling_med[mask]
    return df


def preprocess(
    raw_path: Path | str,
    output_path: Path | str,
    target_col: str = "AEP_MW",
) -> pd.DataFrame:
    """Full preprocessing pipeline: load, resample, outlier-clean, save parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw(raw_path, target_col)
    df = resample_hourly(df, target_col)
    df = remove_outliers(df, target_col)

    df.to_parquet(output_path)
    print(f"Saved {len(df):,} rows to {output_path}")
    return df
