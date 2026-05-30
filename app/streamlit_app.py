"""Streamlit forecast dashboard — PJM AEP 24-hour-ahead energy load forecasting."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import compute_metrics
from src.models.baseline import SeasonalNaive

CONFIG_PATH = Path("configs/config.yaml")
ARTIFACTS = Path("artifacts")
PALETTE = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_resource
def load_saved_models() -> dict:
    return {p.stem: joblib.load(p) for p in sorted(ARTIFACTS.glob("*.joblib"))}


def forecast_band_trace(
    x: list, low: np.ndarray, high: np.ndarray, color: str, name: str
) -> go.Scatter:
    return go.Scatter(
        x=x + x[::-1],
        y=list(high) + list(low[::-1]),
        fill="toself",
        fillcolor=color,
        opacity=0.15,
        line=dict(color="rgba(0,0,0,0)"),
        name=f"{name} 80% CI",
        showlegend=False,
    )


def build_forecast_figure(
    actuals: pd.Series,
    forecast_idx: pd.DatetimeIndex,
    preds: dict[str, dict],
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actuals.index, y=actuals.values,
            name="Actuals", line=dict(color="black", width=1.5),
        )
    )
    for i, (model_name, data) in enumerate(preds.items()):
        color = PALETTE[i % len(PALETTE)]
        if data.get("low") is not None and data.get("high") is not None:
            fig.add_trace(
                forecast_band_trace(
                    list(forecast_idx), data["low"], data["high"], color, model_name
                )
            )
        fig.add_trace(
            go.Scatter(
                x=forecast_idx, y=data["pred"],
                name=model_name, line=dict(color=color, width=2),
            )
        )
    fig.update_layout(
        title="24-Hour-Ahead Energy Load Forecast",
        xaxis_title="Date / Hour",
        yaxis_title="Load (MW)",
        hovermode="x unified",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def main() -> None:
    st.set_page_config(
        page_title="Energy Demand Forecast", page_icon="⚡", layout="wide"
    )
    st.title("⚡ Energy Demand Forecasting Dashboard")
    st.caption(
        "PJM AEP region · 24-hour-ahead forecast · "
        "Seasonal Naive, SARIMA, Prophet, LightGBM, N-BEATS"
    )

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    processed_path = (
        Path(cfg["data"]["processed_path"]) / "aep_hourly_clean.parquet"
    )
    if not processed_path.exists():
        st.error(
            f"Processed data not found at `{processed_path}`. "
            "Run `make data` or `python scripts/train_all.py --data-only` first."
        )
        st.stop()

    df = load_data(str(processed_path))
    target = cfg["forecasting"]["target_col"]
    horizon = cfg["forecasting"]["horizon"]

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Settings")
    max_date = df.index.max().date()
    min_date = (df.index.min() + pd.Timedelta(days=60)).date()

    selected_date = st.sidebar.date_input(
        "Forecast start date",
        value=max_date - pd.Timedelta(days=7),
        min_value=min_date,
        max_value=max_date,
    )
    all_model_names = ["Seasonal Naive"] + [
        p.stem.upper() if p.stem in ("sarima",) else p.stem.capitalize()
        for p in sorted(ARTIFACTS.glob("*.joblib"))
    ]
    # friendlier labels
    label_map = {"sarima": "SARIMA", "prophet": "Prophet", "lgbm": "LightGBM", "nbeats": "N-BEATS"}
    all_model_names = ["Seasonal Naive"] + [
        label_map.get(p.stem, p.stem) for p in sorted(ARTIFACTS.glob("*.joblib"))
    ]
    selected_models = st.sidebar.multiselect(
        "Models to display",
        options=all_model_names,
        default=["Seasonal Naive", "LightGBM"] if "LightGBM" in all_model_names else all_model_names[:2],
    )

    # ── Slice data ────────────────────────────────────────────────────────────
    cutoff = pd.Timestamp(selected_date)
    context_size = cfg["cv"]["initial_train_size"]
    context = df[df.index < cutoff].tail(context_size)
    ground_truth = df[df.index >= cutoff].head(horizon)

    if len(ground_truth) < horizon:
        st.warning(
            "Not enough data after the selected date for a full 24-hour forecast. "
            "Choose an earlier date."
        )
        st.stop()

    forecast_idx = ground_truth.index
    preds: dict[str, dict] = {}

    # ── Seasonal Naive ────────────────────────────────────────────────────────
    if "Seasonal Naive" in selected_models:
        model = SeasonalNaive(24)
        model.fit(context[target].values)
        p = model.predict(horizon)
        preds["Seasonal Naive"] = {"pred": p, "low": None, "high": None}

    # ── Saved models ──────────────────────────────────────────────────────────
    saved = load_saved_models()
    stem_to_label = {v: k for k, v in label_map.items()}  # e.g. "LightGBM" -> "lgbm"
    for display_name in selected_models:
        stem = stem_to_label.get(display_name)
        if stem and stem in saved:
            with st.spinner(f"Running {display_name}…"):
                try:
                    m = saved[stem]
                    if stem == "lgbm":
                        p, lo, hi = m.predict(context, target)
                    else:
                        p, lo, hi = m.predict(horizon)
                    preds[display_name] = {"pred": p[:horizon], "low": lo, "high": hi}
                except Exception as exc:
                    st.warning(f"{display_name} failed: {exc}")

    # ── Layout ────────────────────────────────────────────────────────────────
    actuals_window = pd.concat([context[target].tail(168), ground_truth[target]])

    col_plot, col_metrics = st.columns([3, 1])
    with col_plot:
        fig = build_forecast_figure(actuals_window, forecast_idx, preds)
        st.plotly_chart(fig, use_container_width=True)

    with col_metrics:
        st.subheader("Metrics vs actuals")
        rows = []
        for model_name, data in preds.items():
            m = compute_metrics(
                ground_truth[target].values,
                data["pred"][:horizon],
                data.get("low"),
                data.get("high"),
            )
            row = {"Model": model_name}
            row.update(m.to_dict())
            rows.append(row)
        if rows:
            st.dataframe(
                pd.DataFrame(rows).set_index("Model"),
                use_container_width=True,
            )

    with st.expander("📊 About this dashboard"):
        st.markdown(
            """
            **Dataset**: PJM AEP hourly energy consumption (Kaggle)
            **Horizon**: 24 hours ahead (next-day forecast)
            **Models**: Seasonal Naive · SARIMA · Prophet · LightGBM · N-BEATS
            **Confidence bands**: 80% prediction interval (where available)
            **Metrics**: MAPE, RMSE, MASE, Pinball loss (q=0.1/0.5/0.9)
            """
        )

    st.markdown("---")
    st.caption("Built with Streamlit · Source on GitHub")


if __name__ == "__main__":
    main()
