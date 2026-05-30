# WRITEUP — Time Series Energy Demand Forecasting

## Motivation

Short-term electricity load forecasting is a standard benchmark problem with real economic stakes — a 1% forecasting error at grid scale translates to millions in balancing costs. It also cleanly tests whether you understand the difference between i.i.d. ML assumptions and temporal dependence. Most DS portfolios only show regression or classification; adding a rigorous time-series project demonstrates a distinct skill set.

---

## Dataset

PJM AEP hourly load (2004–2018, ~145 k rows). Chosen because:
- Freely available on Kaggle (no sign-up wall for demo viewers)
- Long enough history for yearly seasonality to be meaningful
- Single time series — keeps the project focused on forecasting methodology rather than multi-site aggregation

**Preprocessing decisions:**
- Forward-fill gaps ≤ 3 hours (clock-change artefacts). Longer gaps are flagged.
- Outliers (|z| > 4) replaced with centred rolling median — keeps energy in the series without inflating variance.

---

## Feature Engineering

LightGBM's best feature class turned out to be **lag features at 24 h and 168 h** (same hour yesterday and same hour last week). Calendar features (hour, day-of-week, month, holiday) capture demand cycles. Cyclical sine/cosine encoding prevents tree models from treating "hour 23 → hour 0" as a large jump.

**What I'd try next:** interaction features (hour × is_weekend), temperature regressors (outside temperature is the strongest driver of energy demand and is publicly available from NOAA).

---

## Models

### Naive baselines
Always included. The seasonal naive (24 h) achieved **3.90% MAPE** — a genuinely strong benchmark. Any model that cannot beat it is not worth deploying.

### SARIMA (2,0,2)(1,0,1)[24]
Fixed order chosen from EDA: d=0 (ADF confirms stationarity), p=q=2 (strong ACF/PACF at lags 1–2), seasonal P=Q=1 with s=24. Fitted via statsmodels SARIMAX on the last 2 years of training data (memory-efficient: only the estimated parameter vector and a 50-hour warm-up window are serialised — **1 KB** on disk vs 2.3 GB for the naive approach of pickling the full Kalman state). Result: **4.85% MAPE**. `pmdarima.auto_arima` was attempted but creates an O(n²) correlation matrix that exhausted RAM on the 17k-row subsample; fixed-order fitting via SARIMAX was the right call.

### Prophet
Facebook Prophet with multiplicative seasonality and US holidays. Result: **8.42% MAPE** — worst among non-trivial models. The multiplicative trend component over-extrapolates the declining demand trend from the training period (2004–2015) when applied to 2016–2018 test dates, producing systematically high forecasts.

### LightGBM (direct multi-step)
One LGBMRegressor per horizon step (24 models). Direct strategy avoids recursive error accumulation. Result: **5.12% MAPE**. This *lost to the seasonal naive*, which was a surprise. Post-hoc analysis suggests two causes: (1) the model was trained on 2004–2015 data with a gradual upward trend; the evaluation window (2016–2018) has a −86 MW/year declining trend, shifting the feature distribution; (2) the direct 24-step strategy builds features at time t only — it can't update its state as new observations arrive within the forecast window, unlike SARIMA.

**Why direct vs. recursive?** Recursive forecasting reuses its own predictions as inputs, compounding errors over the 24-step horizon. Direct training lets each step-h model independently optimise for that specific horizon offset — generally better, but more sensitive to distribution shift.

### N-BEATS (darts) — **winner**
Pure deep-learning baseline with no hand-crafted features. Learns trend/seasonality decomposition stacks internally. Result: **3.52% MAPE**, beating all models including the seasonal naive. Trained on the last 3 years of data (26,280 hours) with 30 epochs on CPU (~20 minutes). The fact that N-BEATS wins without any feature engineering is a strong signal that the structured inductive bias (decomposition stacks) is particularly well-suited to the double-seasonality structure of energy load.

---

## Evaluation

- **Rolling (expanding) window CV** — 5 folds, each fold advances by 1 week (168 h). Training set grows; test window is always 24 h ahead. Never shuffles the time axis.
- **MAPE** — primary metric (interpretable, percentage scale).
- **MASE** — not reportable here: with horizon=24 and seasonal_period=24, the denominator (seasonal naive MAE) is computed on an empty array. Would require a longer test window per fold.
- **Pinball loss** — measures calibration of prediction intervals. N-BEATS and SARIMA produce the best-calibrated intervals.

**Evaluation caveat:** 5 folds × 24h = 120 test observations total. This is a small sample; the ranking is directionally correct but MAPE estimates have high variance. A production evaluation would use hundreds of non-overlapping 24h windows.

---

## Deployment

Streamlit on HuggingFace Spaces. The app loads pre-trained `.joblib` artifacts at startup. Users can select any date in the test window, choose which models to overlay, and inspect the metric table for that specific slice.

**Scheduled retrain:** GitHub Actions cron runs every Monday — downloads the latest data, retrains all models, commits updated artifacts. This means the demo always reflects recent data without manual intervention.

---

## What didn't work

- **LSTM / seq2seq**: tried a basic PyTorch LSTM — underperformed N-BEATS with identical compute budget. The structured inductive bias of N-BEATS (trend/seasonality decomposition stacks) is more sample-efficient than a generic LSTM for energy data.
- **Recursive LightGBM**: worse than direct by ~0.8 MAPE percentage points due to error accumulation.
- **Feature selection with SHAP**: negligible improvement over using all lag/calendar features; added complexity for no gain on this dataset.

---

## What I'd do with more time / compute

1. Add **temperature regressors** from NOAA (strongest external driver of electricity demand).
2. Train a **Temporal Fusion Transformer** (darts) — multi-horizon, interpretable attention weights, native quantile forecasting.
3. **Ensemble** LightGBM + N-BEATS predictions with a simple linear blender.
4. Extend to **multiple PJM regions** and build a multi-variate model.
