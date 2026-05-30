# Project 4 — Time Series Energy Demand Forecasting: PLAN

## Decisions locked

| # | Decision | Choice |
|---|----------|--------|
| D1 | Dataset | PJM AEP hourly (Kaggle: `robikscube/hourly-energy-consumption`) |
| D2 | Deep learning model | N-BEATS via `darts` library |
| D3 | Forecast horizon | 24 hours ahead |
| D4 | Deployment | HuggingFace Spaces (Streamlit) |

---

## File tree

```
project-4-time-series-forecasting/
├── .github/
│   └── workflows/
│       ├── ci.yml               # lint, test on push/PR
│       └── retrain.yml          # Monday 6am UTC cron retrain
├── data/
│   ├── raw/                     # downloaded CSVs — gitignored
│   └── processed/               # cleaned parquet — gitignored
├── notebooks/
│   ├── 01_eda.ipynb             # stationarity, decomposition, ACF/PACF
│   └── 02_model_comparison.ipynb
├── src/
│   ├── data/
│   │   ├── download.py          # Kaggle API fetch + checksum
│   │   └── preprocess.py        # resample, outlier removal, save parquet
│   ├── features/
│   │   └── engineering.py       # lags, rolling stats, calendar, cyclical encoding
│   ├── models/
│   │   ├── baseline.py          # NaiveLastValue + SeasonalNaive (24h / 168h)
│   │   ├── sarima_model.py      # pmdarima auto_arima seasonal
│   │   ├── prophet_model.py     # Facebook Prophet + US holidays
│   │   ├── lgbm_model.py        # LightGBM direct multi-step
│   │   └── nbeats_model.py      # N-BEATS via darts
│   └── evaluation/
│       └── metrics.py           # MAPE, RMSE, MASE, pinball loss, rolling CV splits
├── app/
│   └── streamlit_app.py         # date picker → forecast + CI bands + model table
├── tests/
│   ├── test_features.py
│   ├── test_metrics.py
│   └── test_models.py
├── configs/
│   └── config.yaml              # horizon, splits, all model hyperparams
├── scripts/
│   ├── train_all.py             # trains all 5 models, saves .joblib artifacts
│   └── evaluate_all.py          # rolling CV, prints comparison table
├── artifacts/                   # saved models — gitignored
├── .env.example
├── .gitignore
├── LICENSE
├── Makefile
├── PLAN.md                      # this file
├── pyproject.toml
├── requirements.txt
├── README.md
└── WRITEUP.md
```

---

## 7-step execution checklist

- [x] **Step 1 — Scaffold**: repo structure, dependencies, CI, README skeleton
- [x] **Step 2 — Data**: download AEP_hourly.csv (121,296 rows), validate, preprocess to parquet
- [x] **Step 3 — EDA**: `notebooks/01_eda.ipynb` — 10 sections, 8 figures, ADF/KPSS/STL/ACF/PACF
- [x] **Step 4 — Modeling**: all 5 models trained; 5-fold rolling CV complete
- [ ] **Step 5 — App**: Streamlit dashboard running locally with all model preds + CI bands ← *current*
- [ ] **Step 6 — Deploy**: push to HuggingFace Spaces; polish README (Mermaid, metrics table, demo GIF); WRITEUP.md
- [ ] **Step 7 — CV update**: add resume bullet to `cv/CV.md`, regenerate PDF, commit

---

## Actual results (5-fold rolling CV, 2016–2018 eval window)

| Model | MAPE (%) | RMSE (MW) |
|-------|----------|-----------|
| **N-BEATS** | **3.52** | **591** |
| Seasonal Naive 24h | 3.90 | 651 |
| SARIMA | 4.85 | 757 |
| LightGBM | 5.12 | 890 |
| Seasonal Naive 168h | 5.68 | 965 |
| Prophet | 8.42 | 1290 |
| Naive (last value) | 10.47 | 1666 |

---

## Key design decisions

- **Direct multi-step forecasting** for LightGBM: one model per horizon step (24 models). Avoids error accumulation from recursive forecasting.
- **Rolling (expanding) window CV**, not random KFold. Respects temporal ordering.
- **Cyclical encoding** for hour/day-of-week/month: sine/cosine pairs prevent distance distortion in tree models.
- **Confidence intervals**: SARIMA uses pmdarima's native CI; Prophet uses built-in uncertainty intervals; N-BEATS uses ±5% approximation; LightGBM uses quantile regression variants.
- **Scheduled retrain**: GitHub Actions cron every Monday re-downloads latest data and retrains all models, commits updated artifacts.
