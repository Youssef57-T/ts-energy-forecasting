"""LightGBM direct multi-step forecaster with lag and calendar features."""
from __future__ import annotations

import numpy as np
import pandas as pd


class LGBMForecaster:
    """One LightGBM model per horizon step (direct multi-step strategy).

    Avoids the error accumulation of recursive forecasting by training a
    separate model for each step h in [0, horizon).
    """

    name = "LightGBM"

    def __init__(
        self,
        horizon: int = 24,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        max_depth: int = 7,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.horizon = horizon
        self._lgbm_params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbose=-1,
        )
        self._models: list = []
        self._feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame, target_col: str = "AEP_MW") -> "LGBMForecaster":
        import lightgbm as lgb
        from src.features.engineering import build_feature_matrix, get_feature_columns

        feat_df = build_feature_matrix(df, target_col)
        self._feature_cols = get_feature_columns(feat_df, target_col)
        X = feat_df[self._feature_cols].values

        self._models = []
        for h in range(self.horizon):
            y = feat_df[target_col].shift(-h).dropna().values
            X_h = X[: len(y)]
            m = lgb.LGBMRegressor(**self._lgbm_params)
            m.fit(X_h, y)
            self._models.append(m)

        return self

    def predict(
        self, df_context: pd.DataFrame, target_col: str = "AEP_MW"
    ) -> tuple[np.ndarray, None, None]:
        from src.features.engineering import build_feature_matrix

        feat_df = build_feature_matrix(df_context, target_col)
        X_last = feat_df[self._feature_cols].iloc[[-1]].values
        preds = np.array([m.predict(X_last)[0] for m in self._models])
        return preds, None, None

    @property
    def feature_importances_(self) -> pd.Series:
        if not self._models:
            raise RuntimeError("Model not fitted.")
        avg = np.mean([m.feature_importances_ for m in self._models], axis=0)
        return pd.Series(avg, index=self._feature_cols).sort_values(ascending=False)
