"""N-BEATS univariate forecasting model via the darts library."""
from __future__ import annotations

import numpy as np
import pandas as pd


class NBEATSForecaster:
    """N-BEATS model for univariate energy load forecasting.

    Uses darts' NBEATSModel with a min-max scaler to keep gradients stable.
    Confidence intervals are approximated as ±5% of the point forecast because
    the generic N-BEATS architecture is deterministic; switch to
    darts.models.NBEATSModel with likelihood= for probabilistic intervals.
    """

    name = "N-BEATS"

    def __init__(
        self,
        input_chunk_length: int = 168,
        output_chunk_length: int = 24,
        num_stacks: int = 2,
        num_blocks: int = 3,
        num_layers: int = 4,
        layer_widths: int = 256,
        n_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        random_state: int = 42,
    ) -> None:
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self._kwargs = dict(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_widths=layer_widths,
            n_epochs=n_epochs,
            batch_size=batch_size,
            optimizer_kwargs={"lr": learning_rate},
            random_state=random_state,
            pl_trainer_kwargs={"enable_progress_bar": True},
        )
        self._model = None
        self._scaler = None
        self._series = None

    def fit(self, y: np.ndarray, index: pd.DatetimeIndex) -> "NBEATSForecaster":
        from darts import TimeSeries
        from darts.dataprocessing.transformers import Scaler
        from darts.models import NBEATSModel

        self._model = NBEATSModel(**self._kwargs)
        self._scaler = Scaler()

        series = TimeSeries.from_times_and_values(index, y, freq="h")
        scaled = self._scaler.fit_transform(series)
        self._model.fit(scaled)
        self._series = scaled
        return self

    def predict(
        self, horizon: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")

        pred_scaled = self._model.predict(n=horizon, series=self._series)
        pred = self._scaler.inverse_transform(pred_scaled)
        values = pred.values().flatten()
        margin = values * 0.05
        return values, values - margin, values + margin

    # ── Custom pickling ──────────────────────────────────────────────────────
    # darts NBEATSModel wraps PyTorch Lightning and cannot be pickled directly
    # by joblib.  We serialise the model to a temp .pt file, embed the raw
    # bytes in the state dict, and reconstruct on load.

    def __getstate__(self) -> dict:
        import os, tempfile

        state = self.__dict__.copy()
        if self._model is not None:
            tmp_dir = tempfile.mkdtemp()
            base = os.path.join(tmp_dir, "nbeats")
            self._model.save(base)
            # darts saves `base` + `base.ckpt` (no .pt extension)
            state["_model_files"] = {}
            for fname in os.listdir(tmp_dir):
                fpath = os.path.join(tmp_dir, fname)
                with open(fpath, "rb") as fh:
                    state["_model_files"][fname] = fh.read()
                os.unlink(fpath)
            os.rmdir(tmp_dir)
            state["_model"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        import os, tempfile
        from darts.models import NBEATSModel

        if "_model_files" in state:
            import torch
            tmp_dir = tempfile.mkdtemp()
            for fname, data in state.pop("_model_files").items():
                with open(os.path.join(tmp_dir, fname), "wb") as fh:
                    fh.write(data)
            base = os.path.join(tmp_dir, "nbeats")
            # PyTorch 2.6 changed default weights_only=True; darts checkpoints
            # include optimizer state which is blocked by the new default.
            # Patching at load time is safe — we generated this file ourselves.
            _orig = torch.load
            torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
            try:
                state["_model"] = NBEATSModel.load(base)
            finally:
                torch.load = _orig
            for fname in os.listdir(tmp_dir):
                os.unlink(os.path.join(tmp_dir, fname))
            os.rmdir(tmp_dir)
        self.__dict__.update(state)
