"""Rust-backed boolean feature regressor."""

from __future__ import annotations

import numpy as np

from binaml._core import BRegressorCore

DEFAULT_LEARNING_RATE = 5e-3


class BRegressor:
    """Online regression over an ensemble of batch-learned boolean functions."""

    def __init__(
        self,
        n_features: int,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        l2: float = 1e-4,
        batch_size: int = 16,
        sgd_steps: int = 5,
        parent_top_k: int = 8,
        max_layers: int = 3,
        max_functions: int = 64,
    ) -> None:
        if (
            isinstance(n_features, bool)
            or not isinstance(n_features, int)
            or n_features < 1
            or learning_rate <= 0
            or l2 < 0
            or isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
            or isinstance(sgd_steps, bool)
            or not isinstance(sgd_steps, int)
            or sgd_steps < 1
            or isinstance(max_functions, bool)
            or not isinstance(max_functions, int)
            or max_functions < 1
        ):
            raise ValueError("invalid feature regressor configuration")
        self.n_features = n_features
        self.batch_size = batch_size
        self.sgd_steps = sgd_steps
        self._model = BRegressorCore(
            n_features,
            learning_rate,
            l2,
            batch_size,
            sgd_steps,
            parent_top_k,
            max_layers,
            max_functions,
        )
        self._prediction_pending = False

    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features)
        if values.shape != (self.n_features,):
            raise ValueError(f"features must have shape ({self.n_features},)")
        if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
            raise ValueError("features must be finite")
        if not np.all((values == 0) | (values == 1)):
            raise ValueError("features must be binary")
        return np.ascontiguousarray(values, dtype=np.uint8)

    def predict(self, features: np.ndarray) -> float:
        if self._prediction_pending:
            raise RuntimeError("observe(features, target) must follow each predict(features)")
        prediction = float(self._model.predict(self._validate_features(features)))
        self._prediction_pending = True
        return prediction

    def observe(self, features: np.ndarray, target: float) -> None:
        if not self._prediction_pending:
            raise RuntimeError("observe(features, target) requires a preceding predict(features)")
        target_value = float(target)
        if not np.isfinite(target_value):
            raise ValueError("target must be finite")
        self._model.observe(self._validate_features(features), target_value)
        self._prediction_pending = False

    @property
    def intercept(self) -> float:
        return float(self._model.intercept)

    @property
    def n_observed(self) -> int:
        return int(self._model.n_observed)

    @property
    def function_count(self) -> int:
        return int(self._model.function_count)

    def weight(self, index: int) -> float | None:
        value = self._model.weight(index)
        return None if value is None else float(value)
