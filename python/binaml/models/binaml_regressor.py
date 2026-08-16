"""Rust-backed boolean feature regressor."""

from __future__ import annotations

import numpy as np

from binaml._core import BRegressorCore

from .base import PredictUpdateState, validate_binary_features, validate_finite_float_target

DEFAULT_LEARNING_RATE = 5e-3


class BRegressor(PredictUpdateState):
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
        super().__init__()
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

    def predict(self, features: np.ndarray) -> float:
        values = validate_binary_features(features, self.n_features)
        self._begin_predict()
        return float(self._model.predict(values))

    def update(self, target: float) -> None:
        self._begin_update()
        target_value = validate_finite_float_target(target)
        self._model.update(target_value)
        self._finish_update()

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
