"""Rust-backed boolean feature regressor."""

from __future__ import annotations

import numpy as np

from binaml._core import BRegressorCore

from .base import PredictUpdateState, validate_binary_features, validate_finite_float_target

DEFAULT_LEARNING_RATE = 0.03


class BRegressor(PredictUpdateState):
    """Online regression over an ensemble of batch-learned conjunction experts."""

    def __init__(
        self,
        n_features: int,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        l2: float = 5e-4,
        batch_size: int = 16,
        sgd_steps: int = 20,
        max_conjunctions: int = 8,
        max_conjunction_length: int = 7,
        max_functions: int = 64,
        max_experts: int = 64,
        stale_layers: int = 2,
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
            or isinstance(stale_layers, bool)
            or not isinstance(stale_layers, int)
            or stale_layers < 1
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
            max_conjunctions,
            max_conjunction_length,
            max_functions,
            max_experts,
            stale_layers,
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
