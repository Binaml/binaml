"""Rust-backed boolean feature classifier."""

from __future__ import annotations

import numpy as np

from binaml._core import BClassifierCore

from .base import PredictObserveState, validate_binary_features, validate_class_index

DEFAULT_LEARNING_RATE = 0.016


class BClassifier(PredictObserveState):
    """Online multiclass classification over an ensemble of batch-learned boolean functions."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        l2: float = 0.0,
        batch_size: int = 6,
        sgd_steps: int = 11,
        parent_top_k: int = 8,
        max_layers: int = 4,
        max_functions: int = 96,
    ) -> None:
        super().__init__()
        if (
            isinstance(n_features, bool)
            or not isinstance(n_features, int)
            or n_features < 1
            or isinstance(n_classes, bool)
            or not isinstance(n_classes, int)
            or n_classes < 2
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
            raise ValueError("invalid feature classifier configuration")
        self.n_features = n_features
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.sgd_steps = sgd_steps
        self._model = BClassifierCore(
            n_features,
            n_classes,
            learning_rate,
            l2,
            batch_size,
            sgd_steps,
            parent_top_k,
            max_layers,
            max_functions,
        )

    def predict(self, features: np.ndarray) -> int:
        values = validate_binary_features(features, self.n_features)
        self._begin_predict()
        return int(self._model.predict(values))

    def observe(self, features: np.ndarray, target: int) -> None:
        self._begin_observe()
        values = validate_binary_features(features, self.n_features)
        target_value = validate_class_index(target, self.n_classes)
        self._finish_observe()
        self._model.observe(values, target_value)

    @property
    def n_observed(self) -> int:
        return int(self._model.n_observed)

    @property
    def function_count(self) -> int:
        return int(self._model.function_count)

    def intercept(self, class_index: int) -> float | None:
        value = self._model.intercept(class_index)
        return None if value is None else float(value)

    def weight(self, function_index: int, class_index: int) -> float | None:
        value = self._model.weight(function_index, class_index)
        return None if value is None else float(value)
