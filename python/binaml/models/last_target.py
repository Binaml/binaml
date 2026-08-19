"""Naive last-target persistence baselines."""

from __future__ import annotations

import numpy as np

from .base import PredictUpdateState, validate_binary_features, validate_class_index, validate_finite_float_target


class LastTargetRegressor(PredictUpdateState):
    """Predict zero initially, then always repeat the most recently observed target."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        if isinstance(n_features, bool) or not isinstance(n_features, int) or n_features < 1:
            raise ValueError("invalid n_features")
        self.n_features = n_features
        self.n_observed = 0
        self._last_target = 0.0

    def predict(self, features: np.ndarray) -> float:
        self._begin_predict()
        validate_binary_features(features, self.n_features)
        return self._last_target

    def update(self, target: float) -> None:
        self._begin_update()
        self._last_target = validate_finite_float_target(target)
        self.n_observed += 1
        self._finish_update()


class LastTargetClassifier(PredictUpdateState):
    """Predict class zero initially, then always repeat the most recently observed label."""

    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        if isinstance(n_features, bool) or not isinstance(n_features, int) or n_features < 1:
            raise ValueError("invalid n_features")
        if isinstance(n_classes, bool) or not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError("invalid n_classes")
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_observed = 0
        self._last_target = 0

    def predict(self, features: np.ndarray) -> int:
        self._begin_predict()
        validate_binary_features(features, self.n_features)
        return self._last_target

    def update(self, target: int) -> None:
        self._begin_update()
        self._last_target = validate_class_index(target, self.n_classes)
        self.n_observed += 1
        self._finish_update()
