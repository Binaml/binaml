"""Replay-batch linear SGD baseline."""

from __future__ import annotations

import numpy as np

from .base import ReplayBatch


class SGDLinearRegressor:
    """Online linear regressor with full-batch replay SGD updates."""

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.03,
        l2: float = 1e-4,
        center_binary_features: bool = False,
        batch_size: int = 32,
        sgd_steps: int = 3,
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
        ):
            raise ValueError("invalid regressor configuration")
        self.n_features, self.learning_rate, self.l2 = n_features, learning_rate, l2
        self.center_binary_features = center_binary_features
        self.batch_size, self.sgd_steps = batch_size, sgd_steps
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.intercept = 0.0
        self.feature_probabilities = np.full(n_features, 0.5, dtype=np.float64)
        self._replay = ReplayBatch(batch_size)
        self.n_observed = 0
        self._prediction_pending = False

    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.shape != (self.n_features,):
            raise ValueError(f"features must have shape ({self.n_features},)")
        if not np.all(np.isfinite(values)):
            raise ValueError("features must be finite")
        if self.center_binary_features and not np.all((values == 0.0) | (values == 1.0)):
            raise ValueError("centered features must be binary")
        return values

    def _model_features(self, values: np.ndarray) -> np.ndarray:
        return values - self.feature_probabilities if self.center_binary_features else values

    def _update_replay_batch(self) -> None:
        features, targets = self._replay.arrays()
        model_features = self._model_features(features)
        errors = targets - (self.intercept + model_features @ self.weights)
        rate = self.learning_rate
        self.intercept += rate * float(np.mean(errors))
        self.weights *= 1.0 - rate * self.l2
        self.weights += rate * np.mean(errors[:, None] * model_features, axis=0)
        if self.center_binary_features:
            self.feature_probabilities += rate * (np.mean(features, axis=0) - self.feature_probabilities)

    def predict(self, features: np.ndarray) -> float:
        if self._prediction_pending:
            raise RuntimeError("observe(features, target) must follow each predict(features)")
        values = self._validate_features(features)
        prediction = self.intercept + float(self._model_features(values) @ self.weights)
        self._prediction_pending = True
        return prediction

    def observe(self, features: np.ndarray, target: float) -> None:
        if not self._prediction_pending:
            raise RuntimeError("observe(features, target) requires a preceding predict(features)")
        values = self._validate_features(features)
        target_value = float(target)
        if not np.isfinite(target_value):
            raise ValueError("target must be finite")
        self._replay.append(values, target_value)
        for _ in range(self.sgd_steps):
            self._update_replay_batch()
        self.n_observed += 1
        self._prediction_pending = False


def slow_sgd_linear_regressor(n_features: int) -> SGDLinearRegressor:
    """Conservative linear SGD baseline."""
    return SGDLinearRegressor(n_features, learning_rate=0.003)


def fast_sgd_linear_regressor(n_features: int) -> SGDLinearRegressor:
    """Fast-adapting linear SGD baseline."""
    return SGDLinearRegressor(n_features, learning_rate=0.03)
