"""Replay-batch linear SGD baseline."""

from __future__ import annotations

import numpy as np

from .base import ReplaySGDModel


class SGDLinearRegressor(ReplaySGDModel):
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
        super().__init__(
            n_features,
            learning_rate,
            l2,
            center_binary_features,
            batch_size,
            sgd_steps,
            invalid_config_message="invalid regressor configuration",
        )
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.intercept = 0.0

    def _predict_from_features(self, values: np.ndarray) -> float:
        return self.intercept + float(self._model_features(values) @ self.weights)

    def _validate_target(self, target: float) -> float:
        from .base import validate_finite_float_target

        return validate_finite_float_target(target)

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


def slow_sgd_linear_regressor(n_features: int) -> SGDLinearRegressor:
    """Conservative linear SGD baseline."""
    return SGDLinearRegressor(n_features, learning_rate=0.003)


def fast_sgd_linear_regressor(n_features: int) -> SGDLinearRegressor:
    """Fast-adapting linear SGD baseline."""
    return SGDLinearRegressor(n_features, learning_rate=0.03)
