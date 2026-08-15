"""Replay-batch linear SGD multiclass baseline."""

from __future__ import annotations

import numpy as np

from .base import ReplaySGDModel, validate_class_index


class SGDLinearClassifier(ReplaySGDModel):
    """Online linear classifier with full-batch replay SGD updates."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        learning_rate: float = 0.035,
        l2: float = 0.0,
        center_binary_features: bool = False,
        batch_size: int = 10,
        sgd_steps: int = 6,
    ) -> None:
        if isinstance(n_classes, bool) or not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError("invalid classifier configuration")
        super().__init__(
            n_features,
            learning_rate,
            l2,
            center_binary_features,
            batch_size,
            sgd_steps,
            invalid_config_message="invalid classifier configuration",
        )
        self.n_classes = n_classes
        self.weights = np.zeros((n_features, n_classes), dtype=np.float64)
        self.intercepts = np.zeros(n_classes, dtype=np.float64)

    def _logits(self, values: np.ndarray) -> np.ndarray:
        return self.intercepts + self._model_features(values) @ self.weights

    def _predict_from_features(self, values: np.ndarray) -> int:
        return int(np.argmax(self._logits(values)))

    def _validate_target(self, target: int) -> int:
        return validate_class_index(target, self.n_classes)

    def _update_replay_batch(self) -> None:
        features, targets = self._replay.arrays()
        model_features = self._model_features(features)
        logits = model_features @ self.weights + self.intercepts
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        one_hot = np.zeros_like(probabilities)
        one_hot[np.arange(len(targets)), targets] = 1.0
        errors = probabilities - one_hot
        rate = self.learning_rate
        self.intercepts -= rate * np.mean(errors, axis=0)
        self.weights *= 1.0 - rate * self.l2
        self.weights -= rate * model_features.T @ errors / len(targets)
        if self.center_binary_features:
            self.feature_probabilities += rate * (np.mean(features, axis=0) - self.feature_probabilities)
