"""Shared contracts and replay storage for online models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from typing import Protocol

import numpy as np


class OnlineModel(Protocol):
    def predict(self, features: np.ndarray) -> float: ...
    def observe(self, features: np.ndarray, target: float) -> None: ...


OnlineModelFactory = Callable[[int], OnlineModel]


class OnlineClassifier(Protocol):
    def predict(self, features: np.ndarray) -> int: ...
    def observe(self, features: np.ndarray, target: int) -> None: ...


OnlineClassifierFactory = Callable[[int, int], OnlineClassifier]


class ReplayBatch:
    """Fixed-size, most-recent sample window."""

    def __init__(self, batch_size: int) -> None:
        self.features: deque[np.ndarray] = deque(maxlen=batch_size)
        self.targets: deque[float | int] = deque(maxlen=batch_size)

    def append(self, features: np.ndarray, target: float | int) -> None:
        self.features.append(features.copy())
        self.targets.append(target)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.features), np.asarray(self.targets)


class PredictObserveState:
    """Enforces predict-then-observe ordering for online models."""

    def __init__(self) -> None:
        self._prediction_pending = False

    def _begin_predict(self) -> None:
        if self._prediction_pending:
            raise RuntimeError("observe(features, target) must follow each predict(features)")
        self._prediction_pending = True

    def _begin_observe(self) -> None:
        if not self._prediction_pending:
            raise RuntimeError("observe(features, target) requires a preceding predict(features)")

    def _finish_observe(self) -> None:
        self._prediction_pending = False


def validate_float_features(features: np.ndarray, n_features: int) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.shape != (n_features,):
        raise ValueError(f"features must have shape ({n_features},)")
    if not np.all(np.isfinite(values)):
        raise ValueError("features must be finite")
    return values


def validate_binary_features(features: np.ndarray, n_features: int) -> np.ndarray:
    values = np.asarray(features)
    if values.shape != (n_features,):
        raise ValueError(f"features must have shape ({n_features},)")
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise ValueError("features must be finite")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("features must be binary")
    return np.ascontiguousarray(values, dtype=np.uint8)


def validate_finite_float_target(target: float) -> float:
    target_value = float(target)
    if not np.isfinite(target_value):
        raise ValueError("target must be finite")
    return target_value


def validate_class_index(target: int, n_classes: int) -> int:
    if isinstance(target, bool):
        raise TypeError("target must be an integer class index")
    target_value = int(target)
    if not 0 <= target_value < n_classes:
        raise ValueError("target must lie in [0, n_classes)")
    return target_value


def validate_positive_int(name: str, value: object, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"invalid {name}")
    return value


class ReplaySGDModel(PredictObserveState, ABC):
    """Replay-batch SGD scaffold shared by linear online models."""

    def __init__(
        self,
        n_features: int,
        learning_rate: float,
        l2: float,
        center_binary_features: bool,
        batch_size: int,
        sgd_steps: int,
        *,
        invalid_config_message: str = "invalid model configuration",
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
        ):
            raise ValueError(invalid_config_message)
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.l2 = l2
        self.center_binary_features = center_binary_features
        self.batch_size = batch_size
        self.sgd_steps = sgd_steps
        self.feature_probabilities = np.full(n_features, 0.5, dtype=np.float64)
        self._replay = ReplayBatch(batch_size)
        self.n_observed = 0

    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        values = validate_float_features(features, self.n_features)
        if self.center_binary_features and not np.all((values == 0.0) | (values == 1.0)):
            raise ValueError("centered features must be binary")
        return values

    def _model_features(self, values: np.ndarray) -> np.ndarray:
        return values - self.feature_probabilities if self.center_binary_features else values

    def predict(self, features: np.ndarray):
        values = self._validate_features(features)
        self._begin_predict()
        return self._predict_from_features(values)

    def observe(self, features: np.ndarray, target) -> None:
        self._begin_observe()
        values = self._validate_features(features)
        target_value = self._validate_target(target)
        self._finish_observe()
        self._replay.append(values, target_value)
        for _ in range(self.sgd_steps):
            self._update_replay_batch()
        self.n_observed += 1

    @abstractmethod
    def _predict_from_features(self, values: np.ndarray): ...

    @abstractmethod
    def _validate_target(self, target): ...

    @abstractmethod
    def _update_replay_batch(self) -> None: ...
