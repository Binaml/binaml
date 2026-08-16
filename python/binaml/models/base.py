"""Shared contracts and replay storage for online models."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Protocol

import numpy as np


class OnlineModel(Protocol):
    def predict(self, features: np.ndarray) -> float: ...
    def update(self, target: float) -> None: ...


OnlineModelFactory = Callable[[int], OnlineModel]


class OnlineClassifier(Protocol):
    def predict(self, features: np.ndarray) -> int: ...
    def update(self, target: int) -> None: ...


OnlineClassifierFactory = Callable[[int, int], OnlineClassifier]


class ReplayBatch:
    """Fixed-size, most-recent sample window."""

    def __init__(self, batch_size: int) -> None:
        self.features: deque[np.ndarray] = deque(maxlen=batch_size)
        self.targets: deque[float | int] = deque(maxlen=batch_size)

    def append(self, features: np.ndarray, target: float) -> None:
        self.features.append(features.copy())
        self.targets.append(target)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.features), np.asarray(self.targets)


class PredictUpdateState:
    """Enforces predict-then-update ordering for online models."""

    def __init__(self) -> None:
        self._prediction_pending = False

    def _begin_predict(self) -> None:
        if self._prediction_pending:
            raise RuntimeError("update(target) must follow each predict(features)")
        self._prediction_pending = True

    def _begin_update(self) -> None:
        if not self._prediction_pending:
            raise RuntimeError("update(target) requires a preceding predict(features)")

    def _finish_update(self) -> None:
        self._prediction_pending = False


def validate_float_features(features: np.ndarray, n_features: int) -> np.ndarray:
    values = np.asarray(features)
    if values.shape != (n_features,):
        raise ValueError(f"features must have shape ({n_features},)")
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise ValueError("features must be finite")
    return np.asarray(values, dtype=np.float32)


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
