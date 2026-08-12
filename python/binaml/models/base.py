"""Shared contracts and replay storage for online models."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Protocol

import numpy as np


class OnlineModel(Protocol):
    def predict(self, features: np.ndarray) -> float: ...
    def observe(self, features: np.ndarray, target: float) -> None: ...


OnlineModelFactory = Callable[[int], OnlineModel]


class ReplayBatch:
    """Fixed-size, most-recent sample window."""

    def __init__(self, batch_size: int) -> None:
        self.features: deque[np.ndarray] = deque(maxlen=batch_size)
        self.targets: deque[float] = deque(maxlen=batch_size)

    def append(self, features: np.ndarray, target: float) -> None:
        self.features.append(features.copy())
        self.targets.append(target)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.features), np.asarray(self.targets)
