"""Predict-then-observe evaluation for online models."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from time import perf_counter
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from binaml.models.base import OnlineModel


class RegressionSampleSource(Protocol):
    """Any finite or infinite source of `(features, target)` pairs."""

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]: ...


@dataclass(frozen=True)
class EvaluationTiming:
    total: float
    prediction: float
    observation: float


@dataclass(frozen=True)
class PrequentialResult:
    predictions: np.ndarray
    targets: np.ndarray
    squared_errors: np.ndarray
    timing_seconds: EvaluationTiming

    @property
    def mean_squared_error(self) -> float:
        valid = np.isfinite(self.squared_errors)
        return float(self.squared_errors[valid].mean()) if np.any(valid) else float("nan")

    @property
    def root_mean_squared_error(self) -> float:
        return float(np.sqrt(self.mean_squared_error))


def evaluate_prequentially(
    model: OnlineModel,
    source: Iterable[tuple[np.ndarray, float]],
    n_samples: int | None = None,
    on_step: Callable[[int], None] | None = None,
) -> PrequentialResult:
    """Predict features before observing their target."""
    predictions, targets, errors = [], [], []
    total_seconds = 0.0
    prediction_seconds = 0.0
    observation_seconds = 0.0
    pairs = source if n_samples is None else islice(source, n_samples)
    for position, (features, target) in enumerate(pairs):
        step_start = perf_counter()
        prediction_start = perf_counter()
        prediction = model.predict(features)
        prediction_seconds += perf_counter() - prediction_start
        predictions.append(prediction)
        targets.append(target)
        errors.append((prediction - target) ** 2 if np.isfinite(prediction) else float("nan"))
        observation_start = perf_counter()
        model.observe(features, target)
        observation_seconds += perf_counter() - observation_start
        total_seconds += perf_counter() - step_start
        if on_step is not None:
            on_step(position)
    return PrequentialResult(
        np.asarray(predictions),
        np.asarray(targets),
        np.asarray(errors),
        EvaluationTiming(total_seconds, prediction_seconds, observation_seconds),
    )
