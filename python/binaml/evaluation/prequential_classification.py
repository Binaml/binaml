"""Predict-then-update evaluation for online classifiers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .prequential import EvaluationTiming, _evaluate_prequentially_loop

if TYPE_CHECKING:
    from binaml.models.base import OnlineClassifier


class ClassificationSampleSource(Protocol):
    """Any finite or infinite source of `(features, label)` pairs."""

    def __iter__(self) -> Iterator[tuple[np.ndarray, int]]: ...


@dataclass(frozen=True)
class PrequentialClassificationResult:
    predictions: np.ndarray
    targets: np.ndarray
    correct: np.ndarray
    timing_seconds: EvaluationTiming

    @property
    def accuracy(self) -> float:
        return float(self.correct.mean()) if len(self.correct) else float("nan")


def evaluate_prequentially_classification(
    model: OnlineClassifier,
    source: Iterable[tuple[np.ndarray, int]],
    n_samples: int | None = None,
    on_step: Callable[[int], None] | None = None,
    warmup_samples: int = 0,
) -> PrequentialClassificationResult:
    """Predict features before updating with their label."""
    predictions, targets, correct, timing = _evaluate_prequentially_loop(
        model,
        source,
        n_samples,
        on_step,
        lambda prediction, target: prediction == target,
        warmup_samples,
    )
    return PrequentialClassificationResult(
        np.asarray(predictions, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        np.asarray(correct, dtype=bool),
        timing,
    )
