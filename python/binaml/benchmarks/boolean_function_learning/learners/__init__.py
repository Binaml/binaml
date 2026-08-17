"""Rust-backed batch function learners exposed to the benchmark."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from binaml._core import FunctionLearner as _FunctionLearner


class RustBatchLearner:
    def __init__(self, learner: _FunctionLearner) -> None:
        self._learner = learner

    def fit(self, x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
        result = self._learner.fit(x, y)
        return {"score": int(result["score"]), "elapsed_seconds": float(result["elapsed_seconds"])}

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._learner.predict(x), dtype=bool)

    def fit_predict_with_details(self, x, y):
        result = self._learner.fit_predict(x, y)
        return result["predictions"], result["score"], result["elapsed_seconds"]


def build_function_builder(**parameters: Any) -> RustBatchLearner:
    return RustBatchLearner(_FunctionLearner(**parameters))


LEARNER_FACTORIES: dict[str, Callable[..., RustBatchLearner]] = {
    "function_builder": build_function_builder,
}
