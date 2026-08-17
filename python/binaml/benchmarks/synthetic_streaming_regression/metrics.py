"""Metric helpers for streaming-regression benchmarks."""

from __future__ import annotations

import numpy as np

from binaml.benchmarks._common import aggregate, timing_payload
from binaml.evaluation import PrequentialResult


def record(seed: int, result: PrequentialResult, warmup_samples: int = 0) -> dict[str, object]:
    squared_errors = result.squared_errors[warmup_samples:]
    valid = np.isfinite(squared_errors)
    mse = float(squared_errors[valid].mean()) if np.any(valid) else float("nan")
    return {
        "seed": seed,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "timing_seconds": timing_payload(result),
    }


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "n_seeds": len(records),
        "mse": aggregate([record["mse"] for record in records]),  # type: ignore[list-item]
        "rmse": aggregate([record["rmse"] for record in records]),  # type: ignore[list-item]
        "timing_seconds": {
            "total": aggregate([record["timing_seconds"]["total"] for record in records])  # type: ignore[index]
        },
    }


def extract_metrics(model_summaries: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        "mse": {name: values["mse"] for name, values in model_summaries.items()},
        "rmse": {name: values["rmse"] for name, values in model_summaries.items()},
        "timing_seconds": {name: values["timing_seconds"] for name, values in model_summaries.items()},
        "n_seeds": {name: values["n_seeds"] for name, values in model_summaries.items()},
    }
