"""Metric helpers for streaming-classification benchmarks."""

from __future__ import annotations

from binaml.benchmarks._common import aggregate, timing_payload
from binaml.evaluation import PrequentialClassificationResult


def record(seed: int, result: PrequentialClassificationResult, warmup_samples: int = 0) -> dict[str, object]:
    correct = result.correct[warmup_samples:]
    accuracy = float(correct.mean()) if len(correct) else float("nan")
    return {
        "seed": seed,
        "accuracy": accuracy,
        "timing_seconds": timing_payload(result),
    }


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "n_seeds": len(records),
        "accuracy": aggregate([record["accuracy"] for record in records]),  # type: ignore[list-item]
        "timing_seconds": {
            "total": aggregate([record["timing_seconds"]["total"] for record in records])  # type: ignore[index]
        },
    }


def extract_metrics(model_summaries: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        "accuracy": {name: values["accuracy"] for name, values in model_summaries.items()},
        "timing_seconds": {name: values["timing_seconds"] for name, values in model_summaries.items()},
        "n_seeds": {name: values["n_seeds"] for name, values in model_summaries.items()},
    }
