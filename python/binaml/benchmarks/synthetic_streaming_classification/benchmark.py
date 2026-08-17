"""Streaming-classification benchmark specification."""

from __future__ import annotations

from typing import Any

from binaml.benchmarks.streaming.spec import StreamingBenchmark
from binaml.environments import (
    ClassificationTrajectory,
    SyntheticClassificationStreamConfig,
    generate_classification_trajectory,
)
from binaml.evaluation import evaluate_prequentially_classification

from . import metrics


def _bind_factory(factory, parameters: dict[str, object]):
    return lambda n_features, n_classes, factory=factory, parameters=parameters: factory(
        n_features,
        n_classes,
        **parameters,
    )


def _build_model(trajectory: ClassificationTrajectory, factory, parameters: dict[str, object]) -> Any:
    return factory(trajectory.config.n_features, trajectory.config.n_classes, **parameters)


def _job_result_extras(result: Any) -> dict[str, object]:
    return {
        "predictions": result.predictions.tolist(),
        "correct": result.correct.tolist(),
    }


BENCHMARK = StreamingBenchmark(
    run_prefix="synthetic_streaming_classification",
    job_module="binaml.benchmarks.synthetic_streaming_classification.job",
    default_models=("binaml.models:SGDLinearClassifier",),
    config_schema_version=1,
    summary_schema_version=1,
    metrics_schema_version=1,
    reserved_parameters=frozenset({"n_features", "n_classes"}),
    bind_factory=_bind_factory,
    load_trajectory_from_scenario=lambda scenario, seed: generate_classification_trajectory(
        SyntheticClassificationStreamConfig.from_dict(scenario["environment"]),
        int(scenario["n_samples"]),
        seed,
        return_metadata=True,
    ),
    load_trajectory_from_npz=ClassificationTrajectory.load_npz,
    build_model=_build_model,
    evaluate=evaluate_prequentially_classification,
    record=metrics.record,
    summarize=metrics.summarize,
    extract_metrics=metrics.extract_metrics,
    job_result_extras=_job_result_extras,
)
