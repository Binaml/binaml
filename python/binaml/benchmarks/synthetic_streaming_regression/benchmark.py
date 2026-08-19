"""Streaming-regression benchmark specification."""

from __future__ import annotations

from typing import Any

from binaml.benchmarks.streaming.spec import StreamingBenchmark
from binaml.environments import SyntheticStreamConfig, Trajectory, generate_trajectory
from binaml.evaluation import evaluate_prequentially

from . import metrics


def _bind_factory(factory, parameters: dict[str, object]):
    return lambda n_features, factory=factory, parameters=parameters: factory(n_features, **parameters)


def _build_model(trajectory: Trajectory, factory, parameters: dict[str, object]) -> Any:
    return factory(trajectory.config.n_features, **parameters)


def _job_result_extras(result: Any) -> dict[str, object]:
    return {
        "predictions": result.predictions.tolist(),
        "squared_errors": result.squared_errors.tolist(),
    }


BENCHMARK = StreamingBenchmark(
    run_prefix="synthetic_streaming_regression",
    job_module="binaml.benchmarks.synthetic_streaming_regression.job",
    default_models=("binaml.models:SGDLinearRegressor",),
    reserved_parameters=frozenset({"n_features"}),
    bind_factory=_bind_factory,
    load_trajectory_from_scenario=lambda scenario, seed: generate_trajectory(
        SyntheticStreamConfig.from_dict(scenario["environment"]),
        int(scenario["n_samples"]),
        seed,
        return_metadata=True,
    ),
    load_trajectory_from_npz=Trajectory.load_npz,
    build_model=_build_model,
    evaluate=evaluate_prequentially,
    record=metrics.record,
    summarize=metrics.summarize,
    extract_metrics=metrics.extract_metrics,
    job_result_extras=_job_result_extras,
)
