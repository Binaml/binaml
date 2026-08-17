"""Command-line composition for named streaming-classification scenarios."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from binaml.benchmarks.streaming import run_scenario as _run_scenario
from binaml.benchmarks.streaming import run_streaming_benchmark_cli
from binaml.benchmarks.streaming import run_trajectory as _run_trajectory
from binaml.environments import ClassificationTrajectory
from binaml.evaluation import PrequentialClassificationResult
from binaml.models import SGDLinearClassifier

from . import metrics
from .benchmark import BENCHMARK

ModelFactory = Callable[[int, int], object]
EvaluationCallback = Callable[[ClassificationTrajectory, dict[str, PrequentialClassificationResult]], None]

load_model_config = BENCHMARK.load_model_config
record = metrics.record
summarize = metrics.summarize


def run_scenario(
    path: str | Path,
    model_factories: dict[str, ModelFactory] | ModelFactory | None = None,
    on_evaluation: EvaluationCallback | None = None,
) -> dict[str, object]:
    return _run_scenario(
        BENCHMARK,
        path,
        model_factories,
        on_evaluation,
        default_factory=SGDLinearClassifier,
    )


def run_trajectory(
    path: str | Path,
    model_factories: dict[str, ModelFactory] | ModelFactory | None = None,
    on_evaluation: EvaluationCallback | None = None,
) -> dict[str, object]:
    return _run_trajectory(
        BENCHMARK,
        path,
        model_factories,
        on_evaluation,
        default_factory=SGDLinearClassifier,
    )


def main() -> None:
    def write_plots(*args: object, **kwargs: object) -> None:
        try:
            from .plots import write_job_plots
        except ModuleNotFoundError as error:
            if error.name in {"matplotlib", "seaborn"}:
                raise RuntimeError("plotting requires `pip install 'binaml[benchmarks]'`") from error
            raise
        write_job_plots(*args, **kwargs)  # type: ignore[arg-type]

    run_streaming_benchmark_cli(
        BENCHMARK,
        default_factory=SGDLinearClassifier,
        write_plots=write_plots,
    )


if __name__ == "__main__":
    main()
