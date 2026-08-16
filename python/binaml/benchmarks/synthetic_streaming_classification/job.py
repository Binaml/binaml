"""One isolated model/seed streaming-classification benchmark job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from binaml.benchmarks._common import load_factory, write_json_atomically
from binaml.environments import (
    ClassificationTrajectory,
    SyntheticClassificationStreamConfig,
    generate_classification_trajectory,
)


def _load_trajectory(args: argparse.Namespace) -> ClassificationTrajectory:
    if args.scenario is not None:
        scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
        config = SyntheticClassificationStreamConfig.from_dict(scenario["environment"])
        return generate_classification_trajectory(config, int(scenario["n_samples"]), args.seed, return_metadata=True)
    return ClassificationTrajectory.load_npz(args.trajectory)


def run_job(args: argparse.Namespace) -> dict[str, object]:
    trajectory = _load_trajectory(args)
    scenario = json.loads(args.scenario.read_text(encoding="utf-8")) if args.scenario is not None else None
    warmup_samples = 0
    if scenario is not None:
        from .cli import _warmup_samples

        warmup_samples = _warmup_samples(scenario)

    factory = load_factory(args.factory)
    parameters = json.loads(args.parameters_json)
    if not isinstance(parameters, dict):
        raise TypeError("model parameters must be a JSON object")
    model = factory(trajectory.config.n_features, trajectory.config.n_classes, **parameters)

    from binaml.evaluation import evaluate_prequentially_classification

    from .cli import _record

    result = evaluate_prequentially_classification(model, trajectory, warmup_samples=warmup_samples)
    return {
        "schema_version": 1,
        "model": args.model_name,
        "warmup_samples": warmup_samples,
        **_record(trajectory.seed, result, warmup_samples),
        "timing_seconds": {
            "total": result.timing_seconds.total,
            "prediction": result.timing_seconds.prediction,
            "update": result.timing_seconds.update,
        },
        "predictions": result.predictions.tolist(),
        "correct": result.correct.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scenario", type=Path)
    source.add_argument("--trajectory", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--parameters-json", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.scenario is not None and args.seed is None:
        parser.error("--seed is required with --scenario")
    write_json_atomically(args.output, run_job(args))


if __name__ == "__main__":
    main()
