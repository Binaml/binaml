"""One isolated model/seed streaming-regression benchmark job."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from binaml.environments import SyntheticStreamConfig, Trajectory, generate_trajectory


def _load_factory(specification: str):
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("model must use module:factory syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("model factory must be callable")
    return factory


def _load_trajectory(args: argparse.Namespace) -> Trajectory:
    if args.scenario is not None:
        scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
        config = SyntheticStreamConfig.from_dict(scenario["environment"])
        return generate_trajectory(config, int(scenario["n_samples"]), args.seed, return_metadata=True)
    return Trajectory.load_npz(args.trajectory)


def _write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def run_job(args: argparse.Namespace) -> dict[str, object]:
    trajectory = _load_trajectory(args)
    scenario = json.loads(args.scenario.read_text(encoding="utf-8")) if args.scenario is not None else None
    warmup_samples = 0
    if scenario is not None:
        from .cli import _warmup_samples

        warmup_samples = _warmup_samples(scenario)

    factory = _load_factory(args.factory)
    parameters = json.loads(args.parameters_json)
    if not isinstance(parameters, dict):
        raise TypeError("model parameters must be a JSON object")
    model = factory(trajectory.config.n_features, **parameters)

    from binaml.evaluation import evaluate_prequentially

    from .cli import _record

    result = evaluate_prequentially(model, trajectory)
    return {
        "schema_version": 1,
        "model": args.model_name,
        "warmup_samples": warmup_samples,
        **_record(trajectory.seed, result, warmup_samples),
        "timing_seconds": {
            "total": result.timing_seconds.total,
            "prediction": result.timing_seconds.prediction,
            "observation": result.timing_seconds.observation,
        },
        "predictions": result.predictions.tolist(),
        "squared_errors": result.squared_errors.tolist(),
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
    payload = run_job(args)
    _write_json_atomically(args.output, payload)


if __name__ == "__main__":
    main()
