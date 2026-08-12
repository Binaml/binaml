"""Command-line composition for named streaming-regression scenarios.

Pass ``--model-config`` a JSON file shaped as:
``{"models": [{"name": "...", "factory": "module:callable", "parameters": {...}}]}``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from math import sqrt
from pathlib import Path
from statistics import fmean, stdev

import numpy as np

from binaml.environments import SyntheticStreamConfig, Trajectory, generate_trajectory
from binaml.evaluation import EvaluationTiming, PrequentialResult, evaluate_prequentially
from binaml.models import OnlineModel, SGDLinearRegressor

ModelFactory = Callable[[int], OnlineModel]
EvaluationCallback = Callable[[Trajectory, dict[str, PrequentialResult]], None]


def _timing_payload(result: object) -> dict[str, float]:
    timing = result.timing_seconds  # type: ignore[attr-defined]
    return {"total": timing.total, "prediction": timing.prediction, "observation": timing.observation}


def _aggregate(values: list[float]) -> dict[str, float]:
    return {
        "average": fmean(values),
        "standard_error": stdev(values) / sqrt(len(values)) if len(values) > 1 else 0.0,
    }


def _warmup_samples(scenario: dict[str, object]) -> int:
    warmup_samples = scenario.get("warmup_samples", 0)
    if not isinstance(warmup_samples, int) or isinstance(warmup_samples, bool):
        raise TypeError("warmup_samples must be an integer")
    n_samples = scenario["n_samples"]
    if not isinstance(n_samples, int) or isinstance(n_samples, bool):
        raise TypeError("n_samples must be an integer")
    if not 0 <= warmup_samples < n_samples:
        raise ValueError("warmup_samples must be non-negative and less than n_samples")
    return warmup_samples


def _record(seed: int, result: PrequentialResult, warmup_samples: int = 0) -> dict[str, object]:
    squared_errors = result.squared_errors[warmup_samples:]
    valid = np.isfinite(squared_errors)
    mse = float(squared_errors[valid].mean()) if np.any(valid) else float("nan")
    return {
        "seed": seed,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "timing_seconds": _timing_payload(result),
    }


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {
        "n_seeds": len(records),
        "mse": _aggregate([record["mse"] for record in records]),  # type: ignore[list-item]
        "rmse": _aggregate([record["rmse"] for record in records]),  # type: ignore[list-item]
        "timing_seconds": {
            "total": _aggregate([record["timing_seconds"]["total"] for record in records])  # type: ignore[index]
        },
    }
    return summary


def _normalize_models(model_factories: dict[str, ModelFactory] | ModelFactory | None) -> dict[str, ModelFactory]:
    if model_factories is None:
        return {"SGDLinearRegressor": SGDLinearRegressor}
    if callable(model_factories):
        return {model_factories.__name__: model_factories}
    return model_factories


def run_scenario(
    path: str | Path,
    model_factories: dict[str, ModelFactory] | ModelFactory | None = None,
    on_evaluation: EvaluationCallback | None = None,
) -> dict[str, object]:
    scenario = json.loads(Path(path).read_text(encoding="utf-8"))
    config = SyntheticStreamConfig.from_dict(scenario["environment"])
    warmup_samples = _warmup_samples(scenario)
    model_factories = _normalize_models(model_factories)
    records = {name: [] for name in model_factories}
    for seed in scenario["seeds"]:
        trajectory = generate_trajectory(config, int(scenario["n_samples"]), int(seed), return_metadata=True)
        evaluations = {
            name: evaluate_prequentially(factory(config.n_features), trajectory)
            for name, factory in model_factories.items()
        }
        for name, result in evaluations.items():
            records[name].append(_record(int(seed), result, warmup_samples))
        if on_evaluation:
            on_evaluation(trajectory, evaluations)
    return {
        "scenario": scenario["name"],
        "config_fingerprint": config.fingerprint,
        "models": {name: {"results": value, "summary": _summary(value)} for name, value in records.items()},
    }


def run_trajectory(
    path: str | Path,
    model_factories: dict[str, ModelFactory] | ModelFactory | None = None,
    on_evaluation: EvaluationCallback | None = None,
) -> dict[str, object]:
    trajectory = Trajectory.load_npz(path)
    model_factories = _normalize_models(model_factories)
    evaluations = {
        name: evaluate_prequentially(factory(trajectory.config.n_features), trajectory)
        for name, factory in model_factories.items()
    }
    if on_evaluation:
        on_evaluation(trajectory, evaluations)
    return {
        "source": str(path),
        "config_fingerprint": trajectory.config.fingerprint,
        "seed": trajectory.seed,
        "n_samples": len(trajectory.y),
        "models": {name: _record(trajectory.seed, result) for name, result in evaluations.items()},
    }


def _load_factory(specification: str) -> ModelFactory:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("model must use module:factory syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("model factory must be callable")
    return factory


def _load_models(specifications: list[str]) -> dict[str, ModelFactory]:
    models: dict[str, ModelFactory] = {}
    for specification in specifications:
        name, separator, factory_specification = specification.partition("=")
        factory_specification = factory_specification if separator else specification
        factory = _load_factory(factory_specification)
        model_name = name if separator else factory_specification.rsplit(":", maxsplit=1)[-1]
        if model_name in models:
            raise ValueError(f"duplicate model name: {model_name}")
        models[model_name] = factory
    return models


def _load_model_config(path: Path) -> tuple[dict[str, ModelFactory], list[dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise TypeError("model config must contain a models list")

    models: dict[str, ModelFactory] = {}
    normalized_config: list[dict[str, object]] = []
    for entry in payload["models"]:
        if not isinstance(entry, dict):
            raise TypeError("each model config entry must be an object")
        name, specification = entry.get("name"), entry.get("factory")
        parameters = entry.get("parameters", {})
        if not isinstance(name, str) or not name:
            raise ValueError("each model config entry requires a name")
        if not isinstance(specification, str) or not specification:
            raise ValueError("each model config entry requires a factory")
        if not isinstance(parameters, dict) or not all(isinstance(key, str) for key in parameters):
            raise ValueError("model parameters must be an object with string keys")
        if "n_features" in parameters:
            raise ValueError("n_features is supplied by the benchmark")
        if name in models:
            raise ValueError(f"duplicate model name: {name}")
        factory = _load_factory(specification)
        models[name] = lambda n_features, factory=factory, parameters=parameters: factory(
            n_features, **parameters
        )
        normalized_config.append(
            {"name": name, "factory": specification, "parameters": parameters}
        )
    if not models:
        raise ValueError("model config must contain at least one model")
    return models, normalized_config


def _write_run(
    output_dir: Path,
    result: dict[str, object],
    source: Path,
    model_config: list[dict[str, object]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "schema_version": 1,
        "source": str(source),
        "models": list(result["models"]),  # type: ignore[arg-type]
    }
    if model_config is not None:
        config["model_config"] = model_config
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps({"schema_version": 1, **result}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_plots(
    output_dir: Path,
    evaluations: list[tuple[Trajectory, dict[str, PrequentialResult]]],
    warmup_samples: int = 0,
) -> None:
    try:
        import seaborn as sns

        from .plots import write_model_plot, write_rmse_plot
    except ModuleNotFoundError as error:
        if error.name in {"matplotlib", "seaborn"}:
            raise RuntimeError("plotting requires `pip install 'binaml[benchmarks]'`") from error
        raise

    plots_dir = output_dir / "plots"
    plots_dir.mkdir()
    for trajectory, model_evaluations in evaluations:
        write_rmse_plot(
            plots_dir / f"rmse_seed_{trajectory.seed}.png",
            trajectory,
            model_evaluations,
            warmup_samples,
        )
        for (model_name, evaluation), color in zip(
            model_evaluations.items(), sns.color_palette("colorblind", n_colors=len(model_evaluations)), strict=True
        ):
            model_stem = model_name.replace("/", "_")
            write_model_plot(
                plots_dir / f"model_{model_stem}_seed_{trajectory.seed}.png",
                trajectory,
                model_name,
                evaluation,
                color,
                warmup_samples,
            )


def _model_entries(specifications: list[str], model_config: list[dict[str, object]] | None) -> list[dict[str, object]]:
    if model_config is not None:
        return model_config
    entries: list[dict[str, object]] = []
    for specification in specifications:
        name, separator, factory = specification.partition("=")
        factory = factory if separator else specification
        entries.append(
            {
                "name": name if separator else factory.rsplit(":", maxsplit=1)[-1],
                "factory": factory,
                "parameters": {},
            }
        )
    return entries


def _run_jobs(
    source_argument: str,
    source_path: Path,
    entries: list[dict[str, object]],
    seeds: list[int],
    output_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    completed, failed = [], []
    for model_index, entry in enumerate(entries):
        model_name = str(entry["name"])
        model_stem = "".join(character if character.isalnum() or character in "._-" else "_" for character in model_name)
        for seed in seeds:
            result_path = output_dir / "results" / f"{model_index}_{model_stem}" / f"seed_{seed}.json"
            command = [
                sys.executable,
                "-m",
                "binaml.benchmarks.synthetic_streaming_regression.job",
                source_argument,
                str(source_path),
                "--model-name",
                model_name,
                "--factory",
                str(entry["factory"]),
                "--parameters-json",
                json.dumps(entry["parameters"], sort_keys=True),
                "--output",
                str(result_path),
            ]
            if source_argument == "--scenario":
                command.extend(["--seed", str(seed)])
            process = subprocess.run(command, capture_output=True, text=True, check=False)
            job = {"model": model_name, "seed": seed, "result_path": str(result_path.relative_to(output_dir))}
            if process.returncode == 0 and result_path.exists():
                completed.append({**job, "result": json.loads(result_path.read_text(encoding="utf-8"))})
            else:
                failed.append(
                    {
                        **job,
                        "returncode": process.returncode,
                        "error": process.stderr.strip() or "child job did not write a result",
                    }
                )
    return completed, failed


def _write_summary(
    output_dir: Path,
    source_path: Path,
    entries: list[dict[str, object]],
    completed: list[dict[str, object]],
    failed: list[dict[str, object]],
) -> dict[str, object]:
    records_by_model: dict[str, list[dict[str, object]]] = {str(entry["name"]): [] for entry in entries}
    for job in completed:
        model_name = str(job["model"])
        records_by_model[model_name].append(job["result"])  # type: ignore[arg-type]
    model_summaries = {
        model_name: _summary(records)
        for model_name, records in records_by_model.items()
        if records
    }
    metrics = {
        "mse": {name: values["mse"] for name, values in model_summaries.items()},
        "rmse": {name: values["rmse"] for name, values in model_summaries.items()},
        "timing_seconds": {name: values["timing_seconds"] for name, values in model_summaries.items()},
        "n_seeds": {name: values["n_seeds"] for name, values in model_summaries.items()},
    }
    summary = {
        "schema_version": 1,
        "source": str(source_path),
        "metrics": metrics,
        "failed_jobs": failed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _write_job_plots(
    output_dir: Path,
    source_argument: str,
    source_path: Path,
    completed: list[dict[str, object]],
    metrics: dict[str, dict[str, object]],
) -> None:
    try:
        import seaborn as sns

        from .plots import (
            write_aggregate_scatter,
            write_model_plot,
            write_rmse_plot,
        )
    except ModuleNotFoundError as error:
        if error.name in {"matplotlib", "seaborn"}:
            raise RuntimeError("plotting requires `pip install 'binaml[benchmarks]'`") from error
        raise
    records_by_seed: dict[int, dict[str, dict[str, object]]] = {}
    for job in completed:
        records_by_seed.setdefault(int(job["seed"]), {})[str(job["model"])] = job["result"]  # type: ignore[index]
    plots_dir = output_dir / "plots"
    plots_dir.mkdir()
    scenario = json.loads(source_path.read_text(encoding="utf-8")) if source_argument == "--scenario" else None
    warmup_samples = _warmup_samples(scenario) if scenario is not None else 0
    for seed, records in records_by_seed.items():
        trajectory = (
            generate_trajectory(
                SyntheticStreamConfig.from_dict(scenario["environment"]),
                int(scenario["n_samples"]),
                seed,
                return_metadata=True,
            )
            if scenario is not None
            else Trajectory.load_npz(source_path)
        )
        evaluations = {
            name: PrequentialResult(
                np.asarray(record["predictions"]),
                trajectory.y,
                np.asarray(record["squared_errors"]),
                EvaluationTiming(**record["timing_seconds"]),  # type: ignore[arg-type]
            )
            for name, record in records.items()
        }
        write_rmse_plot(plots_dir / f"rmse_seed_{seed}.png", trajectory, evaluations, warmup_samples)
        for (model_name, evaluation), color in zip(
            evaluations.items(), sns.color_palette("colorblind", n_colors=len(evaluations)), strict=True
        ):
            model_dir = plots_dir / model_name.replace("/", "_")
            model_dir.mkdir(exist_ok=True)
            write_model_plot(
                model_dir / f"seed_{seed}.png",
                trajectory,
                model_name,
                evaluation,
                color,
                warmup_samples,
            )
    write_aggregate_scatter(plots_dir / "model_comparison.png", metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scenario", type=Path)
    source.add_argument("--trajectory", type=Path)
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument("--model", action="append", default=[])
    model_source.add_argument("--model-config", type=Path, help="JSON model factory and parameter configuration")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.model_config:
        models, model_config = _load_model_config(args.model_config)
    else:
        model_specifications = args.model or ["binaml.models:SGDLinearRegressor"]
        models = _load_models(model_specifications)
        model_config = None
    source_path = args.scenario if args.scenario else args.trajectory
    output_dir = args.output_dir or Path("runs") / f"synthetic_streaming_regression_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    entries = _model_entries(args.model or ["binaml.models:SGDLinearRegressor"], model_config)
    if args.scenario:
        scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
        seeds = [int(seed) for seed in scenario["seeds"]]
        warmup_samples = _warmup_samples(scenario)
        source_argument = "--scenario"
    else:
        trajectory = Trajectory.load_npz(args.trajectory)
        seeds = [trajectory.seed]
        warmup_samples = 0
        source_argument = "--trajectory"
    config = {
        "schema_version": 2,
        "source": str(source_path),
        "models": list(models),
        "model_config": entries,
        "warmup_samples": warmup_samples,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    completed, failed = _run_jobs(source_argument, source_path, entries, seeds, output_dir)
    summary = _write_summary(output_dir, source_path, entries, completed, failed)
    (output_dir / "metrics.json").write_text(
        json.dumps({"schema_version": 2, "metrics": summary["metrics"]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_job_plots(
        output_dir,
        source_argument,
        source_path,
        completed,
        summary["metrics"],  # type: ignore[arg-type]
    )
    print(json.dumps({"run_dir": str(output_dir), "completed_jobs": len(completed), "failed_jobs": len(failed)}, indent=2))


if __name__ == "__main__":
    main()
