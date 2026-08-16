"""Command-line composition for named streaming-regression scenarios.

Pass ``--model-config`` a JSON file shaped as:
``{"models": [{"name": "...", "factory": "module:callable", "parameters": {...}}]}``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from binaml.benchmarks._common import (
    aggregate,
    load_factory,
    load_model_config,
    load_models,
    model_entries,
    normalize_models,
    run_jobs,
    timing_payload,
    warmup_samples,
)
from binaml.environments import SyntheticStreamConfig, Trajectory, generate_trajectory
from binaml.evaluation import EvaluationTiming, PrequentialResult, evaluate_prequentially
from binaml.models import OnlineModel, SGDLinearRegressor

ModelFactory = Callable[[int], OnlineModel]
EvaluationCallback = Callable[[Trajectory, dict[str, PrequentialResult]], None]

_warmup_samples = warmup_samples


def _load_model_config(path: Path) -> tuple[dict[str, ModelFactory], list[dict[str, object]]]:
    return load_model_config(
        path,
        reserved_parameters=frozenset({"n_features"}),
        bind_factory=lambda factory, parameters: (
            lambda n_features, factory=factory, parameters=parameters: factory(n_features, **parameters)
        ),
    )


def _record(seed: int, result: PrequentialResult, warmup_samples: int = 0) -> dict[str, object]:
    squared_errors = result.squared_errors[warmup_samples:]
    valid = np.isfinite(squared_errors)
    mse = float(squared_errors[valid].mean()) if np.any(valid) else float("nan")
    return {
        "seed": seed,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "timing_seconds": timing_payload(result),
    }


def _summary(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "n_seeds": len(records),
        "mse": aggregate([record["mse"] for record in records]),  # type: ignore[list-item]
        "rmse": aggregate([record["rmse"] for record in records]),  # type: ignore[list-item]
        "timing_seconds": {
            "total": aggregate([record["timing_seconds"]["total"] for record in records])  # type: ignore[index]
        },
    }


def run_scenario(
    path: str | Path,
    model_factories: dict[str, ModelFactory] | ModelFactory | None = None,
    on_evaluation: EvaluationCallback | None = None,
) -> dict[str, object]:
    scenario = json.loads(Path(path).read_text(encoding="utf-8"))
    config = SyntheticStreamConfig.from_dict(scenario["environment"])
    warmup = _warmup_samples(scenario)
    model_factories = normalize_models(model_factories, SGDLinearRegressor)
    records = {name: [] for name in model_factories}
    for seed in scenario["seeds"]:
        trajectory = generate_trajectory(config, int(scenario["n_samples"]), int(seed), return_metadata=True)
        evaluations = {
            name: evaluate_prequentially(factory(config.n_features), trajectory, warmup_samples=warmup)
            for name, factory in model_factories.items()
        }
        for name, result in evaluations.items():
            records[name].append(_record(int(seed), result, warmup))
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
    model_factories = normalize_models(model_factories, SGDLinearRegressor)
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


def _write_job_plots(
    output_dir: Path,
    source_argument: str,
    source_path: Path,
    completed: list[dict[str, object]],
    metrics: dict[str, dict[str, object]],
) -> None:
    try:
        import seaborn as sns

        from .plots import write_aggregate_scatter, write_model_plot, write_rmse_plot
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
    warmup = _warmup_samples(scenario) if scenario is not None else 0
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
        write_rmse_plot(plots_dir / f"rmse_seed_{seed}.png", trajectory, evaluations, warmup)
        for (model_name, evaluation), color in zip(
            evaluations.items(), sns.color_palette("colorblind", n_colors=len(evaluations)), strict=True
        ):
            model_dir = plots_dir / model_name.replace("/", "_")
            model_dir.mkdir(exist_ok=True)
            write_model_plot(model_dir / f"seed_{seed}.png", trajectory, model_name, evaluation, color, warmup)
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
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    if args.model_config:
        models, model_config = _load_model_config(args.model_config)
    else:
        model_specifications = args.model or ["binaml.models:SGDLinearRegressor"]
        models = load_models(model_specifications)
        model_config = None
    source_path = args.scenario if args.scenario else args.trajectory
    output_dir = args.output_dir or Path("runs") / f"synthetic_streaming_regression_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    entries = model_entries(args.model or ["binaml.models:SGDLinearRegressor"], model_config)
    if args.scenario:
        scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
        seeds = [int(seed) for seed in scenario["seeds"]]
        warmup = _warmup_samples(scenario)
        source_argument = "--scenario"
    else:
        trajectory = Trajectory.load_npz(args.trajectory)
        seeds = [trajectory.seed]
        warmup = 0
        source_argument = "--trajectory"
    config = {
        "schema_version": 2,
        "source": str(source_path),
        "models": list(models),
        "model_config": entries,
        "warmup_samples": warmup,
        "plots": args.plots,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    completed, failed = run_jobs(
        job_module="binaml.benchmarks.synthetic_streaming_regression.job",
        source_argument=source_argument,
        source_path=source_path,
        entries=entries,
        seeds=seeds,
        output_dir=output_dir,
    )
    records_by_model: dict[str, list[dict[str, object]]] = {str(entry["name"]): [] for entry in entries}
    for job in completed:
        records_by_model[str(job["model"])].append(job["result"])  # type: ignore[arg-type]
    model_summaries = {name: _summary(records) for name, records in records_by_model.items() if records}
    metrics = {
        "mse": {name: values["mse"] for name, values in model_summaries.items()},
        "rmse": {name: values["rmse"] for name, values in model_summaries.items()},
        "timing_seconds": {name: values["timing_seconds"] for name, values in model_summaries.items()},
        "n_seeds": {name: values["n_seeds"] for name, values in model_summaries.items()},
    }
    summary = {"schema_version": 1, "source": str(source_path), "metrics": metrics, "failed_jobs": failed}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps({"schema_version": 2, "metrics": metrics}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.plots:
        _write_job_plots(output_dir, source_argument, source_path, completed, metrics)
    print(json.dumps({"run_dir": str(output_dir), "completed_jobs": len(completed), "failed_jobs": len(failed)}, indent=2))


if __name__ == "__main__":
    main()
