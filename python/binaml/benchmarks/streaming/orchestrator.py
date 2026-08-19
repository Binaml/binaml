"""Shared CLI orchestration for trajectory-based streaming benchmarks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from binaml.benchmarks._common import load_models, model_entries, normalize_models, run_jobs
from binaml.benchmarks.scenario import load_scenario, warmup_samples

from .spec import StreamingBenchmark


def run_scenario(
    benchmark: StreamingBenchmark,
    path: str | Path,
    model_factories: dict[str, Callable[..., object]] | Callable[..., object] | None = None,
    on_evaluation: Callable[..., None] | None = None,
    *,
    default_factory: Callable[..., object],
) -> dict[str, object]:
    scenario = load_scenario(path)
    warmup = warmup_samples(scenario)
    model_factories = normalize_models(model_factories, default_factory)
    records = {name: [] for name in model_factories}
    config_fingerprint = None
    for seed in scenario["seeds"]:
        trajectory = benchmark.load_trajectory_from_scenario(scenario, int(seed))
        if config_fingerprint is None:
            config_fingerprint = trajectory.config.fingerprint
        evaluations = {}
        for name, factory in model_factories.items():
            model = benchmark.build_model(trajectory, factory, {})
            evaluations[name] = benchmark.evaluate(model, trajectory, warmup_samples=warmup)
        for name, result in evaluations.items():
            records[name].append(benchmark.record(int(seed), result, warmup))
        if on_evaluation:
            on_evaluation(trajectory, evaluations)
    return {
        "scenario": scenario["name"],
        "config_fingerprint": config_fingerprint,
        "models": {name: {"results": value, "summary": benchmark.summarize(value)} for name, value in records.items()},
    }


def run_trajectory(
    benchmark: StreamingBenchmark,
    path: str | Path,
    model_factories: dict[str, Callable[..., object]] | Callable[..., object] | None = None,
    on_evaluation: Callable[..., None] | None = None,
    *,
    default_factory: Callable[..., object],
) -> dict[str, object]:
    trajectory = benchmark.load_trajectory_from_npz(Path(path))
    model_factories = normalize_models(model_factories, default_factory)
    evaluations = {}
    for name, factory in model_factories.items():
        model = benchmark.build_model(trajectory, factory, {})
        evaluations[name] = benchmark.evaluate(model, trajectory, warmup_samples=0)
    if on_evaluation:
        on_evaluation(trajectory, evaluations)
    return {
        "source": str(path),
        "config_fingerprint": trajectory.config.fingerprint,
        "seed": trajectory.seed,
        "n_samples": len(trajectory.y),
        "models": {name: benchmark.record(trajectory.seed, result) for name, result in evaluations.items()},
    }


def run_streaming_benchmark_cli(
    benchmark: StreamingBenchmark,
    *,
    default_factory: Callable[..., object],
    write_plots: Callable[[Path, str, Path, list[dict[str, object]], dict[str, object]], None] | None = None,
) -> None:
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

    default_models = list(benchmark.default_models)
    if args.model_config:
        models, model_config = benchmark.load_model_config(args.model_config)
    else:
        model_specifications = args.model or list(default_models)
        models = load_models(model_specifications)
        model_config = None

    source_path = args.scenario if args.scenario else args.trajectory
    output_dir = args.output_dir or Path("runs") / f"{benchmark.run_prefix}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    entries = model_entries(args.model or list(default_models), model_config)

    if args.scenario:
        scenario = load_scenario(args.scenario)
        seeds = [int(seed) for seed in scenario["seeds"]]
        warmup = warmup_samples(scenario)
        source_argument = "--scenario"
    else:
        trajectory = benchmark.load_trajectory_from_npz(args.trajectory)
        seeds = [trajectory.seed]
        warmup = 0
        source_argument = "--trajectory"

    config = {
        "source": str(source_path),
        "models": list(models),
        "model_config": entries,
        "warmup_samples": warmup,
        "plots": args.plots,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    completed, failed = run_jobs(
        job_module=benchmark.job_module,
        source_argument=source_argument,
        source_path=source_path,
        entries=entries,
        seeds=seeds,
        output_dir=output_dir,
    )

    records_by_model: dict[str, list[dict[str, object]]] = {str(entry["name"]): [] for entry in entries}
    for job in completed:
        records_by_model[str(job["model"])].append(job["result"])  # type: ignore[arg-type]
    model_summaries = {name: benchmark.summarize(records) for name, records in records_by_model.items() if records}
    metrics = benchmark.extract_metrics(model_summaries)
    summary = {
        "source": str(source_path),
        "metrics": metrics,
        "failed_jobs": failed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps({"metrics": metrics}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    if args.plots:
        if write_plots is None:
            raise RuntimeError("plotting is not configured for this benchmark")
        write_plots(output_dir, source_argument, source_path, completed, metrics)
    print(json.dumps({"run_dir": str(output_dir), "completed_jobs": len(completed), "failed_jobs": len(failed)}, indent=2))
