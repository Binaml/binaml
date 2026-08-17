"""Shared job runner for trajectory-based streaming benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from binaml.benchmarks._common import load_factory, write_json_atomically
from binaml.benchmarks.scenario import load_scenario, warmup_samples

from .spec import StreamingBenchmark


def run_streaming_job(benchmark: StreamingBenchmark, args: argparse.Namespace) -> dict[str, object]:
    if args.scenario is not None:
        scenario = load_scenario(args.scenario)
        trajectory = benchmark.load_trajectory_from_scenario(scenario, args.seed)
        warmup = warmup_samples(scenario)
    else:
        scenario = None
        trajectory = benchmark.load_trajectory_from_npz(args.trajectory)
        warmup = 0

    parameters = json.loads(args.parameters_json)
    if not isinstance(parameters, dict):
        raise TypeError("model parameters must be a JSON object")

    factory = load_factory(args.factory)
    model = benchmark.build_model(trajectory, factory, parameters)
    result = benchmark.evaluate(model, trajectory, warmup_samples=warmup)
    record = benchmark.record(trajectory.seed, result, warmup)
    timing = record["timing_seconds"]
    return {
        "schema_version": 1,
        "model": args.model_name,
        "warmup_samples": warmup,
        **record,
        "timing_seconds": {
            "total": timing["total"],
            "prediction": timing["prediction"],
            "update": timing["update"],
        },
        **benchmark.job_result_extras(result),
    }


def run_streaming_job_cli(benchmark: StreamingBenchmark) -> None:
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
    write_json_atomically(args.output, run_streaming_job(benchmark, args))
