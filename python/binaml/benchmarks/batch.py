"""Shared orchestration for batch (non-streaming) benchmarks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from binaml.benchmarks._common import run_jobs, write_json_atomically
from binaml.benchmarks.scenario import expand_grid, load_scenario


def run_batch_benchmark_cli(
    *,
    run_prefix: str,
    job_module: str,
    entries_key: str,
    default_entries: list[dict[str, object]],
    summarize_variant: Callable[[list[dict[str, object]]], dict[str, object]],
    extract_metrics: Callable[[dict[str, dict[str, object]]], dict[str, object]],
    variant_label: Callable[[dict[str, object], int], str],
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    base_scenario = load_scenario(args.scenario)
    variants = expand_grid(base_scenario)
    output_dir = args.output_dir or Path("runs") / f"{run_prefix}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)

    raw_entries = base_scenario.get(entries_key, default_entries)
    if not isinstance(raw_entries, dict):
        raise TypeError(f"{entries_key} must be an object")
    entries = [{"name": name, "factory": name, "parameters": parameters} for name, parameters in raw_entries.items()]

    seeds = [int(seed) for seed in base_scenario["seeds"]]
    config = {
        "source": str(args.scenario),
        "entries": entries,
        "n_variants": len(variants),
        "seeds": seeds,
    }
    write_json_atomically(output_dir / "config.json", config)

    completed: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    for variant_index, variant in enumerate(variants):
        variant_dir = output_dir / "variants" / variant_label(variant, variant_index)
        variant_path = variant_dir / "scenario.json"
        write_json_atomically(variant_path, variant)
        variant_completed, variant_failed = run_jobs(
            job_module=job_module,
            source_argument="--scenario",
            source_path=variant_path,
            entries=entries,
            seeds=seeds,
            output_dir=variant_dir,
        )
        for job in variant_completed:
            completed.append({**job, "variant_index": variant_index, "variant_label": variant_label(variant, variant_index)})
        for job in variant_failed:
            failed.append({**job, "variant_index": variant_index, "variant_label": variant_label(variant, variant_index)})

    records_by_variant: dict[str, list[dict[str, object]]] = {}
    for job in completed:
        label = str(job["variant_label"])
        records_by_variant.setdefault(label, []).append(job["result"])  # type: ignore[arg-type]
    variant_summaries = {label: summarize_variant(records) for label, records in records_by_variant.items() if records}
    metrics = {label: extract_metrics(summaries) for label, summaries in variant_summaries.items()}
    summary = {
        "source": str(args.scenario),
        "metrics": metrics,
        "failed_jobs": failed,
    }
    write_json_atomically(output_dir / "summary.json", summary)
    write_json_atomically(output_dir / "metrics.json", {"metrics": metrics})
    print(json.dumps({"run_dir": str(output_dir), "completed_jobs": len(completed), "failed_jobs": len(failed)}, indent=2))
