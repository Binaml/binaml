"""Benchmark specification protocol for trajectory-based streaming evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from binaml.benchmarks._common import load_model_config

ModelFactory = Callable[..., Any]


@dataclass(frozen=True)
class StreamingBenchmark:
    run_prefix: str
    job_module: str
    default_models: tuple[str, ...]
    config_schema_version: int
    summary_schema_version: int
    metrics_schema_version: int
    reserved_parameters: frozenset[str]
    bind_factory: Callable[[ModelFactory, dict[str, object]], ModelFactory]
    load_trajectory_from_scenario: Callable[[dict[str, object], int], Any]
    load_trajectory_from_npz: Callable[[Path], Any]
    build_model: Callable[[Any, ModelFactory, dict[str, object]], Any]
    evaluate: Callable[[Any, Any, int], Any]
    record: Callable[[int, Any, int], dict[str, object]]
    summarize: Callable[[list[dict[str, object]]], dict[str, object]]
    extract_metrics: Callable[[dict[str, dict[str, object]]], dict[str, object]]
    job_result_extras: Callable[[Any], dict[str, object]]

    def load_model_config(self, path: Path) -> tuple[dict[str, ModelFactory], list[dict[str, object]]]:
        return load_model_config(
            path,
            reserved_parameters=self.reserved_parameters,
            bind_factory=self.bind_factory,
        )
