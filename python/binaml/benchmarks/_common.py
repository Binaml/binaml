"""Shared helpers for streaming benchmark CLIs and jobs."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from collections.abc import Callable
from math import sqrt
from pathlib import Path
from statistics import fmean, stdev


def timing_payload(result: object) -> dict[str, float]:
    timing = result.timing_seconds  # type: ignore[attr-defined]
    return {"total": timing.total, "prediction": timing.prediction, "update": timing.update}


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "average": fmean(values),
        "standard_error": stdev(values) / sqrt(len(values)) if len(values) > 1 else 0.0,
    }


def warmup_samples(scenario: dict[str, object]) -> int:
    from binaml.benchmarks.scenario import warmup_samples as _warmup_samples

    return _warmup_samples(scenario)


def load_factory(specification: str):
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("model must use module:factory syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("model factory must be callable")
    return factory


def load_models(specifications: list[str]):
    models = {}
    for specification in specifications:
        name, separator, factory_specification = specification.partition("=")
        factory_specification = factory_specification if separator else specification
        factory = load_factory(factory_specification)
        model_name = name if separator else factory_specification.rsplit(":", maxsplit=1)[-1]
        if model_name in models:
            raise ValueError(f"duplicate model name: {model_name}")
        models[model_name] = factory
    return models


def load_model_config(
    path: Path,
    *,
    reserved_parameters: frozenset[str],
    bind_factory: Callable[[Callable[..., object], dict[str, object]], Callable[..., object]],
) -> tuple[dict[str, Callable[..., object]], list[dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise TypeError("model config must contain a models list")

    models: dict[str, Callable[..., object]] = {}
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
        for reserved in reserved_parameters:
            if reserved in parameters:
                raise ValueError(f"{reserved} is supplied by the benchmark")
        if name in models:
            raise ValueError(f"duplicate model name: {name}")
        factory = load_factory(specification)
        models[name] = bind_factory(factory, parameters)
        normalized_config.append({"name": name, "factory": specification, "parameters": parameters})
    if not models:
        raise ValueError("model config must contain at least one model")
    return models, normalized_config


def model_entries(specifications: list[str], model_config: list[dict[str, object]] | None) -> list[dict[str, object]]:
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


def write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def run_jobs(
    *,
    job_module: str,
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
                job_module,
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


def normalize_models(model_factories, default_factory):
    if model_factories is None:
        return {default_factory.__name__: default_factory}
    if callable(model_factories):
        return {model_factories.__name__: model_factories}
    return model_factories
