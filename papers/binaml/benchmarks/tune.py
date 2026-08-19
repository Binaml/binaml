"""Coarse-to-fine hyperparameter search for paper streaming benchmarks."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import numpy as np

from binaml.benchmarks.scenario import load_scenario, warmup_samples
from binaml.environments import (
    SyntheticClassificationStreamConfig,
    SyntheticStreamConfig,
    generate_classification_trajectory,
    generate_trajectory,
)
from binaml.evaluation import evaluate_prequentially, evaluate_prequentially_classification
from binaml.models import BClassifier, BRegressor, MLPClassifier, MLPRegressor, SGDLinearClassifier, SGDLinearRegressor

BENCHMARK_DIR = Path(__file__).resolve().parent
TUNE_SEED = 0
TUNE_SAMPLES = 400
TUNE_WARMUP = 40
FINAL_SEEDS = [0, 1, 2, 3, 4]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: str
    base_parameters: dict[str, Any]
    coarse_axes: dict[str, list[Any]]
    fine_axes: dict[str, Callable[[Any], list[Any]]]
    score_key: str
    lower_is_better: bool


def _float_neighbors(value: float, *, minimum: float = 0.0) -> list[float]:
    candidates = {value * factor for factor in (0.5, 1.0, 1.5)}
    return sorted({max(minimum, candidate) for candidate in candidates})


def _int_neighbors(value: int, *, minimum: int = 1) -> list[int]:
    candidates = {value - 1, value, value + 1}
    return sorted({max(minimum, candidate) for candidate in candidates})


def _load_model_parameters(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for entry in payload["models"]:
        if entry["name"] == name:
            return dict(entry["parameters"])
    raise KeyError(f"model {name!r} not found in {path}")


REGRESSION_MODELS = [
    ModelSpec(
        "BRegressor",
        "binaml.models:BRegressor",
        {},
        {
            "learning_rate": [0.01, 0.015, 0.03],
            "l2": [1e-3, 2e-3, 5e-3],
            "batch_size": [4, 5, 8],
        },
        {
            "learning_rate": lambda value: _float_neighbors(value, minimum=1e-4),
            "l2": lambda value: _float_neighbors(value, minimum=0.0),
            "batch_size": lambda value: _int_neighbors(value, minimum=1),
        },
        "mse",
        True,
    ),
    ModelSpec(
        "SGDLinearRegressor",
        "binaml.models:SGDLinearRegressor",
        {},
        {
            "learning_rate": [0.01, 0.02, 0.03],
            "l2": [5e-3, 7.5e-3, 1e-2],
            "sgd_steps": [4, 8, 16],
        },
        {
            "learning_rate": lambda value: _float_neighbors(value, minimum=1e-5),
            "l2": lambda value: _float_neighbors(value, minimum=0.0),
            "sgd_steps": lambda value: _int_neighbors(value, minimum=1),
        },
        "mse",
        True,
    ),
    ModelSpec(
        "MLPRegressor",
        "binaml.models:MLPRegressor",
        {},
        {
            "learning_rate": [0.02, 0.03, 0.05],
            "alpha": [5e-4, 1e-3, 2e-3],
            "sgd_steps": [12, 24, 32],
        },
        {
            "learning_rate": lambda value: _float_neighbors(value, minimum=1e-5),
            "alpha": lambda value: _float_neighbors(value, minimum=0.0),
            "sgd_steps": lambda value: _int_neighbors(value, minimum=1),
        },
        "mse",
        True,
    ),
]

CLASSIFICATION_MODELS = [
    ModelSpec(
        "BClassifier",
        "binaml.models:BClassifier",
        {},
        {
            "learning_rate": [0.04, 0.06, 0.08],
            "l2": [1e-3, 3e-3, 5e-3],
            "batch_size": [4, 6, 8],
            "sgd_steps": [20, 29, 40],
        },
        {
            "learning_rate": lambda value: _float_neighbors(value, minimum=1e-4),
            "l2": lambda value: _float_neighbors(value, minimum=0.0),
            "batch_size": lambda value: _int_neighbors(value, minimum=1),
            "sgd_steps": lambda value: _int_neighbors(value, minimum=1),
        },
        "accuracy",
        False,
    ),
    ModelSpec(
        "SGDLinearClassifier",
        "binaml.models:SGDLinearClassifier",
        {},
        {
            "learning_rate": [0.01, 0.02, 0.03],
            "l2": [0.0, 1e-3, 5e-3],
            "sgd_steps": [8, 16, 24],
        },
        {
            "learning_rate": lambda value: _float_neighbors(value, minimum=1e-5),
            "l2": lambda value: _float_neighbors(value, minimum=0.0),
            "sgd_steps": lambda value: _int_neighbors(value, minimum=1),
        },
        "accuracy",
        False,
    ),
    ModelSpec(
        "MLPClassifier",
        "binaml.models:MLPClassifier",
        {},
        {
            "learning_rate": [0.01, 0.015, 0.025],
            "alpha": [0.0, 1e-4, 5e-4],
            "sgd_steps": [8, 16, 24],
        },
        {
            "learning_rate": lambda value: _float_neighbors(value, minimum=1e-5),
            "alpha": lambda value: _float_neighbors(value, minimum=0.0),
            "sgd_steps": lambda value: _int_neighbors(value, minimum=1),
        },
        "accuracy",
        False,
    ),
]


def _expand_grid(base: dict[str, Any], axes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not axes:
        return [dict(base)]
    keys = list(axes)
    grids = []
    for values in product(*(axes[key] for key in keys)):
        parameters = dict(base)
        parameters.update(dict(zip(keys, values, strict=True)))
        grids.append(parameters)
    return grids


def _expand_fine_grid(base: dict[str, Any], best: dict[str, Any], fine_axes: dict[str, Callable[[Any], list[Any]]]) -> list[dict[str, Any]]:
    axes = {key: neighbor_fn(best[key]) for key, neighbor_fn in fine_axes.items()}
    return _expand_grid(base, axes)


def _score(record: dict[str, float], spec: ModelSpec) -> float:
    value = record[spec.score_key]
    return value if spec.lower_is_better else -value


def _evaluate_regression(parameters: dict[str, Any], factory_name: str, trajectory) -> dict[str, float]:
    n_features = trajectory.config.n_features
    if factory_name.endswith("BRegressor"):
        model = BRegressor(n_features, **parameters)
    elif factory_name.endswith("SGDLinearRegressor"):
        model = SGDLinearRegressor(n_features, **parameters)
    else:
        model = MLPRegressor(n_features, **parameters)
    result = evaluate_prequentially(model, trajectory, warmup_samples=TUNE_WARMUP)
    mse = float(result.squared_errors[TUNE_WARMUP:].mean())
    return {"mse": mse, "rmse": float(np.sqrt(mse))}


def _evaluate_classification(parameters: dict[str, Any], factory_name: str, trajectory) -> dict[str, float]:
    n_features = trajectory.config.n_features
    n_classes = trajectory.config.n_classes
    if factory_name.endswith("BClassifier"):
        model = BClassifier(n_features, n_classes, **parameters)
    elif factory_name.endswith("SGDLinearClassifier"):
        model = SGDLinearClassifier(n_features, n_classes, **parameters)
    else:
        model = MLPClassifier(n_features, n_classes, **parameters)
    result = evaluate_prequentially_classification(model, trajectory, warmup_samples=TUNE_WARMUP)
    accuracy = float(result.correct[TUNE_WARMUP:].mean())
    return {"accuracy": accuracy}


def _with_base_parameters(spec: ModelSpec, base_parameters: dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        spec.name,
        spec.factory,
        base_parameters,
        spec.coarse_axes,
        spec.fine_axes,
        spec.score_key,
        spec.lower_is_better,
    )


def _search_model(
    spec: ModelSpec,
    trajectory,
    evaluate: Callable[[dict[str, Any], str, Any], dict[str, float]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    history: list[dict[str, Any]] = []

    def run_stage(stage: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        best_parameters = candidates[0]
        best_metrics = evaluate(best_parameters, spec.factory, trajectory)
        best_score = _score(best_metrics, spec)
        for parameters in candidates:
            metrics = evaluate(parameters, spec.factory, trajectory)
            score = _score(metrics, spec)
            history.append({"stage": stage, "parameters": parameters, **metrics, "score": score})
            if score < best_score:
                best_score = score
                best_parameters = parameters
                best_metrics = metrics
        return best_parameters

    coarse_candidates = _expand_grid(spec.base_parameters, spec.coarse_axes)
    coarse_best = run_stage("coarse", coarse_candidates)
    fine_candidates = _expand_fine_grid(spec.base_parameters, coarse_best, spec.fine_axes)
    fine_best = run_stage("fine", fine_candidates)
    return fine_best, history


def _load_tune_trajectory(task: str, scenario: dict[str, object]):
    environment = scenario["environment"]
    if task == "regression":
        config = SyntheticStreamConfig.from_dict(environment)
        return generate_trajectory(config, TUNE_SAMPLES, TUNE_SEED)
    config = SyntheticClassificationStreamConfig.from_dict(environment)
    return generate_classification_trajectory(config, TUNE_SAMPLES, TUNE_SEED)


def _final_eval_regression(parameters_by_name: dict[str, dict[str, Any]], scenario: dict[str, object]) -> dict[str, Any]:
    environment = scenario["environment"]
    config = SyntheticStreamConfig.from_dict(environment)
    warmup = warmup_samples(scenario)
    records: dict[str, list[dict[str, float]]] = {name: [] for name in parameters_by_name}
    for seed in FINAL_SEEDS:
        trajectory = generate_trajectory(config, int(scenario["n_samples"]), seed)
        for name, parameters in parameters_by_name.items():
            if name == "BRegressor":
                model = BRegressor(config.n_features, **parameters)
            elif name == "SGDLinearRegressor":
                model = SGDLinearRegressor(config.n_features, **parameters)
            else:
                model = MLPRegressor(config.n_features, **parameters)
            result = evaluate_prequentially(model, trajectory, warmup_samples=warmup)
            mse = float(result.squared_errors[warmup:].mean())
            records[name].append(
                {
                    "seed": seed,
                    "mse": mse,
                    "rmse": float(np.sqrt(mse)),
                    "timing_seconds": result.timing_seconds.total,
                }
            )
    return _summarize_records(records)


def _final_eval_classification(parameters_by_name: dict[str, dict[str, Any]], scenario: dict[str, object]) -> dict[str, Any]:
    environment = scenario["environment"]
    config = SyntheticClassificationStreamConfig.from_dict(environment)
    warmup = warmup_samples(scenario)
    records: dict[str, list[dict[str, float]]] = {name: [] for name in parameters_by_name}
    for seed in FINAL_SEEDS:
        trajectory = generate_classification_trajectory(config, int(scenario["n_samples"]), seed)
        for name, parameters in parameters_by_name.items():
            if name == "BClassifier":
                model = BClassifier(config.n_features, config.n_classes, **parameters)
            elif name == "SGDLinearClassifier":
                model = SGDLinearClassifier(config.n_features, config.n_classes, **parameters)
            else:
                model = MLPClassifier(config.n_features, config.n_classes, **parameters)
            result = evaluate_prequentially_classification(model, trajectory, warmup_samples=warmup)
            records[name].append(
                {
                    "seed": seed,
                    "accuracy": float(result.correct[warmup:].mean()),
                    "timing_seconds": result.timing_seconds.total,
                }
            )
    return _summarize_records(records)


def _summarize_records(records: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, values in records.items():
        metric_keys = [key for key in values[0] if key not in {"seed"}]
        summary[name] = {
            "n_seeds": len(values),
            **{
                key: {
                    "average": fmean(item[key] for item in values),
                    "standard_error": (stdev(item[key] for item in values) / math.sqrt(len(values))) if len(values) > 1 else 0.0,
                }
                for key in metric_keys
            },
        }
    return summary


def _write_model_config(
    path: Path,
    specs: list[ModelSpec],
    parameters_by_name: dict[str, dict[str, Any]],
    *,
    template_path: Path,
) -> None:
    template = json.loads(template_path.read_text(encoding="utf-8"))
    tuned_names = {spec.name for spec in specs}
    models = []
    for entry in template["models"]:
        if entry["name"] in tuned_names:
            models.append(
                {
                    "name": entry["name"],
                    "factory": entry["factory"],
                    "parameters": parameters_by_name[entry["name"]],
                }
            )
        else:
            models.append(entry)
    path.write_text(json.dumps({"models": models}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tune_task(
    task: str,
    output_dir: Path,
    *,
    scenario_path: Path,
    model_config_path: Path,
) -> dict[str, Any]:
    if task == "regression":
        model_specs = REGRESSION_MODELS
        evaluate = _evaluate_regression
        final_eval = _final_eval_regression
    else:
        model_specs = CLASSIFICATION_MODELS
        evaluate = _evaluate_classification
        final_eval = _final_eval_classification

    model_specs = [
        _with_base_parameters(spec, _load_model_parameters(model_config_path, spec.name)) for spec in model_specs
    ]
    scenario = load_scenario(scenario_path)
    trajectory = _load_tune_trajectory(task, scenario)
    parameters_by_name: dict[str, dict[str, Any]] = {}
    tuning_history: dict[str, list[dict[str, Any]]] = {}
    for spec in model_specs:
        best, history = _search_model(spec, trajectory, evaluate)
        parameters_by_name[spec.name] = best
        tuning_history[spec.name] = history
        print(f"[{task}] {spec.name} best tuning params: {json.dumps(best, sort_keys=True)}")

    _write_model_config(
        model_config_path,
        model_specs,
        parameters_by_name,
        template_path=model_config_path,
    )
    final_summary = final_eval(parameters_by_name, scenario)

    report = {
        "task": task,
        "scenario": str(scenario_path),
        "n_functions": scenario["environment"]["n_functions"],
        "tune_seed": TUNE_SEED,
        "tune_samples": TUNE_SAMPLES,
        "tune_warmup": TUNE_WARMUP,
        "final_seeds": FINAL_SEEDS,
        "best_parameters": parameters_by_name,
        "tuning_history": tuning_history,
        "final_summary": final_summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{task}_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("regression", "classification", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tuning"))
    parser.add_argument(
        "--regression-scenario",
        type=Path,
        default=BENCHMARK_DIR / "scenarios" / "default.json",
    )
    parser.add_argument(
        "--classification-scenario",
        type=Path,
        default=BENCHMARK_DIR / "scenarios" / "default_multiclass.json",
    )
    args = parser.parse_args()

    task_runs = {
        "regression": (args.regression_scenario, BENCHMARK_DIR / "models.json"),
        "classification": (args.classification_scenario, BENCHMARK_DIR / "models_classification.json"),
    }
    tasks = ("regression", "classification") if args.task == "all" else (args.task,)
    reports = [
        tune_task(
            task,
            args.output_dir,
            scenario_path=task_runs[task][0],
            model_config_path=task_runs[task][1],
        )
        for task in tasks
    ]
    print(json.dumps({"output_dir": str(args.output_dir), "reports": reports}, indent=2, default=str))


if __name__ == "__main__":
    main()
