#!/usr/bin/env python3
"""Two-phase tuning: random search on 1 seed, validate top-k on 5 seeds."""

from __future__ import annotations

import json
import random
from pathlib import Path

from binaml.benchmarks.synthetic_streaming_classification.cli import run_scenario as run_cls
from binaml.benchmarks.synthetic_streaming_regression.cli import run_scenario as run_reg
from binaml.models import BClassifier, BRegressor

REG_TUNE = Path("papers/binaml/benchmarks/scenarios/default_tune.json")
CLS_TUNE = Path("papers/binaml/benchmarks/scenarios/default_multiclass_tune.json")
REG_FULL = Path("papers/binaml/benchmarks/scenarios/default.json")
CLS_FULL = Path("papers/binaml/benchmarks/scenarios/default_multiclass.json")


def _avg(summary: dict[str, object], name: str, key: str) -> float:
    entry = summary["models"][name]["summary"][key]
    return float(entry["average"])


def _sample(space: dict[str, list[object]], rng: random.Random) -> dict[str, object]:
    return {key: rng.choice(values) for key, values in space.items()}


def _search(
    *,
    label: str,
    tune_path: Path,
    full_path: Path,
    model_name: str,
    metric: str,
    higher: bool,
    space: dict[str, list[object]],
    n_samples: int,
    top_k: int,
    run_tune,
    run_full,
    make_model,
) -> tuple[float, dict[str, object]]:
    rng = random.Random(0)
    scored: list[tuple[float, dict[str, object]]] = []
    for i in range(n_samples):
        params = _sample(space, rng)
        summary = run_tune(tune_path, {model_name: lambda *args, p=params: make_model(*args, p)})
        score = _avg(summary, model_name, metric)
        if metric == "mse" and (score != score or score > 1e6):
            continue
        scored.append((score, params))
        if i % 20 == 0:
            print(f"{label} progress {i}/{n_samples}")

    scored.sort(key=lambda item: item[0], reverse=higher)
    candidates = scored[:top_k]
    print(f"=== {label} validate top {top_k} on 5 seeds ===")
    best = (float("-inf") if higher else float("inf"), {})
    for score_tune, params in candidates:
        summary = run_full(full_path, {model_name: lambda *args, p=params: make_model(*args, p)})
        score = _avg(summary, model_name, metric)
        improved = score > best[0] if higher else score < best[0]
        if improved:
            best = (score, params)
        print(f"{label} tune={score_tune:.4f} full={score:.4f} {params}")
    return best


def main() -> None:
    reg = _search(
        label="reg",
        tune_path=REG_TUNE,
        full_path=REG_FULL,
        model_name="BRegressor",
        metric="mse",
        higher=False,
        n_samples=100,
        top_k=12,
        space={
            "learning_rate": [0.015, 0.02, 0.024, 0.03, 0.035, 0.04, 0.05],
            "l2": [0.0, 0.0001, 0.0003, 0.0005, 0.001, 0.002],
            "sgd_steps": [10, 15, 20, 25, 30, 35, 40, 50],
            "batch_size": [1, 2, 4, 6, 8, 12],
            "parent_top_k": [4, 6, 8, 10, 12, 16],
            "l_pat": [2, 3, 4, 5, 6, 8],
            "max_functions": [32, 48, 64, 80, 96, 128],
            "max_expert_nodes": [32, 48, 64, 80, 96],
        },
        run_tune=run_reg,
        run_full=run_reg,
        make_model=lambda n_features, params: BRegressor(n_features, **params),
    )
    print(f"reg best full mse={reg[0]:.4f} params={json.dumps(reg[1], sort_keys=True)}")

    cls = _search(
        label="cls",
        tune_path=CLS_TUNE,
        full_path=CLS_FULL,
        model_name="BClassifier",
        metric="accuracy",
        higher=True,
        n_samples=100,
        top_k=12,
        space={
            "learning_rate": [0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22],
            "l2": [0.0, 0.0001, 0.0003, 0.0005, 0.001],
            "sgd_steps": [10, 20, 30, 40, 50, 60],
            "batch_size": [1, 2, 4, 6, 8, 12, 16],
            "parent_top_k": [4, 6, 8, 10, 12, 16],
            "l_pat": [2, 3, 4, 5, 6],
            "max_functions": [48, 64, 80, 96, 128],
            "max_expert_nodes": [32, 48, 64, 80, 96],
        },
        run_tune=run_cls,
        run_full=run_cls,
        make_model=lambda n_features, n_classes, params: BClassifier(n_features, n_classes, **params),
    )
    print(f"cls best full acc={cls[0]:.4f} params={json.dumps(cls[1], sort_keys=True)}")


if __name__ == "__main__":
    main()
