"""Batch evaluation metrics for the boolean function learning benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np

from binaml.benchmarks._common import aggregate
from binaml.benchmarks.boolean_function_learning.batches import (
    LearnerSplit,
    draw_split,
    upsample_balanced,
)
from binaml.benchmarks.boolean_function_learning.learners import LEARNER_FACTORIES, RustBatchLearner


def association_score(values: np.ndarray, target: np.ndarray) -> int:
    values = np.asarray(values, dtype=bool)
    target = np.asarray(target, dtype=bool)
    n = len(target)
    nx = int(values.sum())
    ny = int(target.sum())
    nxy = int(np.logical_and(values, target).sum())
    return n * nxy - nx * ny


def accuracy(predictions: np.ndarray, target: np.ndarray) -> float:
    target = np.asarray(target, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    return float(np.mean(predictions == target))


def majority_baseline(target: np.ndarray) -> float:
    positive_rate = float(np.mean(target))
    return max(positive_rate, 1.0 - positive_rate)


def evaluate_learner(
    split: LearnerSplit,
    learner: RustBatchLearner,
    *,
    upsample_train_target: bool = False,
    seed: int = 0,
) -> dict[str, object]:
    x_train, y_train = split.x_train, split.y_train
    if upsample_train_target:
        x_train, y_train = upsample_balanced(x_train, y_train, np.random.default_rng(seed))
    fit_result = learner.fit(x_train, y_train)
    train_predictions = learner.predict(split.x_train)
    test_predictions = learner.predict(split.x_test)
    train_accuracy = accuracy(train_predictions, split.y_train)
    test_accuracy = accuracy(test_predictions, split.y_test)
    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "improvement_over_majority_train": train_accuracy - majority_baseline(split.y_train),
        "improvement_over_majority_test": test_accuracy - majority_baseline(split.y_test),
        "association_train": association_score(train_predictions, split.y_train),
        "association_test": association_score(test_predictions, split.y_test),
        "learner_score": int(fit_result["score"]),
        "elapsed_seconds": float(fit_result["elapsed_seconds"]),
        "p_target_empirical_train": split.p_target_empirical_train,
        "p_target_empirical_test": split.p_target_empirical_test,
        "upsample_train_target": upsample_train_target,
        "n_train_fit": len(y_train),
        "p_feature_empirical_train": split.p_feature_empirical_train.tolist(),
        "p_feature_empirical_test": split.p_feature_empirical_test.tolist(),
        "target_function": split.target_function.to_dict(),
    }


def run_job(scenario: dict[str, object], learner_name: str, parameters: dict[str, Any], seed: int) -> dict[str, object]:
    if learner_name not in LEARNER_FACTORIES:
        raise ValueError(f"unknown learner: {learner_name}")
    split = draw_split(scenario, seed)
    parameters = dict(parameters)
    upsample_train_target = bool(parameters.pop("upsample_train_target", False))
    learner = LEARNER_FACTORIES[learner_name](**parameters)
    result = evaluate_learner(split, learner, upsample_train_target=upsample_train_target, seed=seed)
    return {
        "schema_version": 1,
        "seed": seed,
        "learner": learner_name,
        "parameters": parameters,
        **result,
    }


def summarize_records(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "n_seeds": len(records),
        "train_accuracy": aggregate([float(record["train_accuracy"]) for record in records]),
        "test_accuracy": aggregate([float(record["test_accuracy"]) for record in records]),
        "improvement_over_majority_train": aggregate(
            [float(record["improvement_over_majority_train"]) for record in records]
        ),
        "improvement_over_majority_test": aggregate(
            [float(record["improvement_over_majority_test"]) for record in records]
        ),
        "association_train": aggregate([float(record["association_train"]) for record in records]),
        "association_test": aggregate([float(record["association_test"]) for record in records]),
        "elapsed_seconds": aggregate([float(record["elapsed_seconds"]) for record in records]),
    }


def extract_metrics(summaries_by_learner: dict[str, dict[str, object]]) -> dict[str, object]:
    metrics = {
        "train_accuracy": {name: values["train_accuracy"] for name, values in summaries_by_learner.items()},
        "test_accuracy": {name: values["test_accuracy"] for name, values in summaries_by_learner.items()},
        "improvement_over_majority_train": {
            name: values["improvement_over_majority_train"] for name, values in summaries_by_learner.items()
        },
        "improvement_over_majority_test": {
            name: values["improvement_over_majority_test"] for name, values in summaries_by_learner.items()
        },
        "association_train": {name: values["association_train"] for name, values in summaries_by_learner.items()},
        "association_test": {name: values["association_test"] for name, values in summaries_by_learner.items()},
        "elapsed_seconds": {name: values["elapsed_seconds"] for name, values in summaries_by_learner.items()},
        "n_seeds": {name: values["n_seeds"] for name, values in summaries_by_learner.items()},
    }
    return metrics
