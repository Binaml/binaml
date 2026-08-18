import json
import sys

import numpy as np
from binaml._core import FunctionLearner
from binaml.benchmarks.boolean_function_learning.batches import draw_split
from binaml.benchmarks.boolean_function_learning.evaluate import association_score
from binaml.benchmarks.boolean_function_learning.learners import (
    build_function_builder,
)
from binaml.benchmarks.scenario import expand_grid


def test_association_score_hand_computed() -> None:
    values = np.array([True, True, False, False])
    target = np.array([True, False, True, False])
    assert association_score(values, target) == 0


def test_function_builder_rust_learner_on_synthetic_batch() -> None:
    x = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ],
        dtype=np.uint8,
    )
    y = np.array([False, True, True, True])
    learner = build_function_builder(parent_top_k=2)
    predictions, score, elapsed = learner.fit_predict_with_details(x, y)
    assert predictions.shape == (4,)
    assert score >= 3
    assert elapsed >= 0.0


def test_function_builder_learns_negated_literal_target() -> None:
    x = np.array([[0], [1], [0], [1], [0], [1], [0], [1]], dtype=np.uint8)
    y = np.array([True, False, True, False, True, False, True, False])
    learner = build_function_builder(parent_top_k=2)
    predictions, score, elapsed = learner.fit_predict_with_details(x, y)
    assert np.array_equal(predictions, y)
    assert score == 8
    assert elapsed >= 0.0


def test_function_builder_predicts_on_holdout_after_fit() -> None:
    x_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    y_train = np.array([False, True, True, True])
    x_test = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    learner = build_function_builder(parent_top_k=2)
    learner.fit(x_train, y_train)
    predictions = learner.predict(x_test)
    assert np.array_equal(predictions, np.array([True, False]))


def test_function_builder_learns_xor_target() -> None:
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    y = np.array([False, True, True, False])
    learner = build_function_builder(parent_top_k=2)
    predictions, score, _elapsed = learner.fit_predict_with_details(x, y)
    assert np.array_equal(predictions, y)
    assert score == 4


def test_draw_split_uses_same_target_on_train_and_test() -> None:
    scenario = {
        "n_train": 8,
        "n_test": 4,
        "environment": {
            "schema_version": 4,
            "n_features": 4,
            "q_max": 0,
        },
    }
    split = draw_split(scenario, seed=0)
    assert split.x_train.shape == (8, 4)
    assert split.x_test.shape == (4, 4)
    assert split.y_train.dtype == bool
    assert split.y_test.dtype == bool
    for rows, labels in ((split.x_train, split.y_train), (split.x_test, split.y_test)):
        recomputed = np.array([bool(split.target_function.evaluate(row)) for row in rows], dtype=bool)
        assert np.array_equal(recomputed, labels)


def test_draw_split_respects_min_target_arity() -> None:
    scenario = {
        "n_train": 32,
        "n_test": 16,
        "environment": {
            "schema_version": 4,
            "n_features": 32,
            "q_max": 0,
            "min_truth_table_function_arity": 9,
            "max_truth_table_function_arity": 12,
        },
    }
    split = draw_split(scenario, seed=0)
    assert len(split.target_function.feature_indices) >= 9


def test_expand_grid_creates_environment_variants() -> None:
    scenario = {
        "name": "tiny",
        "n_train": 4,
        "n_test": 2,
        "seeds": [0],
        "environment": {"p_min": 0.1, "p_activation_min": 0.2},
        "grid": {"p_min": [0.1, 0.5], "p_activation_min": [0.2, 0.8]},
    }
    variants = expand_grid(scenario)
    assert len(variants) == 4
    assert variants[0]["environment"]["p_min"] == 0.1


def test_cli_smoke(tmp_path, monkeypatch) -> None:
    scenario = tmp_path / "scenario.json"
    scenario.write_text(
        json.dumps(
            {
                "name": "tiny",
                "n_train": 24,
                "n_test": 8,
                "seeds": [0],
                "environment": {
                    "schema_version": 4,
                    "n_features": 8,
                    "q_max": 0,
                },
                "learners": {
                    "function_builder": {"parent_top_k": 4},
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli",
            "--scenario",
            str(scenario),
            "--output-dir",
            str(output_dir),
        ],
    )
    from binaml.benchmarks.boolean_function_learning.cli import main

    main()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["failed_jobs"] == []
    first_variant = next(iter(summary["metrics"].values()))
    assert "function_builder" in first_variant["test_accuracy"]


def test_upsample_balanced_equalizes_classes() -> None:
    from binaml.benchmarks.boolean_function_learning.batches import upsample_balanced

    x = np.arange(10, dtype=np.uint8).reshape(-1, 1)
    y = np.array([True, True, True, True, True, True, True, False, False, False])
    x_bal, y_bal = upsample_balanced(x, y, np.random.default_rng(0))
    assert len(y_bal) == 14
    assert float(y_bal.mean()) == 0.5
    assert x_bal.shape == (14, 1)

    x_large = np.arange(300, dtype=np.uint8).reshape(-1, 1)
    y_large = np.array([True] * 150 + [False] * 150)
    _x_capped, y_capped = upsample_balanced(x_large, y_large, np.random.default_rng(0))
    assert len(y_capped) == 254
    assert float(y_capped.mean()) == 0.5


def test_rust_learner_is_constructible() -> None:
    assert isinstance(FunctionLearner(), FunctionLearner)
