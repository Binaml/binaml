from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from binaml.environments import (
    SyntheticClassificationStreamConfig,
    SyntheticDriftingClassificationStream,
    generate_classification_trajectory,
)
from binaml.evaluation import evaluate_prequentially_classification
from binaml.models import BClassifier, SGDLinearClassifier

_SMALL_CLAUSE = dict(min_term_degree=1, max_term_degree=3)


def test_same_seed_replays_exactly() -> None:
    config = SyntheticClassificationStreamConfig(
        n_features=4, n_functions=3, n_classes=3, q_max=2, p_sample_min_x=0.3, p_sample_max_x=0.3, **_SMALL_CLAUSE
    )
    first = generate_classification_trajectory(config, 20, seed=5, return_metadata=True)
    second = generate_classification_trajectory(config, 20, seed=5, return_metadata=True)
    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.y, second.y)
    assert first.metadata == second.metadata


def test_state_restoration_continues_exactly() -> None:
    config = SyntheticClassificationStreamConfig(
        n_features=4,
        n_functions=2,
        n_classes=3,
        q_max=2,
        p_x=0.2,
        p_g=0.3,
        p_sample_min_x=0.4,
        p_sample_max_x=0.4,
        p_sample_min_g=0.5,
        p_sample_max_g=0.5,
        p_b=0.2,
        **_SMALL_CLAUSE,
    )
    stream = SyntheticDriftingClassificationStream(config, 8)
    stream.next_sample()
    state = stream.get_state()
    expected = [stream.next_sample() for _ in range(5)]
    restored = SyntheticDriftingClassificationStream(config, 8)
    restored.set_state(state)
    actual = [restored.next_sample() for _ in range(5)]
    assert all(np.array_equal(left[0], right[0]) and left[1] == right[1] for left, right in zip(expected, actual, strict=True))


def test_labels_match_noisy_score_argmax() -> None:
    trajectory = generate_classification_trajectory(
        SyntheticClassificationStreamConfig(n_features=6, n_functions=4, n_classes=3, **_SMALL_CLAUSE),
        50,
        seed=11,
        return_metadata=True,
    )
    assert trajectory.metadata is not None
    for label, metadata in zip(trajectory.y, trajectory.metadata, strict=True):
        assert label == int(np.argmax(metadata["noisy_class_scores"]))
        assert 0 <= label < 3


def test_metadata_uses_functions_key() -> None:
    trajectory = generate_classification_trajectory(
        SyntheticClassificationStreamConfig(n_features=4, n_functions=2, n_classes=2, **_SMALL_CLAUSE),
        5,
        seed=3,
        return_metadata=True,
    )
    assert trajectory.metadata is not None
    assert "functions" in trajectory.metadata[0]
    assert "feature_indices" in trajectory.metadata[0]["functions"][0]
    assert "negated" in trajectory.metadata[0]["functions"][0]


def test_binary_scenario_labels_match_function() -> None:
    config = SyntheticClassificationStreamConfig(
        n_features=8,
        n_functions=1,
        n_classes=2,
        noise_std=0,
        p_x=0,
        p_g=0,
        p_b=0,
        p_sample_min_g=1,
        p_sample_max_g=1,
        min_term_degree=1,
        max_term_degree=3,
        weights=((2.0,), (1.0,)),
        intercepts=(0.0, 0.5),
    )
    stream = SyntheticDriftingClassificationStream(config, seed=0)
    function = stream.functions[0]
    for _ in range(50):
        x, label, metadata = stream.next_sample(return_metadata=True)
        assert label == int(np.argmax(metadata["class_scores"]))
        gate = metadata["gate_states"][0][0]
        if gate == 1:
            value = function.evaluate(x)
            expected = 0 if value == 1 else 1
            assert label == expected


def test_npz_round_trip(tmp_path: Path) -> None:
    trajectory = generate_classification_trajectory(
        SyntheticClassificationStreamConfig(n_features=5, n_functions=3, n_classes=4, **_SMALL_CLAUSE),
        10,
        seed=2,
    )
    path = tmp_path / "trajectory.npz"
    trajectory.save_npz(path)
    restored = type(trajectory).load_npz(path)
    assert np.array_equal(restored.X, trajectory.X)
    assert np.array_equal(restored.y, trajectory.y)
    assert restored.config.fingerprint == trajectory.config.fingerprint


def test_invalid_class_count_rejected() -> None:
    with pytest.raises(ValueError):
        SyntheticClassificationStreamConfig(n_features=4, n_functions=2, n_classes=1)


def test_prequential_evaluation_runs() -> None:
    trajectory = generate_classification_trajectory(
        SyntheticClassificationStreamConfig(n_features=8, n_functions=3, n_classes=3, **_SMALL_CLAUSE),
        30,
        seed=0,
    )
    result = evaluate_prequentially_classification(SGDLinearClassifier(8, 3), trajectory)
    assert result.predictions.shape == (30,)
    assert result.correct.shape == (30,)
    assert 0.0 <= result.accuracy <= 1.0


def test_b_classifier_runs_on_stream() -> None:
    trajectory = generate_classification_trajectory(
        SyntheticClassificationStreamConfig(n_features=8, n_functions=3, n_classes=3, **_SMALL_CLAUSE),
        20,
        seed=1,
    )
    result = evaluate_prequentially_classification(BClassifier(8, 3, batch_size=4), trajectory)
    assert len(result.predictions) == 20
