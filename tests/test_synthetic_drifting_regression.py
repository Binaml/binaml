from __future__ import annotations

from pathlib import Path

import numpy as np
from binaml.environments import (
    ConjunctionTerm,
    FunctionSpec,
    SyntheticDriftingRegressionStream,
    SyntheticStreamConfig,
    Trajectory,
    generate_trajectory,
)
from binaml.evaluation import evaluate_prequentially
from binaml.models import BRegressor, SGDLinearRegressor

_SMALL_CLAUSE = dict(min_term_degree=1, max_term_degree=3)


def test_same_seed_replays_exactly() -> None:
    config = SyntheticStreamConfig(n_features=4, n_functions=3, q_max=2, p_sample_min_x=0.3, p_sample_max_x=0.3, **_SMALL_CLAUSE)
    first = generate_trajectory(config, 20, seed=5, return_metadata=True)
    second = generate_trajectory(config, 20, seed=5, return_metadata=True)
    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.y, second.y)
    assert first.metadata == second.metadata


def test_state_restoration_continues_exactly() -> None:
    config = SyntheticStreamConfig(
        n_features=4,
        n_functions=2,
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
    stream = SyntheticDriftingRegressionStream(config, 8)
    stream.next_sample()
    state = stream.get_state()
    expected = [stream.next_sample() for _ in range(5)]
    restored = SyntheticDriftingRegressionStream(config, 8)
    restored.set_state(state)
    actual = [restored.next_sample() for _ in range(5)]
    assert all(np.array_equal(left[0], right[0]) and left[1] == right[1] for left, right in zip(expected, actual, strict=True))


def test_conjunction_term_positive_literals() -> None:
    term = ConjunctionTerm((0, 1), (False, False))
    assert term.evaluate(np.array([1, 1, 0], dtype=np.uint8)) == 1
    assert term.evaluate(np.array([1, 0, 0], dtype=np.uint8)) == 0


def test_conjunction_term_negated_literal() -> None:
    term = ConjunctionTerm((0, 1), (False, True))
    assert term.evaluate(np.array([1, 0, 0], dtype=np.uint8)) == 1
    assert term.evaluate(np.array([1, 1, 0], dtype=np.uint8)) == 0


def test_dags_are_acyclic_and_parent_limited() -> None:
    stream = SyntheticDriftingRegressionStream(SyntheticStreamConfig(n_features=7, n_functions=5, q_max=2, **_SMALL_CLAUSE), 13)
    for dag in (stream.input_dag, stream.gate_dag):
        positions = {node: position for position, node in enumerate(dag.order)}
        assert len(dag.order) == len(dag.parents)
        for node, parents in enumerate(dag.parents):
            assert len(parents) <= 2
            assert all(positions[parent] < positions[node] for parent in parents)


def test_distribution_sampling_retains_binary_states_and_fixed_target() -> None:
    config = SyntheticStreamConfig(
        n_features=4,
        n_functions=3,
        q_max=1,
        p_x=1,
        p_g=1,
        p_sample_min_x=0,
        p_sample_max_x=0,
        p_sample_min_g=0,
        p_sample_max_g=0,
        **_SMALL_CLAUSE,
    )
    stream = SyntheticDriftingRegressionStream(config, 7)
    input_state, gate_state = stream.input_state.copy(), stream.gate_state.copy()
    functions, weights = stream.functions.copy(), stream.weights.copy()
    _, _, metadata = stream.next_sample(return_metadata=True)
    assert np.array_equal(stream.input_state, input_state)
    assert np.array_equal(stream.gate_state, gate_state)
    assert stream.functions == functions
    assert np.array_equal(stream.weights, weights)
    assert metadata["input_distribution_sampling_indicator"]
    assert metadata["gate_distribution_sampling_indicator"]


def test_gates_never_appear_in_model_features() -> None:
    config = SyntheticStreamConfig(n_features=3, n_functions=8, **_SMALL_CLAUSE)
    stream = SyntheticDriftingRegressionStream(config, 12)
    X = np.asarray([stream.next_sample()[0] for _ in range(5)])
    assert X.shape == (5, config.n_features)


def test_prequential_protocol_learns_after_prediction() -> None:
    config = SyntheticStreamConfig(n_features=3, n_functions=1, noise_std=0, **_SMALL_CLAUSE)
    result = evaluate_prequentially(SGDLinearRegressor(3), SyntheticDriftingRegressionStream(config, 1), 10)
    assert result.predictions.shape == result.targets.shape == (10,)
    assert np.isfinite(result.mean_squared_error)
    assert result.timing_seconds.total >= result.timing_seconds.prediction >= 0
    assert result.timing_seconds.total >= result.timing_seconds.update >= 0


def test_warmup_is_excluded_from_timing() -> None:
    config = SyntheticStreamConfig(n_features=3, n_functions=1, noise_std=0, **_SMALL_CLAUSE)
    stream = SyntheticDriftingRegressionStream(config, 1)
    result = evaluate_prequentially(SGDLinearRegressor(3), stream, 8, warmup_samples=8)
    assert result.timing_seconds.total == 0.0
    assert result.timing_seconds.prediction == 0.0
    assert result.timing_seconds.update == 0.0


def test_feature_regressor_runs_on_the_binary_synthetic_stream() -> None:
    config = SyntheticStreamConfig(n_features=3, n_functions=1, noise_std=0, **_SMALL_CLAUSE)
    model = BRegressor(
        3,
        learning_rate=0.05,
        l2=0.01,
        batch_size=2,
        parent_top_k=2,
        max_functions=4,
    )
    result = evaluate_prequentially(model, SyntheticDriftingRegressionStream(config, 1), 8)

    assert result.predictions.shape == result.targets.shape == (8,)
    assert np.isfinite(result.mean_squared_error)
    assert model.n_observed == 8


def test_stored_trajectory_is_a_valid_evaluation_source(tmp_path: Path) -> None:
    config = SyntheticStreamConfig(n_features=3, n_functions=1, **_SMALL_CLAUSE)
    path = tmp_path / "trajectory.npz"
    generate_trajectory(config, 8, seed=2).save_npz(path)
    restored = Trajectory.load_npz(path)
    result = evaluate_prequentially(SGDLinearRegressor(3), restored)
    assert len(result.targets) == 8
