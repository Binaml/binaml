from importlib.metadata import version

import numpy as np
import pytest
from binaml import (
    BRegressor,
    MLPRegressor,
    SGDLinearRegressor,
    __version__,
)
from binaml.models import LastTargetClassifier, LastTargetRegressor


def test_package_version_matches_installed_distribution() -> None:
    assert __version__ == version("binaml")


def test_models_package_imports_without_instantiating_baselines() -> None:
    from binaml import models

    assert models.BRegressor is BRegressor
    assert models.SGDLinearRegressor is SGDLinearRegressor


def test_sgd_updates_from_pending_finite_features() -> None:
    model = SGDLinearRegressor(2, learning_rate=0.01, sgd_steps=1)

    with pytest.raises(ValueError, match="finite"):
        model.predict(np.array([np.nan, 0.0]))

    model.predict(np.array([0.0, 1.0]))
    model.update(1.0)
    np.testing.assert_allclose(model.weights, [0.0, 0.01], rtol=1e-5, atol=1e-6)
    assert model.n_observed == 1

    model.predict(np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="target must be finite"):
        model.update(np.nan)

    model.update(0.0)
    assert model.n_observed == 2


def test_update_requires_preceding_predict() -> None:
    model = SGDLinearRegressor(1)
    with pytest.raises(RuntimeError, match="preceding predict"):
        model.update(0.0)
    model.predict(np.array([1.0]))
    model.update(1.0)
    with pytest.raises(RuntimeError, match="preceding predict"):
        model.update(0.0)


def test_predict_requires_following_update() -> None:
    model = SGDLinearRegressor(1)
    model.predict(np.array([1.0]))
    with pytest.raises(RuntimeError, match="must follow"):
        model.predict(np.array([0.0]))


def test_sgd_can_center_binary_features_with_a_learning_rate_ema() -> None:
    model = SGDLinearRegressor(
        1,
        learning_rate=0.5,
        l2=0.0,
        center_binary_features=True,
        sgd_steps=1,
    )

    model.predict(np.array([1.0]))
    model.update(1.0)

    np.testing.assert_allclose(model.feature_probabilities, [0.75], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(model.weights, [0.25], rtol=1e-5, atol=1e-6)

    with pytest.raises(ValueError, match="binary"):
        model.predict(np.array([0.5]))


def test_sgd_replays_latest_batch_with_full_batch_gradients() -> None:
    model = SGDLinearRegressor(1, learning_rate=1.0, l2=0.0, batch_size=2, sgd_steps=1)

    model.predict(np.array([1.0]))
    model.update(1.0)
    model.predict(np.array([0.0]))
    model.update(0.0)
    model.predict(np.array([0.0]))
    model.update(0.0)

    np.testing.assert_allclose(model.weights, [0.5], rtol=1e-5, atol=1e-6)
    assert model.intercept == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(np.asarray(model._replay.features), [[0.0], [0.0]])


def test_sgd_steps_repeat_full_batch_updates() -> None:
    model = SGDLinearRegressor(1, learning_rate=0.1, l2=0.0, batch_size=1, sgd_steps=2)

    model.predict(np.array([1.0]))
    model.update(1.0)

    np.testing.assert_allclose(model.weights, [0.18], rtol=1e-5, atol=1e-6)
    assert model.intercept == pytest.approx(0.18, rel=1e-5, abs=1e-6)


def test_feature_regressor_requires_binary_predict_then_update() -> None:
    model = BRegressor(2, batch_size=2)

    with pytest.raises(ValueError, match="binary"):
        model.predict(np.array([0.5, 0.0]))
    with pytest.raises(RuntimeError, match="preceding predict"):
        model.update(0.0)

    assert model.predict(np.array([1, 0], dtype=np.uint8)) == 0.0
    model.update(1.0)
    assert model.n_observed == 1
    assert model.function_count >= 0

    model.predict(np.array([1, 0], dtype=np.uint8))
    with pytest.raises(RuntimeError, match="must follow"):
        model.predict(np.array([1, 0], dtype=np.uint8))


def test_feature_regressor_repeats_replay_batch_updates() -> None:
    model = BRegressor(1, learning_rate=0.1, l2=0.0, batch_size=1, sgd_steps=2)

    model.predict(np.array([0], dtype=np.uint8))
    model.update(1.0)

    assert model.intercept == pytest.approx(0.19)


def test_b_regressor_learns_negated_literal_via_output_inversion() -> None:
    model = BRegressor(
        1,
        batch_size=8,
        parent_top_k=2,
        max_functions=4,
        sgd_steps=1,
    )
    features = np.array([[0], [1], [0], [1], [0], [1], [0], [1]], dtype=np.uint8)
    targets = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    for row, target in zip(features, targets):
        model.predict(row)
        model.update(target)
    assert model.function_count == 1


def test_mlp_replays_the_latest_batch_for_each_sgd_step() -> None:
    model = MLPRegressor(1, batch_size=2, sgd_steps=2, random_state=0)

    model.predict(np.array([0.0]))
    model.update(0.0)
    model.predict(np.array([1.0]))
    model.update(1.0)
    model.predict(np.array([2.0]))
    model.update(2.0)

    assert model.n_observed == 3
    assert len(model._replay.features) == 2
    np.testing.assert_allclose(np.asarray(model._replay.features), [[1.0], [2.0]])


def test_mlp_rejects_unknown_kwargs() -> None:
    with pytest.raises(TypeError):
        MLPRegressor(1, momentum=0.9)


def test_last_target_regressor_starts_at_zero_then_repeats_target() -> None:
    model = LastTargetRegressor(2)

    assert model.predict(np.array([1, 0], dtype=np.uint8)) == 0.0
    model.update(1.5)
    assert model.predict(np.array([0, 1], dtype=np.uint8)) == 1.5
    model.update(-0.25)
    assert model.predict(np.array([1, 1], dtype=np.uint8)) == -0.25


def test_last_target_classifier_starts_at_zero_then_repeats_label() -> None:
    model = LastTargetClassifier(2, n_classes=3)

    assert model.predict(np.array([1, 0], dtype=np.uint8)) == 0
    model.update(2)
    assert model.predict(np.array([0, 1], dtype=np.uint8)) == 2
    model.update(1)
    assert model.predict(np.array([1, 1], dtype=np.uint8)) == 1
