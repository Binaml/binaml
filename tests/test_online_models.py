import warnings
from importlib.metadata import version

import numpy as np
import pytest
from binaml import (
    BRegressor,
    MLPRegressor,
    SGDLinearRegressor,
    __version__,
)


def test_package_version_matches_installed_distribution() -> None:
    assert __version__ == version("binaml")


def test_sgd_recomputes_prediction_from_observed_finite_features() -> None:
    model = SGDLinearRegressor(2, learning_rate=0.01, sgd_steps=1)

    with pytest.raises(ValueError, match="finite"):
        model.predict(np.array([np.nan, 0.0]))

    model.predict(np.array([1.0, 0.0]))
    model.observe(np.array([0.0, 1.0]), 1.0)
    np.testing.assert_allclose(model.weights, [0.0, 0.01])
    assert model.n_observed == 1

    model.predict(np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="target must be finite"):
        model.observe(np.array([1.0, 0.0]), np.nan)

    model.observe(np.array([1.0, 0.0]), 0.0)
    assert model.n_observed == 2


def test_sgd_can_center_binary_features_with_a_learning_rate_ema() -> None:
    model = SGDLinearRegressor(
        1,
        learning_rate=0.5,
        l2=0.0,
        center_binary_features=True,
        sgd_steps=1,
    )

    model.predict(np.array([1.0]))
    model.observe(np.array([1.0]), 1.0)

    np.testing.assert_allclose(model.feature_probabilities, [0.75])
    np.testing.assert_allclose(model.weights, [0.25])

    with pytest.raises(ValueError, match="binary"):
        model.predict(np.array([0.5]))


def test_sgd_replays_latest_batch_with_full_batch_gradients() -> None:
    model = SGDLinearRegressor(1, learning_rate=1.0, l2=0.0, batch_size=2, sgd_steps=1)

    model.predict(np.array([1.0]))
    model.observe(np.array([1.0]), 1.0)
    model.predict(np.array([0.0]))
    model.observe(np.array([0.0]), 0.0)
    model.predict(np.array([0.0]))
    model.observe(np.array([0.0]), 0.0)

    np.testing.assert_allclose(model.weights, [0.5])
    assert model.intercept == 0.0
    np.testing.assert_allclose(np.asarray(model._replay.features), [[0.0], [0.0]])


def test_sgd_steps_repeat_full_batch_updates() -> None:
    model = SGDLinearRegressor(1, learning_rate=0.1, l2=0.0, batch_size=1, sgd_steps=2)

    model.predict(np.array([1.0]))
    model.observe(np.array([1.0]), 1.0)

    np.testing.assert_allclose(model.weights, [0.18])
    assert model.intercept == pytest.approx(0.18)


def test_feature_regressor_requires_binary_predict_then_observe() -> None:
    model = BRegressor(2, batch_size=2)

    with pytest.raises(ValueError, match="binary"):
        model.predict(np.array([0.5, 0.0]))
    with pytest.raises(RuntimeError, match="preceding predict"):
        model.observe(np.array([0, 0]), 0.0)

    assert model.predict(np.array([1, 0], dtype=np.uint8)) == 0.0
    model.observe(np.array([1, 0], dtype=np.uint8), 1.0)
    assert model.n_observed == 1
    assert model.function_count >= 0

    model.predict(np.array([1, 0], dtype=np.uint8))
    with pytest.raises(RuntimeError, match="must follow"):
        model.predict(np.array([1, 0], dtype=np.uint8))


def test_feature_regressor_repeats_replay_batch_updates() -> None:
    model = BRegressor(1, learning_rate=0.1, l2=0.0, batch_size=1, sgd_steps=2)

    model.predict(np.array([0], dtype=np.uint8))
    model.observe(np.array([0], dtype=np.uint8), 1.0)

    assert model.intercept == pytest.approx(0.19)


def test_mlp_replays_the_latest_batch_for_each_sgd_step() -> None:
    pytest.importorskip("sklearn")
    model = MLPRegressor(1, batch_size=2, sgd_steps=2, random_state=0)

    assert model.predict(np.array([0.0])) == 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.observe(np.array([0.0]), 0.0)
    model.predict(np.array([1.0]))
    model.observe(np.array([1.0]), 1.0)
    model.predict(np.array([2.0]))
    model.observe(np.array([2.0]), 2.0)

    assert model.n_observed == 3
    assert len(model._replay.features) == 2
    np.testing.assert_allclose(np.asarray(model._replay.features), [[1.0], [2.0]])

    with pytest.raises(ValueError, match="early_stopping"):
        MLPRegressor(1, early_stopping=True)
