import numpy as np
from binaml._core import FunctionLearner


def _build_function_builder(**parameters):
    return FunctionLearner(**parameters)


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
    learner = _build_function_builder(parent_top_k=2)
    result = learner.fit_predict(x, y)
    predictions = result["predictions"]
    score = result["score"]
    elapsed = result["elapsed_seconds"]
    assert predictions.shape == (4,)
    assert score >= 3
    assert elapsed >= 0.0


def test_function_builder_learns_negated_literal_target() -> None:
    x = np.array([[0], [1], [0], [1], [0], [1], [0], [1]], dtype=np.uint8)
    y = np.array([True, False, True, False, True, False, True, False])
    learner = _build_function_builder(parent_top_k=2)
    result = learner.fit_predict(x, y)
    assert np.array_equal(result["predictions"], y)
    assert result["score"] == 8
    assert result["elapsed_seconds"] >= 0.0


def test_function_builder_predicts_on_holdout_after_fit() -> None:
    x_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    y_train = np.array([False, True, True, True])
    x_test = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    learner = _build_function_builder(parent_top_k=2)
    learner.fit(x_train, y_train)
    predictions = learner.predict(x_test)
    assert np.array_equal(predictions, np.array([True, False]))


def test_function_builder_learns_xor_target() -> None:
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    y = np.array([False, True, True, False])
    learner = _build_function_builder(parent_top_k=2)
    result = learner.fit_predict(x, y)
    assert np.array_equal(result["predictions"], y)
    assert result["score"] == 4


def test_rust_learner_is_constructible() -> None:
    assert isinstance(FunctionLearner(), FunctionLearner)
