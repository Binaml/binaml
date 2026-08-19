import numpy as np
from binaml._core import ConjunctionLearner


def _build_conjunction_learner(**parameters):
    return ConjunctionLearner(**parameters)


def test_conjunction_builder_rust_learner_on_synthetic_batch() -> None:
    x = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ],
        dtype=np.uint8,
    )
    y = np.array([False, False, False, True])
    learner = _build_conjunction_learner(max_conjunctions=4)
    result = learner.fit_predict(x, y)
    predictions = result["predictions"]
    score = result["score"]
    elapsed = result["elapsed_seconds"]
    assert predictions.shape == (4,)
    assert score == 4
    assert elapsed >= 0.0


def test_conjunction_builder_learns_negated_literal_target() -> None:
    x = np.array([[0], [1], [0], [1], [0], [1], [0], [1]], dtype=np.uint8)
    y = np.array([True, False, True, False, True, False, True, False])
    learner = _build_conjunction_learner(max_conjunctions=2)
    result = learner.fit_predict(x, y)
    assert np.array_equal(result["predictions"], y)
    assert result["score"] == 8
    assert result["elapsed_seconds"] >= 0.0


def test_conjunction_builder_predicts_on_holdout_after_fit() -> None:
    x_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    y_train = np.array([False, False, False, True])
    x_test = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    learner = _build_conjunction_learner(max_conjunctions=4)
    learner.fit(x_train, y_train)
    predictions = learner.predict(x_test)
    assert np.array_equal(predictions, np.array([True, False]))


def test_conjunction_builder_learns_two_literal_conjunction() -> None:
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    y = np.array([False, False, False, True])
    learner = _build_conjunction_learner(max_conjunctions=4)
    result = learner.fit_predict(x, y)
    assert np.array_equal(result["predictions"], y)
    assert result["score"] == 4


def test_rust_learner_is_constructible() -> None:
    assert isinstance(ConjunctionLearner(), ConjunctionLearner)
