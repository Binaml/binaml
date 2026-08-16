import json

import numpy as np
import pytest
from binaml.benchmarks.synthetic_streaming_classification.cli import (
    _load_model_config,
    _record,
)
from binaml.evaluation import EvaluationTiming, PrequentialClassificationResult
from binaml.models import BClassifier, MLPClassifier, SGDLinearClassifier


def test_model_config_creates_models_with_configured_parameters(tmp_path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "name": "sgd-lr-1e-3",
                        "factory": "binaml.models:SGDLinearClassifier",
                        "parameters": {"learning_rate": 1e-3},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    models, config = _load_model_config(path)

    classifier = models["sgd-lr-1e-3"](3, 4)
    assert isinstance(classifier, SGDLinearClassifier)
    assert classifier.learning_rate == 1e-3
    assert config[0]["parameters"] == {"learning_rate": 1e-3}


def test_model_config_cannot_override_dimensions(tmp_path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "name": "invalid-features",
                        "factory": "binaml.models:SGDLinearClassifier",
                        "parameters": {"n_features": 2},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="n_features"):
        _load_model_config(path)

    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "name": "invalid-classes",
                        "factory": "binaml.models:SGDLinearClassifier",
                        "parameters": {"n_classes": 2},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="n_classes"):
        _load_model_config(path)


def test_warmup_is_excluded_from_reported_metrics() -> None:
    result = PrequentialClassificationResult(
        predictions=np.array([0, 1, 2]),
        targets=np.array([0, 0, 2]),
        correct=np.array([True, False, True]),
        timing_seconds=EvaluationTiming(0.0, 0.0, 0.0),
    )

    record = _record(0, result, warmup_samples=1)

    assert record["accuracy"] == 0.5


def test_sgd_classifier_predict_then_update() -> None:
    model = SGDLinearClassifier(2, 3, learning_rate=0.01, sgd_steps=1)
    assert model.predict(np.array([1.0, 0.0])) in {0, 1, 2}
    model.update(1)
    assert model.n_observed == 1


def test_b_classifier_requires_binary_features() -> None:
    model = BClassifier(2, 3, batch_size=2)
    with pytest.raises(ValueError, match="binary"):
        model.predict(np.array([0.5, 0.0]))
    model.predict(np.array([1, 0], dtype=np.uint8))
    model.update(1)


def test_mlp_classifier_runs_on_a_short_stream() -> None:
    model = MLPClassifier(2, 3, batch_size=2, sgd_steps=1, random_state=0)
    assert model.predict(np.array([0.0, 1.0])) in {0, 1, 2}
    model.update(1)
    model.predict(np.array([1.0, 0.0]))
    model.update(2)
    assert model.n_observed == 2


def test_mlp_classifier_rejects_unknown_kwargs() -> None:
    with pytest.raises(TypeError):
        MLPClassifier(2, 3, momentum=0.9)
