import json

import numpy as np
import pytest
from binaml.benchmarks.synthetic_streaming_regression.cli import (
    _load_model_config,
    _record,
    _warmup_samples,
)
from binaml.evaluation import EvaluationTiming, PrequentialResult
from binaml.models import SGDLinearRegressor


def test_model_config_creates_models_with_configured_parameters(tmp_path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "name": "sgd-lr-1e-3",
                        "factory": "binaml.models:SGDLinearRegressor",
                        "parameters": {"learning_rate": 1e-3},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    models, config = _load_model_config(path)

    sgd = models["sgd-lr-1e-3"](3)
    assert isinstance(sgd, SGDLinearRegressor)
    assert sgd.learning_rate == 1e-3
    assert config[0]["parameters"] == {"learning_rate": 1e-3}


def test_model_config_cannot_override_feature_count(tmp_path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "name": "invalid",
                        "factory": "binaml.models:SGDLinearRegressor",
                        "parameters": {"n_features": 2},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="n_features"):
        _load_model_config(path)


def test_warmup_is_excluded_from_reported_metrics() -> None:
    result = PrequentialResult(
        predictions=np.array([0.0, 0.0, 0.0]),
        targets=np.array([0.0, 0.0, 0.0]),
        squared_errors=np.array([100.0, 4.0, 9.0]),
        timing_seconds=EvaluationTiming(0.0, 0.0, 0.0),
    )

    record = _record(0, result, warmup_samples=1)

    assert record["mse"] == 6.5
    assert record["rmse"] == pytest.approx(np.sqrt(6.5))


@pytest.mark.parametrize("warmup_samples", [-1, 10])
def test_warmup_must_be_a_non_negative_prefix(warmup_samples) -> None:
    with pytest.raises(ValueError, match="warmup_samples"):
        _warmup_samples({"n_samples": 10, "warmup_samples": warmup_samples})


@pytest.mark.parametrize("warmup_samples", [True, 1.5])
def test_warmup_must_be_an_integer(warmup_samples) -> None:
    with pytest.raises(TypeError, match="warmup_samples"):
        _warmup_samples({"n_samples": 10, "warmup_samples": warmup_samples})
