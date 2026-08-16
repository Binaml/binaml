import json
import sys

import numpy as np
import pytest
from binaml.benchmarks.synthetic_streaming_regression.cli import (
    _load_model_config,
    _record,
    _warmup_samples,
    main as regression_main,
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


def _tiny_regression_scenario(tmp_path) -> object:
    path = tmp_path / "scenario.json"
    path.write_text(
        json.dumps(
            {
                "name": "tiny",
                "schema_version": 2,
                "n_samples": 4,
                "warmup_samples": 0,
                "seeds": [0],
                "environment": {"schema_version": 2, "n_features": 3, "n_functions": 1},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cli_skips_plots_by_default(tmp_path, monkeypatch) -> None:
    scenario = _tiny_regression_scenario(tmp_path)
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
            "--model",
            "binaml.models:SGDLinearRegressor",
        ],
    )
    regression_main()
    assert not (output_dir / "plots").exists()
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["plots"] is False
    result = json.loads(next((output_dir / "results").rglob("*.json")).read_text(encoding="utf-8"))
    assert "update" in result["timing_seconds"]
    assert "observation" not in result["timing_seconds"]


def test_cli_writes_plots_when_requested(tmp_path, monkeypatch) -> None:
    pytest.importorskip("seaborn")
    scenario = _tiny_regression_scenario(tmp_path)
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
            "--model",
            "binaml.models:SGDLinearRegressor",
            "--plots",
        ],
    )
    regression_main()
    assert (output_dir / "plots").is_dir()
    assert any((output_dir / "plots").glob("*.png"))
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["plots"] is True
