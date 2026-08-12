from importlib.metadata import version

from .environments import (
    SyntheticDriftingRegressionStream,
    SyntheticStreamConfig,
    generate_trajectory,
)
from .evaluation import evaluate_prequentially

__version__ = version("binaml")
_MODEL_EXPORTS = {"BRegressor", "MLPRegressor", "OnlineModel", "SGDLinearRegressor"}


def __getattr__(name: str):
    if name in _MODEL_EXPORTS:
        from . import models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BRegressor",
    "MLPRegressor",
    "OnlineModel",
    "SGDLinearRegressor",
    "SyntheticDriftingRegressionStream",
    "SyntheticStreamConfig",
    "__version__",
    "evaluate_prequentially",
    "generate_trajectory",
]
