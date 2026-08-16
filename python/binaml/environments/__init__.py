"""Reproducible data-generating environments."""

from .synthetic_drifting_classification import (
    ClassificationTrajectory,
    SyntheticClassificationStreamConfig,
    SyntheticDriftingClassificationStream,
    generate_classification_trajectory,
)
from .synthetic_drifting_regression import (
    BinaryFunctionSpec,
    SyntheticDriftingRegressionStream,
    SyntheticStreamConfig,
    Trajectory,
    generate_trajectory,
)

__all__ = [
    "BinaryFunctionSpec",
    "ClassificationTrajectory",
    "SyntheticClassificationStreamConfig",
    "SyntheticDriftingClassificationStream",
    "SyntheticDriftingRegressionStream",
    "SyntheticStreamConfig",
    "Trajectory",
    "generate_classification_trajectory",
    "generate_trajectory",
]
