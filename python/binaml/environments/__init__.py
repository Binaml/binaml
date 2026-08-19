"""Reproducible data-generating environments."""

from .boolean_dgp import (
    BooleanDgpConfig,
    ConditionalDag,
    ConjunctionTerm,
    FunctionSpec,
)
from .synthetic_drifting_classification import (
    ClassificationTrajectory,
    SyntheticClassificationStreamConfig,
    SyntheticDriftingClassificationStream,
    generate_classification_trajectory,
)
from .synthetic_drifting_regression import (
    SyntheticDriftingRegressionStream,
    SyntheticStreamConfig,
    Trajectory,
    generate_trajectory,
)

__all__ = [
    "BooleanDgpConfig",
    "ClassificationTrajectory",
    "ConditionalDag",
    "ConjunctionTerm",
    "FunctionSpec",
    "SyntheticClassificationStreamConfig",
    "SyntheticDriftingClassificationStream",
    "SyntheticDriftingRegressionStream",
    "SyntheticStreamConfig",
    "Trajectory",
    "generate_classification_trajectory",
    "generate_trajectory",
]
