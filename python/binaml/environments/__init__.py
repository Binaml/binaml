"""Reproducible data-generating environments."""

from .boolean_dgp import (
    BinaryFunctionSpec,
    BooleanBatch,
    BooleanDgpConfig,
    ConditionalDag,
    generate_boolean_batch,
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
    "BinaryFunctionSpec",
    "BooleanBatch",
    "BooleanDgpConfig",
    "ClassificationTrajectory",
    "ConditionalDag",
    "SyntheticClassificationStreamConfig",
    "SyntheticDriftingClassificationStream",
    "SyntheticDriftingRegressionStream",
    "SyntheticStreamConfig",
    "Trajectory",
    "generate_boolean_batch",
    "generate_classification_trajectory",
    "generate_trajectory",
]
