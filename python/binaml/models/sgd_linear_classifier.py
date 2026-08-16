"""Replay-batch linear SGD multiclass baseline."""

from __future__ import annotations

import numpy as np

from .base import validate_class_index
from .jax.replay import ReplayJAXModel


class SGDLinearClassifier(ReplayJAXModel):
    """Online linear classifier with full-batch replay SGD updates."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        learning_rate: float = 0.035,
        l2: float = 0.0,
        center_binary_features: bool = False,
        batch_size: int = 10,
        sgd_steps: int = 6,
    ) -> None:
        if isinstance(n_classes, bool) or not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError("invalid classifier configuration")
        try:
            from .jax.linear import init_linear, linear_forward
            from .jax.losses import softmax_cross_entropy
        except ImportError as error:
            raise ImportError(
                "SGDLinearClassifier requires JAX; install binaml[benchmarks]"
            ) from error
        params, mask = init_linear(n_features, n_classes)
        super().__init__(
            n_features,
            n_classes,
            learning_rate,
            l2,
            center_binary_features,
            batch_size,
            sgd_steps,
            params,
            mask,
            linear_forward,
            softmax_cross_entropy,
            classification=True,
            validate_target=lambda target: validate_class_index(target, n_classes),
            invalid_config_message="invalid classifier configuration",
        )
        self.n_classes = n_classes

    @property
    def weights(self) -> np.ndarray:
        return np.asarray(self.params["weights"])

    @property
    def intercepts(self) -> np.ndarray:
        return np.asarray(self.params["bias"])
