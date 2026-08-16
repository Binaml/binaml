"""Replay-batch JAX MLP multiclass baseline."""

from __future__ import annotations

from collections.abc import Sequence

from .base import validate_class_index
from .jax.replay import ReplayJAXModel
from .mlp import _hidden_layer_sizes


class MLPClassifier(ReplayJAXModel):
    """Online MLP classifier with full-batch replay SGD updates."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_layer_sizes: int | Sequence[int] = (75,),
        learning_rate: float = 0.012,
        alpha: float = 0.0,
        batch_size: int = 12,
        sgd_steps: int = 6,
        random_state: int | None = 0,
    ) -> None:
        if isinstance(n_classes, bool) or not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError("invalid classifier configuration")
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, int)
        ):
            raise ValueError("invalid random_state")
        try:
            from .jax.losses import softmax_cross_entropy
            from .jax.mlp import init_mlp, mlp_forward
        except ImportError as error:
            raise ImportError(
                "MLPClassifier requires JAX; install binaml[benchmarks]"
            ) from error
        sizes = _hidden_layer_sizes(hidden_layer_sizes)
        params, mask = init_mlp(n_features, sizes, n_classes, random_state)
        super().__init__(
            n_features,
            n_classes,
            learning_rate,
            alpha,
            False,
            batch_size,
            sgd_steps,
            params,
            mask,
            mlp_forward,
            softmax_cross_entropy,
            classification=True,
            validate_target=lambda target: validate_class_index(target, n_classes),
            invalid_config_message="invalid classifier configuration",
        )
        self.n_classes = n_classes
        self.hidden_layer_sizes = sizes
        self.alpha = alpha
        self.random_state = random_state
