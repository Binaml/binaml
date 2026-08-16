"""Replay-batch JAX MLP baseline."""

from __future__ import annotations

from collections.abc import Sequence

from .jax.replay import ReplayJAXModel


def _hidden_layer_sizes(value: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, bool):
        raise TypeError("invalid hidden_layer_sizes")
    if isinstance(value, int):
        sizes = (value,)
    else:
        sizes = tuple(value)
    if not sizes or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in sizes
    ):
        raise ValueError("invalid hidden_layer_sizes")
    return sizes


class MLPRegressor(ReplayJAXModel):
    """Online MLP regressor with full-batch replay SGD updates."""

    def __init__(
        self,
        n_features: int,
        hidden_layer_sizes: int | Sequence[int] = (50,),
        learning_rate: float = 0.003,
        alpha: float = 1e-4,
        batch_size: int = 32,
        sgd_steps: int = 3,
        random_state: int | None = 0,
    ) -> None:
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, int)
        ):
            raise ValueError("invalid random_state")
        try:
            from .jax.losses import mse
            from .jax.mlp import init_mlp, mlp_forward
        except ImportError as error:
            raise ImportError(
                "MLPRegressor requires JAX; install binaml[benchmarks]"
            ) from error
        sizes = _hidden_layer_sizes(hidden_layer_sizes)
        params, mask = init_mlp(n_features, sizes, 1, random_state)
        super().__init__(
            n_features,
            1,
            learning_rate,
            alpha,
            False,
            batch_size,
            sgd_steps,
            params,
            mask,
            mlp_forward,
            mse,
            invalid_config_message="invalid regressor configuration",
        )
        self.hidden_layer_sizes = sizes
        self.alpha = alpha
        self.random_state = random_state
