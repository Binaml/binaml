"""Replay-batch linear SGD baseline."""

from __future__ import annotations

import numpy as np

from .jax.replay import ReplayJAXModel


class SGDLinearRegressor(ReplayJAXModel):
    """Online linear regressor with full-batch replay SGD updates."""

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.01,
        l2: float = 1e-4,
        center_binary_features: bool = False,
        batch_size: int = 1,
        sgd_steps: int = 1,
    ) -> None:
        try:
            from .jax.linear import init_linear, linear_forward
            from .jax.losses import mse
        except ImportError as error:
            raise ImportError(
                "SGDLinearRegressor requires JAX; install binaml[benchmarks]"
            ) from error
        params, mask = init_linear(n_features, 1)
        super().__init__(
            n_features,
            1,
            learning_rate,
            l2,
            center_binary_features,
            batch_size,
            sgd_steps,
            params,
            mask,
            linear_forward,
            mse,
            invalid_config_message="invalid regressor configuration",
        )

    @property
    def weights(self) -> np.ndarray:
        return np.asarray(self.params["weights"]).reshape(self.n_features)

    @property
    def intercept(self) -> float:
        return float(np.asarray(self.params["bias"]).reshape(()))


def slow_sgd_linear_regressor(n_features: int) -> SGDLinearRegressor:
    """Conservative linear SGD baseline."""
    return SGDLinearRegressor(n_features, learning_rate=0.003)


def fast_sgd_linear_regressor(n_features: int) -> SGDLinearRegressor:
    """Fast-adapting linear SGD baseline."""
    return SGDLinearRegressor(n_features, learning_rate=0.03)
