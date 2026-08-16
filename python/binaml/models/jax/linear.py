"""Zero-initialized linear head for replay-batch JAX baselines."""

from __future__ import annotations

import jax.numpy as jnp


def init_linear(n_features: int, n_outputs: int) -> tuple[dict, dict]:
    params = {
        "weights": jnp.zeros((n_features, n_outputs), dtype=jnp.float32),
        "bias": jnp.zeros((n_outputs,), dtype=jnp.float32),
    }
    mask = {"weights": True, "bias": False}
    return params, mask


def linear_forward(params, features):
    return features @ params["weights"] + params["bias"]
