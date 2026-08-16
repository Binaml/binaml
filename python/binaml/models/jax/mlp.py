"""One-or-more hidden-layer ReLU MLP for replay-batch JAX baselines."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def init_mlp(
    n_features: int,
    hidden_layer_sizes: tuple[int, ...],
    n_outputs: int,
    random_state: int | None,
) -> tuple[dict, dict]:
    key = jax.random.key(0 if random_state is None else random_state)
    keys = jax.random.split(key, len(hidden_layer_sizes) + 1)
    he_uniform = jax.nn.initializers.he_uniform()
    glorot_uniform = jax.nn.initializers.glorot_uniform()
    hidden = []
    hidden_mask = []
    n_in = n_features
    for index, n_hidden in enumerate(hidden_layer_sizes):
        hidden.append(
            {
                "weights": he_uniform(keys[index], (n_in, n_hidden), jnp.float32),
                "bias": jnp.zeros((n_hidden,), dtype=jnp.float32),
            }
        )
        hidden_mask.append({"weights": True, "bias": False})
        n_in = n_hidden
    params = {
        "hidden": hidden,
        "output": {
            "weights": glorot_uniform(keys[-1], (n_in, n_outputs), jnp.float32),
            "bias": jnp.zeros((n_outputs,), dtype=jnp.float32),
        },
    }
    mask = {
        "hidden": hidden_mask,
        "output": {"weights": True, "bias": False},
    }
    return params, mask


def mlp_forward(params, features):
    hidden = features
    for layer in params["hidden"]:
        hidden = jax.nn.relu(hidden @ layer["weights"] + layer["bias"])
    return hidden @ params["output"]["weights"] + params["output"]["bias"]
