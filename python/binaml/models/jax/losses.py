"""Losses for replay-batch JAX baselines."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def mse(outputs, targets):
    """Half-mean squared error so SGD matches residual linear updates."""
    return 0.5 * jnp.mean((outputs.reshape(-1) - targets.reshape(-1)) ** 2)


def softmax_cross_entropy(logits, targets):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(targets, logits.shape[-1], dtype=log_probs.dtype)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))
