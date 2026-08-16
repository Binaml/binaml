"""Float32 feature centering for replay-batch JAX models."""

from __future__ import annotations

import jax.numpy as jnp


def centered(features, probabilities):
    return features - probabilities


def ema_update(probabilities, features, learning_rate):
    return probabilities + learning_rate * (jnp.mean(features, axis=0) - probabilities)
