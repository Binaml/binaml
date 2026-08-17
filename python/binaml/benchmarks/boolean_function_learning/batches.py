"""Draw static learner batches from the input DGP and function sampler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from binaml.environments.boolean_dgp import (
    BinaryFunctionSpec,
    BooleanDgpConfig,
    generate_boolean_batch,
)

MAX_BATCH_SIZE = 255


@dataclass(frozen=True)
class LearnerSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    target_function: BinaryFunctionSpec
    p_feature_empirical_train: np.ndarray
    p_feature_empirical_test: np.ndarray
    p_target_empirical_train: float
    p_target_empirical_test: float


def draw_split(scenario: dict[str, object], seed: int) -> LearnerSplit:
    config = BooleanDgpConfig.from_dict(scenario["environment"])  # type: ignore[arg-type]
    n_train = int(scenario["n_train"])
    n_test = int(scenario["n_test"])
    if n_train < 1 or n_test < 1:
        raise ValueError("n_train and n_test must be positive")

    batch = generate_boolean_batch(config, n_train + n_test, seed)
    y = np.array([bool(batch.target_function.evaluate(row)) for row in batch.X], dtype=bool)

    p_noise = float(scenario.get("p_noise", 0.0))
    if p_noise > 0.0:
        rng = np.random.default_rng(seed)
        flip = rng.random(n_train + n_test) < p_noise
        y = np.logical_xor(y, flip)

    x_train, x_test = batch.X[:n_train], batch.X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    return LearnerSplit(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        target_function=batch.target_function,
        p_feature_empirical_train=x_train.mean(axis=0),
        p_feature_empirical_test=x_test.mean(axis=0),
        p_target_empirical_train=float(y_train.mean()),
        p_target_empirical_test=float(y_test.mean()),
    )


def upsample_balanced(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Upsample the minority class with replacement until both classes are equal."""
    positive = np.flatnonzero(y)
    negative = np.flatnonzero(~y)
    if len(positive) == 0 or len(negative) == 0:
        return x, y
    if len(positive) < len(negative):
        minority, majority = positive, negative
    else:
        minority, majority = negative, positive
    upsampled_minority = rng.choice(minority, size=len(majority), replace=True)
    indices = np.concatenate([majority, upsampled_minority])
    rng.shuffle(indices)
    x_bal, y_bal = x[indices], y[indices]
    if len(y_bal) <= MAX_BATCH_SIZE:
        return x_bal, y_bal
    per_class = MAX_BATCH_SIZE // 2
    positive = np.flatnonzero(y_bal)
    negative = np.flatnonzero(~y_bal)
    positive = rng.choice(positive, size=per_class, replace=False)
    negative = rng.choice(negative, size=per_class, replace=False)
    indices = np.concatenate([positive, negative])
    rng.shuffle(indices)
    return x_bal[indices], y_bal[indices]
