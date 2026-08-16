"""Replay-window SGD scaffold shared by JAX linear and MLP baselines."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..base import (
    PredictUpdateState,
    ReplayBatch,
    validate_finite_float_target,
    validate_float_features,
)


class ReplayJAXModel(PredictUpdateState):
    """Full-batch replay SGD with a JAX parameter tree and optax optimizer."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        learning_rate: float,
        l2: float,
        center_binary_features: bool,
        batch_size: int,
        sgd_steps: int,
        params,
        mask,
        forward: Callable,
        loss: Callable,
        *,
        classification: bool = False,
        validate_target: Callable | None = None,
        invalid_config_message: str = "invalid model configuration",
    ) -> None:
        super().__init__()
        try:
            import jax
            import optax
        except ImportError as error:
            raise ImportError(
                "JAX baselines require JAX and Optax; install binaml[benchmarks]"
            ) from error

        jax.config.update("jax_enable_x64", False)
        if (
            isinstance(n_features, bool)
            or not isinstance(n_features, int)
            or n_features < 1
            or isinstance(n_outputs, bool)
            or not isinstance(n_outputs, int)
            or n_outputs < 1
            or learning_rate <= 0
            or l2 < 0
            or isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
            or isinstance(sgd_steps, bool)
            or not isinstance(sgd_steps, int)
            or sgd_steps < 1
        ):
            raise ValueError(invalid_config_message)
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.center_binary_features = center_binary_features
        self.batch_size = batch_size
        self.sgd_steps = sgd_steps
        self.classification = classification
        self._validate_target_fn = validate_target or validate_finite_float_target
        self._forward = forward
        self._loss = loss
        self.params = params
        self.feature_probabilities = np.full(n_features, 0.5, dtype=np.float32)
        self._replay = ReplayBatch(batch_size)
        self._pending_features: np.ndarray | None = None
        self.n_observed = 0
        optimizer = optax.chain(optax.add_decayed_weights(l2, mask=mask), optax.sgd(learning_rate))
        self._opt_state = optimizer.init(params)
        self._train = _make_train_fn(
            jax,
            optax,
            optimizer,
            forward,
            loss,
            sgd_steps,
            center_binary_features,
            learning_rate,
        )

    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        values = validate_float_features(features, self.n_features)
        if self.center_binary_features and not np.all((values == 0.0) | (values == 1.0)):
            raise ValueError("centered features must be binary")
        return values

    def _model_features(self, values: np.ndarray) -> np.ndarray:
        return values - self.feature_probabilities if self.center_binary_features else values

    def _predict_from_features(self, values: np.ndarray):
        import jax.numpy as jnp

        features = jnp.asarray(self._model_features(values), dtype=jnp.float32)
        outputs = self._forward(self.params, features)
        if self.classification:
            return int(jnp.argmax(outputs))
        return float(np.asarray(outputs).reshape(()))

    def predict(self, features: np.ndarray):
        values = self._validate_features(features)
        self._begin_predict()
        self._pending_features = values
        return self._predict_from_features(values)

    def update(self, target) -> None:
        import jax.numpy as jnp

        self._begin_update()
        target_value = self._validate_target_fn(target)
        values = self._pending_features
        if values is None:
            raise RuntimeError("update(target) requires a preceding predict(features)")
        self._pending_features = None
        self._finish_update()
        self._replay.append(values, target_value)
        features, targets = self._replay.arrays()
        target_dtype = jnp.int32 if self.classification else jnp.float32
        self.params, self._opt_state, probabilities = self._train(
            self.params,
            self._opt_state,
            jnp.asarray(self.feature_probabilities, dtype=jnp.float32),
            jnp.asarray(features, dtype=jnp.float32),
            jnp.asarray(targets, dtype=target_dtype),
        )
        self.feature_probabilities = np.asarray(probabilities, dtype=np.float32)
        self.n_observed += 1


def _make_train_fn(jax, optax, optimizer, forward, loss, sgd_steps, center, learning_rate):
    from .features import centered, ema_update

    def train(params, opt_state, probabilities, features, targets):
        def step(state, _):
            params, opt_state, probabilities = state

            def loss_for_params(params):
                model_features = centered(features, probabilities) if center else features
                return loss(forward(params, model_features), targets)

            _value, grads = jax.value_and_grad(loss_for_params)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            if center:
                probabilities = ema_update(probabilities, features, learning_rate)
            return (params, opt_state, probabilities), None

        (params, opt_state, probabilities), _ = jax.lax.scan(
            step, (params, opt_state, probabilities), None, length=sgd_steps
        )
        return params, opt_state, probabilities

    return jax.jit(train)
