"""Replay-batch scikit-learn MLP baseline."""

from __future__ import annotations

import numpy as np

from .base import ReplayBatch


class MLPRegressor:
    """Online wrapper around scikit-learn's SGD MLP regressor."""

    def __init__(
        self,
        n_features: int,
        batch_size: int = 32,
        sgd_steps: int = 3,
        hidden_layer_sizes: int | tuple[int, ...] = (50,),
        activation: str = "relu",
        alpha: float = 0.0001,
        learning_rate: str = "constant",
        learning_rate_init: float = 0.003,
        power_t: float = 0.5,
        shuffle: bool = True,
        random_state: int | None = 0,
        tol: float = 0.0001,
        verbose: bool = False,
        momentum: float = 0.0,
        nesterovs_momentum: bool = False,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        n_iter_no_change: int = 10,
    ) -> None:
        if isinstance(n_features, bool) or not isinstance(n_features, int) or n_features < 1:
            raise ValueError("n_features must be a positive integer")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(sgd_steps, bool) or not isinstance(sgd_steps, int) or sgd_steps < 1:
            raise ValueError("sgd_steps must be a positive integer")
        if early_stopping:
            raise ValueError("early_stopping is not supported with replay partial_fit updates")

        try:
            from sklearn.neural_network import MLPRegressor as SklearnMLPRegressor
        except ImportError as error:
            raise ImportError(
                "MLPRegressor requires scikit-learn; install binaml[benchmarks]"
            ) from error

        self.n_features, self.batch_size, self.sgd_steps = n_features, batch_size, sgd_steps
        self._model = SklearnMLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver="sgd",
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=1,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=False,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
        )
        self._replay = ReplayBatch(batch_size)
        self.n_observed = 0
        self._prediction_pending = False

    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.shape != (self.n_features,):
            raise ValueError(f"features must have shape ({self.n_features},)")
        if not np.all(np.isfinite(values)):
            raise ValueError("features must be finite")
        return values

    def predict(self, features: np.ndarray) -> float:
        if self._prediction_pending:
            raise RuntimeError("observe(features, target) must follow each predict(features)")
        values = self._validate_features(features)
        prediction = 0.0 if self.n_observed == 0 else float(self._model.predict(values[None, :])[0])
        self._prediction_pending = True
        return prediction

    def observe(self, features: np.ndarray, target: float) -> None:
        if not self._prediction_pending:
            raise RuntimeError("observe(features, target) requires a preceding predict(features)")
        values = self._validate_features(features)
        target_value = float(target)
        if not np.isfinite(target_value):
            raise ValueError("target must be finite")
        self._replay.append(values, target_value)
        replay_features, replay_targets = self._replay.arrays()
        self._model.batch_size = len(replay_targets)
        for _ in range(self.sgd_steps):
            self._model.partial_fit(replay_features, replay_targets)
        self.n_observed += 1
        self._prediction_pending = False
