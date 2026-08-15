"""Replay-batch scikit-learn MLP baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import PredictObserveState, ReplayBatch, validate_class_index, validate_finite_float_target, validate_float_features


class SklearnMLPOnlineModel(PredictObserveState):
    """Online wrapper around scikit-learn's SGD MLP regressor or classifier."""

    def __init__(
        self,
        n_features: int,
        batch_size: int,
        sgd_steps: int,
        *,
        n_classes: int | None = None,
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
        super().__init__()
        if isinstance(n_features, bool) or not isinstance(n_features, int) or n_features < 1:
            raise ValueError("n_features must be a positive integer")
        if n_classes is not None and (isinstance(n_classes, bool) or not isinstance(n_classes, int) or n_classes < 2):
            raise ValueError("n_classes must be an integer greater than 1")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(sgd_steps, bool) or not isinstance(sgd_steps, int) or sgd_steps < 1:
            raise ValueError("sgd_steps must be a positive integer")
        if early_stopping:
            raise ValueError("early_stopping is not supported with replay partial_fit updates")

        if n_classes is None:
            try:
                from sklearn.neural_network import MLPRegressor as SklearnMLP
            except ImportError as error:
                raise ImportError("MLPRegressor requires scikit-learn; install binaml[benchmarks]") from error
        else:
            try:
                from sklearn.neural_network import MLPClassifier as SklearnMLP
            except ImportError as error:
                raise ImportError("MLPClassifier requires scikit-learn; install binaml[benchmarks]") from error

        self.n_features = n_features
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.sgd_steps = sgd_steps
        self._classes = None if n_classes is None else np.arange(n_classes, dtype=np.int64)
        model_kwargs: dict[str, Any] = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": activation,
            "solver": "sgd",
            "alpha": alpha,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "learning_rate_init": learning_rate_init,
            "power_t": power_t,
            "max_iter": 1,
            "shuffle": shuffle,
            "random_state": random_state,
            "tol": tol,
            "verbose": verbose,
            "warm_start": False,
            "momentum": momentum,
            "nesterovs_momentum": nesterovs_momentum,
            "validation_fraction": validation_fraction,
            "beta_1": beta_1,
            "beta_2": beta_2,
            "epsilon": epsilon,
            "n_iter_no_change": n_iter_no_change,
        }
        self._model = SklearnMLP(**model_kwargs)
        self._replay = ReplayBatch(batch_size)
        self.n_observed = 0

    def predict(self, features: np.ndarray):
        values = validate_float_features(features, self.n_features)
        self._begin_predict()
        if self.n_observed == 0:
            return 0.0 if self.n_classes is None else 0
        prediction = self._model.predict(values[None, :])[0]
        return float(prediction) if self.n_classes is None else int(prediction)

    def observe(self, features: np.ndarray, target) -> None:
        self._begin_observe()
        values = validate_float_features(features, self.n_features)
        if self.n_classes is None:
            target_value = validate_finite_float_target(target)
        else:
            target_value = validate_class_index(target, self.n_classes)
        self._finish_observe()
        self._replay.append(values, target_value)
        replay_features, replay_targets = self._replay.arrays()
        self._model.batch_size = len(replay_targets)
        for _ in range(self.sgd_steps):
            if self._classes is None:
                self._model.partial_fit(replay_features, replay_targets)
            else:
                self._model.partial_fit(replay_features, replay_targets, classes=self._classes)
        self.n_observed += 1
