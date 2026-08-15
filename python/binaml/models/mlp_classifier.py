"""Replay-batch scikit-learn MLP multiclass baseline."""

from __future__ import annotations

from .sklearn_mlp import SklearnMLPOnlineModel


class MLPClassifier(SklearnMLPOnlineModel):
    """Online wrapper around scikit-learn's SGD MLP classifier."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        batch_size: int = 12,
        sgd_steps: int = 6,
        hidden_layer_sizes: int | tuple[int, ...] = (75,),
        activation: str = "relu",
        alpha: float = 0.0,
        learning_rate: str = "constant",
        learning_rate_init: float = 0.012,
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
        super().__init__(
            n_features,
            batch_size,
            sgd_steps,
            n_classes=n_classes,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
        )
