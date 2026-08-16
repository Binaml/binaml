import numpy as np
import pytest

pytest.importorskip("jax")


def test_mse_decreases_on_a_linear_batch() -> None:
    import jax
    import jax.numpy as jnp
    from binaml.models.jax.linear import init_linear, linear_forward
    from binaml.models.jax.losses import mse

    params, _mask = init_linear(2, 1)
    features = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    targets = jnp.asarray([1.0, -1.0], dtype=jnp.float32)
    initial = mse(linear_forward(params, features), targets)

    def loss_fn(params):
        return mse(linear_forward(params, features), targets)

    grads = jax.grad(loss_fn)(params)
    updated = {
        "weights": params["weights"] - 0.1 * grads["weights"],
        "bias": params["bias"] - 0.1 * grads["bias"],
    }
    assert mse(linear_forward(updated, features), targets) < initial


def test_softmax_cross_entropy_is_finite() -> None:
    import jax.numpy as jnp
    from binaml.models.jax.linear import init_linear, linear_forward
    from binaml.models.jax.losses import softmax_cross_entropy

    params, _mask = init_linear(2, 3)
    features = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    targets = jnp.asarray([1, 2], dtype=jnp.int32)
    loss = softmax_cross_entropy(linear_forward(params, features), targets)
    assert jnp.isfinite(loss)


def test_mlp_forward_shapes() -> None:
    import jax.numpy as jnp
    from binaml.models.jax.mlp import init_mlp, mlp_forward

    params, _mask = init_mlp(3, (4,), 2, random_state=0)
    outputs = mlp_forward(params, jnp.ones((5, 3), dtype=jnp.float32))
    assert outputs.shape == (5, 2)


def test_linear_init_is_float32_zeros() -> None:
    import jax.numpy as jnp
    from binaml.models.jax.linear import init_linear

    params, mask = init_linear(3, 1)
    assert params["weights"].dtype == jnp.float32
    assert mask == {"weights": True, "bias": False}
    np.testing.assert_array_equal(np.asarray(params["weights"]), np.zeros((3, 1)))
