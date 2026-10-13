from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from rl2.model.encoders.kinetix import KinetixEncoder


@pytest.mark.parametrize("shape", [(1, 1), (2, 7)], ids=["single-feature", "batched-features"])
def test_symbolic_shapes(shape: tuple[int, int]) -> None:
    model = KinetixEncoder(hidden_size=4, embedding_size=3)
    obs = jax.ShapeDtypeStruct(shape, jnp.float32)  # (B,F)
    variables = jax.eval_shape(model.init, jax.random.key(0), obs)
    output = jax.eval_shape(model.apply, variables, obs)  # (B,E)
    assert output.shape == (shape[0], 3) and output.dtype == jnp.float32


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_feature_slots_and_gradients(dtype: jax.typing.DTypeLike) -> None:
    model = KinetixEncoder(hidden_size=4, embedding_size=3, dtype=dtype)
    obs = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [10.0, -10.0, 1.0]])  # (B=4,F=3)
    variables = model.init(jax.random.key(0), obs)
    output = jax.jit(model.apply)(variables, obs)  # (B,E=3)
    assert output.shape == (4, 3) and output.dtype == dtype
    assert np.isfinite(output).all()
    assert not np.allclose(output[0].astype(jnp.float32), output[1].astype(jnp.float32))
    assert not np.allclose(output[1].astype(jnp.float32), output[2].astype(jnp.float32))
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(variables))

    def loss(params: optax.Params) -> jax.Array:
        features = model.apply({"params": params}, obs).astype(jnp.float32)  # (B,E)
        return jnp.square(features).mean()  # ()

    gradients = jax.jit(jax.grad(loss))(variables["params"])
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradients))
    assert np.any(np.asarray(gradients["dense_0"]["kernel"]) != 0)


@pytest.mark.parametrize(
    "shape,dtype",
    [((3,), jnp.float32), ((1, 3, 3), jnp.float32), ((1, 3), jnp.uint8)],
    ids=["missing-batch", "spatial-input", "integer-input"],
)
def test_invalid_observation_contract(shape: tuple[int, ...], dtype: jax.typing.DTypeLike) -> None:
    model = KinetixEncoder(hidden_size=4, embedding_size=3)
    obs = jax.ShapeDtypeStruct(shape, dtype)  # invalid (B,F) floating-point contract
    with pytest.raises(AssertionError):
        jax.eval_shape(partial(model.init, jax.random.key(0)), obs)
