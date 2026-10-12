from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from xminigrid.core.constants import NUM_COLORS, NUM_TILES

from rl2.model.encoders.xminigrid import XMiniGridEncoder


@pytest.mark.parametrize(
    "shape,blocks",
    [((2, 3, 3, 2), 1), ((1, 3, 5, 2), 1), ((1, 1, 1, 2), 0)],
    ids=["batched", "rectangular", "single-cell-without-residual-blocks"],
)
def test_grid_and_batch_shapes(shape: tuple[int, int, int, int], blocks: int) -> None:
    model = XMiniGridEncoder(token_size=2, channels=4, blocks=blocks, embedding_size=8)
    obs = jax.ShapeDtypeStruct(shape, jnp.uint8)  # (B, H, W, 2)
    variables = jax.eval_shape(model.init, jax.random.key(0), obs)  # parameter pytree
    output = jax.eval_shape(model.apply, variables, obs)  # (B, E)
    assert output.shape == (shape[0], 8)
    assert output.dtype == jnp.float32


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_categories_positions_and_gradients(dtype: jax.typing.DTypeLike) -> None:
    model = XMiniGridEncoder(token_size=2, channels=4, blocks=1, embedding_size=8, dtype=dtype)
    obs = jnp.zeros((4, 3, 3, 2), jnp.uint8)  # (B, H, W, 2)
    obs = obs.at[1, 0, 0, 0].set(NUM_TILES - 1)  # (B, H, W, 2): tile changes
    obs = obs.at[2, 0, 0, 1].set(NUM_COLORS - 1)  # (B, H, W, 2): color changes
    obs = obs.at[3, 2, 2, 0].set(NUM_TILES - 1)  # (B, H, W, 2): tile position changes
    variables = model.init(jax.random.key(0), obs)  # parameter pytree
    output = jax.jit(model.apply)(variables, obs)  # (B, E)
    assert output.shape == (4, 8) and output.dtype == dtype
    assert np.isfinite(output).all()
    for left, right in ((0, 1), (0, 2), (1, 3)):
        assert not np.allclose(output[left].astype(jnp.float32), output[right].astype(jnp.float32))
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(variables))

    def loss(params: optax.Params) -> jax.Array:  # parameter pytree -> ()
        features = model.apply({"params": params}, obs).astype(jnp.float32)  # (B, E)
        return jnp.square(features).mean()  # ()

    gradients = jax.jit(jax.grad(loss))(variables["params"])  # parameter pytree
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradients))
    for name in ("tile_embedding", "color_embedding"):
        assert np.any(np.asarray(gradients[name]["embedding"]) != 0)


@pytest.mark.parametrize(
    "shape,dtype",
    [((1, 3, 3, 3), jnp.uint8), ((3, 3, 2), jnp.uint8), ((1, 3, 3, 2), jnp.float32)],
    ids=["wrong-channels", "missing-batch", "noninteger-ids"],
)
def test_invalid_observation_contract(shape: tuple[int, ...], dtype: jax.typing.DTypeLike) -> None:
    model = XMiniGridEncoder(token_size=2, channels=4, blocks=0, embedding_size=8)
    obs = jax.ShapeDtypeStruct(shape, dtype)  # invalid (B, H, W, 2) contract
    with pytest.raises(AssertionError):
        jax.eval_shape(partial(model.init, jax.random.key(0)), obs)
