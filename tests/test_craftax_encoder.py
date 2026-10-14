import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.envs.observations import CraftaxObservation
from rl2.model.encoders.craftax import CraftaxEncoder


@pytest.mark.parametrize(
    "grid_shape,feature_count,batch_size,dtype",
    [((9, 11, 83), 51, 1, jnp.float32), ((7, 9, 21), 22, 2, jnp.bfloat16)],
    ids=["full", "classic-batched-bfloat16"],
)
def test_shapes_and_both_branches(
    grid_shape: tuple[int, int, int], feature_count: int, batch_size: int, dtype: jax.typing.DTypeLike
) -> None:
    obs = CraftaxObservation(
        grid=jax.random.uniform(jax.random.key(0), (batch_size, *grid_shape)),  # (B,H,W,C)
        features=jax.random.uniform(jax.random.key(1), (batch_size, feature_count)),  # (B,F)
    )
    encoder = CraftaxEncoder(channels=4, blocks=1, hidden_size=4, embedding_size=4, dtype=dtype)
    params = encoder.init(jax.random.key(2), obs)
    apply = jax.jit(encoder.apply)
    result = apply(params, obs)  # (B,E=4)
    assert result.shape == (batch_size, 4) and result.dtype == dtype
    without_grid = apply(params, obs._replace(grid=jnp.zeros_like(obs.grid)))  # (B,E)
    without_features = apply(params, obs._replace(features=jnp.zeros_like(obs.features)))  # (B,E)
    assert not np.array_equal(result, without_grid)
    assert not np.array_equal(result, without_features)

    def loss(inputs: CraftaxObservation) -> jax.Array:  # grid: (B,H,W,C), features: (B,F)
        return jnp.square(encoder.apply(params, inputs).astype(jnp.float32)).sum()  # ()

    gradients = jax.jit(jax.grad(loss))(obs)  # grid: (B,H,W,C), features: (B,F)
    for original, gradient in zip(jax.tree.leaves(obs), jax.tree.leaves(gradients)):
        assert gradient.shape == original.shape
        assert bool(jnp.any(gradient != 0))
