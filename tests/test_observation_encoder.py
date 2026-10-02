import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from rl2.observation_encoder import ConvObservationEncoder


@pytest.mark.parametrize(
    "height,width,limit,flattened,stages",
    [
        (210, 160, 8192, 6144, 6),
        (84, 84, 8192, 4608, 5),
        (64, 64, 8192, 8192, 4),
        (84, 84, None, 18432, 4),
        (210, 160, None, 71680, 4),
        (84, 84, 512, 512, 7),
        (1, 257, 8192, 4608, 5),
    ],
)
def test_projection_budget(height: int, width: int, limit: int | None, flattened: int, stages: int) -> None:
    model = ConvObservationEncoder(max_flattened_size=limit)
    obs = jax.ShapeDtypeStruct((2, 1, height, width, 3), jnp.uint8)
    variables = jax.eval_shape(model.init, jax.random.key(0), obs)
    params = variables["params"]
    assert params["Dense_0"]["kernel"].shape == (flattened, 768)
    assert len([name for name in params if name.startswith("Conv_")]) == stages
    output = jax.eval_shape(model.apply, variables, obs)
    chex.assert_shape(output, (2, 768))
    chex.assert_type(output, jnp.float32)


@pytest.mark.parametrize("limit", [0, -1, 3, 8.5])
def test_invalid_projection_budget(limit: float) -> None:
    model = ConvObservationEncoder(encoder_channels=(4,), max_flattened_size=limit)
    with pytest.raises(AssertionError):
        model.init(jax.random.key(0), jnp.zeros((1, 1, 8, 8), dtype=jnp.uint8))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_extra_stages_support_jit_and_gradients(dtype: jax.typing.DTypeLike) -> None:
    model = ConvObservationEncoder(encoder_channels=(4,), embedding_size=8, dtype=dtype, max_flattened_size=16)
    obs = jax.random.randint(jax.random.key(0), (2, 2, 17, 13), 0, 256, dtype=jnp.uint8)
    variables = model.init(jax.random.key(1), obs)
    output = jax.jit(model.apply)(variables, obs)
    chex.assert_shape(output, (2, 8))
    chex.assert_type(output, dtype)
    assert variables["params"]["Dense_0"]["kernel"].shape == (8, 8)

    def loss(params: optax.Params) -> jax.Array:
        chex.assert_trees_all_equal_shapes_and_dtypes(params, variables["params"])
        values = model.apply({"params": params}, obs)
        chex.assert_shape(values, (2, 8))
        chex.assert_type(values, dtype)
        return jnp.square(values.astype(jnp.float32)).mean()

    gradients = jax.jit(jax.grad(loss))(variables["params"])
    chex.assert_trees_all_equal_shapes_and_dtypes(gradients, variables["params"])
    for gradient in jax.tree.leaves(gradients):
        assert np.isfinite(gradient).all()
    assert np.any(np.asarray(gradients["stage_3_block_0"]["Conv_0"]["kernel"]) != 0)
