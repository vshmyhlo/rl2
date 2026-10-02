from functools import partial

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
        (210, 160, 8192, 3072, 6),
        (84, 84, 8192, 2304, 5),
        (64, 64, 8192, 4096, 4),
        (128, 64, 8192, 8192, 4),
        (84, 84, None, 9216, 4),
        (210, 160, None, 35840, 4),
        (84, 84, 256, 256, 7),
        (1, 257, 8192, 4352, 4),
    ],
)
def test_projection_budget(height: int, width: int, limit: int | None, flattened: int, stages: int) -> None:
    model = ConvObservationEncoder(max_flattened_size=limit)
    obs = jax.ShapeDtypeStruct((2, 1, height, width, 3), jnp.uint8)
    variables = jax.eval_shape(model.init, jax.random.key(0), obs)
    params = variables["params"]
    assert params["stem"]["kernel"].shape == (7, 7, 3, model.encoder_channels[0])
    assert params["Dense_0"]["kernel"].shape == (flattened, 768)
    assert len([name for name in params if name.startswith("Conv_")]) == len(model.encoder_channels)
    assert len([name for name in params if name.startswith("stage_")]) == len(model.encoder_channels) + stages
    for stage in range(len(model.encoder_channels), stages):
        assert f"stage_{stage}_block_0" in params
        assert f"stage_{stage}_block_1" not in params
    output = jax.eval_shape(model.apply, variables, obs)
    chex.assert_shape(output, (2, 768))
    chex.assert_type(output, jnp.float32)


def test_native_encoder_parameter_budget() -> None:
    model = ConvObservationEncoder(embedding_size=128)
    obs = jax.ShapeDtypeStruct((1, 1, 210, 160, 3), jnp.uint8)
    variables = jax.eval_shape(model.init, jax.random.key(0), obs)
    count = sum(parameter.size for parameter in jax.tree.leaves(variables["params"]))
    assert 6_000_000 < count < 7_000_000
    assert variables["params"]["Dense_0"]["kernel"].shape == (3072, 128)


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
    output, captured = jax.jit(partial(model.apply, capture_intermediates=True, mutable=["intermediates"]))(
        variables, obs
    )
    stem = captured["intermediates"]["stem"]["__call__"][0]
    chex.assert_shape(stem, (2, 17, 13, 4))
    chex.assert_type(stem, dtype)
    normalized_stem = captured["intermediates"]["stem_norm"]["__call__"][0]
    chex.assert_equal_shape((stem, normalized_stem))
    chex.assert_type(normalized_stem, dtype)
    assert variables["params"]["stem"]["kernel"].shape == (7, 7, 2, 4)
    chex.assert_shape(captured["intermediates"]["Conv_0"]["__call__"][0], (2, 9, 7, 4))
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
    assert np.any(np.asarray(gradients["stem"]["kernel"]) != 0)
    assert np.any(np.asarray(gradients["stage_3_block_0"]["Conv_0"]["kernel"]) != 0)
