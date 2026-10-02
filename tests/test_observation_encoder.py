from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from rl2.observation_encoder import DEFAULT_STAGES, ConvObservationEncoder, ConvStage, ConvStages, ResidualBlock


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_residual_branch_normalizes_and_applies_silu_after_final_convolution(dtype: jax.typing.DTypeLike) -> None:
    model = ResidualBlock(4, dtype=dtype)
    x = jnp.full((1, 3, 5, 4), -3, dtype=dtype)
    params = model.init(jax.random.key(0), x)["params"]
    params["Conv_1"]["kernel"] = jnp.zeros_like(params["Conv_1"]["kernel"])
    params["Conv_1"]["bias"] = jnp.zeros_like(params["Conv_1"]["bias"])
    params["LayerNorm_1"]["bias"] = jnp.asarray([2, -2, 1, -1], dtype=jnp.float32)

    output = model.apply({"params": params}, x)
    chex.assert_shape(output, x.shape)
    chex.assert_type(output, dtype)
    bias = jnp.asarray([2, -2, 1, -1], dtype=dtype)
    expected = (x + jax.nn.silu(bias)) * jnp.asarray(2**-0.5, dtype=dtype)
    np.testing.assert_allclose(output.astype(jnp.float32), expected.astype(jnp.float32))


@pytest.mark.parametrize(
    "height,width,stages,flattened",
    [
        (210, 160, DEFAULT_STAGES, 3072),
        (84, 84, DEFAULT_STAGES, 1024),
        (84, 84, DEFAULT_STAGES[:5], 2304),
        (64, 64, DEFAULT_STAGES[:4], 4096),
        (128, 64, DEFAULT_STAGES[:4], 8192),
        (84, 84, DEFAULT_STAGES[:4], 9216),
        (210, 160, DEFAULT_STAGES[:4], 35840),
        (1, 257, DEFAULT_STAGES, 1280),
    ],
)
def test_explicit_stages(height: int, width: int, stages: ConvStages, flattened: int) -> None:
    model = ConvObservationEncoder(stages=stages)
    obs = jax.ShapeDtypeStruct((2, 1, height, width, 3), jnp.uint8)
    variables = jax.eval_shape(model.init, jax.random.key(0), obs)
    params = variables["params"]
    assert params["stem"]["kernel"].shape == (7, 7, 3, stages[0].channels)
    assert params["Dense_0"]["kernel"].shape == (flattened, 768)
    assert len([name for name in params if name.startswith("stage_")]) == len(stages)
    for index, stage in enumerate(stages):
        stage_params = params[f"stage_{index}"]
        assert stage_params["resize_conv"]["conv"]["kernel"].shape[-1] == stage.channels
        assert len([name for name in stage_params if name.startswith("block_")]) == stage.blocks
    output = jax.eval_shape(model.apply, variables, obs)
    chex.assert_shape(output, (2, 768))
    chex.assert_type(output, jnp.float32)


def test_native_encoder_parameter_budget() -> None:
    model = ConvObservationEncoder(embedding_size=128)
    obs = jax.ShapeDtypeStruct((1, 1, 210, 160, 3), jnp.uint8)
    variables = jax.eval_shape(model.init, jax.random.key(0), obs)
    count = sum(parameter.size for parameter in jax.tree.leaves(variables["params"]))
    assert 7_000_000 < count < 8_000_000
    assert variables["params"]["Dense_0"]["kernel"].shape == (3072, 128)


@pytest.mark.parametrize("options", [{"channels": 0}, {"channels": 3.5}, {"channels": 4, "blocks": 0}])
def test_invalid_stage(options: dict[str, Any]) -> None:
    with pytest.raises(AssertionError):
        ConvStage(**options)


@pytest.mark.parametrize("stages", [()])
def test_invalid_stage_sequence(stages: ConvStages) -> None:
    model = ConvObservationEncoder(stages=stages)
    with pytest.raises((AssertionError, ValueError)):
        model.init(jax.random.key(0), jnp.zeros((1, 1, 8, 8), dtype=jnp.uint8))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_explicit_stages_support_jit_and_gradients(dtype: jax.typing.DTypeLike) -> None:
    stages = (ConvStage(4), ConvStage(4, 1), ConvStage(4, 1), ConvStage(4, 1))
    model = ConvObservationEncoder(stages=stages, embedding_size=8, dtype=dtype)
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
    chex.assert_shape(captured["intermediates"]["stage_0"]["resize_conv"]["__call__"][0], (2, 9, 7, 4))
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
    assert np.any(np.asarray(gradients["stage_3"]["block_0"]["Conv_0"]["kernel"]) != 0)
