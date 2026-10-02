from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from rl2.observation_decoder import ConvObservationDecoder
from rl2.observation_encoder import DEFAULT_STAGES, ConvObservationEncoder, ConvStage


@pytest.mark.parametrize("shape", [(1, 210, 160, 3), (4, 84, 84), (2, 17, 13, 3), (1, 1, 257)])
def test_default_decoder_mirrors_encoder_shapes(shape: tuple[int, ...]) -> None:
    encoder = ConvObservationEncoder(embedding_size=16)
    decoder = ConvObservationDecoder(shape)
    obs = jax.ShapeDtypeStruct((2, *shape), jnp.uint8)
    encoded, encoder_variables = jax.eval_shape(encoder.init_with_output, jax.random.key(0), obs)
    decoded, decoder_variables = jax.eval_shape(decoder.init_with_output, jax.random.key(1), encoded)
    chex.assert_shape(decoded, obs.shape)
    chex.assert_type(decoded, jnp.float32)
    input_width = encoder_variables["params"]["Dense_0"]["kernel"].shape[0]
    assert decoder_variables["params"]["projection"]["kernel"].shape == (16, input_width)
    for index, stage in enumerate(DEFAULT_STAGES):
        assert (f"stage_{index}_projection" in decoder_variables["params"]) == stage.project
        for block in range(stage.blocks):
            assert f"stage_{index}_block_{block}" in decoder_variables["params"]


@pytest.mark.parametrize("rgb", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_decoder_jit_gradients_and_frame_layout(rgb: bool, dtype: jax.typing.DTypeLike) -> None:
    shape = (2, 17, 13, 3) if rgb else (2, 17, 13)
    stages = (ConvStage(4), ConvStage(8), ConvStage(8, blocks=1, project=False))
    model = ConvObservationDecoder(shape, stages=stages, dtype=dtype)
    latent = jax.random.normal(jax.random.key(0), (2, 8)).astype(dtype)
    params = model.init(jax.random.key(1), latent)["params"]
    output, captured = jax.jit(partial(model.apply, capture_intermediates=True, mutable=["intermediates"]))(
        {"params": params}, latent
    )
    chex.assert_shape(output, (2, *shape))
    chex.assert_type(output, dtype)
    chex.assert_type(jax.tree.leaves(params), jnp.float32)
    for index, (height, width) in enumerate(((9, 7), (5, 4), (3, 2))):
        intermediate = captured["intermediates"][f"stage_{index}_block_0"]["__call__"][0]
        chex.assert_shape(intermediate, (2, height, width, stages[index].channels))
        chex.assert_type(intermediate, dtype)

    def loss(parameters: optax.Params, features: jax.Array) -> jax.Array:
        chex.assert_trees_all_equal_shapes_and_dtypes(parameters, params)
        chex.assert_shape(features, (2, 8))
        chex.assert_type(features, dtype)
        prediction = model.apply({"params": parameters}, features)
        return jnp.square(prediction.astype(jnp.float32) - 0.5).mean()

    loss_value, (gradients, latent_gradient) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(params, latent)
    assert np.isfinite(loss_value)
    chex.assert_trees_all_equal_shapes_and_dtypes(gradients, params)
    for gradient in jax.tree.leaves((gradients, latent_gradient)):
        assert np.isfinite(gradient).all()
    assert np.any(np.asarray(latent_gradient) != 0)
    assert np.any(np.asarray(gradients["projection"]["kernel"]) != 0)
    assert np.any(np.asarray(gradients["output"]["kernel"]) != 0)

    # Distinct biases verify that frames and RGB channels are unpacked in the
    # same order used by the encoder, and that predictions are not clipped.
    params["output"]["kernel"] = jnp.zeros_like(params["output"]["kernel"])
    colors = 3 if rgb else 1
    params["output"]["bias"] = jnp.arange(2 * colors, dtype=jnp.float32) - 2
    output = model.apply({"params": params}, latent).astype(jnp.float32)
    for frame in range(2):
        if rgb:
            for color in range(3):
                np.testing.assert_array_equal(output[:, frame, :, :, color], frame * 3 + color - 2)
        else:
            np.testing.assert_array_equal(output[:, frame], frame - 2)


@pytest.mark.parametrize("shape", [(4, 4), (1, 0, 4), (1, 4, 4, 2)])
def test_invalid_observation_shape(shape: tuple[int, ...]) -> None:
    decoder = ConvObservationDecoder(shape, stages=(ConvStage(4),))
    with pytest.raises((ValueError, AssertionError)):
        decoder.init(jax.random.key(0), jnp.zeros((2, 8)))


def test_invalid_latent_shape_and_dtype() -> None:
    decoder = ConvObservationDecoder((1, 4, 4), stages=(ConvStage(4),))
    for latent in (jnp.zeros((2, 3, 8)), jnp.zeros((2, 8), jnp.int32)):
        with pytest.raises(AssertionError):
            decoder.init(jax.random.key(0), latent)
