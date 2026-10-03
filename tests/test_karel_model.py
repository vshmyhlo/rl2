from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.grpo import generation_logits
from rl2.karel import TOKENS, KarelConfig, KarelProgramEnv, sample_task
from rl2.karel_model import KarelProgramModel
from rl2.transformer import AttentionImplementation


@pytest.fixture(scope="module")
def inputs() -> tuple[jax.Array, jax.Array, jax.Array]:
    rng = np.random.default_rng(42)
    tasks = [sample_task(rng, KarelConfig(height=4, width=4)) for _ in range(2)]
    initial = jnp.asarray(np.stack([task.initial for task in tasks]))
    target = jnp.asarray(np.stack([task.target for task in tasks]))
    tokens = jnp.asarray([[1, 1], [2, 2], [3, 3]], dtype=jnp.int32)
    return initial, target, tokens


@pytest.fixture(scope="module", params=["mamba3", "transformer"])
def model_and_variables(
    inputs: tuple[jax.Array, jax.Array, jax.Array], request: pytest.FixtureRequest
) -> tuple[KarelProgramModel, dict[str, Any]]:
    model = KarelProgramModel(
        d_model=8,
        num_layers=1,
        d_state=8,
        headdim=4,
        backbone_type=request.param,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=inputs[2].shape[0] + 1,
    )
    variables = model.init(jax.random.key(0), *inputs)
    return model, variables


@pytest.fixture(scope="module")
def nonzero_head_model_and_variables(
    model_and_variables: tuple[KarelProgramModel, dict[str, Any]],
) -> tuple[KarelProgramModel, dict[str, Any]]:
    """Use a nonzero head so sequence-equivalence and causality checks are nontrivial."""
    model, variables = model_and_variables
    params = variables["params"]
    kernel = params["head"]["kernel"]
    head = {**params["head"], "kernel": jax.random.normal(jax.random.key(1), kernel.shape) / np.sqrt(model.d_model)}
    return model, {"params": {**params, "head": head}}


def test_fresh_policy_is_uniform_except_pad(
    inputs: tuple[jax.Array, jax.Array, jax.Array], model_and_variables: tuple[KarelProgramModel, dict[str, Any]]
) -> None:
    model, variables = model_and_variables
    initial, target, tokens = inputs
    _, logits = jax.jit(model.apply)(variables, *inputs)
    chex.assert_shape(logits, (tokens.shape[0] + 1, initial.shape[0], len(TOKENS)))
    chex.assert_type(logits, jnp.float32)
    expected = np.full(len(TOKENS), 1 / (len(TOKENS) - 1), dtype=np.float32)
    expected[KarelProgramEnv.pad_token_id] = 0
    np.testing.assert_allclose(jax.nn.softmax(generation_logits(logits)), np.broadcast_to(expected, logits.shape))
    carry, first = model.apply(variables, initial, target, method=model.prefill)
    _, next_logits = model.apply(variables, tokens[0], carry, method=model.step)
    for prediction in (first, next_logits):
        np.testing.assert_allclose(
            jax.nn.softmax(generation_logits(prediction)), np.broadcast_to(expected, prediction.shape)
        )


def test_prefill_and_steps_match_teacher_forcing(
    inputs: tuple[jax.Array, jax.Array, jax.Array],
    nonzero_head_model_and_variables: tuple[KarelProgramModel, dict[str, Any]],
) -> None:
    model, variables = nonzero_head_model_and_variables
    initial, target, tokens = inputs
    expected_carry, expected = jax.jit(model.apply)(variables, *inputs)
    assert expected.shape == (tokens.shape[0] + 1, 2, len(TOKENS))
    carry, first = jax.jit(partial(model.apply, method=model.prefill))(variables, initial, target)
    outputs = [first]
    step = jax.jit(partial(model.apply, method=model.step))
    for token in tokens:
        carry, logits = step(variables, token, carry)
        outputs.append(logits)
    np.testing.assert_allclose(jnp.stack(outputs), expected, rtol=3e-5, atol=3e-6)
    chex.assert_trees_all_close(carry, expected_carry, rtol=3e-5, atol=3e-6)


def test_future_tokens_do_not_change_earlier_predictions(
    inputs: tuple[jax.Array, jax.Array, jax.Array],
    nonzero_head_model_and_variables: tuple[KarelProgramModel, dict[str, Any]],
) -> None:
    model, variables = nonzero_head_model_and_variables
    initial, target, tokens = inputs
    forward = jax.jit(model.apply)
    _, original = forward(variables, *inputs)
    _, changed = forward(variables, initial, target, tokens.at[-1].set(7))
    np.testing.assert_allclose(original[:-1], changed[:-1], rtol=3e-5, atol=3e-6)
    assert not np.allclose(original[-1], changed[-1])
    # Both halves of the observation must influence the context prediction.
    for other_initial, other_target in ((target, target), (initial, initial)):
        _, other = forward(variables, other_initial, other_target, tokens)
        assert not np.allclose(original[0], other[0])


def test_token_loss_trains_encoder_and_backbone(
    inputs: tuple[jax.Array, jax.Array, jax.Array], model_and_variables: tuple[KarelProgramModel, dict[str, Any]]
) -> None:
    model, variables = model_and_variables
    initial, target, tokens = inputs
    terminal = jnp.full((1, tokens.shape[1]), KarelProgramEnv.terminal_token_id, dtype=jnp.int32)
    labels = jnp.concatenate((tokens, terminal), axis=0)

    def loss(params: dict[str, Any]) -> jax.Array:
        _, logits = model.apply({"params": params}, initial, target, tokens)
        log_probs = jax.nn.log_softmax(generation_logits(logits))
        return -jnp.take_along_axis(log_probs, labels[..., None], axis=-1).mean()

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    value, grads = value_and_grad(variables["params"])
    assert np.isfinite(value)
    assert np.any(np.asarray(grads["head"]["kernel"]) != 0)
    for name in ("context_projection", "token_embedding", "backbone"):
        for leaf in jax.tree.leaves(grads[name]):
            np.testing.assert_array_equal(leaf, 0)

    def gradient_step(param: jax.Array, grad: jax.Array) -> jax.Array:
        chex.assert_equal_shape((param, grad))
        chex.assert_type((param, grad), jnp.float32)
        return param - 0.1 * grad

    # A first head update unlocks gradients through the encoder and backbone.
    updated = jax.tree.map(gradient_step, variables["params"], grads)
    value, grads = value_and_grad(updated)
    assert np.isfinite(value)
    for name in ("context_projection", "token_embedding", "backbone", "head"):
        leaves = jax.tree.leaves(grads[name])
        assert all(np.isfinite(leaf).all() for leaf in leaves)
        assert any(np.any(np.asarray(leaf) != 0) for leaf in leaves), name


def test_invalid_backbone_is_rejected(inputs: tuple[jax.Array, jax.Array, jax.Array]) -> None:
    model = KarelProgramModel(backbone_type="unknown")
    with pytest.raises(ValueError, match="backbone_type"):
        model.init(jax.random.key(0), *inputs)


@pytest.mark.parametrize(
    "implementation",
    [
        "xla",
        pytest.param(
            "cudnn",
            marks=pytest.mark.skipif(
                not any(device.platform == "gpu" for device in jax.devices()), reason="cuDNN requires an NVIDIA GPU"
            ),
        ),
    ],
)
def test_bf16_transformer_generation_and_gradients(
    inputs: tuple[jax.Array, jax.Array, jax.Array], implementation: AttentionImplementation
) -> None:
    model = KarelProgramModel(
        backbone_type="transformer",
        d_model=128,
        num_layers=1,
        num_heads=2,
        max_seq_len=8,
        dtype=jnp.bfloat16,
        attention_implementation=implementation,
    )
    params = model.init(jax.random.key(5), *inputs)["params"]
    chex.assert_type(jax.tree.leaves(params), jnp.float32)
    params = {
        **params,
        "head": {
            **params["head"],
            "kernel": jax.random.normal(jax.random.key(6), params["head"]["kernel"].shape) * 0.05,
        },
    }
    variables = {"params": params}
    initial, target, tokens = inputs
    encoded = model.apply(variables, initial, target, method=model.encode_pair)
    chex.assert_type(encoded, jnp.bfloat16)
    carry, logits = jax.jit(model.apply)(variables, *inputs)
    chex.assert_type(logits, jnp.float32)
    for layer in carry:
        chex.assert_type((layer.key, layer.value), jnp.bfloat16)
    carry, first = jax.jit(partial(model.apply, method=model.prefill))(variables, initial, target)
    outputs = [first]
    step = jax.jit(partial(model.apply, method=model.step))
    for token in tokens:
        carry, prediction = step(variables, token, carry)
        chex.assert_type(prediction, jnp.float32)
        outputs.append(prediction)
    np.testing.assert_allclose(jnp.stack(outputs), logits, rtol=0.05, atol=0.02)

    def loss(parameters: dict[str, Any]) -> jax.Array:
        _, predictions = model.apply({"params": parameters}, *inputs)
        return -jax.nn.log_softmax(generation_logits(predictions))[..., 1].mean()

    value, grads = jax.jit(jax.value_and_grad(loss))(params)
    chex.assert_type((value, *jax.tree.leaves(grads)), jnp.float32)
    assert np.isfinite(value)
    for name in ("context_projection", "token_embedding", "backbone", "head"):
        leaves = jax.tree.leaves(grads[name])
        assert all(np.isfinite(leaf).all() for leaf in leaves)
        assert any(np.any(np.asarray(leaf) != 0) for leaf in leaves), name
