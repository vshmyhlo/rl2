from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.karel import TOKENS, KarelConfig, KarelProgramEnv, sample_task
from rl2.karel_model import KarelProgramModel


@pytest.fixture(scope="module")
def inputs() -> tuple[jax.Array, jax.Array, jax.Array]:
    rng = np.random.default_rng(42)
    tasks = [sample_task(rng, KarelConfig(height=4, width=4)) for _ in range(2)]
    initial = jnp.asarray(np.stack([task.initial for task in tasks]))
    target = jnp.asarray(np.stack([task.target for task in tasks]))
    tokens = jnp.asarray([[1, 1], [2, 2], [3, 3]], dtype=jnp.int32)
    return initial, target, tokens


@pytest.fixture(scope="module")
def model_and_variables(inputs: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[KarelProgramModel, dict[str, Any]]:
    model = KarelProgramModel(d_model=8, num_layers=1, d_state=8, headdim=4, conv_channels=(4, 4))
    variables = model.init(jax.random.key(0), *inputs)
    return model, variables


def test_prefill_and_steps_match_teacher_forcing(
    inputs: tuple[jax.Array, jax.Array, jax.Array], model_and_variables: tuple[KarelProgramModel, dict[str, Any]]
) -> None:
    model, variables = model_and_variables
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
    inputs: tuple[jax.Array, jax.Array, jax.Array], model_and_variables: tuple[KarelProgramModel, dict[str, Any]]
) -> None:
    model, variables = model_and_variables
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
        log_probs = jax.nn.log_softmax(logits)
        return -jnp.take_along_axis(log_probs, labels[..., None], axis=-1).mean()

    value, grads = jax.jit(jax.value_and_grad(loss))(variables["params"])
    assert np.isfinite(value)
    for name in ("conv_0", "context_projection", "token_embedding", "backbone", "head"):
        leaves = jax.tree.leaves(grads[name])
        assert all(np.isfinite(leaf).all() for leaf in leaves)
        assert any(np.any(np.asarray(leaf) != 0) for leaf in leaves), name
