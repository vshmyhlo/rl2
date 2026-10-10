"""Episode-reset contract for the recurrent GDN-2 stack adapter."""

from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.gdn2 import GatedDeltaNet2Config, GatedDeltaNet2Recurrent, GatedDeltaNet2Stack, GatedDeltaNet2StackCarry
from rl2.sequence_model import RecurrentSequenceModel


def assert_tree_close(actual: Any, expected: Any, tolerance: float = 2e-6) -> None:
    chex.assert_trees_all_equal_shapes_and_dtypes(actual, expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    "dtype,use_short_conv",
    [(jnp.float32, True), (jnp.bfloat16, False)],
    ids=["convolution-history", "bf16-without-convolution"],
)
def test_recurrent_stack_contract(dtype: jax.typing.DTypeLike, use_short_conv: bool) -> None:
    config = GatedDeltaNet2Config(
        hidden_size=4, head_dim=2, num_heads=1, conv_size=2, dtype=dtype, use_short_conv=use_short_conv
    )
    model = GatedDeltaNet2Recurrent(config, num_layers=2, intermediate_size=6)
    assert isinstance(model, RecurrentSequenceModel)
    fresh = model.initial_carry(num_envs=2)
    assert len(fresh) == 2
    for layer in fresh:
        assert layer.state.shape == (2, 1, 2, 2)
        for history in (layer.q, layer.k, layer.v):
            assert history.shape == (2, int(use_short_conv), 2)
    for leaf in jax.tree.leaves(fresh):
        assert leaf.dtype == jnp.float32
        np.testing.assert_array_equal(leaf, 0)

    x = jax.random.normal(jax.random.key(0), (4, 2, 4))
    # Reset one environment twice, including consecutive episode starts.
    starts = jnp.zeros((4, 2), jnp.bool_).at[2:, 0].set(True)
    variables = model.init(jax.random.key(1), x, fresh, starts)
    step_variables = model.init(jax.random.key(1), x[0], fresh, starts[0], method=model.step)
    assert_tree_close(variables, step_variables, tolerance=0)
    apply = jax.jit(model.apply)
    step = jax.jit(partial(model.apply, method=model.step))
    tolerance = 0.01 if dtype == jnp.bfloat16 else 2e-6

    for incoming in (fresh, jax.tree.map(partial(jnp.full_like, fill_value=0.1), fresh)):
        final, output = apply(variables, x, incoming, starts)
        memory = incoming
        outputs = []
        for t in range(x.shape[0]):
            memory, value = step(variables, x[t], memory, starts[t])
            outputs.append(value)
        assert output.dtype == dtype
        assert_tree_close((memory, jnp.stack(outputs)), (final, output), tolerance)
        memory, prefix = apply(variables, x[:2], incoming, starts[:2])
        memory, suffix = apply(variables, x[2:], memory, starts[2:])
        assert_tree_close((memory, jnp.concatenate((prefix, suffix))), (final, output), tolerance)

    # Compare against the wrapped stack directly with no resets.
    stack = GatedDeltaNet2Stack(config, num_layers=2, intermediate_size=6)
    expected_memory, expected = stack.apply(
        {"params": variables["params"]["stack"]}, x.swapaxes(0, 1), jnp.full((2,), 4, jnp.int32), fresh
    )
    assert_tree_close(
        apply(variables, x, fresh, jnp.zeros_like(starts)), (expected_memory, expected.swapaxes(0, 1)), tolerance
    )

    # A reset discards even NaN histories and still processes the current input.
    def poison(leaf: jax.Array) -> jax.Array:
        return leaf.at[0].set(jnp.nan)

    poisoned = jax.tree.map(poison, final)
    reset_memory, reset_output = step(variables, x[0], poisoned, jnp.array([True, False]))
    fresh_memory, fresh_output = step(variables, x[0], fresh, starts[0])
    continued_memory, continued_output = step(variables, x[0], final, starts[0])
    for actual, reset, continued in zip(
        jax.tree.leaves((reset_memory, reset_output)),
        jax.tree.leaves((fresh_memory, fresh_output)),
        jax.tree.leaves((continued_memory, continued_output)),
        strict=True,
    ):
        np.testing.assert_allclose(actual[0], reset[0], atol=tolerance, rtol=tolerance)
        np.testing.assert_allclose(actual[1], continued[1], atol=tolerance, rtol=tolerance)


def test_recurrent_stack_reset_stops_history_gradients() -> None:
    model = GatedDeltaNet2Recurrent(GatedDeltaNet2Config(hidden_size=4, head_dim=2, num_heads=1, conv_size=2), 1, 6)
    x = jax.random.normal(jax.random.key(2), (4, 2, 4))
    starts = jnp.zeros((4, 2), jnp.bool_).at[2, 0].set(True)
    incoming = model.initial_carry(2)
    variables = model.init(jax.random.key(1), x, incoming, starts)

    def loss(inputs: jax.Array, memory: GatedDeltaNet2StackCarry) -> jax.Array:
        chex.assert_shape(inputs, (4, 2, 4))
        chex.assert_type(inputs, jnp.float32)
        return model.apply(variables, inputs, memory, starts)[1][-1].sum()

    inputs_grad, memory_grad = jax.jit(jax.grad(loss, argnums=(0, 1)))(x, incoming)
    np.testing.assert_array_equal(inputs_grad[:2, 0], 0)
    assert float(jnp.linalg.norm(inputs_grad[:2, 1])) > 0
    assert float(jnp.linalg.norm(inputs_grad[2:, 0])) > 0
    for leaf in jax.tree.leaves(memory_grad):
        np.testing.assert_array_equal(leaf[0], 0)
    # Query history affects only the first output, not the recurrent state.
    np.testing.assert_array_equal(memory_grad[0].q, 0)
    for leaf in (memory_grad[0].state, memory_grad[0].k, memory_grad[0].v):
        assert float(jnp.linalg.norm(leaf[1])) > 0


@pytest.mark.parametrize("bad_input", ["input_dtype", "layer_count", "empty_time", "empty_batch"])
def test_recurrent_stack_validation(bad_input: str) -> None:
    model = GatedDeltaNet2Recurrent(GatedDeltaNet2Config(hidden_size=4, head_dim=2, num_heads=1), 1, 6)
    shape = (0, 1, 4) if bad_input == "empty_time" else (1, 0, 4) if bad_input == "empty_batch" else (1, 1, 4)
    x = jax.ShapeDtypeStruct(shape, jnp.int32 if bad_input == "input_dtype" else jnp.float32)
    starts = jax.ShapeDtypeStruct(shape[:2], jnp.bool_)
    carry = () if bad_input == "layer_count" else model.initial_carry(1)
    with pytest.raises((AssertionError, ValueError)):
        jax.eval_shape(model.init, jax.random.key(0), x, carry, starts)


@pytest.mark.parametrize("num_layers,num_envs", [(0, 2), (1, 0)])
def test_recurrent_stack_initial_carry_rejects_empty_dimensions(num_layers: int, num_envs: int) -> None:
    model = GatedDeltaNet2Recurrent(GatedDeltaNet2Config(hidden_size=4, head_dim=2, num_heads=1), num_layers, 6)
    with pytest.raises((AssertionError, ValueError)):
        model.initial_carry(num_envs)
