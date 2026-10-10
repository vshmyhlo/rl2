from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.lstm import LSTM, LSTMCarry, LSTMStack, LSTMStackCarry, initial_carry
from rl2.sequence_model import RecurrentSequenceModel


def test_initial_carry_is_zero_float32_even_with_bf16_compute() -> None:
    carry = LSTMStack(4, num_layers=2, dtype=jnp.bfloat16).initial_carry(2)
    assert len(carry) == 2
    for state in jax.tree.leaves(carry):
        assert state.shape == (2, 4)
        assert state.dtype == jnp.float32
        np.testing.assert_array_equal(state, 0)


def test_bf16_lstm_preserves_float32_gate_and_memory_precision() -> None:
    model = LSTM(1, dtype=jnp.bfloat16)
    x = jnp.zeros((2, 1, 1), dtype=jnp.bfloat16)
    starts = jnp.zeros((2, 1), dtype=jnp.bool_)
    carry = (jnp.full((1, 1), 0.75, dtype=jnp.float32), jnp.zeros((1, 1), dtype=jnp.float32))
    variables = model.init(jax.random.key(0), x, carry, starts)
    variables = jax.tree.map(jnp.zeros_like, variables)
    cell_params = variables["params"]["OptimizedLSTMCell_0"]
    for gate, bias in {"i": 0.75, "f": 7.0, "g": 0.5, "o": -0.25}.items():
        cell_params[f"h{gate}"]["bias"] = jnp.full((1,), bias, dtype=jnp.float32)

    # Exactly representable logits isolate activation/update precision from
    # projection rounding. In BF16, sigmoid(7) would round to exactly one.
    input_gate, forget_gate, output_gate = 1 / (1 + np.exp(-np.array([0.75, 7.0, -0.25], dtype=np.float32)))
    expected_cell = np.float32(0.75)
    expected_hidden = []
    for _ in range(2):
        expected_cell = forget_gate * expected_cell + input_gate * np.tanh(np.float32(0.5))
        expected_hidden.append(output_gate * np.tanh(expected_cell))

    final, output = jax.jit(model.apply)(variables, x, carry, starts)
    np.testing.assert_allclose(final[0], expected_cell, rtol=1e-6, atol=0)
    np.testing.assert_allclose(final[1], expected_hidden[-1], rtol=1e-6, atol=0)
    np.testing.assert_array_equal(output, jnp.array(expected_hidden, dtype=jnp.bfloat16).reshape(2, 1, 1))


def test_bf16_lstm_accepts_weakly_typed_float32_carry() -> None:
    model = LSTM(1, dtype=jnp.bfloat16)
    x = jax.ShapeDtypeStruct((2, 1, 1), jnp.bfloat16)
    starts = jax.ShapeDtypeStruct((2, 1), jnp.bool_)
    # Arrays made from Python floats can have weak_type=True while still
    # satisfying the float32 carry contract; promotion must not narrow them.
    memory = jax.ShapeDtypeStruct((1, 1), jnp.float32, weak_type=True)
    (final, output), _ = jax.eval_shape(model.init_with_output, jax.random.key(0), x, (memory, memory), starts)
    for state in final:
        assert state.dtype == jnp.float32
        assert not state.weak_type
    assert output.dtype == jnp.bfloat16


@pytest.mark.parametrize(
    "dtype,supplied_carry,num_layers,intermediate_size",
    [
        (jnp.float32, False, 0, 0),
        (jnp.float32, True, 0, 0),
        (jnp.bfloat16, True, 0, 0),
        (jnp.float32, False, 1, 0),
        (jnp.float32, True, 2, 8),
        (jnp.bfloat16, True, 2, 8),
    ],
    ids=["fresh", "continued", "bf16", "stack-no-mlp", "stack-continued", "stack-bf16"],
)
@jax.default_matmul_precision("highest")
def test_lstm_sequence_step_chunks_and_resets(
    dtype: jax.typing.DTypeLike, supplied_carry: bool, num_layers: int, intermediate_size: int
) -> None:
    model = LSTMStack(4, num_layers, intermediate_size, dtype=dtype) if num_layers else LSTM(4, dtype=dtype)
    assert isinstance(model, RecurrentSequenceModel)
    x = jax.random.normal(jax.random.key(0), (4, 2, 4 if num_layers else 3)).astype(dtype)
    starts = jnp.zeros((4, 2), dtype=jnp.bool_).at[2:, 0].set(True)
    fresh_carry = model.initial_carry(2)
    incoming = jax.tree.map(partial(jnp.full_like, fill_value=0.5), fresh_carry) if supplied_carry else fresh_carry
    params = model.init(jax.random.key(1), x, incoming, starts)
    apply = jax.jit(model.apply)
    step = jax.jit(partial(model.apply, method=model.step))
    final, output = apply(params, x, incoming, starts)
    carry = incoming
    outputs = []
    for t in range(x.shape[0]):
        carry, value = step(params, x[t], carry, starts[t])
        outputs.append(value)
    tolerance = 0.008 if dtype == jnp.bfloat16 else 1e-6
    np.testing.assert_allclose(output, jnp.stack(outputs), atol=tolerance, rtol=0)
    for actual, expected in zip(jax.tree.leaves(carry), jax.tree.leaves(final), strict=True):
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)
        assert actual.dtype == jnp.float32
    assert output.shape == (4, 2, 4)
    assert output.dtype == dtype
    for leaf in jax.tree.leaves(params):
        assert leaf.dtype == jnp.float32

    # Both entry points initialize the same parameters, allowing step-only use.
    step_params = model.init(jax.random.key(1), x[0], incoming, starts[0], method=model.step)
    for actual, expected in zip(jax.tree.leaves(step_params), jax.tree.leaves(params), strict=True):
        np.testing.assert_array_equal(actual, expected)

    prefix_carry, prefix = apply(params, x[:2], incoming, starts[:2])
    chunk_carry, suffix = apply(params, x[2:], prefix_carry, starts[2:])
    np.testing.assert_allclose(jnp.concatenate((prefix, suffix)), output, atol=tolerance, rtol=0)
    for actual, expected in zip(jax.tree.leaves(chunk_carry), jax.tree.leaves(final), strict=True):
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)
    _, fresh = apply(params, x[2:], fresh_carry, starts[2:])
    np.testing.assert_allclose(suffix[:, 0], fresh[:, 0], atol=tolerance, rtol=0)
    assert float(jnp.max(jnp.abs(suffix[:, 1] - fresh[:, 1]))) > 1e-3


@pytest.mark.parametrize("stacked", (False, True))
def test_lstm_gradients_follow_history_but_stop_at_episode_reset(stacked: bool) -> None:
    model = LSTMStack(4, 2, 8) if stacked else LSTM(4)
    inputs = jax.random.normal(jax.random.key(2), (4, 2, 4 if stacked else 3))
    starts = jnp.zeros((4, 2), dtype=jnp.bool_).at[2, 0].set(True)
    carry = model.initial_carry(2)
    params = model.init(jax.random.key(1), inputs, carry, starts)

    def loss(x: jax.Array, memory: LSTMCarry | LSTMStackCarry) -> jax.Array:
        return model.apply(params, x, memory, starts)[1][-1].sum()

    grads, carry_grads = jax.jit(jax.grad(loss, argnums=(0, 1)))(inputs, carry)
    np.testing.assert_array_equal(grads[:2, 0], 0)
    assert float(jnp.linalg.norm(grads[:2, 1])) > 0
    assert float(jnp.linalg.norm(grads[2:, 0])) > 0
    for grad in jax.tree.leaves(carry_grads):
        np.testing.assert_array_equal(grad[0], 0)
        assert float(jnp.linalg.norm(grad[1])) > 0


@pytest.mark.parametrize("shape", [(0, 1, 3), (1, 0, 3), (1, 1, 0)], ids=["time", "batch", "features"])
def test_lstm_rejects_empty_input_dimensions(shape: tuple[int, int, int]) -> None:
    model = LSTM(4)
    x = jax.ShapeDtypeStruct(shape, jnp.float32)
    starts = jax.ShapeDtypeStruct(shape[:2], jnp.bool_)
    carry = initial_carry(1, 4)
    with pytest.raises(AssertionError):
        jax.eval_shape(model.init, jax.random.key(0), x, carry, starts)


@pytest.mark.parametrize("num_envs,hidden_size", [(0, 4), (2, 0)])
def test_lstm_initial_carry_rejects_empty_dimensions(num_envs: int, hidden_size: int) -> None:
    with pytest.raises(AssertionError):
        initial_carry(num_envs, hidden_size)


@pytest.mark.parametrize("field", ("features", "num_layers", "intermediate_size"))
def test_lstm_stack_rejects_invalid_dimensions(field: str) -> None:
    settings = {"features": 4, "num_layers": 2, "intermediate_size": 8}
    settings[field] = -1 if field == "intermediate_size" else 0
    with pytest.raises(AssertionError):
        LSTMStack(**settings).initial_carry(2)


@pytest.mark.parametrize("method", ["__call__", "step"])
def test_lstm_stack_validates_layer_count(method: str) -> None:
    model = LSTMStack(4, 2, 8)
    shape = (1, 4) if method == "step" else (1, 1, 4)
    x = jax.ShapeDtypeStruct(shape, jnp.float32)
    starts = jax.ShapeDtypeStruct(shape[:-1], jnp.bool_)
    carry = model.initial_carry(1)[:1]
    with pytest.raises(ValueError, match="one LSTMCarry per layer"):
        jax.eval_shape(partial(model.init, method=method), jax.random.key(0), x, carry, starts)
