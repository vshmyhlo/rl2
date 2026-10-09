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


@pytest.mark.parametrize(
    "method,bad_input",
    [
        ("step", "input_rank"),
        ("__call__", "input_rank"),
        ("step", "input_dtype"),
        ("__call__", "mask_dtype"),
        ("step", "mask_shape"),
        ("__call__", "mask_shape"),
        ("step", "carry_batch"),
        ("__call__", "carry_width"),
        ("step", "cell_dtype"),
        ("__call__", "hidden_dtype"),
        ("__call__", "empty_time"),
        ("step", "empty_batch"),
        ("step", "empty_features"),
    ],
)
def test_lstm_validates_its_inputs(method: str, bad_input: str) -> None:
    model = LSTM(4)
    x_shape = (2, 3) if method == "step" else (4, 2, 3)
    if bad_input == "input_rank":
        x_shape = x_shape + (1,)
    elif bad_input in ("empty_time", "empty_batch"):
        x_shape = (0, *x_shape[1:])
    elif bad_input == "empty_features":
        x_shape = (*x_shape[:-1], 0)
    x = jax.ShapeDtypeStruct(x_shape, jnp.int32 if bad_input == "input_dtype" else jnp.float32)
    mask_shape = (2,) if method == "step" else (4, 2)
    if bad_input == "mask_shape":
        mask_shape = (*mask_shape[:-1], 1)
    elif bad_input in ("empty_time", "empty_batch"):
        mask_shape = (0, *mask_shape[1:])
    starts = jax.ShapeDtypeStruct(mask_shape, jnp.int32 if bad_input == "mask_dtype" else jnp.bool_)
    carry_shape = (1, 4) if bad_input == "carry_batch" else (2, 3) if bad_input == "carry_width" else (2, 4)
    carry = (
        jax.ShapeDtypeStruct(carry_shape, jnp.bfloat16 if bad_input == "cell_dtype" else jnp.float32),
        jax.ShapeDtypeStruct(carry_shape, jnp.bfloat16 if bad_input == "hidden_dtype" else jnp.float32),
    )
    with pytest.raises(AssertionError):
        jax.eval_shape(partial(model.init, method=method), jax.random.key(0), x, carry, starts)


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


@pytest.mark.parametrize(
    "method,bad_input",
    [
        ("__call__", "carry_layers"),
        ("step", "carry_layers"),
        ("__call__", "carry_width"),
        ("step", "carry_dtype"),
        ("__call__", "input_width"),
        ("step", "mask_dtype"),
    ],
)
def test_lstm_stack_validates_inputs(method: str, bad_input: str) -> None:
    model = LSTMStack(4, 2, 8)
    shape = (2, 4) if method == "step" else (4, 2, 4)
    if bad_input == "input_width":
        shape = (*shape[:-1], 3)
    x = jax.ShapeDtypeStruct(shape, jnp.float32)
    starts = jax.ShapeDtypeStruct(shape[:-1], jnp.int32 if bad_input == "mask_dtype" else jnp.bool_)
    carry = model.initial_carry(2)
    if bad_input == "carry_layers":
        carry = carry[:1]
    elif bad_input == "carry_width":
        carry = (carry[0], (carry[1][0][:, :3], carry[1][1]))
    elif bad_input == "carry_dtype":
        carry = (carry[0], (carry[1][0], carry[1][1].astype(jnp.bfloat16)))
    with pytest.raises(ValueError if bad_input == "carry_layers" else AssertionError):
        jax.eval_shape(partial(model.init, method=method), jax.random.key(0), x, carry, starts)
