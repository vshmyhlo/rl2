from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.attention import AttentionState, _rope, attention
from rl2.shape_checker import ShapeChecker


@pytest.mark.parametrize("causal", [True, False])
def test_attention_uniform_values_and_zero_length(causal: bool) -> None:
    query = jnp.zeros((1, 2, 2, 2), jnp.float32)
    key = jnp.zeros((1, 2, 1, 2), jnp.float32)
    value = jnp.broadcast_to(jnp.asarray([2.0, 6.0])[None, :, None, None], key.shape)
    attend = jax.jit(partial(attention, max_seq_len=3, rope_theta=10000.0, causal=causal, implementation="xla"))
    carry, output = attend(query, key, value)
    sc = ShapeChecker(B=1, T=2, H=2, K=1, F=2, C=3, U=1)
    sc.check(output, "BTHF", jnp.float32)
    expected = jnp.asarray([2.0, 4.0] if causal else [4.0, 4.0])[None, :, None, None]
    np.testing.assert_allclose(output, jnp.broadcast_to(expected, output.shape))
    sc.check((carry[0], carry[1]), "BCKF", jnp.float32)
    sc.check(carry[2], "B", jnp.int32)
    np.testing.assert_array_equal(carry[2], [2])
    np.testing.assert_array_equal(carry[1][:, :2], value)
    np.testing.assert_array_equal(carry[1][:, 2:], 0)

    next_value = jnp.full((1, 1, 1, 2), 10.0)
    continued, output = attend(query[:, :1], key[:, :1], next_value, carry)
    sc.check(output, "BUHF", jnp.float32)
    np.testing.assert_allclose(output, 6.0)
    np.testing.assert_array_equal(continued[2], [3])
    skipped, output = attend(query[:, :1], key[:, :1], next_value, continued, jnp.zeros((1,), jnp.int32))
    sc.check(output, "BUHF", jnp.float32)
    np.testing.assert_array_equal(output, 0)
    chex.assert_trees_all_equal(skipped, continued)
    empty, output = attend(query, key, value, x_len=jnp.zeros((1,), jnp.int32))
    np.testing.assert_array_equal(output, 0)
    np.testing.assert_array_equal(empty[2], 0)
    np.testing.assert_array_equal(empty[1], 0)


@pytest.mark.parametrize("causal", [True, False])
def test_cached_attention_ignores_unused_slots_and_preserves_gradients(causal: bool) -> None:
    positions = jnp.asarray([1, 2, 0], jnp.int32)
    lengths = jnp.asarray([2, 0, 0], jnp.int32)
    old_valid = jnp.arange(3)[None, :] < positions[:, None]
    valid = jnp.arange(2)[None, :] < lengths[:, None]
    cache_key = jnp.broadcast_to(jnp.where(old_valid[..., None, None], 0.0, jnp.nan), (3, 3, 1, 2))
    cache_value = jnp.where(old_valid[..., None, None], 2.0, cache_key)
    query = jnp.broadcast_to(jnp.where(valid[..., None, None], 0.0, jnp.nan), (3, 2, 2, 2))
    key = query[:, :, :1]
    value = jnp.where(valid[..., None, None], jnp.asarray([6.0, 10.0])[None, :, None, None], key)

    def loss(
        q: jax.Array, k: jax.Array, v: jax.Array, old_k: jax.Array, old_v: jax.Array
    ) -> tuple[jax.Array, tuple[AttentionState, jax.Array]]:
        sc = ShapeChecker(B=3, T=2, H=2, K=1, F=2, C=3)
        sc.check(q, "BTHF", jnp.float32)
        sc.check((k, v), "BTKF", jnp.float32)
        sc.check((old_k, old_v), "BCKF", jnp.float32)
        state, output = attention(
            q,
            k,
            v,
            (old_k, old_v, positions),
            lengths,
            max_seq_len=3,
            rope_theta=10000.0,
            causal=causal,
            implementation="xla",
        )
        return output.sum(), (state, output)

    (_, (state, output)), grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4), has_aux=True))(
        query, key, value, cache_key, cache_value
    )
    expected = jnp.asarray([4.0, 6.0] if causal else [6.0, 6.0])[:, None, None]
    np.testing.assert_allclose(output[0], jnp.broadcast_to(expected, output[0].shape))
    np.testing.assert_array_equal(output[1:], 0)
    np.testing.assert_array_equal(state[2], [3, 2, 0])
    np.testing.assert_array_equal(state[1][0, :, 0, 0], [2, 6, 10])
    for before, after in zip((cache_key, cache_value), state[:2]):
        np.testing.assert_array_equal(after[1:], before[1:])
    for gradient in grads:
        assert np.isfinite(gradient).all()
    for gradient in grads[:3]:
        np.testing.assert_array_equal(gradient[~valid], 0)
    for gradient in grads[3:]:
        np.testing.assert_array_equal(gradient[~old_valid], 0)
        np.testing.assert_array_equal(gradient[1:], 0)
    # Both old and appended values receive gradients through the packed cache.
    expected_new = jnp.asarray([5 / 3, 2 / 3] if causal else [4 / 3, 4 / 3])[:, None, None]
    np.testing.assert_allclose(grads[2][0], jnp.broadcast_to(expected_new, grads[2][0].shape))
    np.testing.assert_allclose(grads[4][0, 0], 5 / 3 if causal else 4 / 3)


@pytest.mark.parametrize(
    "value_shape,value_dtype",
    [
        pytest.param((1, 2, 2), jnp.float32, id="rank"),
        pytest.param((1, 1, 1, 2), jnp.float32, id="time-dimension"),
        pytest.param((1, 2, 1, 2), jnp.bfloat16, id="dtype"),
    ],
)
def test_attention_rejects_incompatible_values(value_shape: tuple[int, ...], value_dtype: jax.typing.DTypeLike) -> None:
    with pytest.raises(AssertionError):
        attention(
            jnp.zeros((1, 2, 2, 2), jnp.float32),
            jnp.zeros((1, 2, 1, 2), jnp.float32),
            jnp.zeros(value_shape, value_dtype),
            max_seq_len=3,
            rope_theta=10000.0,
            causal=True,
            implementation="xla",
        )


def test_attention_rejects_invalid_configuration() -> None:
    x = jnp.zeros((1, 1, 1, 2), jnp.float32)
    with pytest.raises(ValueError, match="rope_theta must be positive and finite"):
        attention(x, x, x, max_seq_len=2, rope_theta=float("inf"), causal=True, implementation="xla")


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 0, 1, 2), id="empty-time"),
        pytest.param((1, 1, 0, 2), id="empty-heads"),
        pytest.param((1, 1, 1, 0), id="empty-head-width"),
    ],
)
def test_attention_rejects_empty_dimensions(shape: tuple[int, ...]) -> None:
    x = jnp.zeros(shape, jnp.float32)
    with pytest.raises((ValueError, AssertionError)):
        attention(x, x, x, max_seq_len=2, rope_theta=10000.0, causal=True, implementation="xla")


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_rope_matches_llama_adjacent_pairs_in_float32(dtype: jax.typing.DTypeLike) -> None:
    x = jnp.asarray([[[[1.25, -0.75, 0.5, 1.75]], [[1.25, -0.75, 0.5, 1.75]], [[0.25, 1.5, -1.25, -0.5]]]], dtype=dtype)
    positions = jnp.asarray([[0, 3, 17]], jnp.int32)
    # Independent complex64 oracle matching Meta's adjacent-pair rotation.
    sc = ShapeChecker(B=1, T=3, H=1, F=4)
    sc.check(x, "BTHF", dtype)
    sc.check(positions, "BT", jnp.int32)
    angles = np.asarray(positions, np.float32)[..., None, None] * np.asarray([1, 0.01], np.float32)
    values = np.asarray(x, np.float32)
    complex_values = values[..., ::2] + 1j * values[..., 1::2]
    rotated = complex_values * np.exp(1j * angles)
    expected = np.stack((rotated.real, rotated.imag), axis=-1).reshape(x.shape).astype(x.dtype)
    actual = _rope(x, positions, 10000.0)
    sc.check(actual, "BTHF", dtype)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[:, 0], x[:, 0])
    compiled = jax.jit(_rope, static_argnums=2)(x, positions, 10000.0)
    np.testing.assert_allclose(
        compiled.astype(jnp.float32), expected.astype(np.float32), atol=0, rtol=2 * jnp.finfo(dtype).eps
    )


@pytest.mark.parametrize(
    "theta",
    [
        pytest.param(1e40, id="base-overflows-float32"),
        pytest.param(1e-40, id="base-is-subnormal-float32"),
    ],
)
def test_rope_preserves_frequencies_for_extreme_finite_theta(theta: float) -> None:
    x = jnp.tile(jnp.asarray([1.0, 0.0], jnp.float32), 8).reshape(1, 1, 1, 16)
    positions = jnp.ones((1, 1), jnp.int32)
    # The base need not fit float32 even though all resulting frequencies do.
    frequencies = np.power(theta, -np.arange(8, dtype=np.float64) / 8).astype(np.float32)
    expected = np.stack((np.cos(frequencies), np.sin(frequencies)), axis=-1).reshape(x.shape)
    for actual in (_rope(x, positions, theta), jax.jit(_rope, static_argnums=2)(x, positions, theta)):
        sc = ShapeChecker(B=1, T=1, H=1, F=16)
        sc.check(actual, "BTHF", jnp.float32)
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize(
    "positions_shape,positions_dtype",
    [
        pytest.param((2,), jnp.int32, id="position-rank"),
        pytest.param((2, 1), jnp.int32, id="position-dimensions"),
        pytest.param((1, 2), jnp.float32, id="position-dtype"),
    ],
)
def test_rope_rejects_invalid_positions(
    positions_shape: tuple[int, ...], positions_dtype: jax.typing.DTypeLike
) -> None:
    with pytest.raises(AssertionError):
        _rope(jnp.zeros((1, 2, 1, 2), jnp.float32), jnp.zeros(positions_shape, positions_dtype), 10000.0)
