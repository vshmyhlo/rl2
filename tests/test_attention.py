from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.attention import AttentionState, AttentionType, _check_attention_config, _rope, attention
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
    "theta,capacity",
    [
        pytest.param(1e-100, 1, id="frequency-overflow-even-at-position-zero"),
        pytest.param(1e-76, 5, id="angle-overflow-at-last-cache-position"),
    ],
)
def test_attention_rejects_overflowing_rope_configuration(theta: float, capacity: int) -> None:
    with pytest.raises(ValueError, match="RoPE.*float32"):
        _check_attention_config(
            num_heads=1,
            num_kv_heads=1,
            head_dim=4,
            max_seq_len=capacity,
            rope_theta=theta,
            implementation="xla",
            dtype=jnp.float32,
        )


def test_attention_accepts_large_finite_rope_angles() -> None:
    # The largest angle is about 3e38; one more position would overflow float32.
    x = jnp.ones((1, 4, 1, 4), jnp.float32)
    state, output = jax.jit(partial(attention, max_seq_len=4, rope_theta=1e-76, causal=True, implementation="xla"))(
        x, x, x
    )
    assert np.isfinite(state[0]).all()
    np.testing.assert_allclose(output, 1.0)


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


def test_biased_attention_without_rope_matches_numpy_outputs_and_bias_gradients() -> None:
    query = jax.random.normal(jax.random.key(21), (3, 3, 2, 3))
    key, value = jax.random.normal(jax.random.key(22), (2, 3, 3, 1, 3))
    lengths = jnp.asarray([3, 2, 0], jnp.int32)
    valid = jnp.arange(3)[None, :] < lengths[:, None]
    bias_valid = valid[:, None, :, None] & valid[:, None, None, :]
    bias = jnp.where(bias_valid, jax.random.normal(jax.random.key(23), (3, 2, 3, 3)), jnp.nan)

    def loss(bias: jax.Array) -> tuple[jax.Array, tuple[AttentionState, jax.Array]]:
        sc = ShapeChecker(B=3, H=2, T=3)
        sc.check(bias, "BHTT", jnp.float32)
        state, output = attention(
            query,
            key,
            value,
            x_len=lengths,
            max_seq_len=3,
            rope_theta=float("inf"),
            causal=False,
            implementation="xla",
            bias=bias,
            use_rope=False,
        )
        return output.sum(), (state, output)

    (_, (state, output)), gradient = jax.jit(jax.value_and_grad(loss, has_aux=True))(bias)
    expected = np.zeros(query.shape, np.float32)
    expected_gradient = np.zeros(bias.shape, np.float32)
    q, k, v, b = map(np.asarray, (query, key, value, bias))
    for batch, length in enumerate(np.asarray(lengths)):
        if length == 0:
            continue
        for head in range(2):
            scores = q[batch, :length, head] @ k[batch, :length, 0].T / np.sqrt(3)
            scores += b[batch, head, :length, :length]
            weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            weights /= weights.sum(axis=-1, keepdims=True)
            result = weights @ v[batch, :length, 0]
            expected[batch, :length, head] = result
            expected_gradient[batch, head, :length, :length] = weights * (
                v[batch, :length, 0].sum(axis=-1)[None, :] - result.sum(axis=-1)[:, None]
            )
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-6)
    assert np.linalg.norm(expected_gradient) > 0
    # Disabling RoPE must also leave cached keys unrotated, including odd head widths.
    np.testing.assert_array_equal(state[0], jnp.where(valid[..., None, None], key, 0))
    np.testing.assert_array_equal(state[2], lengths)


@pytest.mark.parametrize("use_rope", [True, False])
def test_biased_causal_chunks_match_full_outputs_and_gradients(use_rope: bool) -> None:
    query = jax.random.normal(jax.random.key(24), (1, 3, 2, 2))
    key, value = jax.random.normal(jax.random.key(25), (2, 1, 3, 1, 2))
    bias = jax.random.normal(jax.random.key(26), (1, 2, 3, 3))
    attend = partial(attention, max_seq_len=3, rope_theta=10000.0, causal=True, implementation="xla", use_rope=use_rope)

    def loss(bias: jax.Array, chunked: bool) -> tuple[jax.Array, tuple[AttentionState, jax.Array]]:
        sc = ShapeChecker(B=1, H=2, T=3)
        sc.check(bias, "BHTT", jnp.float32)
        if chunked:
            state, prefix = attend(query[:, :1], key[:, :1], value[:, :1], bias=bias[:, :, :1, :1])
            state, suffix = attend(query[:, 1:], key[:, 1:], value[:, 1:], state, bias=bias[:, :, 1:])
            output = jnp.concatenate((prefix, suffix), axis=1)
        else:
            state, output = attend(query, key, value, bias=bias)
        return jnp.sum(output**2), (state, output)

    full = jax.jit(jax.value_and_grad(partial(loss, chunked=False), has_aux=True))(bias)
    chunks = jax.jit(jax.value_and_grad(partial(loss, chunked=True), has_aux=True))(bias)
    chex.assert_trees_all_close(chunks, full, rtol=1e-5, atol=1e-6)
    gradient = full[1]
    np.testing.assert_array_equal(gradient[:, :, np.triu_indices(3, 1)[0], np.triu_indices(3, 1)[1]], 0)
    assert np.linalg.norm(gradient) > 0


@pytest.mark.parametrize(
    "shape,dtype,cached",
    [
        pytest.param((2, 2, 2), jnp.float32, False, id="rank"),
        pytest.param((1, 1, 2, 2), jnp.float32, False, id="no-head-broadcast"),
        pytest.param((1, 2, 1, 2), jnp.float32, False, id="query-length"),
        pytest.param((1, 2, 2, 3), jnp.float32, False, id="fresh-keys-use-input-length"),
        pytest.param((1, 2, 2, 2), jnp.float32, True, id="cached-keys-use-capacity"),
        pytest.param((1, 2, 2, 2), jnp.int32, False, id="nonfloating-bias"),
    ],
)
def test_attention_rejects_invalid_bias(shape: tuple[int, ...], dtype: jax.typing.DTypeLike, cached: bool) -> None:
    query = jnp.zeros((1, 2, 2, 2), jnp.float32)
    key = query[:, :, :1]
    carry = (jnp.zeros((1, 3, 1, 2)), jnp.zeros((1, 3, 1, 2)), jnp.zeros((1,), jnp.int32)) if cached else None
    with pytest.raises(AssertionError):
        attention(
            query,
            key,
            key,
            carry,
            max_seq_len=3,
            rope_theta=10000.0,
            causal=False,
            implementation="xla",
            bias=jnp.zeros(shape, dtype),
        )


def test_disabling_rope_preserves_backend_head_width_constraints() -> None:
    with pytest.raises(AssertionError):
        _check_attention_config(
            num_heads=1,
            num_kv_heads=1,
            head_dim=3,
            max_seq_len=3,
            rope_theta=float("inf"),
            implementation="cudnn",
            dtype=jnp.bfloat16,
            use_rope=False,
        )


@pytest.mark.parametrize(
    "cached,padded,real_cudnn",
    [
        pytest.param(False, False, False, id="fresh-implicit-lengths"),
        pytest.param(False, True, False, id="fresh-padding-and-empty-example"),
        pytest.param(True, True, False, id="cached-causal-mask-and-bias"),
        pytest.param(
            False,
            True,
            True,
            id="real-cudnn",
            marks=pytest.mark.skipif(
                not any(d.platform == "gpu" for d in jax.devices()), reason="cuDNN requires a GPU"
            ),
        ),
    ],
)
def test_cudnn_bias_padding_preserves_outputs_and_gradients(
    monkeypatch: pytest.MonkeyPatch, cached: bool, padded: bool, real_cudnn: bool
) -> None:
    query = jax.random.normal(jax.random.key(27), (3, 3, 2, 8), jnp.bfloat16)
    key, value = jax.random.normal(jax.random.key(28), (2, 3, 3, 1, 8), jnp.bfloat16)
    lengths = jnp.asarray([3, 2, 0], jnp.int32) if padded else None
    carry = None
    if cached:
        carry = (jnp.ones((3, 5, 1, 8), jnp.bfloat16), jnp.ones((3, 5, 1, 8), jnp.bfloat16), jnp.ones((3,), jnp.int32))
    bias = jax.random.normal(jax.random.key(29), (3, 2, 3, 5 if cached else 3))
    original_attention = jax.nn.dot_product_attention
    routed = []

    def portable_attention(
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        *,
        bias: jax.Array,
        mask: jax.Array | None,
        query_seq_lengths: jax.Array | None,
        key_value_seq_lengths: jax.Array | None,
        is_causal: bool,
        local_window_size: tuple[int, int] | None,
        implementation: AttentionType,
    ) -> jax.Array:
        sc = ShapeChecker(B=3, H=2, K=1, F=8, U=1)
        sc.check(query, "BQHF", jnp.bfloat16)
        sc.check((key, value), "BSKF", jnp.bfloat16)
        sc.check(bias, "BHQS", jnp.bfloat16)
        assert query_seq_lengths is not None and key_value_seq_lengths is not None
        sc.check((query_seq_lengths, key_value_seq_lengths), "B", jnp.int32)
        if mask is not None:
            sc.check(mask, "BUQS", jnp.bool_)
        if implementation == "cudnn":
            assert query.shape[1] == 4
            assert key.shape[1] == (6 if cached else 4)
            routed.append(True)
        return original_attention(
            query,
            key,
            value,
            bias=bias,
            mask=mask,
            query_seq_lengths=query_seq_lengths,
            key_value_seq_lengths=key_value_seq_lengths,
            is_causal=is_causal,
            local_window_size=local_window_size,
            implementation="xla",
        )

    if not real_cudnn:
        monkeypatch.setattr(jax.nn, "dot_product_attention", portable_attention)

    def loss(
        q: jax.Array, k: jax.Array, v: jax.Array, b: jax.Array, implementation: AttentionType
    ) -> tuple[jax.Array, jax.Array]:
        sc = ShapeChecker(B=3, T=3, H=2, K=1, F=8, S=5 if cached else 3)
        sc.check(q, "BTHF", jnp.bfloat16)
        sc.check((k, v), "BTKF", jnp.bfloat16)
        sc.check(b, "BHTS", jnp.float32)
        _, output = attention(
            q,
            k,
            v,
            carry,
            lengths,
            max_seq_len=5,
            rope_theta=10000.0,
            causal=cached,
            implementation=implementation,
            bias=b,
            use_rope=False,
        )
        return jnp.sum(output.astype(jnp.float32) ** 2), output

    expected = jax.jit(jax.value_and_grad(partial(loss, implementation="xla"), argnums=(0, 1, 2, 3), has_aux=True))(
        query, key, value, bias
    )
    actual = jax.jit(jax.value_and_grad(partial(loss, implementation="cudnn"), argnums=(0, 1, 2, 3), has_aux=True))(
        query, key, value, bias
    )
    assert real_cudnn or routed
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(np.asarray(a, np.float32), np.asarray(b, np.float32), rtol=0.05, atol=0.02)
    for gradient in actual[1]:
        assert np.isfinite(gradient).all()
        assert np.linalg.norm(np.asarray(gradient, np.float32)) > 0
        if padded:
            np.testing.assert_array_equal(gradient[2], 0)
