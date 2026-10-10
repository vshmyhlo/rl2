"""Projected attention with optional RoPE/bias, KV caching, and XLA/cuDNN backends.

This module operates on arrays and cache tuples without depending on
transformer model classes. Projections, residual layers, and model-specific
carry wrappers belong to the caller.
"""

import math
from typing import Literal

import chex
import jax
import jax.numpy as jnp

__all__ = ["AttentionState", "AttentionType", "attention"]

type AttentionType = Literal["xla", "cudnn"]
type AttentionState = tuple[jax.Array, jax.Array, jax.Array]


def _positive_integer(value: int, name: str) -> None:
    chex.assert_scalar_positive(value, custom_message=f"{name} must be positive")


def _check_range(values: jax.Array, maximum: int, message: str) -> None:
    """Reject out-of-range counts eagerly and inside compiled calls."""
    invalid = jnp.any((values < 0) | (values > maximum))

    def fail_if_invalid(value: jax.Array) -> None:
        if bool(value):
            raise ValueError(message)

    def report_failure() -> None:
        # vmap may evaluate both branches, so check the predicate in the callback.
        jax.debug.callback(fail_if_invalid, invalid)

    def success() -> None:
        pass

    if isinstance(invalid, jax.core.Tracer):
        jax.lax.cond(invalid, report_failure, success)
    elif bool(invalid):
        fail_if_invalid(invalid)


def _rope(x: jax.Array, positions: jax.Array, theta: float) -> jax.Array:
    """Rotate [batch,time,heads,head_dim] using [batch,time] positions."""
    chex.assert_type(x, jnp.floating)
    chex.assert_is_divisible(x.shape[-1], 2)
    chex.assert_scalar_positive(theta)
    half = x.shape[-1] // 2
    # Theta and the head width are static. Compute frequencies before casting:
    # a finite theta can overflow or lose precision in float32 even when its
    # inverse powers are representable. Trigonometry remains in float32.
    frequencies = jnp.asarray([theta ** (-i / half) for i in range(half)], dtype=jnp.float32)
    angles = positions.astype(jnp.float32)[..., None, None] * frequencies
    real, imag = x[..., ::2].astype(jnp.float32), x[..., 1::2].astype(jnp.float32)
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    pairs = jnp.stack((real * cos - imag * sin, imag * cos + real * sin), axis=-1)
    rotated = pairs.reshape(x.shape).astype(x.dtype)
    return rotated


def _check_attention_config(
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    rope_theta: float,
    implementation: AttentionType,
    dtype: jax.typing.DTypeLike,
    use_rope: bool = True,
) -> None:
    """Keep module and standalone attention validation consistent."""
    for name, value in (
        ("num_heads", num_heads),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
        ("max_seq_len", max_seq_len),
    ):
        _positive_integer(value, name)
    chex.assert_is_divisible(num_heads, num_kv_heads)
    if use_rope:
        chex.assert_is_divisible(head_dim, 2)
        if not 0 < rope_theta < math.inf:
            raise ValueError("rope_theta must be positive and finite")
        # Bases below one increase frequencies. Check in log space so validation
        # itself cannot overflow, including when only position zero is used (0*inf).
        max_log_frequency = max(0.0, -(1 - 2 / head_dim) * math.log(rope_theta))
        max_log_angle = max_log_frequency + math.log(max(1, max_seq_len - 1))
        if max_log_angle > math.log(float(jnp.finfo(jnp.float32).max)):
            raise ValueError("RoPE frequencies and angles must fit in float32 for all cache positions")
    dtype = jnp.dtype(dtype)
    if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
        raise ValueError("dtype must be float32, bfloat16, or float16")
    if implementation == "cudnn":
        if dtype == jnp.float32:
            raise ValueError("cuDNN attention requires dtype=float16 or bfloat16")
        chex.assert_is_divisible(head_dim, 8)


def attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    carry: AttentionState | None = None,
    x_len: jax.Array | None = None,
    *,
    max_seq_len: int,
    rope_theta: float,
    causal: bool,
    implementation: AttentionType,
    bias: jax.Array | None = None,
    use_rope: bool = True,
) -> tuple[AttentionState, jax.Array]:
    """Attend to projected Q/K/V and update the sequence cache.

    Queries are [batch,time,query_heads,head_dim]; keys and values are
    [batch,time,kv_heads,head_dim], all with the same floating dtype and
    nonempty dimensions. Query heads must be divisible by KV heads; head width
    must be even when use_rope=True (the default). cuDNN requires
    float16/bfloat16 and head width divisible by 8 regardless of use_rope.
    With RoPE enabled, frequencies and angles through max_seq_len must fit in
    float32. With RoPE disabled, rope_theta is ignored and keys are unrotated.
    Optional int32 [batch] x_len gives valid prefix lengths in [0, time].
    Padding produces zero output and does not update the cache.
    Carry is (keys, values, next positions), shaped [batch,capacity,
    kv_heads,head_dim], [batch,capacity,kv_heads,head_dim], and int32 [batch].
    Cached calls append valid tokens before attending over the updated cache.
    Sequence lengths exclude padding; causal chunks also mask future positions.
    Keep use_rope and rope_theta consistent when reusing a cache.

    Optional floating bias has exact shape [batch,query_heads,time,key_length],
    without broadcasting. key_length is time for fresh calls and max_seq_len
    for cached calls; cached columns address absolute cache slots after append.
    Bias is added to scaled attention scores before softmax, cast to the
    attention compute dtype. Padded rows/columns are ignored. It is not cached
    and cannot override causal or sequence-length masks.
    Return the updated carry tuple and [batch,time,query_heads,head_dim] output.
    """
    dtype = query.dtype
    chex.assert_type(query, jnp.floating)
    batch, steps, num_heads, head_dim = query.shape
    kv_heads = key.shape[2]
    _positive_integer(batch, "batch_size")
    _positive_integer(steps, "sequence_length")
    _check_attention_config(
        num_heads=num_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        rope_theta=rope_theta,
        implementation=implementation,
        dtype=dtype,
        use_rope=use_rope,
    )
    fresh = carry is None
    if bias is not None:
        chex.assert_type(bias, jnp.floating)
    if carry is None:
        carry = (
            jnp.zeros((batch, max_seq_len, kv_heads, head_dim), dtype),
            jnp.zeros((batch, max_seq_len, kv_heads, head_dim), dtype),
            jnp.zeros((batch,), jnp.int32),
        )
    cache_key, cache_value, cache_position = carry
    native_attention = fresh and x_len is None and bias is None
    if x_len is None:
        x_len = jnp.full((batch,), steps, jnp.int32)
    _check_range(x_len, steps, "x_len must be between 0 and the input sequence length")
    _check_range(cache_position, max_seq_len, "Invalid cache position for max_seq_len cache capacity")
    next_position = cache_position + x_len
    _check_range(next_position, max_seq_len, "Sequence exceeds max_seq_len cache capacity")

    index = jnp.arange(steps, dtype=jnp.int32)[None, :]
    valid_tokens = index < x_len[:, None]
    positions = jnp.where(valid_tokens, cache_position[:, None] + index, 0)
    query, key, value = (jnp.where(valid_tokens[..., None, None], v, 0) for v in (query, key, value))
    if use_rope:
        query, key = _rope(query, positions, rope_theta), _rope(key, positions, rope_theta)
    slots = jnp.arange(max_seq_len, dtype=jnp.int32)[None, :]

    # Append each valid prefix immediately after that example's cached history.
    source = slots - cache_position[:, None]
    gather = jnp.clip(source, 0, steps - 1)[..., None, None]
    new_key, new_value = (jnp.take_along_axis(v, gather, axis=1) for v in (key, value))
    written = ((source >= 0) & (source < x_len[:, None]))[..., None, None]
    carry = (
        jnp.where(written, new_key, cache_key),
        jnp.where(written, new_value, cache_value),
        next_position,
    )

    mask = None
    if fresh:
        # Prefill need only attend over the input, not the entire cache capacity.
        keys, values = key, value
        kv_lengths = x_len
    else:
        key_valid = slots < next_position[:, None]
        # Sanitize unused slots without changing the returned cache. Masking
        # alone would not isolate NaNs in these slots from attention gradients.
        keys, values = (jnp.where(key_valid[..., None, None], v, 0) for v in carry[:2])
        kv_lengths = next_position
        if causal and steps > 1:
            # Queries start at each example's old cache position, so the
            # backend's zero-offset causal triangle would exclude history.
            # Invalid queries have position zero and can safely attend slot 0.
            mask = (positions[:, :, None] >= slots[:, None, :])[:, None]
    if bias is not None:
        key_valid = jnp.arange(keys.shape[1])[None, :] < kv_lengths[:, None]
        bias_valid = valid_tokens[:, None, :, None] & key_valid[:, None, None, :]
        bias = jnp.where(bias_valid, bias, 0)
    query_seq_lengths = key_value_seq_lengths = None
    if not native_attention:
        # Give empty examples one safe dummy query/key instead of an empty
        # softmax in fused backends. Padded inputs are zero; outputs are zeroed
        # below, and only the original valid prefixes enter the returned cache.
        query_seq_lengths = jnp.maximum(x_len, 1)
        key_value_seq_lengths = jnp.maximum(kv_lengths, 1)
    queries = query
    if implementation == "cudnn" and (mask is not None or bias is not None):
        # cuDNN masked/biased backward requires even Q and KV lengths. Give
        # a padded query the preceding valid mask to avoid all-masked
        # softmax rows; its output is discarded below.
        if steps % 2:
            queries = jnp.pad(queries, ((0, 0), (0, 1), (0, 0), (0, 0)))
            if mask is not None:
                mask = jnp.concatenate((mask, mask[:, :, -1:]), axis=2)
            if bias is not None:
                bias = jnp.pad(bias, ((0, 0), (0, 0), (0, 1), (0, 0)))
        if keys.shape[1] % 2:
            keys, values = (jnp.pad(v, ((0, 0), (0, 1), (0, 0), (0, 0))) for v in (keys, values))
            if mask is not None:
                mask = jnp.pad(mask, ((0, 0), (0, 0), (0, 0), (0, 1)))
            if bias is not None:
                bias = jnp.pad(bias, ((0, 0), (0, 0), (0, 0), (0, 1)))
    # JAX's F16_F16_F32 dot algorithm is unsupported on CPU. Keep
    # projections/cache in float16 but use portable float32 attention.
    attention_dtype = jnp.float32 if implementation == "xla" and jnp.dtype(dtype) == jnp.float16 else dtype
    queries, keys, values = (v.astype(attention_dtype) for v in (queries, keys, values))
    if bias is not None:
        bias = bias.astype(attention_dtype)
    attended = jax.nn.dot_product_attention(
        queries,
        keys,
        values,
        bias=bias,
        mask=mask,
        query_seq_lengths=query_seq_lengths,
        key_value_seq_lengths=key_value_seq_lengths,
        is_causal=causal and fresh,
        local_window_size=None,
        implementation=implementation,
    )
    attended = jnp.where(valid_tokens[..., None, None], attended[:, :steps].astype(dtype), 0)

    return carry, attended
