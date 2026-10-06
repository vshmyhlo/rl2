"""Projected RoPE attention with padding, KV caching, and XLA/cuDNN backends.

This module operates on arrays and cache tuples without depending on
transformer model classes. Projections, residual layers, and model-specific
carry wrappers belong to the caller.
"""

import math
from typing import Literal

import chex
import jax
import jax.numpy as jnp

from rl2.shape_checker import ShapeChecker

__all__ = ["AttentionState", "AttentionType", "attention"]

type AttentionType = Literal["xla", "cudnn"]
type AttentionState = tuple[jax.Array, jax.Array, jax.Array]


def _positive_integer(value: int, name: str) -> None:
    chex.assert_scalar_positive(value, custom_message=f"{name} must be positive")


def _check_range(values: jax.Array, maximum: int, message: str) -> None:
    """Reject out-of-range counts eagerly and inside compiled calls."""
    sc = ShapeChecker()
    sc.check(values, "B", jnp.int32)
    invalid = jnp.any((values < 0) | (values > maximum))
    sc.check(invalid, "", jnp.bool_)

    def fail_if_invalid(value: jax.Array) -> None:
        sc = ShapeChecker()
        sc.check(value, "", jnp.bool_)
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
    sc = ShapeChecker(U=1)
    sc.check(x, "BTHF")
    chex.assert_type(x, jnp.floating)
    sc.check(positions, "BT", jnp.int32)
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
    sc.check(frequencies, "R", jnp.float32)
    sc.check(angles, "BTUR", jnp.float32)
    sc.check((real, imag), "BTHR", jnp.float32)
    sc.check((cos, sin), "BTUR", jnp.float32)
    pairs = jnp.stack((real * cos - imag * sin, imag * cos + real * sin), axis=-1)
    sc.check(pairs, "BTHRP", jnp.float32)
    rotated = pairs.reshape(x.shape).astype(x.dtype)
    sc.check(rotated, "BTHF", x.dtype)
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
    chex.assert_is_divisible(head_dim, 2)
    if not 0 < rope_theta < math.inf:
        raise ValueError("rope_theta must be positive and finite")
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
) -> tuple[AttentionState, jax.Array]:
    """Apply RoPE attention to projected Q/K/V and update the sequence cache.

    Queries are [batch,time,query_heads,head_dim]; keys and values are
    [batch,time,kv_heads,head_dim], all with the same floating dtype and
    nonempty dimensions. Query heads must be divisible by KV heads; head width
    must be even. cuDNN requires float16/bfloat16 and head width divisible by 8.
    Optional int32 [batch] x_len gives valid prefix lengths in [0, time].
    Padding produces zero output and does not update the cache.
    Carry is (rotated keys, values, next positions), shaped [batch,capacity,
    kv_heads,head_dim], [batch,capacity,kv_heads,head_dim], and int32 [batch].
    Return the updated carry tuple and [batch,time,query_heads,head_dim] output.
    """
    dtype = query.dtype
    sc = ShapeChecker(C=max_seq_len, U=1)
    sc.check(query, "BTHF", dtype)
    chex.assert_type(query, jnp.floating)
    sc.check((key, value), "BTKF", dtype)
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
    )
    fresh = carry is None
    if carry is None:
        carry = (
            jnp.zeros(sc["BCKF"], dtype),
            jnp.zeros(sc["BCKF"], dtype),
            jnp.zeros(sc["B"], jnp.int32),
        )
    cache_key, cache_value, cache_position = carry
    sc.check((cache_key, cache_value), "BCKF", dtype)
    sc.check(cache_position, "B", jnp.int32)
    native_attention = fresh and x_len is None
    if x_len is None:
        x_len = jnp.full(sc["B"], steps, jnp.int32)
    sc.check(x_len, "B", jnp.int32)
    _check_range(x_len, steps, "x_len must be between 0 and the input sequence length")
    _check_range(cache_position, max_seq_len, "Invalid cache position for max_seq_len cache capacity")
    next_position = cache_position + x_len
    sc.check(next_position, "B", jnp.int32)
    _check_range(next_position, max_seq_len, "Sequence exceeds max_seq_len cache capacity")

    index = jnp.arange(steps, dtype=jnp.int32)[None, :]
    valid_tokens = index < x_len[:, None]
    positions = jnp.where(valid_tokens, cache_position[:, None] + index, 0)
    sc.check(valid_tokens, "BT", jnp.bool_)
    sc.check(positions, "BT", jnp.int32)
    query, key, value = (jnp.where(valid_tokens[..., None, None], v, 0) for v in (query, key, value))
    query, key = _rope(query, positions, rope_theta), _rope(key, positions, rope_theta)
    slots = jnp.arange(max_seq_len, dtype=jnp.int32)[None, :]
    old_valid = slots < cache_position[:, None]
    sc.check(old_valid, "BC", jnp.bool_)

    mask = None
    if native_attention:
        # Let the backend handle causality without a dense mask.
        keys, values = key, value
    else:
        if fresh:
            keys, values, key_positions, key_valid = key, value, positions, valid_tokens
        else:
            old_key, old_value = (jnp.where(old_valid[..., None, None], v, 0) for v in (cache_key, cache_value))
            keys, values = jnp.concatenate((old_key, key), 1), jnp.concatenate((old_value, value), 1)
            key_positions = jnp.concatenate((jnp.broadcast_to(slots, sc["BC"]), positions), 1)
            key_valid = jnp.concatenate((old_valid, valid_tokens), 1)
        sc.check(key_positions, "BS", jnp.int32)
        sc.check(key_valid, "BS", jnp.bool_)
        mask = jnp.broadcast_to(key_valid[:, None, :], sc["BTS"])
        if causal:
            mask &= positions[:, :, None] >= key_positions[:, None, :]
        # Invalid queries attend a harmless slot, avoiding all-masked softmax
        # rows (and NaN gradients in fused backends). Their output is zeroed.
        fallback = jnp.arange(keys.shape[1]) == 0
        mask = jnp.where(valid_tokens[:, :, None], mask, fallback)
        mask = mask[:, None]
        sc.check(mask, "BUTS", jnp.bool_)
    queries = query
    if implementation == "cudnn" and mask is not None:
        # cuDNN masked backward requires even Q and KV lengths. Give
        # a padded query the preceding valid mask to avoid all-masked
        # softmax rows; its output is discarded below.
        if steps % 2:
            queries = jnp.pad(queries, ((0, 0), (0, 1), (0, 0), (0, 0)))
            mask = jnp.concatenate((mask, mask[:, :, -1:]), axis=2)
        if keys.shape[1] % 2:
            keys, values = (jnp.pad(v, ((0, 0), (0, 1), (0, 0), (0, 0))) for v in (keys, values))
            mask = jnp.pad(mask, ((0, 0), (0, 0), (0, 0), (0, 1)))
    # JAX's F16_F16_F32 dot algorithm is unsupported on CPU. Keep
    # projections/cache in float16 but use portable float32 attention.
    attention_dtype = jnp.float32 if implementation == "xla" and jnp.dtype(dtype) == jnp.float16 else dtype
    # Q/S may include cuDNN padding, independently of the chunk length T.
    attention_sc = ShapeChecker(B=batch, H=num_heads, K=kv_heads, F=head_dim, U=1)
    queries, keys, values = (v.astype(attention_dtype) for v in (queries, keys, values))
    attention_sc.check(queries, "BQHF", attention_dtype)
    attention_sc.check((keys, values), "BSKF", attention_dtype)
    if mask is not None:
        attention_sc.check(mask, "BUQS", jnp.bool_)
    attended = jax.nn.dot_product_attention(
        queries,
        keys,
        values,
        mask=mask,
        is_causal=causal and native_attention,
        local_window_size=None,
        implementation=implementation,
    )
    attention_sc.check(attended, "BQHF", attention_dtype)
    attended = jnp.where(valid_tokens[..., None, None], attended[:, :steps].astype(dtype), 0)

    # Append each valid prefix immediately after that example's cached history.
    source = slots - cache_position[:, None]
    sc.check(source, "BC", jnp.int32)
    gather = jnp.clip(source, 0, steps - 1)[..., None, None]
    new_key, new_value = (jnp.take_along_axis(v, gather, axis=1) for v in (key, value))
    sc.check((new_key, new_value), "BCKF", dtype)
    written = ((source >= 0) & (source < x_len[:, None]))[..., None, None]
    sc.check(written, "BCUU", jnp.bool_)
    carry = (
        jnp.where(written, new_key, cache_key),
        jnp.where(written, new_value, cache_value),
        next_position,
    )
    sc.check((carry[0], carry[1]), "BCKF", dtype)
    sc.check(carry[2], "B", jnp.int32)
    sc.check(attended, "BTHF", dtype)
    return carry, attended
