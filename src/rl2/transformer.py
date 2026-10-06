"""Llama 3 decoder backbone with batch-major sequences and explicit KV caches.

``TransformerBlock`` combines RoPE attention with pre-RMSNorm and SwiGLU
residual layers; ``TransformerStack`` repeats it and adds a final RMSNorm.
The attention and SwiGLU architecture follow Meta's reference:
https://github.com/meta-llama/llama3/blob/main/llama/model.py
MLP width is ``round(dim * mlp_expansion)``, with a default expansion of 2.
Model sizes and KV head counts remain configurable.
Neither includes embeddings or a prediction head. Projections are bias-free,
parameters/norm statistics are float32, and ``dtype`` controls projections and
cached keys/values. Block outputs use the residual operands' promoted dtype;
stack outputs are cast to ``dtype`` even when the final norm is disabled.
All projection weights use normal initialization with standard deviation
``initializer_range`` (default 0.02), without depth-dependent scaling.
The XLA float16 path evaluates attention in float32 for CPU portability.
For mixed-precision training, set ``dtype=jnp.bfloat16`` on the block or stack;
inputs may be float32 or bfloat16. Keep the initialized parameters and optimizer
state in float32 and compute the loss in float32. KV entries use bfloat16, while
RoPE trigonometry and normalization statistics stay float32.
RoPE rotates adjacent feature pairs in float32, then casts back to the
activation dtype, matching Meta's Llama 3 reference. Queries and keys have
no additional normalization. Residual additions use their operands' dtypes
without explicitly promoting to float32.

Attention defaults to causal; set ``causal=False`` for
bidirectional attention over the current chunk and valid cached history in
the sequence. Cached states are not recomputed, so non-causal outputs
depend on chunk boundaries. Step calls require ``causal=True`` and raise
``ValueError`` for bidirectional attention.
The fixed-size cache retains
all valid tokens, up to ``max_seq_len``; exceeding that capacity raises an
error, including under JIT. Optional int32 ``x_len[batch]`` delimits each
left-aligned valid prefix; remaining input tokens are right padding. Only valid
tokens advance the cache and RoPE positions. Lengths must be between zero and
the chunk length; omission means all tokens are valid. Padded outputs are zero.
Pass a fresh carry to restart a sequence. With causal attention, sequence, chunk and step calls
have identical semantics; prefill computes attention in parallel, without a
token-by-token attention scan. Gradients flow through supplied caches unless
the caller applies ``jax.lax.stop_gradient``.

``attention_implementation="xla"`` is portable (the default). Select ``"cudnn"``
with float16/bfloat16 and a supported NVIDIA GPU for JAX's cuDNN fused attention.
Backend shape/device restrictions are reported by JAX, without silent fallback.
Fresh, unpadded prefill uses the native attention mode; cached or padded
sequences use an explicit boolean mask to handle offsets and valid lengths.
Masked cuDNN calls pad odd sequence lengths for its backward-pass constraints.

Example::

    model = TransformerStack(dim=256, num_layers=4, num_heads=8,
                             num_kv_heads=2, max_seq_len=1024)
    x = jnp.zeros((8, 16, 256))
    variables = model.init(jax.random.key(0), x)
    carry, y = model.apply(variables, x)
    carry, next_y = model.apply(variables, x[:, 0], carry=carry, method=model.step)
"""

import math
from typing import Literal, NamedTuple

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from rl2.shape_checker import ShapeChecker

type AttentionImplementation = Literal["xla", "cudnn"]


class TransformerCarry(NamedTuple):
    """Batch-leading KV cache and next RoPE position within each sequence.

    key/value: [B,max_seq_len,num_kv_heads,headdim], in projection dtype.
    position: [B], int32. Slot ``p`` stores position ``p``.
    Keys are already rotated. Unused slots are initialized to zero.
    """

    key: jax.Array
    value: jax.Array
    position: jax.Array


type TransformerStackCarry = tuple[TransformerCarry, ...]


def _positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    chex.assert_scalar_positive(value)


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
    frequencies = theta ** (-jnp.arange(half, dtype=jnp.float32) / half)
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


def _expanded_mlp_width(dim: int, mlp_expansion: float) -> int:
    """Scale the model dimension and round to the nearest integer width."""
    _positive_integer(dim, "dim")
    if not 0 < mlp_expansion < math.inf:
        raise ValueError("mlp_expansion must be positive and finite")
    width = round(dim * mlp_expansion)
    _positive_integer(width, "MLP width")
    return width


def _check_attention_config(
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    rope_theta: float,
    causal: bool,
    implementation: AttentionImplementation,
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
    if implementation not in ("xla", "cudnn"):
        raise ValueError("attention_implementation must be 'xla' or 'cudnn'")
    if not isinstance(causal, bool):
        raise TypeError("causal must be a bool")
    dtype = jnp.dtype(dtype)
    if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
        raise ValueError("dtype must be float32, bfloat16, or float16")
    if implementation == "cudnn":
        if dtype == jnp.float32:
            raise ValueError("cuDNN attention requires dtype=float16 or bfloat16")
        chex.assert_is_divisible(head_dim, 8)


def _attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    carry: TransformerCarry | None = None,
    x_len: jax.Array | None = None,
    *,
    max_seq_len: int,
    rope_theta: float,
    causal: bool,
    implementation: AttentionImplementation,
) -> tuple[TransformerCarry, jax.Array]:
    """Apply RoPE attention to projected Q/K/V and update the sequence cache.

    Queries are [batch,time,query_heads,head_dim]; keys and values are
    [batch,time,kv_heads,head_dim], all with the same floating dtype and
    nonempty dimensions. Configuration constraints match TransformerBlock.
    Optional int32 [batch] x_len gives valid prefix lengths in [0, time].
    Padding produces zero output and does not update the cache.
    Return the updated cache and [batch,time,query_heads,head_dim] output.
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
        causal=causal,
        implementation=implementation,
        dtype=dtype,
    )
    fresh = carry is None
    if carry is None:
        carry = TransformerCarry(
            jnp.zeros(sc["BCKF"], dtype),
            jnp.zeros(sc["BCKF"], dtype),
            jnp.zeros(sc["B"], jnp.int32),
        )
    if not isinstance(carry, TransformerCarry):
        raise TypeError("carry must be a TransformerCarry")
    sc.check((carry.key, carry.value), "BCKF", dtype)
    sc.check(carry.position, "B", jnp.int32)
    native_attention = fresh and x_len is None
    if x_len is None:
        x_len = jnp.full(sc["B"], steps, jnp.int32)
    sc.check(x_len, "B", jnp.int32)
    _check_range(x_len, steps, "x_len must be between 0 and the input sequence length")
    _check_range(carry.position, max_seq_len, "Invalid cache position for max_seq_len cache capacity")
    next_position = carry.position + x_len
    sc.check(next_position, "B", jnp.int32)
    _check_range(next_position, max_seq_len, "Sequence exceeds max_seq_len cache capacity")

    index = jnp.arange(steps, dtype=jnp.int32)[None, :]
    valid_tokens = index < x_len[:, None]
    positions = jnp.where(valid_tokens, carry.position[:, None] + index, 0)
    sc.check(valid_tokens, "BT", jnp.bool_)
    sc.check(positions, "BT", jnp.int32)
    query, key, value = (jnp.where(valid_tokens[..., None, None], v, 0) for v in (query, key, value))
    query, key = _rope(query, positions, rope_theta), _rope(key, positions, rope_theta)
    slots = jnp.arange(max_seq_len, dtype=jnp.int32)[None, :]
    old_valid = slots < carry.position[:, None]
    sc.check(old_valid, "BC", jnp.bool_)

    mask = None
    if native_attention:
        # Let the backend handle causality without a dense mask.
        keys, values = key, value
    else:
        if fresh:
            keys, values, key_positions, key_valid = key, value, positions, valid_tokens
        else:
            old_key, old_value = (jnp.where(old_valid[..., None, None], v, 0) for v in (carry.key, carry.value))
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
    source = slots - carry.position[:, None]
    sc.check(source, "BC", jnp.int32)
    gather = jnp.clip(source, 0, steps - 1)[..., None, None]
    new_key, new_value = (jnp.take_along_axis(v, gather, axis=1) for v in (key, value))
    sc.check((new_key, new_value), "BCKF", dtype)
    written = ((source >= 0) & (source < x_len[:, None]))[..., None, None]
    sc.check(written, "BCUU", jnp.bool_)
    carry = TransformerCarry(
        jnp.where(written, new_key, carry.key),
        jnp.where(written, new_value, carry.value),
        next_position,
    )
    sc.check((carry.key, carry.value), "BCKF", dtype)
    sc.check(carry.position, "B", jnp.int32)
    sc.check(attended, "BTHF", dtype)
    return carry, attended


class TransformerBlock(nn.Module):
    """Llama 3 pre-RMSNorm attention and SwiGLU MLP with residual connections.

    MLP width is ``round(dim * mlp_expansion)``; expansion defaults to 2.
    ``num_kv_heads=None`` gives ordinary multi-head attention.
    ``causal=False`` allows attention to future tokens in the current chunk,
    within the sequence. Defaults to causal attention.

    Fewer KV heads enable grouped-query attention (one gives multi-query
    attention). ``dim`` must be divisible by ``num_heads``, and the query
    head count must be divisible by the KV head count. Head width must be even
    for RoPE.
    """

    dim: int
    mlp_expansion: float = 2.0
    num_heads: int = 8
    num_kv_heads: int | None = None
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_epsilon: float = 1e-5
    attention_implementation: AttentionImplementation = "xla"
    dtype: jax.typing.DTypeLike = jnp.float32
    initializer_range: float = 0.02
    causal: bool = True

    @nn.nowrap
    def _mlp_width(self) -> int:
        return _expanded_mlp_width(self.dim, self.mlp_expansion)

    @nn.nowrap
    def _dimensions(self) -> tuple[int, int]:
        for name in ("dim", "num_heads"):
            _positive_integer(getattr(self, name), name)
        self._mlp_width()
        kv_heads = self.num_heads if self.num_kv_heads is None else self.num_kv_heads
        chex.assert_is_divisible(self.dim, self.num_heads)
        head_dim = self.dim // self.num_heads
        _check_attention_config(
            num_heads=self.num_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            max_seq_len=self.max_seq_len,
            rope_theta=self.rope_theta,
            causal=self.causal,
            implementation=self.attention_implementation,
            dtype=self.dtype,
        )
        for name in ("initializer_range", "norm_epsilon"):
            if not 0 < getattr(self, name) < math.inf:
                raise ValueError(f"{name} must be positive and finite")
        return kv_heads, head_dim

    @nn.nowrap
    def initial_carry(self, batch_size: int) -> TransformerCarry:
        """Allocate a fixed-size empty cache without initializing parameters."""
        kv_heads, head_dim = self._dimensions()
        _positive_integer(batch_size, "batch_size")
        sc = ShapeChecker(B=batch_size, C=self.max_seq_len, K=kv_heads, F=head_dim)
        carry = TransformerCarry(
            jnp.zeros(sc["BCKF"], self.dtype),
            jnp.zeros(sc["BCKF"], self.dtype),
            jnp.zeros(sc["B"], jnp.int32),
        )
        sc.check((carry.key, carry.value), "BCKF", self.dtype)
        sc.check(carry.position, "B", jnp.int32)
        return carry

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        x_len: jax.Array | None = None,
        carry: TransformerCarry | None = None,
    ) -> tuple[TransformerCarry, jax.Array]:
        """Map [batch,time,dim] to (updated KV cache, same-shaped output).

        Input dimensions must be nonempty.
        ``x_len`` is int32 [batch], in [0, time], defaulting to time.
        Only the left-aligned valid prefix updates history; right-padded
        outputs are zero. A zero length preserves that example's cache.
        """
        kv_heads, head_dim = self._dimensions()
        width = self._mlp_width()
        # B/T: batch/time, D: model width, H/K: query/KV heads,
        # F: head width, C: cache capacity, I: MLP width.
        sc = ShapeChecker(D=self.dim, H=self.num_heads, K=kv_heads, F=head_dim, C=self.max_seq_len, I=width)
        sc.check(x, "BTD")
        chex.assert_type(x, jnp.floating)
        _positive_integer(x.shape[0], "batch_size")
        _positive_integer(x.shape[1], "sequence_length")
        if carry is not None:
            if not isinstance(carry, TransformerCarry):
                raise TypeError("carry must be a TransformerCarry")
            sc.check((carry.key, carry.value), "BCKF", self.dtype)
            sc.check(carry.position, "B", jnp.int32)
        if x_len is not None:
            sc.check(x_len, "B", jnp.int32)

        valid_tokens = None
        if x_len is not None:
            valid_tokens = jnp.arange(x.shape[1])[None, :] < x_len[:, None]
            sc.check(valid_tokens, "BT", jnp.bool_)
            x = jnp.where(valid_tokens[..., None], x, 0)

        # Attention projections use [batch,time,heads,head_dim].
        residual = x
        x = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype, name="norm")(x)
        sc.check(x, "BTD", self.dtype)
        kernel_init = nn.initializers.normal(stddev=self.initializer_range)
        query = nn.Dense(self.dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init, name="q_proj")(x)
        key = nn.Dense(kv_heads * head_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init, name="k_proj")(x)
        value = nn.Dense(kv_heads * head_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init, name="v_proj")(
            x
        )
        sc.check(query, "BTD", self.dtype)
        query = query.reshape(sc["BTHF"])
        key, value = (v.reshape(sc["BTKF"]) for v in (key, value))
        sc.check(query, "BTHF", self.dtype)
        sc.check((key, value), "BTKF", self.dtype)
        carry, attended = _attention(
            query,
            key,
            value,
            carry,
            x_len,
            max_seq_len=self.max_seq_len,
            rope_theta=self.rope_theta,
            causal=self.causal,
            implementation=self.attention_implementation,
        )
        attended = attended.reshape(sc["BTD"])
        y = nn.Dense(self.dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init, name="out_proj")(attended)
        sc.check(y, "BTD", self.dtype)
        x = residual + y
        residual_dtype = jnp.result_type(residual.dtype, self.dtype)
        sc.check(x, "BTD", residual_dtype)
        y = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype, name="norm2")(x)
        sc.check(y, "BTD", self.dtype)
        gate = nn.Dense(width, use_bias=False, dtype=self.dtype, kernel_init=kernel_init, name="gate_proj")(y)
        value = nn.Dense(width, use_bias=False, dtype=self.dtype, kernel_init=kernel_init, name="up_proj")(y)
        sc.check((gate, value), "BTI", self.dtype)
        y = nn.silu(gate) * value
        sc.check(y, "BTI", self.dtype)
        y = nn.Dense(self.dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init, name="down_proj")(y)
        sc.check(y, "BTD", self.dtype)
        output = x + y
        if valid_tokens is not None:
            output = jnp.where(valid_tokens[..., None], output, 0)
        sc.check(output, "BTD", residual_dtype)
        return carry, output

    def step(
        self,
        x: jax.Array,
        x_len: jax.Array | None = None,
        carry: TransformerCarry | None = None,
    ) -> tuple[TransformerCarry, jax.Array]:
        """Process [batch,dim] with optional int32 [batch] x_len of 0 or 1.

        Zero lengths skip the example and return zero output. Requires causal=True.
        """
        if not self.causal:
            raise ValueError("step() requires causal=True; use __call__() for bidirectional attention")
        sc = ShapeChecker(D=self.dim)
        sc.check(x, "BD")
        chex.assert_type(x, jnp.floating)
        if x_len is not None:
            sc.check(x_len, "B", jnp.int32)
        carry, y = self(x[:, None], x_len, carry)
        output = y[:, 0]
        sc.check(output, "BD", jnp.result_type(x.dtype, self.dtype))
        return carry, output


class TransformerStack(nn.Module):
    """Llama 3 decoder backbone with independent pre-RMSNorm/SwiGLU layers.

    Sequences are batch-major; optional int32 x_len has shape [batch]. Carry is a tuple of one
    TransformerCarry per layer. MLP width is ``round(dim * mlp_expansion)``,
    with a default expansion of 2, shared by every block.
    Final RMSNorm defaults on; residual additions
    preserve the operands' normal dtype promotion rules.
    All projections use normal initialization with standard deviation
    ``initializer_range`` (default 0.02), independent of depth.
    ``causal`` controls attention in every block and defaults to True.
    Non-causal attention sees the current chunk and cached history within
    each sequence; results depend on chunk boundaries.
    """

    dim: int
    num_layers: int
    num_heads: int = 8
    num_kv_heads: int | None = None
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    mlp_expansion: float = 2.0
    norm_epsilon: float = 1e-5
    final_norm: bool = True
    initializer_range: float = 0.02
    attention_implementation: AttentionImplementation = "xla"
    dtype: jax.typing.DTypeLike = jnp.float32
    causal: bool = True

    @nn.nowrap
    def _mlp_width(self) -> int:
        for name in ("dim", "num_layers"):
            _positive_integer(getattr(self, name), name)
        if not 0 < self.norm_epsilon < math.inf:
            raise ValueError("norm_epsilon must be positive and finite")
        return _expanded_mlp_width(self.dim, self.mlp_expansion)

    @nn.nowrap
    def _make_block(self) -> TransformerBlock:
        block = TransformerBlock(
            dim=self.dim,
            mlp_expansion=self.mlp_expansion,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            max_seq_len=self.max_seq_len,
            rope_theta=self.rope_theta,
            norm_epsilon=self.norm_epsilon,
            attention_implementation=self.attention_implementation,
            dtype=self.dtype,
            initializer_range=self.initializer_range,
            causal=self.causal,
            parent=None,
        )
        block._dimensions()
        return block

    def setup(self) -> None:
        self._mlp_width()
        self.layers = tuple(self._make_block().clone(parent=self, name=f"layers_{i}") for i in range(self.num_layers))
        if self.final_norm:
            self.norm_f = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)

    @nn.nowrap
    def initial_carry(self, batch_size: int) -> TransformerStackCarry:
        """Allocate independent caches for all layers without parameter init."""
        self._mlp_width()
        block = self._make_block()
        return tuple(block.initial_carry(batch_size) for _ in range(self.num_layers))

    def __call__(
        self,
        x: jax.Array,
        x_len: jax.Array | None = None,
        carry: TransformerStackCarry | None = None,
    ) -> tuple[TransformerStackCarry, jax.Array]:
        """Map [batch,time,dim] to (per-layer KV caches, output).

        Optional int32 x_len[batch] counts valid prefix tokens in this chunk.
        Right padding returns zero output and leaves cached history unchanged.
        """
        sc = ShapeChecker(D=self.dim)
        sc.check(x, "BTD")
        chex.assert_type(x, jnp.floating)
        if carry is not None and (not isinstance(carry, tuple) or len(carry) != self.num_layers):
            raise ValueError("carry must be a tuple with one TransformerCarry per layer")
        if x_len is not None:
            sc.check(x_len, "B", jnp.int32)
        next_carry = []
        for i, layer in enumerate(self.layers):
            state = None if carry is None else carry[i]
            if carry is not None and not isinstance(state, TransformerCarry):
                raise TypeError("each layer carry must be a TransformerCarry")
            state, x = layer(x, x_len, state)
            next_carry.append(state)
        if self.final_norm:
            x = self.norm_f(x)
        output = x.astype(self.dtype)
        sc.check(output, "BTD", self.dtype)
        return tuple(next_carry), output

    def step(
        self,
        x: jax.Array,
        x_len: jax.Array | None = None,
        carry: TransformerStackCarry | None = None,
    ) -> tuple[TransformerStackCarry, jax.Array]:
        """Process [batch,dim] with optional int32 [batch] x_len of 0 or 1.

        Zero lengths skip the example and return zero output. Requires causal=True.
        """
        if not self.causal:
            raise ValueError("step() requires causal=True; use __call__() for bidirectional attention")
        sc = ShapeChecker(D=self.dim)
        sc.check(x, "BD")
        chex.assert_type(x, jnp.floating)
        if x_len is not None:
            sc.check(x_len, "B", jnp.int32)
        carry, y = self(x[:, None], x_len, carry)
        output = y[:, 0]
        sc.check(output, "BD", self.dtype)
        return carry, output
