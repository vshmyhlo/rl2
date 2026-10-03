"""Causal RoPE transformer with the time-major, explicit-carry Mamba3 API.

``Transformer`` is the attention mixer; ``TransformerStack`` adds pre-RMSNorm,
SwiGLU residual blocks and a final RMSNorm. Neither includes embeddings or a
prediction head. Projections are bias-free, parameters/norm statistics are
float32, and ``dtype`` controls projections, outputs and cached keys/values.
The XLA float16 path evaluates attention in float32 for CPU portability.

The fixed-size ring cache implements sliding causal attention: each query sees
at most ``max_seq_len`` tokens, including itself. Set it to the maximum episode
length for full-context attention. RoPE positions continue across cache wraps
and restart on per-example ``episode_starts``. Sequence, chunk and step calls
have identical semantics; prefill computes attention in parallel, without a
token-by-token attention scan. Gradients flow through supplied caches unless
the caller applies ``jax.lax.stop_gradient``.

``attention_implementation="xla"`` is portable (the default). Select ``"cudnn"``
with float16/bfloat16 and a supported NVIDIA GPU for JAX's cuDNN fused attention.
Backend shape/device restrictions are reported by JAX, without silent fallback.
Fresh, unsegmented prefill uses the native causal/window mask; cached or packed
sequences use an explicit boolean mask to handle offsets and episode resets.
Masked cuDNN calls pad odd sequence lengths for its backward-pass constraints.

Example::

    model = TransformerStack(d_model=256, num_layers=4, num_heads=8,
                             num_kv_heads=2, max_seq_len=1024)
    x = jnp.zeros((16, 8, 256))
    variables = model.init(jax.random.key(0), x)
    carry, y = model.apply(variables, x)
    carry, next_y = model.apply(variables, x[0], carry, method=model.step)
"""

import math
from typing import Literal, NamedTuple

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from rl2.block_stack import BlockStack

type AttentionImplementation = Literal["xla", "cudnn"]


class TransformerCarry(NamedTuple):
    """Batch-leading KV ring buffer and next RoPE position within each episode.

    key/value: [B,max_seq_len,num_kv_heads,headdim], in projection dtype.
    position: [B], int32. Slot ``p % max_seq_len`` stores position ``p``.
    Keys are already rotated. Unused slots are zero, including after resets.
    """

    key: jax.Array
    value: jax.Array
    position: jax.Array


type TransformerStackCarry = tuple[TransformerCarry, ...]


def _positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    chex.assert_scalar_positive(value)


def _rope(x: jax.Array, positions: jax.Array, theta: float) -> jax.Array:
    """Full-head RoPE with split-half pairs and float32 trigonometry."""
    chex.assert_rank(x, 4)
    chex.assert_type(x, jnp.floating)
    chex.assert_shape(positions, x.shape[:2])
    chex.assert_type(positions, jnp.int32)
    chex.assert_is_divisible(x.shape[-1], 2)
    chex.assert_scalar_positive(theta)
    half = x.shape[-1] // 2
    frequencies = theta ** (-jnp.arange(half, dtype=jnp.float32) / half)
    angles = positions.astype(jnp.float32)[..., None, None] * frequencies
    real, imag = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    return jnp.concatenate((real * cos - imag * sin, imag * cos + real * sin), axis=-1).astype(x.dtype)


class Transformer(nn.Module):
    """RoPE attention mixer; ``num_kv_heads=None`` gives ordinary multi-head attention.

    Fewer KV heads enable grouped-query attention (one gives multi-query
    attention). ``d_model`` must be divisible by ``num_heads``, and the query
    head count must be divisible by the KV head count. Head width must be even
    for RoPE.
    """

    d_model: int
    num_heads: int = 8
    num_kv_heads: int | None = None
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    attention_implementation: AttentionImplementation = "xla"
    dtype: jax.typing.DTypeLike = jnp.float32
    out_proj_init_scale: float = 1.0

    @nn.nowrap
    def _dimensions(self) -> tuple[int, int]:
        for name in ("d_model", "num_heads", "max_seq_len"):
            _positive_integer(getattr(self, name), name)
        kv_heads = self.num_heads if self.num_kv_heads is None else self.num_kv_heads
        _positive_integer(kv_heads, "num_kv_heads")
        chex.assert_is_divisible(self.d_model, self.num_heads)
        chex.assert_is_divisible(self.num_heads, kv_heads)
        head_dim = self.d_model // self.num_heads
        chex.assert_is_divisible(head_dim, 2)
        for name in ("rope_theta", "out_proj_init_scale"):
            if not 0 < getattr(self, name) < math.inf:
                raise ValueError(f"{name} must be positive and finite")
        if self.attention_implementation not in ("xla", "cudnn"):
            raise ValueError("attention_implementation must be 'xla' or 'cudnn'")
        dtype = jnp.dtype(self.dtype)
        if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
            raise ValueError("dtype must be float32, bfloat16, or float16")
        if self.attention_implementation == "cudnn" and dtype == jnp.float32:
            raise ValueError("cuDNN attention requires dtype=float16 or bfloat16")
        if self.attention_implementation == "cudnn":
            chex.assert_is_divisible(head_dim, 8)
        return kv_heads, head_dim

    @nn.nowrap
    def initial_carry(self, batch_size: int) -> TransformerCarry:
        """Allocate a fixed-size empty cache without initializing parameters."""
        kv_heads, head_dim = self._dimensions()
        _positive_integer(batch_size, "batch_size")
        shape = (batch_size, self.max_seq_len, kv_heads, head_dim)
        return TransformerCarry(
            jnp.zeros(shape, self.dtype), jnp.zeros(shape, self.dtype), jnp.zeros((batch_size,), jnp.int32)
        )

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        carry: TransformerCarry | None = None,
        episode_starts: jax.Array | None = None,
    ) -> tuple[TransformerCarry, jax.Array]:
        """Map [time,batch,d_model] to (updated KV cache, same-shaped output).

        A True entry in ``episode_starts[time,batch]`` discards all preceding
        history for that example before processing its current input.
        """
        kv_heads, head_dim = self._dimensions()
        chex.assert_shape(x, (None, None, self.d_model))
        chex.assert_type(x, jnp.floating)
        steps, batch = x.shape[:2]
        _positive_integer(batch, "batch_size")
        fresh = carry is None
        if carry is None:
            carry = self.initial_carry(batch)
        if not isinstance(carry, TransformerCarry):
            raise TypeError("carry must be a TransformerCarry")
        chex.assert_shape((carry.key, carry.value), (batch, self.max_seq_len, kv_heads, head_dim))
        chex.assert_type((carry.key, carry.value), self.dtype)
        chex.assert_shape(carry.position, (batch,))
        chex.assert_type(carry.position, jnp.int32)
        unsegmented = episode_starts is None
        if episode_starts is None:
            episode_starts = jnp.zeros((steps, batch), jnp.bool_)
        chex.assert_shape(episode_starts, (steps, batch))
        chex.assert_type(episode_starts, jnp.bool_)

        # Batch-major projections for jax.nn.dot_product_attention (BTNH).
        x = jnp.swapaxes(x, 0, 1)
        query = nn.Dense(self.d_model, use_bias=False, dtype=self.dtype, name="q_proj")(x)
        key = nn.Dense(kv_heads * head_dim, use_bias=False, dtype=self.dtype, name="k_proj")(x)
        value = nn.Dense(kv_heads * head_dim, use_bias=False, dtype=self.dtype, name="v_proj")(x)
        query = query.reshape((batch, steps, self.num_heads, head_dim))
        key, value = (v.reshape((batch, steps, kv_heads, head_dim)) for v in (key, value))
        if steps:
            starts = episode_starts.T
            index = jnp.arange(steps, dtype=jnp.int32)[None, :]
            # The virtual start of a continued episode is -carry.position.
            last_reset = jax.lax.associative_scan(
                jnp.maximum, jnp.where(starts, index, -carry.position[:, None]), axis=1
            )
            positions = index - last_reset
            query, key = _rope(query, positions, self.rope_theta), _rope(key, positions, self.rope_theta)
            slots = jnp.arange(self.max_seq_len, dtype=jnp.int32)[None, :]
            old_positions = carry.position[:, None] - 1
            old_positions = old_positions - (old_positions - slots) % self.max_seq_len

            mask, window = None, None
            native_causal = fresh and unsegmented
            if native_causal:
                # Let cuDNN handle causality without materializing a dense mask.
                window = (self.max_seq_len - 1, 0) if steps > self.max_seq_len else None
                keys, values = key, value
            else:
                segments = jnp.cumsum(starts, axis=1, dtype=jnp.int32)
                if fresh:
                    keys, values, key_positions, key_segments = key, value, positions, segments
                else:
                    keys, values = jnp.concatenate((carry.key, key), 1), jnp.concatenate((carry.value, value), 1)
                    key_positions = jnp.concatenate((old_positions, positions), 1)
                    key_segments = jnp.concatenate((jnp.zeros_like(old_positions), segments), 1)
                distance = positions[:, :, None] - key_positions[:, None, :]
                mask = (distance >= 0) & (distance < self.max_seq_len)
                mask &= key_positions[:, None, :] >= 0
                mask &= segments[:, :, None] == key_segments[:, None, :]
                mask = mask[:, None]
            queries = query
            if self.attention_implementation == "cudnn" and mask is not None:
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
            attention_dtype = (
                jnp.float32
                if self.attention_implementation == "xla" and jnp.dtype(self.dtype) == jnp.float16
                else self.dtype
            )
            attended = jax.nn.dot_product_attention(
                queries.astype(attention_dtype),
                keys.astype(attention_dtype),
                values.astype(attention_dtype),
                mask=mask,
                is_causal=native_causal,
                local_window_size=window,
                implementation=self.attention_implementation,
            )[:, :steps].astype(self.dtype)

            next_position = positions[:, -1] + 1
            # Gather the last write to each ring slot. Unlike a scatter with
            # duplicate indices, this is deterministic for chunks > cache size.
            slot_positions = next_position[:, None] - 1
            slot_positions = slot_positions - (slot_positions - slots) % self.max_seq_len
            source = steps - next_position[:, None] + slot_positions
            gather = jnp.clip(source, 0, steps - 1)[..., None, None]
            new_key, new_value = (jnp.take_along_axis(v, gather, axis=1) for v in (key, value))
            written, valid = (source >= 0)[..., None, None], (slot_positions >= 0)[..., None, None]
            carry = TransformerCarry(
                jnp.where(valid, jnp.where(written, new_key, carry.key), 0),
                jnp.where(valid, jnp.where(written, new_value, carry.value), 0),
                next_position,
            )
        else:
            attended = query
        attended = attended.reshape((batch, steps, self.d_model))
        out_init = nn.initializers.variance_scaling(self.out_proj_init_scale**2, "fan_in", "truncated_normal")
        y = nn.Dense(self.d_model, use_bias=False, dtype=self.dtype, kernel_init=out_init, name="out_proj")(attended)
        return carry, jnp.swapaxes(y, 0, 1)

    def step(
        self,
        x: jax.Array,
        carry: TransformerCarry | None = None,
        episode_starts: jax.Array | None = None,
    ) -> tuple[TransformerCarry, jax.Array]:
        """Process [batch,d_model] using the same parameters as sequence calls."""
        chex.assert_shape(x, (None, self.d_model))
        chex.assert_type(x, jnp.floating)
        if episode_starts is not None:
            chex.assert_shape(episode_starts, (x.shape[0],))
            chex.assert_type(episode_starts, jnp.bool_)
        starts = None if episode_starts is None else episode_starts[None]
        carry, y = self(x[None], carry, starts)
        return carry, y[0]


class _TransformerBlock(nn.Module):
    mixer: Transformer
    d_intermediate: int
    norm_epsilon: float
    residual_in_fp32: bool

    @nn.compact
    def __call__(
        self, x: jax.Array, carry: TransformerCarry | None, episode_starts: jax.Array | None
    ) -> tuple[TransformerCarry, jax.Array]:
        chex.assert_shape(x, (None, None, self.mixer.d_model))
        chex.assert_type(x, jnp.floating)
        dtype = self.mixer.dtype
        residual_dtype = jnp.float32 if self.residual_in_fp32 else dtype
        x = x.astype(residual_dtype)
        normalized = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=dtype, name="norm")(x)
        # The mixer validates carry and episode_starts.
        carry, y = self.mixer(normalized, carry, episode_starts)
        x = x + y.astype(residual_dtype)
        y = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=dtype, name="norm2")(x)
        gate = nn.Dense(self.d_intermediate, use_bias=False, dtype=dtype, name="gate_proj")(y)
        value = nn.Dense(self.d_intermediate, use_bias=False, dtype=dtype, name="up_proj")(y)
        y = nn.silu(gate) * value
        out_init = nn.initializers.variance_scaling(self.mixer.out_proj_init_scale**2, "fan_in", "truncated_normal")
        y = nn.Dense(self.mixer.d_model, use_bias=False, dtype=dtype, kernel_init=out_init, name="down_proj")(y)
        return carry, x + y.astype(residual_dtype)


class TransformerStack(BlockStack[TransformerStackCarry]):
    """Modern decoder backbone with independent pre-RMSNorm/SwiGLU layers.

    Same sequence/step/reset interface as Mamba3Stack. Carry is a tuple of one
    TransformerCarry per layer. The default MLP width is 8*d_model/3 rounded
    up to ``mlp_multiple_of``. Final RMSNorm and float32 residuals default on.
    Output projections initialize with depth scale 1/sqrt(2*num_layers).
    """

    num_heads: int = 8
    num_kv_heads: int | None = None
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    d_intermediate: int | None = None
    mlp_multiple_of: int = 128
    norm_epsilon: float = 1e-5
    residual_in_fp32: bool = True
    final_norm: bool = True
    rescale_prenorm_residual: bool = True
    attention_implementation: AttentionImplementation = "xla"
    dtype: jax.typing.DTypeLike = jnp.float32

    @nn.nowrap
    def _mlp_width(self) -> int:
        for name in ("d_model", "num_layers", "mlp_multiple_of"):
            _positive_integer(getattr(self, name), name)
        if not 0 < self.norm_epsilon < math.inf:
            raise ValueError("norm_epsilon must be positive and finite")
        width = int(8 * self.d_model / 3) if self.d_intermediate is None else self.d_intermediate
        _positive_integer(width, "d_intermediate")
        return (width + self.mlp_multiple_of - 1) // self.mlp_multiple_of * self.mlp_multiple_of

    @nn.nowrap
    def _make_mixer(self) -> Transformer:
        self._mlp_width()
        mixer = Transformer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            max_seq_len=self.max_seq_len,
            rope_theta=self.rope_theta,
            attention_implementation=self.attention_implementation,
            dtype=self.dtype,
            out_proj_init_scale=1 / math.sqrt(2 * self.num_layers) if self.rescale_prenorm_residual else 1.0,
            parent=None,
        )
        mixer._dimensions()
        return mixer

    def setup(self) -> None:
        width = self._mlp_width()
        self.layers = tuple(
            _TransformerBlock(self._make_mixer(), width, self.norm_epsilon, self.residual_in_fp32, name=f"layers_{i}")
            for i in range(self.num_layers)
        )
        if self.final_norm:
            self.norm_f = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)

    @nn.nowrap
    def initial_carry(self, batch_size: int) -> TransformerStackCarry:
        """Allocate independent caches for all layers without parameter init."""
        mixer = self._make_mixer()
        return tuple(mixer.initial_carry(batch_size) for _ in range(self.num_layers))

    def __call__(
        self,
        x: jax.Array,
        carry: TransformerStackCarry | None = None,
        episode_starts: jax.Array | None = None,
    ) -> tuple[TransformerStackCarry, jax.Array]:
        """Map [time,batch,d_model] to (per-layer KV caches, output)."""
        chex.assert_shape(x, (None, None, self.d_model))
        chex.assert_type(x, jnp.floating)
        if carry is not None and (not isinstance(carry, tuple) or len(carry) != self.num_layers):
            raise ValueError("carry must be a tuple with one TransformerCarry per layer")
        if episode_starts is not None:
            chex.assert_shape(episode_starts, x.shape[:2])
            chex.assert_type(episode_starts, jnp.bool_)
        next_carry = []
        for i, layer in enumerate(self.layers):
            state = None if carry is None else carry[i]
            if carry is not None and not isinstance(state, TransformerCarry):
                raise TypeError("each layer carry must be a TransformerCarry")
            state, x = layer(x, state, episode_starts)
            next_carry.append(state)
        if self.final_norm:
            x = self.norm_f(x)
        return tuple(next_carry), x.astype(self.dtype)
