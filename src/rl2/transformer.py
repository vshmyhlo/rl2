"""Llama 3 decoder backbone with batch-major sequences and explicit KV caches.

The internal block combines RoPE attention with pre-RMSNorm and SwiGLU
residual layers; a shared internal stack repeats it and adds a final RMSNorm.
``ARTransformer`` implements the autoregressive sequence/carry contract;
``BDTransformer`` exposes bidirectional sequence outputs without carry.
Both specialized stacks require explicit ``x_len`` arrays for sequence calls.
RoPE can be disabled with ``use_rope=False``. ``BDTransformer`` accepts an
optional tuple of additive attention biases, one per layer, supplied by callers.
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

``ARTransformer`` uses causal attention and maintains an explicit KV cache.
Its fixed-size cache retains all valid tokens up to ``max_seq_len``; exceeding
that capacity raises an error, including under JIT. Sequence, chunk, and step
calls have identical semantics. Prefill computes attention in parallel, without
a token-by-token attention scan. Gradients flow through supplied caches unless
the caller applies ``jax.lax.stop_gradient``.

``BDTransformer`` attends bidirectionally over each complete valid sequence.
Its sequence call accepts no carry and returns only the output array.

Sequence calls require int32 ``x_len[batch]`` to delimit each left-aligned
valid prefix; remaining input tokens are right padding. Lengths must be between
zero and the input length. Padded outputs are zero. Only valid tokens advance
an autoregressive cache and its RoPE positions; omit carry to start fresh.
Autoregressive steps require a boolean ``x_active[batch]`` mask immediately after
``x``. Inactive examples return zero output and preserve their carry.

``attention_implementation="xla"`` is portable (the default). Select ``"cudnn"``
with float16/bfloat16 and a supported NVIDIA GPU for JAX's cuDNN fused attention.
Backend shape/device restrictions are reported by JAX, without silent fallback.
Sequence lengths handle padding; explicit masks handle causal offsets in cached chunks.
Masked cuDNN calls pad odd sequence lengths for its backward-pass constraints.

Example::

    model = ARTransformer(dim=256, num_layers=4, num_heads=8,
                             num_kv_heads=2, max_seq_len=1024)
    x = jnp.zeros((8, 16, 256))
    x_len = jnp.full((8,), 16, jnp.int32)
    variables = model.init(jax.random.key(0), x, x_len)
    carry, y = model.apply(variables, x, x_len)
    carry, next_y = model.apply(variables, x[:, 0], jnp.ones((8,), jnp.bool_), carry, method=model.step)
"""

import math
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from rl2 import attention as _attention
from rl2.sequence_model import ARSequenceModel, BDSequenceModel
from rl2.shape_checker import ShapeChecker

__all__ = ["ARTransformer", "BDTransformer", "TransformerCarry", "TransformerStackCarry"]


class TransformerCarry(NamedTuple):
    """Batch-leading KV cache and next RoPE position within each sequence.

    key/value: [B,max_seq_len,num_kv_heads,headdim], in projection dtype.
    position: [B], int32. Slot ``p`` stores position ``p``.
    Keys are already rotated when use_rope=True. Unused slots start at zero.
    """

    key: jax.Array
    value: jax.Array
    position: jax.Array


type TransformerStackCarry = tuple[TransformerCarry, ...]
type LayerAttentionBiases = tuple[jax.Array, ...]


def _positive_integer(value: int, name: str) -> None:
    chex.assert_scalar_positive(value, custom_message=f"{name} must be positive")


def _expanded_mlp_width(dim: int, mlp_expansion: float) -> int:
    """Scale the model dimension and round to the nearest integer width."""
    _positive_integer(dim, "dim")
    if not 0 < mlp_expansion < math.inf:
        raise ValueError("mlp_expansion must be positive and finite")
    width = round(dim * mlp_expansion)
    _positive_integer(width, "MLP width")
    return width


class _TransformerBlock(nn.Module):
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
    attention_implementation: _attention.AttentionType = "xla"
    dtype: jax.typing.DTypeLike = jnp.float32
    initializer_range: float = 0.02
    causal: bool = True
    use_rope: bool = True

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
        _attention._check_attention_config(
            num_heads=self.num_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            max_seq_len=self.max_seq_len,
            rope_theta=self.rope_theta,
            implementation=self.attention_implementation,
            dtype=self.dtype,
            use_rope=self.use_rope,
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
        *,
        bias: jax.Array | None = None,
    ) -> tuple[TransformerCarry, jax.Array]:
        """Map [batch,time,dim] to (updated KV cache, same-shaped output).

        Input dimensions must be nonempty.
        ``x_len`` is int32 [batch], in [0, time], defaulting to time.
        Only the left-aligned valid prefix updates history; right-padded
        outputs are zero. A zero length preserves that example's cache.
        Optional floating bias is [batch,num_heads,time,key_length], with
        key_length=time for fresh calls and max_seq_len for cached calls.
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
            sc.check((carry.key, carry.value), "BCKF", self.dtype)
            sc.check(carry.position, "B", jnp.int32)
        if x_len is not None:
            sc.check(x_len, "B", jnp.int32)
        if bias is not None:
            sc.check(bias, "BHTT" if carry is None else "BHTC")
            chex.assert_type(bias, jnp.floating)

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
        attention_state, attended = _attention.attention(
            query,
            key,
            value,
            carry,
            x_len,
            max_seq_len=self.max_seq_len,
            rope_theta=self.rope_theta,
            causal=self.causal,
            implementation=self.attention_implementation,
            bias=bias,
            use_rope=self.use_rope,
        )
        carry = TransformerCarry(*attention_state)
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
        x_active: jax.Array,
        carry: TransformerCarry | None = None,
    ) -> tuple[TransformerCarry, jax.Array]:
        """Process [batch,dim] with a required boolean x_active[batch] mask.

        Inactive examples return zero output and preserve carry.
        Requires causal=True.
        """
        if not self.causal:
            raise ValueError("step() requires causal=True; use __call__() for bidirectional attention")
        sc = ShapeChecker(D=self.dim)
        sc.check(x, "BD")
        chex.assert_type(x, jnp.floating)
        sc.check(x_active, "B", jnp.bool_)
        x_len = x_active.astype(jnp.int32)
        sc.check(x_len, "B", jnp.int32)
        carry, y = self(x[:, None], x_len, carry)
        output = y[:, 0]
        sc.check(output, "BD", jnp.result_type(x.dtype, self.dtype))
        return carry, output


class _TransformerStack(nn.Module):
    """Shared Llama 3 layers and computation without a public decoding interface.

    MLP width is ``round(dim * mlp_expansion)``,
    with a default expansion of 2, shared by every block.
    Final RMSNorm defaults on; residual additions
    preserve the operands' normal dtype promotion rules.
    All projections use normal initialization with standard deviation
    ``initializer_range`` (default 0.02), independent of depth.
    ``causal`` controls attention in every block and defaults to True.
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
    attention_implementation: _attention.AttentionType = "xla"
    dtype: jax.typing.DTypeLike = jnp.float32
    causal: bool = True
    use_rope: bool = True

    @nn.nowrap
    def _mlp_width(self) -> int:
        for name in ("dim", "num_layers"):
            _positive_integer(getattr(self, name), name)
        if not 0 < self.norm_epsilon < math.inf:
            raise ValueError("norm_epsilon must be positive and finite")
        return _expanded_mlp_width(self.dim, self.mlp_expansion)

    @nn.nowrap
    def _make_block(self) -> _TransformerBlock:
        block = _TransformerBlock(
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
            use_rope=self.use_rope,
            parent=None,
        )
        block._dimensions()
        return block

    def setup(self) -> None:
        self._mlp_width()
        self.layers = tuple(self._make_block().clone(parent=self, name=f"layers_{i}") for i in range(self.num_layers))
        if self.final_norm:
            self.norm_f = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)

    def _forward(
        self,
        x: jax.Array,
        x_len: jax.Array | None = None,
        carry: TransformerStackCarry | None = None,
        *,
        biases: LayerAttentionBiases | None = None,
    ) -> tuple[TransformerStackCarry, jax.Array]:
        """Map [batch,time,dim] to (per-layer KV caches, output).

        Optional int32 x_len[batch] counts valid prefix tokens in this chunk.
        Right padding returns zero output and leaves cached history unchanged.
        Optional biases contains one floating [batch,num_heads,time,key_length]
        array per layer. key_length is time without carry, else max_seq_len.
        """
        sc = ShapeChecker(D=self.dim, H=self.num_heads, C=self.max_seq_len)
        sc.check(x, "BTD")
        chex.assert_type(x, jnp.floating)
        if carry is not None and len(carry) != self.num_layers:
            raise ValueError("carry must contain one TransformerCarry per layer")
        if biases is not None:
            if len(biases) != self.num_layers:
                raise ValueError("biases must contain one attention bias per layer")
            sc.check(biases, "BHTT" if carry is None else "BHTC")
            chex.assert_type(biases, jnp.floating)
        if x_len is not None:
            sc.check(x_len, "B", jnp.int32)
        next_carry = []
        for i, layer in enumerate(self.layers):
            state = None if carry is None else carry[i]
            state, x = layer(x, x_len, state, bias=None if biases is None else biases[i])
            next_carry.append(state)
        if self.final_norm:
            x = self.norm_f(x)
        output = x.astype(self.dtype)
        sc.check(output, "BTD", self.dtype)
        return tuple(next_carry), output


class Transformer(_TransformerStack):
    """Internal stack exposing optional lengths and per-layer KV caches.

    Sequences are batch-major; optional int32 x_len has shape [batch].
    Carry is a tuple of one layer cache per layer. Non-causal attention sees
    the current chunk and cached history; its results depend on chunk boundaries.
    """

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
        """Process [batch,time,dim], returning per-layer caches and output."""
        return self._forward(x, x_len, carry)

    def step(
        self,
        x: jax.Array,
        x_active: jax.Array,
        carry: TransformerStackCarry | None = None,
    ) -> tuple[TransformerStackCarry, jax.Array]:
        """Process [batch,dim] with a required boolean x_active[batch] mask.

        Inactive examples return zero output and preserve carry.
        Requires causal=True.
        """
        if not self.causal:
            raise ValueError("step() requires causal=True; use __call__() for bidirectional attention")
        sc = ShapeChecker(D=self.dim)
        sc.check(x, "BD")
        chex.assert_type(x, jnp.floating)
        sc.check(x_active, "B", jnp.bool_)
        x_len = x_active.astype(jnp.int32)
        sc.check(x_len, "B", jnp.int32)
        carry, y = self(x[:, None], x_len, carry)
        output = y[:, 0]
        sc.check(output, "BD", self.dtype)
        return carry, output


class ARTransformer(Transformer, ARSequenceModel[TransformerStackCarry]):
    """Autoregressive stack with sequence lengths, step activity masks, and KV carry.

    Full sequences, chunks, and repeated steps produce equivalent outputs and
    final carry, whether starting fresh or continuing supplied history.
    ``causal`` must remain True. Parameter names match ``Transformer``.
    """

    def setup(self) -> None:
        if not self.causal:
            raise ValueError("ARTransformer requires causal=True")
        super().setup()

    def __call__(
        self,
        x: jax.Array,
        x_len: jax.Array,
        carry: TransformerStackCarry | None = None,
    ) -> tuple[TransformerStackCarry, jax.Array]:
        """Process [batch,time,dim] with required int32 [batch] prefix lengths."""
        sc = ShapeChecker(D=self.dim)
        sc.check(x, "BTD")
        chex.assert_type(x, jnp.floating)
        sc.check(x_len, "B", jnp.int32)
        carry, output = super().__call__(x, x_len, carry)
        sc.check(output, "BTD", self.dtype)
        return carry, output


class BDTransformer(_TransformerStack, BDSequenceModel):
    """Bidirectional stack with required valid lengths and array-only output.

    Every call processes a fresh sequence, attending only to its valid prefix.
    ``causal`` must remain False. Carry is neither accepted nor returned;
    there is no single-step decoding or carry initialization interface.
    Parameter names match ``Transformer``.
    """

    causal: bool = False

    def setup(self) -> None:
        if self.causal:
            raise ValueError("BDTransformer requires causal=False")
        super().setup()

    def __call__(self, x: jax.Array, x_len: jax.Array, *, biases: LayerAttentionBiases | None = None) -> jax.Array:
        """Process [batch,time,dim]; int32 [batch] lengths delimit valid prefixes.

        Optional biases contains one floating [batch,num_heads,time,time] array
        per layer, added before attention softmax. Biases cannot override padding.
        Callers own any learned bias parameters; the stack stays domain-agnostic.
        """
        sc = ShapeChecker(D=self.dim)
        sc.check(x, "BTD")
        chex.assert_type(x, jnp.floating)
        sc.check(x_len, "B", jnp.int32)
        _, output = self._forward(x, x_len, biases=biases)
        sc.check(output, "BTD", self.dtype)
        return output
