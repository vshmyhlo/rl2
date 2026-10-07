# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Flax Linen GDN-2 mixer, residual backbone, and recurrent language model.

JAX adaptation of NVlabs/GatedDeltaNet-2, revision
a5552fe3c67e0ebc7ef1220df68ae8896ec62d56 (lit_gpt/gdn2.py).
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Distributed under the NVIDIA Source Code License-NC; see LICENSE in this directory.
"""

import math
from dataclasses import dataclass
from typing import Literal, NamedTuple

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from rl2.gdn2.core import delta_rule_step
from rl2.gdn2.masking import prefix_mask
from rl2.sequence_model import ARSequenceModel
from rl2.shape_checker import ShapeChecker


@dataclass(frozen=True)
class GatedDeltaNet2Config:
    """Mixer dimensions and precision; defaults match the upstream mixer.

    hidden_size need not equal num_heads * head_dim. Parameters and carry
    remain float32; dtype controls dense projections and returned activations.
    """

    hidden_size: int = 2048
    head_dim: int = 128
    num_heads: int = 16
    num_v_heads: int | None = None
    expand_v: float = 1.0
    use_short_conv: bool = True
    conv_size: int = 4
    conv_bias: bool = False
    allow_neg_eigval: bool = False
    norm_eps: float = 1e-5
    dtype: jax.typing.DTypeLike = jnp.float32

    def __post_init__(self) -> None:
        for name in ("hidden_size", "head_dim", "num_heads", "conv_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.value_heads < self.num_heads or self.value_heads % self.num_heads:
            raise ValueError("num_v_heads must be a positive multiple of num_heads")
        width = self.head_dim * self.expand_v
        if not math.isfinite(width) or width < 1 or not math.isclose(width, round(width), rel_tol=1e-5):
            raise ValueError("head_dim * expand_v must be a positive integer")
        if not 0 < self.norm_eps < math.inf:
            raise ValueError("norm_eps must be positive and finite")
        if jnp.dtype(self.dtype) not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
            raise ValueError("dtype must be float32, bfloat16, or float16")

    @property
    def value_heads(self) -> int:
        return self.num_heads if self.num_v_heads is None else self.num_v_heads

    @property
    def value_head_dim(self) -> int:
        return round(self.head_dim * self.expand_v)


class GatedDeltaNet2Carry(NamedTuple):
    """Float32 state [B,Hv,K,V] and oldest-first convolution histories.

    q/k histories are [B,C-1,Hk*K], v is [B,C-1,Hv*V]. With convolutions
    disabled the history length is zero. All leaves use batch as the first axis.
    """

    state: jax.Array
    q: jax.Array
    k: jax.Array
    v: jax.Array


type GatedDeltaNet2Backend = Literal["jax", "triton"]


type GatedDeltaNet2StackCarry = tuple[GatedDeltaNet2Carry, ...]
type MixerInputs = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


def _initial_carry(config: GatedDeltaNet2Config, batch_size: int) -> GatedDeltaNet2Carry:
    chex.assert_scalar_positive(batch_size)
    history = config.conv_size - 1 if config.use_short_conv else 0
    key_width = config.num_heads * config.head_dim
    value_width = config.value_heads * config.value_head_dim
    return GatedDeltaNet2Carry(
        jnp.zeros((batch_size, config.value_heads, config.head_dim, config.value_head_dim), jnp.float32),
        jnp.zeros((batch_size, history, key_width), jnp.float32),
        jnp.zeros((batch_size, history, key_width), jnp.float32),
        jnp.zeros((batch_size, history, value_width), jnp.float32),
    )


def _select(x_active: jax.Array, yes: jax.Array, no: jax.Array) -> jax.Array:
    sc = ShapeChecker()
    sc.check(x_active, "B", jnp.bool_)
    # Shapes vary between recurrence and convolution histories.
    names = "B" + "XYZ"[: yes.ndim - 1]
    sc.check([yes, no], names, jnp.float32)
    return jnp.where(x_active.reshape((x_active.shape[0],) + (1,) * (yes.ndim - 1)), yes, no)


def _linear_init(key: jax.Array, shape: tuple[int, ...], dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
    return nn.initializers.variance_scaling(2**-5, "fan_avg", "uniform")(key, shape, dtype)


def _rate_init(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    return jnp.log(jax.random.uniform(key, shape, dtype=jnp.float32, minval=1.0, maxval=16.0))


def _dt_init(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    dt = jnp.exp(jax.random.uniform(key, shape, dtype=jnp.float32, minval=math.log(0.001), maxval=math.log(0.1)))
    return dt + jnp.log(-jnp.expm1(-dt))


def _sequence_mask(x: jax.Array, x_len: jax.Array, hidden_size: int) -> jax.Array:
    """Validate batch-first sequence inputs and construct their valid prefixes."""
    sc = ShapeChecker(D=hidden_size)
    sc.check(x, "BTD")
    chex.assert_type(x, jnp.floating)
    return prefix_mask(x_len, x.shape[0], x.shape[1])


class GatedDeltaNet2(nn.Module, ARSequenceModel[GatedDeltaNet2Carry]):
    """Batch-first autoregressive token mixer with explicit recurrent state.

    ``__call__(x, x_len, carry=None)`` accepts floating x [B,T,D] and required
    int32 prefix lengths [B], returning (carry, y [B,T,D]). ``step`` accepts
    x [B,D] and required boolean x_active [B]. Padding returns zero and
    preserves every carry leaf. No hidden mutable cache is used.
    backend="triton" enables optional GPU kernels with shared parameters/carry.
    """

    config: GatedDeltaNet2Config = GatedDeltaNet2Config()
    backend: GatedDeltaNet2Backend = "jax"

    @nn.nowrap
    def initial_carry(self, batch_size: int) -> GatedDeltaNet2Carry:
        return _initial_carry(self.config, batch_size)

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        x_len: jax.Array,
        carry: GatedDeltaNet2Carry | None = None,
    ) -> tuple[GatedDeltaNet2Carry, jax.Array]:
        valid = _sequence_mask(x, x_len, self.config.hidden_size)
        c = self.config
        sc = ShapeChecker(D=c.hidden_size, H=c.value_heads, K=c.head_dim, V=c.value_head_dim)
        sc.check(x, "BTD")
        chex.assert_type(x, jnp.floating)
        carry = self.initial_carry(x.shape[0]) if carry is None else carry
        sc.check(carry.state, "BHKV", jnp.float32)
        history = c.conv_size - 1 if c.use_short_conv else 0
        key_width, value_width = c.num_heads * c.head_dim, c.value_heads * c.value_head_dim
        conv_sc = ShapeChecker(B=x.shape[0], C=history, Q=key_width, W=value_width)
        conv_sc.check([carry.q, carry.k], "BCQ", jnp.float32)
        conv_sc.check(carry.v, "BCW", jnp.float32)
        # Padding values must not enter projections (including NaNs in padding).
        x = jnp.where(valid[..., None], x, 0).astype(c.dtype)

        def dense(inputs: jax.Array, width: int, name: str, bias: bool = False) -> jax.Array:
            sc = ShapeChecker(B=x.shape[0], T=x.shape[1], O=width)
            sc.check(inputs, "BTI", c.dtype)
            result = nn.Dense(width, use_bias=bias, dtype=c.dtype, kernel_init=_linear_init, name=name)(inputs)
            sc.check(result, "BTO", c.dtype)
            return result

        q, k, v = (
            dense(x, width, name).astype(jnp.float32)
            for width, name in ((key_width, "q_proj"), (key_width, "k_proj"), (value_width, "v_proj"))
        )
        f = dense(dense(x, c.value_head_dim, "f_proj_in"), key_width, "f_proj_out").astype(jnp.float32)
        rate = self.param("A_log", _rate_init, (c.num_heads,))
        dt_bias = self.param("dt_bias", _dt_init, (key_width,))
        log_decay = -jnp.repeat(jnp.exp(rate), c.head_dim) * jax.nn.softplus(f + dt_bias)
        erase = jax.nn.sigmoid(dense(x, key_width, "b_proj").astype(jnp.float32))
        write = jax.nn.sigmoid(dense(x, value_width, "w_proj").astype(jnp.float32))
        if c.allow_neg_eigval:
            erase = 2 * erase

        kernels: list[jax.Array] = []
        biases: list[jax.Array] = []
        if c.use_short_conv:
            # PyTorch Conv1d initialization, with [oldest,...,current] taps.
            def conv_init(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
                bound = 1 / math.sqrt(c.conv_size)
                return jax.random.uniform(key, shape, dtype=jnp.float32, minval=-bound, maxval=bound)

            for name, width in (("q", key_width), ("k", key_width), ("v", value_width)):
                kernels.append(self.param(f"{name}_conv_kernel", conv_init, (c.conv_size, width)))
                biases.append(
                    self.param(f"{name}_conv_bias", conv_init, (width,))
                    if c.conv_bias
                    else jnp.zeros((width,), jnp.float32)
                )

        def step(previous: GatedDeltaNet2Carry, inputs: MixerInputs) -> tuple[GatedDeltaNet2Carry, jax.Array]:
            qt, kt, vt, gt, bt, wt, x_active = inputs

            def preserve_padding(new: jax.Array, old: jax.Array) -> jax.Array:
                return _select(x_active, new, old)

            projections = [qt, kt, vt]
            histories = [previous.q, previous.k, previous.v]
            for i in range(3):
                if c.use_short_conv:
                    window = jnp.concatenate((histories[i], projections[i][:, None]), axis=1)
                    projections[i] = jnp.sum(window * kernels[i], axis=1) + biases[i]
                    histories[i] = window[:, 1:]
                projections[i] = jax.nn.silu(projections[i])
            qt, kt, vt = projections
            shape = (x.shape[0], c.num_heads, c.head_dim)
            qt, kt, gt, bt = (a.reshape(shape) for a in (qt, kt, gt, bt))
            # Exactly the epsilon placement of upstream recurrent kernels.
            qt = qt * jax.lax.rsqrt(jnp.sum(qt * qt, axis=-1, keepdims=True) + 1e-6) * c.head_dim**-0.5
            kt = kt * jax.lax.rsqrt(jnp.sum(kt * kt, axis=-1, keepdims=True) + 1e-6)
            repeats = c.value_heads // c.num_heads
            qt, kt, gt, bt = (jnp.repeat(a, repeats, axis=1) for a in (qt, kt, gt, bt))
            vt, wt = (a.reshape((x.shape[0], c.value_heads, c.value_head_dim)) for a in (vt, wt))
            state, output = delta_rule_step(previous.state, qt, kt, vt, gt, bt, wt)
            updated = GatedDeltaNet2Carry(state, *histories)
            updated = jax.tree.map(preserve_padding, updated, previous)
            output = _select(x_active, output, jnp.zeros_like(output))
            return updated, output

        if self.backend == "triton":
            from rl2.gdn2.triton_backend import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule, short_conv

            projections = [q, k, v]
            histories = [carry.q, carry.k, carry.v]
            for i in range(3):
                if c.use_short_conv:
                    histories[i], projections[i] = short_conv(
                        projections[i], x_len, histories[i], kernels[i], biases[i]
                    )
                else:
                    projections[i] = jax.nn.silu(projections[i])
            q, k, v = projections
            shape = (*x.shape[:2], c.num_heads, c.head_dim)
            q, k, log_decay, erase = (a.reshape(shape) for a in (q, k, log_decay, erase))
            q = q * jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6) * c.head_dim**-0.5
            k = k * jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
            repeats = c.value_heads // c.num_heads
            q, k, log_decay, erase = (jnp.repeat(a, repeats, axis=2) for a in (q, k, log_decay, erase))
            v, write = (a.reshape((*x.shape[:2], c.value_heads, c.value_head_dim)) for a in (v, write))
            # Zero decay/keys/values make padding an identity state transition.
            q, k, v, log_decay, erase, write = (
                jnp.where(valid[..., None, None], a, 0) for a in (q, k, v, log_decay, erase, write)
            )
            rule = fused_recurrent_gated_delta_rule if x.shape[1] == 1 else chunk_gated_delta_rule
            state, output = rule(q, k, v, log_decay, erase, write, carry.state)
            carry = GatedDeltaNet2Carry(state, *histories)
        else:
            inputs = tuple(a.swapaxes(0, 1) for a in (q, k, v, log_decay, erase, write, valid))
            carry, output = jax.lax.scan(step, carry, inputs)
            output = output.swapaxes(0, 1)
        sc.check(output, "BTHV", jnp.float32)
        gate = dense(dense(x, c.value_head_dim, "g_proj_in"), value_width, "g_proj_out", bias=True)
        gate = gate.astype(jnp.float32).reshape(output.shape)
        norm_weight = self.param("o_norm_scale", nn.initializers.ones, (c.value_head_dim,), jnp.float32)
        if self.backend == "triton":
            from rl2.gdn2.triton_backend import gated_rms_norm

            output = gated_rms_norm(output, gate, norm_weight, c.norm_eps)
        else:
            output = output * jax.lax.rsqrt(jnp.mean(output**2, axis=-1, keepdims=True) + c.norm_eps)
            output = output * norm_weight * jax.nn.silu(gate)
        y = dense(output.reshape((*x.shape[:2], value_width)).astype(c.dtype), c.hidden_size, "o_proj")
        y = jnp.where(valid[..., None], y, 0)
        sc.check(y, "BTD", c.dtype)
        return carry, y

    def step(
        self,
        x: jax.Array,
        x_active: jax.Array,
        carry: GatedDeltaNet2Carry | None = None,
    ) -> tuple[GatedDeltaNet2Carry, jax.Array]:
        sc = ShapeChecker(D=self.config.hidden_size)
        sc.check(x, "BD")
        chex.assert_type(x, jnp.floating)
        sc.check(x_active, "B", jnp.bool_)
        carry, y = self(x[:, None], x_active.astype(jnp.int32), carry)
        output = y[:, 0]
        sc.check(output, "BD", self.config.dtype)
        return carry, output


class GatedDeltaNet2Stack(nn.Module, ARSequenceModel[GatedDeltaNet2StackCarry]):
    """Recurrent-only pre-RMSNorm residual blocks with SwiGLU and final RMSNorm.

    Implements the same batch-first length/active-mask interface as the mixer.
    The mixer config is shared; each layer has independent parameters and carry.
    ``intermediate_size`` is the SwiGLU width (upstream 1.3B recipe uses 6208).
    Residual accumulation and normalization statistics stay float32.
    """

    config: GatedDeltaNet2Config
    num_layers: int
    intermediate_size: int
    backend: GatedDeltaNet2Backend = "jax"

    def setup(self) -> None:
        if self.num_layers <= 0 or self.intermediate_size <= 0:
            raise ValueError("num_layers and intermediate_size must be positive")
        self.mixers = [
            GatedDeltaNet2(self.config, backend=self.backend, name=f"mixer_{i}") for i in range(self.num_layers)
        ]

    @nn.nowrap
    def initial_carry(self, batch_size: int) -> GatedDeltaNet2StackCarry:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        return tuple(_initial_carry(self.config, batch_size) for _ in range(self.num_layers))

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        x_len: jax.Array,
        carry: GatedDeltaNet2StackCarry | None = None,
    ) -> tuple[GatedDeltaNet2StackCarry, jax.Array]:
        valid = _sequence_mask(x, x_len, self.config.hidden_size)
        c = self.config
        sc = ShapeChecker(D=c.hidden_size, I=self.intermediate_size)
        sc.check(x, "BTD")
        chex.assert_type(x, jnp.floating)
        carry = self.initial_carry(x.shape[0]) if carry is None else carry
        if len(carry) != self.num_layers:
            raise ValueError("carry must contain one state per layer")
        x = x.astype(jnp.float32)
        x = jnp.where(valid[..., None], x, 0)
        updated: list[GatedDeltaNet2Carry] = []
        # Upstream mamba_init uses PyTorch linear defaults and scales the
        # MLP residual output by sqrt(2 * depth).
        mlp_init = nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform")
        residual_init = nn.initializers.variance_scaling(1 / (6 * self.num_layers), "fan_in", "uniform")
        for i, mixer in enumerate(self.mixers):
            normed = nn.RMSNorm(epsilon=c.norm_eps, dtype=c.dtype, name=f"norm_mixer_{i}")(x)
            state, mixed = mixer(normed, x_len, carry[i])
            updated.append(state)
            x = x + mixed.astype(jnp.float32)
            normed = nn.RMSNorm(epsilon=c.norm_eps, dtype=c.dtype, name=f"norm_mlp_{i}")(x)
            gate = nn.Dense(
                self.intermediate_size, use_bias=False, dtype=c.dtype, kernel_init=mlp_init, name=f"mlp_gate_{i}"
            )(normed)
            value = nn.Dense(
                self.intermediate_size, use_bias=False, dtype=c.dtype, kernel_init=mlp_init, name=f"mlp_up_{i}"
            )(normed)
            sc.check([gate, value], "BTI", c.dtype)
            hidden = jax.nn.silu(gate) * value
            x = x + nn.Dense(
                c.hidden_size, use_bias=False, dtype=c.dtype, kernel_init=residual_init, name=f"mlp_down_{i}"
            )(hidden).astype(jnp.float32)
        y = nn.RMSNorm(epsilon=c.norm_eps, dtype=c.dtype, name="final_norm")(x)
        y = jnp.where(valid[..., None], y, 0)
        sc.check(y, "BTD", c.dtype)
        return tuple(updated), y

    def step(
        self,
        x: jax.Array,
        x_active: jax.Array,
        carry: GatedDeltaNet2StackCarry | None = None,
    ) -> tuple[GatedDeltaNet2StackCarry, jax.Array]:
        sc = ShapeChecker(D=self.config.hidden_size)
        sc.check(x, "BD")
        chex.assert_type(x, jnp.floating)
        sc.check(x_active, "B", jnp.bool_)
        carry, y = self(x[:, None], x_active.astype(jnp.int32), carry)
        output = y[:, 0]
        sc.check(output, "BD", self.config.dtype)
        return carry, output


class GatedDeltaNet2LM(nn.Module):
    """Causal language model with untied embedding/head and explicit layer state.

    ``__call__(tokens, x_len, carry=None)`` accepts integer tokens [B,T] and
    required int32 prefix lengths [B], returning (carry, float32 logits
    [B,T,vocab_size]). ``step(tokens, x_active, carry=None)`` accepts tokens
    [B] and required boolean x_active [B]. Padding returns zero logits and
    preserves every carry leaf, following the sequence model convention.
    Valid token IDs must be in [0,vocab_size); callers validate external input.
    This is the recurrent-only architecture, without hybrid sliding attention.
    """

    config: GatedDeltaNet2Config
    num_layers: int
    intermediate_size: int
    vocab_size: int
    backend: GatedDeltaNet2Backend = "jax"

    def setup(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.backbone = GatedDeltaNet2Stack(self.config, self.num_layers, self.intermediate_size, backend=self.backend)
        self.embedding = nn.Embed(
            self.vocab_size,
            self.config.hidden_size,
            dtype=self.config.dtype,
            embedding_init=nn.initializers.normal(0.02),
        )
        self.lm_head = nn.Dense(
            self.vocab_size,
            use_bias=False,
            dtype=jnp.float32,
            kernel_init=nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform"),
        )

    @nn.nowrap
    def initial_carry(self, batch_size: int) -> GatedDeltaNet2StackCarry:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        return tuple(_initial_carry(self.config, batch_size) for _ in range(self.num_layers))

    def __call__(
        self,
        tokens: jax.Array,
        x_len: jax.Array,
        carry: GatedDeltaNet2StackCarry | None = None,
    ) -> tuple[GatedDeltaNet2StackCarry, jax.Array]:
        sc = ShapeChecker(V=self.vocab_size)
        sc.check(tokens, "BT")
        chex.assert_type(tokens, jnp.integer)
        valid = prefix_mask(x_len, tokens.shape[0], tokens.shape[1])
        tokens = jnp.where(valid, tokens, 0)
        carry, hidden = self.backbone(self.embedding(tokens), x_len, carry)
        logits = self.lm_head(hidden.astype(jnp.float32))
        sc.check(logits, "BTV", jnp.float32)
        return carry, logits

    def step(
        self,
        tokens: jax.Array,
        x_active: jax.Array,
        carry: GatedDeltaNet2StackCarry | None = None,
    ) -> tuple[GatedDeltaNet2StackCarry, jax.Array]:
        sc = ShapeChecker(V=self.vocab_size)
        sc.check(tokens, "B")
        chex.assert_type(tokens, jnp.integer)
        sc.check(x_active, "B", jnp.bool_)
        carry, logits = self(tokens[:, None], x_active.astype(jnp.int32), carry)
        output = logits[:, 0]
        sc.check(output, "BV", jnp.float32)
        return carry, output


def gdn2_370m(
    dtype: jax.typing.DTypeLike = jnp.float32,
    *,
    backend: GatedDeltaNet2Backend = "jax",
) -> GatedDeltaNet2LM:
    """380,603,648-parameter architecture used by the paper-matched 370M checkpoint.

    The checkpoint has 16 mixer heads, independent of the GPT config's n_head.
    This constructs the module without allocating or initializing parameters.
    """
    return GatedDeltaNet2LM(
        GatedDeltaNet2Config(hidden_size=1024, head_dim=128, num_heads=16, dtype=dtype),
        num_layers=16,
        intermediate_size=2048,
        vocab_size=32000,
        backend=backend,
    )
