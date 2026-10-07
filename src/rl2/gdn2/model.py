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
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from rl2.gdn2.core import delta_rule_step
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


type GatedDeltaNet2StackCarry = tuple[GatedDeltaNet2Carry, ...]
type MixerInputs = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


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


def _select(mask: jax.Array, yes: jax.Array, no: jax.Array) -> jax.Array:
    sc = ShapeChecker()
    sc.check(mask, "B", jnp.bool_)
    # Shapes vary between recurrence and convolution histories.
    names = "B" + "XYZ"[: yes.ndim - 1]
    sc.check([yes, no], names, jnp.float32)
    return jnp.where(mask.reshape((mask.shape[0],) + (1,) * (yes.ndim - 1)), yes, no)


def _linear_init(key: jax.Array, shape: tuple[int, ...], dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
    return nn.initializers.variance_scaling(2**-5, "fan_avg", "uniform")(key, shape, dtype)


def _rate_init(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    return jnp.log(jax.random.uniform(key, shape, dtype=jnp.float32, minval=1.0, maxval=16.0))


def _dt_init(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    dt = jnp.exp(jax.random.uniform(key, shape, dtype=jnp.float32, minval=math.log(0.001), maxval=math.log(0.1)))
    return dt + jnp.log(-jnp.expm1(-dt))


class GatedDeltaNet2(nn.Module):
    """Time-major token mixer with explicit state, resets, and padding masks.

    Call with x [T,B,D], optional carry, episode_starts [T,B] and mask [T,B].
    Returns (carry, y [T,B,D]). True episode_starts reset all history before
    that token; false mask entries skip the token and return zero. Padding
    takes precedence over resets, so a masked reset does not affect carry.
    No hidden mutable cache is used. Use ``method=model.step`` for [B,D].
    """

    config: GatedDeltaNet2Config = GatedDeltaNet2Config()

    @nn.nowrap
    def initial_carry(self, batch_size: int) -> GatedDeltaNet2Carry:
        return _initial_carry(self.config, batch_size)

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        carry: GatedDeltaNet2Carry | None = None,
        episode_starts: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> tuple[GatedDeltaNet2Carry, jax.Array]:
        c = self.config
        sc = ShapeChecker(D=c.hidden_size, H=c.value_heads, K=c.head_dim, V=c.value_head_dim)
        sc.check(x, "TBD")
        chex.assert_type(x, jnp.floating)
        starts = jnp.zeros(sc["TB"], jnp.bool_) if episode_starts is None else episode_starts
        valid = jnp.ones(sc["TB"], jnp.bool_) if mask is None else mask
        sc.check([starts, valid], "TB", jnp.bool_)
        carry = self.initial_carry(x.shape[1]) if carry is None else carry
        sc.check(carry.state, "BHKV", jnp.float32)
        history = c.conv_size - 1 if c.use_short_conv else 0
        key_width, value_width = c.num_heads * c.head_dim, c.value_heads * c.value_head_dim
        conv_sc = ShapeChecker(B=x.shape[1], C=history, Q=key_width, W=value_width)
        conv_sc.check([carry.q, carry.k], "BCQ", jnp.float32)
        conv_sc.check(carry.v, "BCW", jnp.float32)
        # Padding values must not enter projections (including NaNs in padding).
        x = jnp.where(valid[..., None], x, 0).astype(c.dtype)

        def dense(inputs: jax.Array, width: int, name: str, bias: bool = False) -> jax.Array:
            sc = ShapeChecker(T=x.shape[0], B=x.shape[1], O=width)
            sc.check(inputs, "TBI", c.dtype)
            result = nn.Dense(width, use_bias=bias, dtype=c.dtype, kernel_init=_linear_init, name=name)(inputs)
            sc.check(result, "TBO", c.dtype)
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
            qt, kt, vt, gt, bt, wt, reset, active = inputs

            def reset_leaf(leaf: jax.Array) -> jax.Array:
                return _select(reset, jnp.zeros_like(leaf), leaf)

            def preserve_padding(new: jax.Array, old: jax.Array) -> jax.Array:
                return _select(active, new, old)

            reset_carry = jax.tree.map(reset_leaf, previous)
            projections = [qt, kt, vt]
            histories = [reset_carry.q, reset_carry.k, reset_carry.v]
            for i in range(3):
                if c.use_short_conv:
                    window = jnp.concatenate((histories[i], projections[i][:, None]), axis=1)
                    projections[i] = jnp.sum(window * kernels[i], axis=1) + biases[i]
                    histories[i] = window[:, 1:]
                projections[i] = jax.nn.silu(projections[i])
            qt, kt, vt = projections
            shape = (x.shape[1], c.num_heads, c.head_dim)
            qt, kt, gt, bt = (a.reshape(shape) for a in (qt, kt, gt, bt))
            # Exactly the epsilon placement of upstream recurrent kernels.
            qt = qt * jax.lax.rsqrt(jnp.sum(qt * qt, axis=-1, keepdims=True) + 1e-6) * c.head_dim**-0.5
            kt = kt * jax.lax.rsqrt(jnp.sum(kt * kt, axis=-1, keepdims=True) + 1e-6)
            repeats = c.value_heads // c.num_heads
            qt, kt, gt, bt = (jnp.repeat(a, repeats, axis=1) for a in (qt, kt, gt, bt))
            vt, wt = (a.reshape((x.shape[1], c.value_heads, c.value_head_dim)) for a in (vt, wt))
            state, output = delta_rule_step(reset_carry.state, qt, kt, vt, gt, bt, wt)
            updated = GatedDeltaNet2Carry(state, *histories)
            updated = jax.tree.map(preserve_padding, updated, previous)
            output = _select(active, output, jnp.zeros_like(output))
            return updated, output

        carry, output = jax.lax.scan(step, carry, (q, k, v, log_decay, erase, write, starts, valid))
        sc.check(output, "TBHV", jnp.float32)
        gate = dense(dense(x, c.value_head_dim, "g_proj_in"), value_width, "g_proj_out", bias=True)
        gate = gate.astype(jnp.float32).reshape(output.shape)
        norm_weight = self.param("o_norm_scale", nn.initializers.ones, (c.value_head_dim,), jnp.float32)
        output = output * jax.lax.rsqrt(jnp.mean(output**2, axis=-1, keepdims=True) + c.norm_eps)
        output = output * norm_weight * jax.nn.silu(gate)
        y = dense(output.reshape((*x.shape[:2], value_width)).astype(c.dtype), c.hidden_size, "o_proj")
        y = jnp.where(valid[..., None], y, 0)
        sc.check(y, "TBD", c.dtype)
        return carry, y

    def step(
        self,
        x: jax.Array,
        carry: GatedDeltaNet2Carry | None = None,
        episode_starts: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> tuple[GatedDeltaNet2Carry, jax.Array]:
        sc = ShapeChecker(D=self.config.hidden_size)
        sc.check(x, "BD")
        chex.assert_type(x, jnp.floating)
        for value in (episode_starts, mask):
            if value is not None:
                sc.check(value, "B", jnp.bool_)
        carry, y = self(
            x[None],
            carry,
            None if episode_starts is None else episode_starts[None],
            None if mask is None else mask[None],
        )
        return carry, y[0]


class GatedDeltaNet2Stack(nn.Module):
    """Recurrent-only pre-RMSNorm residual blocks with SwiGLU and final RMSNorm.

    The mixer config is shared; each layer has independent parameters and carry.
    ``intermediate_size`` is the SwiGLU width (upstream 1.3B recipe uses 6208).
    Residual accumulation and normalization statistics stay float32.
    """

    config: GatedDeltaNet2Config
    num_layers: int
    intermediate_size: int

    def setup(self) -> None:
        if self.num_layers <= 0 or self.intermediate_size <= 0:
            raise ValueError("num_layers and intermediate_size must be positive")
        self.mixers = [GatedDeltaNet2(self.config, name=f"mixer_{i}") for i in range(self.num_layers)]

    @nn.nowrap
    def initial_carry(self, batch_size: int) -> GatedDeltaNet2StackCarry:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        return tuple(_initial_carry(self.config, batch_size) for _ in range(self.num_layers))

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        carry: GatedDeltaNet2StackCarry | None = None,
        episode_starts: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> tuple[GatedDeltaNet2StackCarry, jax.Array]:
        c = self.config
        sc = ShapeChecker(D=c.hidden_size, I=self.intermediate_size)
        sc.check(x, "TBD")
        chex.assert_type(x, jnp.floating)
        for value in (episode_starts, mask):
            if value is not None:
                sc.check(value, "TB", jnp.bool_)
        carry = self.initial_carry(x.shape[1]) if carry is None else carry
        if len(carry) != self.num_layers:
            raise ValueError("carry must contain one state per layer")
        x = x.astype(jnp.float32)
        if mask is not None:
            x = jnp.where(mask[..., None], x, 0)
        updated: list[GatedDeltaNet2Carry] = []
        # Upstream mamba_init uses PyTorch linear defaults and scales the
        # MLP residual output by sqrt(2 * depth).
        mlp_init = nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform")
        residual_init = nn.initializers.variance_scaling(1 / (6 * self.num_layers), "fan_in", "uniform")
        for i, mixer in enumerate(self.mixers):
            normed = nn.RMSNorm(epsilon=c.norm_eps, dtype=c.dtype, name=f"norm_mixer_{i}")(x)
            state, mixed = mixer(normed, carry[i], episode_starts, mask)
            updated.append(state)
            x = x + mixed.astype(jnp.float32)
            normed = nn.RMSNorm(epsilon=c.norm_eps, dtype=c.dtype, name=f"norm_mlp_{i}")(x)
            gate = nn.Dense(
                self.intermediate_size, use_bias=False, dtype=c.dtype, kernel_init=mlp_init, name=f"mlp_gate_{i}"
            )(normed)
            value = nn.Dense(
                self.intermediate_size, use_bias=False, dtype=c.dtype, kernel_init=mlp_init, name=f"mlp_up_{i}"
            )(normed)
            sc.check([gate, value], "TBI", c.dtype)
            hidden = jax.nn.silu(gate) * value
            x = x + nn.Dense(
                c.hidden_size, use_bias=False, dtype=c.dtype, kernel_init=residual_init, name=f"mlp_down_{i}"
            )(hidden).astype(jnp.float32)
        y = nn.RMSNorm(epsilon=c.norm_eps, dtype=c.dtype, name="final_norm")(x)
        if mask is not None:
            y = jnp.where(mask[..., None], y, 0)
        sc.check(y, "TBD", c.dtype)
        return tuple(updated), y

    def step(
        self,
        x: jax.Array,
        carry: GatedDeltaNet2StackCarry | None = None,
        episode_starts: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> tuple[GatedDeltaNet2StackCarry, jax.Array]:
        sc = ShapeChecker(D=self.config.hidden_size)
        sc.check(x, "BD")
        chex.assert_type(x, jnp.floating)
        for value in (episode_starts, mask):
            if value is not None:
                sc.check(value, "B", jnp.bool_)
        carry, y = self(
            x[None],
            carry,
            None if episode_starts is None else episode_starts[None],
            None if mask is None else mask[None],
        )
        return carry, y[0]


class GatedDeltaNet2LM(nn.Module):
    """Causal language model with untied embedding/head and explicit layer state.

    Accepts integer tokens [T,B]; returns (carry, float32 logits [T,B,vocab_size]).
    Token IDs must be in [0,vocab_size); callers validate external token input.
    This is the recurrent-only architecture, without hybrid sliding attention.
    """

    config: GatedDeltaNet2Config
    num_layers: int
    intermediate_size: int
    vocab_size: int

    def setup(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.backbone = GatedDeltaNet2Stack(self.config, self.num_layers, self.intermediate_size)
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
        carry: GatedDeltaNet2StackCarry | None = None,
        episode_starts: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> tuple[GatedDeltaNet2StackCarry, jax.Array]:
        sc = ShapeChecker(V=self.vocab_size)
        sc.check(tokens, "TB")
        chex.assert_type(tokens, jnp.integer)
        if mask is not None:
            sc.check(mask, "TB", jnp.bool_)
            tokens = jnp.where(mask, tokens, 0)
        carry, hidden = self.backbone(self.embedding(tokens), carry, episode_starts, mask)
        logits = self.lm_head(hidden.astype(jnp.float32))
        sc.check(logits, "TBV", jnp.float32)
        return carry, logits

    def step(
        self,
        tokens: jax.Array,
        carry: GatedDeltaNet2StackCarry | None = None,
        episode_starts: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> tuple[GatedDeltaNet2StackCarry, jax.Array]:
        sc = ShapeChecker()
        sc.check(tokens, "B")
        chex.assert_type(tokens, jnp.integer)
        for value in (episode_starts, mask):
            if value is not None:
                sc.check(value, "B", jnp.bool_)
        carry, logits = self(
            tokens[None],
            carry,
            None if episode_starts is None else episode_starts[None],
            None if mask is None else mask[None],
        )
        return carry, logits[0]
