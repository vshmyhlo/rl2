"""Portable JAX/Flax Mamba-3 sequence mixer (SISO and MIMO).

Implements exponential-trapezoidal discretization, data-dependent rotary B/C,
B/C RMS normalization and biases, and factorized MIMO projections from
https://arxiv.org/abs/2603.15569. Parameterization follows the authors' module:
https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py.

Sequences are time-major, matching rl2's recurrent policies. This reference uses
``lax.scan``, not the upstream fused CUDA/chunked SSD kernels. It is a mixer,
without an outer residual connection, pre-norm, embedding, or prediction head.
Parameters and recurrent accumulation remain float32 with bf16 projections.

Example::

    model = Mamba3(d_model=128, d_state=64, headdim=32, mimo_rank=4)
    x = jnp.zeros((16, 8, 128))  # time, batch, features
    variables = model.init(jax.random.key(0), x)
    carry, y = model.apply(variables, x)
    carry, y_next = model.apply(variables, x[0], carry, method=model.step)
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn


class Mamba3Carry(NamedTuple):
    """Float32 recurrent state; all leaves have batch as their leading axis.

    Shapes: state [B,H,P,N], key [B,H,R,N], value [B,H,P], angle [B,H,Q].
    H is heads, P head width, N state width, R MIMO rank, Q rotary pairs.
    Use ``jax.tree.map(jax.lax.stop_gradient, carry)`` for truncated BPTT.
    """

    state: jax.Array
    key: jax.Array
    value: jax.Array
    angle: jax.Array


type StepInputs = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


def _rotate(x: jax.Array, angle: jax.Array) -> jax.Array:
    """Rotate adjacent pairs in the leading rotary portion of the state axis."""
    width = 2 * angle.shape[-1]
    pairs = x[..., :width].reshape((*x.shape[:-1], -1, 2))
    cos, sin = jnp.cos(angle)[..., None, :], jnp.sin(angle)[..., None, :]
    real, imag = pairs[..., 0], pairs[..., 1]
    rotated = jnp.stack((real * cos - imag * sin, real * sin + imag * cos), axis=-1)
    return jnp.concatenate((rotated.reshape((*x.shape[:-1], width)), x[..., width:]), axis=-1)


def _ssm_step(carry: Mamba3Carry, inputs: StepInputs, mimo_x: jax.Array) -> tuple[Mamba3Carry, jax.Array]:
    """Trapezoidal recurrence in the rotating B/C coordinate system."""
    x, b, c, dt, a, trap, angle_delta, starts = inputs

    def reset(leaf: jax.Array) -> jax.Array:
        mask = starts.reshape((starts.shape[0],) + (1,) * (leaf.ndim - 1))
        return jnp.where(mask, 0.0, leaf)

    carry = jax.tree.map(reset, carry)
    angle = carry.angle + angle_delta
    b, c = _rotate(b, angle), _rotate(c, angle)
    alpha = jnp.exp(dt * a)[..., None, None]
    beta = ((1.0 - trap) * dt)[..., None, None] * alpha
    gamma = (trap * dt)[..., None, None]
    # Contract over rank, keeping the recurrent state independent of MIMO rank.
    current = jnp.einsum("bhrn,bhp,hrp->bhpn", b, x, mimo_x)
    previous = jnp.einsum("bhrn,bhp,hrp->bhpn", carry.key, carry.value, mimo_x)
    state = alpha * carry.state + beta * previous + gamma * current
    y = jnp.einsum("bhpn,bhrn->bhrp", state, c)
    return Mamba3Carry(state, b, x, angle), y


class Mamba3(nn.Module):
    """Mamba-3 mixer with explicit carry and per-example episode resets.

    ``mimo_rank=1`` selects SISO; larger ranks enable MIMO. B/C projections
    are shared within ``ngroups``, with separate learned biases per head/rank.
    Adjacent rotary pairs are used for both modes (upstream MIMO uses a
    permuted layout). This module does not load upstream checkpoints directly.
    """

    d_model: int
    d_state: int = 128
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    mimo_rank: int = 1
    rope_fraction: float = 0.5
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    a_floor: float = 1e-4
    outproj_norm: bool = False
    dtype: jax.typing.DTypeLike = jnp.float32

    @nn.nowrap
    def _dimensions(self) -> tuple[int, int, int]:
        for name in ("d_model", "d_state", "expand", "headdim", "ngroups", "mimo_rank"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        inner = self.expand * self.d_model
        if inner % self.headdim:
            raise ValueError("expand * d_model must be divisible by headdim")
        heads = inner // self.headdim
        if heads % self.ngroups:
            raise ValueError("number of heads must be divisible by ngroups")
        if self.d_state % 2 or self.rope_fraction not in (0.5, 1.0):
            raise ValueError("d_state must be even and rope_fraction must be 0.5 or 1.0")
        pairs = int(self.d_state * self.rope_fraction) // 2
        if pairs < 1:
            raise ValueError("rope_fraction * d_state must allow at least one rotary pair")
        if not (0 < self.dt_min <= self.dt_max < math.inf and 0 < self.dt_init_floor <= self.dt_max):
            raise ValueError("require 0 < dt_min <= dt_max and 0 < dt_init_floor <= dt_max, all finite")
        if not 0 < self.a_floor < math.inf:
            raise ValueError("a_floor must be positive and finite")
        if jnp.dtype(self.dtype) not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)):
            raise ValueError("dtype must be float32, bfloat16, or float16")
        return inner, heads, pairs

    @nn.nowrap
    def initial_carry(self, batch_size: int) -> Mamba3Carry:
        """Allocate zero history, usable without initializing model parameters."""
        _, heads, pairs = self._dimensions()
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return Mamba3Carry(
            jnp.zeros((batch_size, heads, self.headdim, self.d_state), jnp.float32),
            jnp.zeros((batch_size, heads, self.mimo_rank, self.d_state), jnp.float32),
            jnp.zeros((batch_size, heads, self.headdim), jnp.float32),
            jnp.zeros((batch_size, heads, pairs), jnp.float32),
        )

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        carry: Mamba3Carry | None = None,
        episode_starts: jax.Array | None = None,
    ) -> tuple[Mamba3Carry, jax.Array]:
        """Map [time,batch,d_model] to (final carry, same-shaped output).

        ``episode_starts[time,batch]`` resets all history *before* that input.
        Omitting carry starts fresh; passing it continues a previous chunk.
        Gradients flow through supplied carry unless the caller detaches it.
        """
        inner, heads, pairs = self._dimensions()
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("x must have shape [time, batch, d_model]")
        steps, batch = x.shape[:2]
        fresh = self.initial_carry(batch)
        if carry is None:
            carry = fresh
        if not isinstance(carry, Mamba3Carry) or any(a.shape != b.shape for a, b in zip(carry, fresh)):
            raise ValueError("carry shapes must match the model and input batch size")

        def to_float32(leaf: jax.Array) -> jax.Array:
            return leaf.astype(jnp.float32)

        carry = jax.tree.map(to_float32, carry)
        if episode_starts is None:
            episode_starts = jnp.zeros((steps, batch), dtype=jnp.bool_)
        elif episode_starts.shape != (steps, batch):
            raise ValueError("episode_starts must have shape [time, batch]")
        episode_starts = episode_starts.astype(jnp.bool_)
        rank = self.mimo_rank
        bc_size = self.ngroups * rank * self.d_state
        sizes = (inner, inner, bc_size, bc_size, heads, heads, heads, pairs)
        projected = nn.Dense(sum(sizes), use_bias=False, dtype=self.dtype, name="in_proj")(x).astype(jnp.float32)
        offsets = tuple(sum(sizes[:i]) for i in range(1, len(sizes)))
        z, value, b, c, raw_dt, raw_a, raw_trap, raw_angle = jnp.split(projected, offsets, axis=-1)
        z, value = (v.reshape((steps, batch, heads, self.headdim)) for v in (z, value))

        def normalize_bc(v: jax.Array, name: str) -> jax.Array:
            v = v.reshape((steps, batch, rank, self.ngroups, self.d_state))
            v = nn.RMSNorm(epsilon=1e-5, dtype=jnp.float32, name=f"{name}_norm")(v)
            v = jnp.repeat(jnp.swapaxes(v, -3, -2), heads // self.ngroups, axis=-3)
            bias = self.param(f"{name}_bias", nn.initializers.ones_init(), (heads, rank, self.d_state))
            return v + bias

        b, c = normalize_bc(b, "B"), normalize_bc(c, "C")

        def init_dt(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
            log_dt = jax.random.uniform(key, shape, minval=math.log(self.dt_min), maxval=math.log(self.dt_max))
            dt = jnp.maximum(jnp.exp(log_dt), self.dt_init_floor)
            return dt + jnp.log(-jnp.expm1(-dt))  # Inverse softplus.

        dt = nn.softplus(raw_dt + self.param("dt_bias", init_dt, (heads,)))
        # Positive heavy-tail activation from the official Mamba-3 module.
        a = -jnp.maximum(jnp.maximum(raw_a, 0) + 1 / (1 - jnp.minimum(raw_a, 0)), self.a_floor)
        trap = nn.sigmoid(raw_trap)
        angle_delta = jnp.pi * jnp.tanh(raw_angle)[..., None, :] * dt[..., None]
        if rank == 1:
            mimo_x = mimo_z = mimo_o = jnp.ones((heads, 1, self.headdim), jnp.float32)
        else:
            mimo_x = self.param("mimo_x", nn.initializers.constant(1 / rank), (heads, rank, self.headdim))
            mimo_z = self.param("mimo_z", nn.initializers.ones_init(), (heads, rank, self.headdim))
            mimo_o = self.param("mimo_o", nn.initializers.constant(1 / rank), (heads, rank, self.headdim))

        def scan_step(state: Mamba3Carry, inputs: StepInputs) -> tuple[Mamba3Carry, jax.Array]:
            return _ssm_step(state, inputs, mimo_x)

        carry, y = jax.lax.scan(scan_step, carry, (value, b, c, dt, a, trap, angle_delta, episode_starts))
        skip = self.param("D", nn.initializers.ones_init(), (heads,))
        y = y + skip[:, None, None] * value[..., None, :] * mimo_x
        if self.outproj_norm:
            scale = self.param("out_norm_scale", nn.initializers.ones_init(), (heads, self.headdim))
            y = y * jax.lax.rsqrt(jnp.mean(jnp.square(y), axis=-1, keepdims=True) + 1e-5) * scale[:, None, :]
        y = y * nn.silu(z[..., None, :] * mimo_z)
        y = jnp.sum(y * mimo_o, axis=-2).reshape((steps, batch, inner))
        y = nn.Dense(self.d_model, use_bias=False, dtype=self.dtype, name="out_proj")(y)
        return carry, y

    def step(
        self,
        x: jax.Array,
        carry: Mamba3Carry | None = None,
        episode_starts: jax.Array | None = None,
    ) -> tuple[Mamba3Carry, jax.Array]:
        """One recurrent step on [batch,d_model], using the same parameters."""
        if x.ndim != 2:
            raise ValueError("step x must have shape [batch, d_model]")
        if episode_starts is not None and episode_starts.shape != (x.shape[0],):
            raise ValueError("step episode_starts must have shape [batch]")
        starts = None if episode_starts is None else episode_starts[None]
        carry, y = self(x[None], carry, starts)
        return carry, y[0]
