"""LSTM sequence model with per-example episode resets and float32 memory."""

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from rl2.sequence_model import RecurrentSequenceModel
from rl2.shape_checker import ShapeChecker

type LSTMCarry = tuple[jax.Array, jax.Array]
type LSTMStackCarry = tuple[LSTMCarry, ...]


def _sigmoid_float32(x: jax.Array) -> jax.Array:
    sc = ShapeChecker()
    sc.check(x, "BD")
    chex.assert_type(x, float)
    output = jax.nn.sigmoid(x.astype(jnp.float32))
    sc.check(output, "BD", jnp.float32)
    return output


def _tanh_float32(x: jax.Array) -> jax.Array:
    sc = ShapeChecker()
    sc.check(x, "BD")
    chex.assert_type(x, float)
    output = jnp.tanh(x.astype(jnp.float32))
    sc.check(output, "BD", jnp.float32)
    return output


def initial_carry(num_envs: int, hidden_size: int) -> LSTMCarry:
    chex.assert_scalar_positive(num_envs)
    chex.assert_scalar_positive(hidden_size)
    carry = (
        jnp.zeros((num_envs, hidden_size), dtype=jnp.float32),
        jnp.zeros((num_envs, hidden_size), dtype=jnp.float32),
    )
    sc = ShapeChecker(B=num_envs, D=hidden_size)
    sc.check(carry, "BD", jnp.float32)
    return carry


class LSTM(nn.Module, RecurrentSequenceModel[LSTMCarry]):
    """Time-major LSTM with inputs/outputs in ``dtype`` and float32 carry.

    Carry is (cell state, hidden state), each [batch, features], and is required.
    Supply ``initial_carry(num_envs)`` to start from zeros; episode starts reset
    both states before consuming input.
    Input width is inferred from the inputs and may differ from ``features``.
    Time, batch, and input feature dimensions must be nonempty.
    Projections use ``dtype``; gate activations and memory updates use float32
    so reduced-precision gates do not prematurely saturate or round updates.
    """

    features: int
    dtype: jax.typing.DTypeLike = jnp.float32

    def setup(self) -> None:
        chex.assert_scalar_positive(self.features)
        # Retain the parameter names used by PPO's original LSTM cell.
        self.cell = nn.OptimizedLSTMCell(
            self.features,
            dtype=self.dtype,
            gate_fn=_sigmoid_float32,
            activation_fn=_tanh_float32,
            name="OptimizedLSTMCell_0",
        )

    @nn.nowrap
    def initial_carry(self, num_envs: int) -> LSTMCarry:
        return initial_carry(num_envs, self.features)

    def __call__(self, x: jax.Array, carry: LSTMCarry, episode_starts: jax.Array) -> tuple[LSTMCarry, jax.Array]:
        sc = ShapeChecker(D=self.features)
        sc.check(x, "TBI", self.dtype)
        sc.check(episode_starts, "TB", jnp.bool_)
        for size in sc["TBI"]:
            chex.assert_scalar_positive(size)
        sc.check(carry, "BD", jnp.float32)

        def recurrent_step(
            model: LSTM, memory: LSTMCarry, inputs: tuple[jax.Array, jax.Array]
        ) -> tuple[LSTMCarry, jax.Array]:
            inputs_x, starts = inputs
            return model.step(inputs_x, memory, starts)

        carry, output = nn.scan(
            recurrent_step,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )(self, carry, (x, episode_starts))
        sc.check(carry, "BD", jnp.float32)
        sc.check(output, "TBD", self.dtype)
        return carry, output

    def step(self, x: jax.Array, carry: LSTMCarry, episode_starts: jax.Array) -> tuple[LSTMCarry, jax.Array]:
        sc = ShapeChecker(D=self.features)
        sc.check(x, "BI", self.dtype)
        sc.check(episode_starts, "B", jnp.bool_)
        for size in sc["BI"]:
            chex.assert_scalar_positive(size)
        sc.check(carry, "BD", jnp.float32)
        cell, hidden = carry
        carry = (
            jnp.where(episode_starts[:, None], 0, cell),
            jnp.where(episode_starts[:, None], 0, hidden),
        )
        carry, output = self.cell(carry, x)
        output = output.astype(self.dtype)
        sc.check(carry, "BD", jnp.float32)
        sc.check(output, "BD", self.dtype)
        return carry, output


class LSTMStack(nn.Module, RecurrentSequenceModel[LSTMStackCarry]):
    """Pre-norm LSTM layers with residual connections and optional SwiGLU MLPs.

    Each layer computes ``x += LSTM(RMSNorm(x))`` followed by
    ``x += SwiGLU(RMSNorm(x))`` when ``intermediate_size > 0``. A final
    RMSNorm produces outputs in ``dtype``. Inputs must have width ``features``.
    Parameters, residual additions, and per-layer (cell, hidden) carry stay
    float32. Episode starts reset every layer's carry for that example.
    """

    features: int
    num_layers: int
    intermediate_size: int = 0
    dtype: jax.typing.DTypeLike = jnp.float32

    @nn.nowrap
    def _validate_config(self) -> None:
        chex.assert_scalar_positive(self.features)
        chex.assert_scalar_positive(self.num_layers)
        chex.assert_scalar_non_negative(self.intermediate_size)

    @nn.nowrap
    def initial_carry(self, num_envs: int) -> LSTMStackCarry:
        self._validate_config()
        return tuple(initial_carry(num_envs, self.features) for _ in range(self.num_layers))

    @nn.compact
    def __call__(
        self, x: jax.Array, carry: LSTMStackCarry, episode_starts: jax.Array
    ) -> tuple[LSTMStackCarry, jax.Array]:
        self._validate_config()
        sc = ShapeChecker(D=self.features, I=self.intermediate_size)
        sc.check(x, "TBD", self.dtype)
        sc.check(episode_starts, "TB", jnp.bool_)
        for size in sc["TB"]:
            chex.assert_scalar_positive(size)
        if len(carry) != self.num_layers:
            raise ValueError("carry must contain one LSTMCarry per layer")
        for state in carry:
            sc.check(state, "BD", jnp.float32)

        x = x.astype(jnp.float32)
        updated: list[LSTMCarry] = []
        mlp_init = nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform")
        residual_init = nn.initializers.variance_scaling(1 / (6 * self.num_layers), "fan_in", "uniform")
        for i, state in enumerate(carry):
            normalized = nn.RMSNorm(epsilon=1e-5, dtype=self.dtype, name=f"norm_mixer_{i}")(x)
            state, mixed = LSTM(self.features, dtype=self.dtype, name=f"mixer_{i}")(normalized, state, episode_starts)
            updated.append(state)
            x = x + mixed.astype(jnp.float32)
            if self.intermediate_size:
                normalized = nn.RMSNorm(epsilon=1e-5, dtype=self.dtype, name=f"norm_mlp_{i}")(x)
                gate = nn.Dense(
                    self.intermediate_size,
                    use_bias=False,
                    dtype=self.dtype,
                    kernel_init=mlp_init,
                    name=f"mlp_gate_{i}",
                )(normalized)
                value = nn.Dense(
                    self.intermediate_size,
                    use_bias=False,
                    dtype=self.dtype,
                    kernel_init=mlp_init,
                    name=f"mlp_up_{i}",
                )(normalized)
                sc.check([gate, value], "TBI", self.dtype)
                hidden = nn.silu(gate) * value
                mixed = nn.Dense(
                    self.features,
                    use_bias=False,
                    dtype=self.dtype,
                    kernel_init=residual_init,
                    name=f"mlp_down_{i}",
                )(hidden)
                sc.check(mixed, "TBD", self.dtype)
                x = x + mixed.astype(jnp.float32)
            sc.check(x, "TBD", jnp.float32)
        output = nn.RMSNorm(epsilon=1e-5, dtype=self.dtype, name="final_norm")(x)
        sc.check(output, "TBD", self.dtype)
        return tuple(updated), output

    def step(self, x: jax.Array, carry: LSTMStackCarry, episode_starts: jax.Array) -> tuple[LSTMStackCarry, jax.Array]:
        sc = ShapeChecker(D=self.features)
        sc.check(x, "BD", self.dtype)
        sc.check(episode_starts, "B", jnp.bool_)
        carry, output = self(x[None], carry, episode_starts[None])
        output = output[0]
        sc.check(output, "BD", self.dtype)
        return carry, output
