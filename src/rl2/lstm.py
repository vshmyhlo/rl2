"""LSTM sequence model with per-example episode resets and float32 memory."""

from typing import NamedTuple, cast

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.recurrent import DenseParams

from rl2.sequence_model import RecurrentSequenceModel

type LSTMCarry = tuple[jax.Array, jax.Array]  # cell, hidden: (B, H)
type LSTMStackCarry = tuple[LSTMCarry, ...]  # Per-layer leaves: (B, H)


def _sigmoid_float32(x: jax.Array) -> jax.Array:  # (*S,) -> (*S,)
    chex.assert_type(x, float)
    return jax.nn.sigmoid(x.astype(jnp.float32))  # (*S,)


def _tanh_float32(x: jax.Array) -> jax.Array:  # (*S,) -> (*S,)
    chex.assert_type(x, float)
    return jnp.tanh(x.astype(jnp.float32))  # (*S,)


class _LSTMWeights(NamedTuple):
    input: jax.Array  # (I, 4H)
    recurrent: jax.Array  # (H, 4H), float32 for scan gradient accumulation
    bias: jax.Array  # (4H,), float32


class _LSTMCell(nn.OptimizedLSTMCell):
    @nn.compact
    def weights(self, x: jax.Array, hidden: jax.Array) -> _LSTMWeights:  # (T, B, I); (B, H)
        """Pack the original Flax parameters without changing checkpoint keys."""
        inputs: list[jax.Array] = []  # each (I, H)
        recurrent: list[jax.Array] = []  # each (H, H)
        biases: list[jax.Array] = []  # each (H,)
        for gate in "ifgo":
            kernel, _ = DenseParams(self.features, use_bias=False, kernel_init=self.kernel_init, name=f"i{gate}")(
                x
            )  # (I, H); None
            inputs.append(kernel)
            kernel, bias = DenseParams(self.features, kernel_init=self.recurrent_kernel_init, name=f"h{gate}")(
                hidden
            )  # (H, H); (H,)
            recurrent.append(kernel)
            biases.append(cast(jax.Array, bias))
        return _LSTMWeights(
            jnp.concatenate(inputs, axis=-1).astype(self.dtype),  # (I, 4H)
            jnp.concatenate(recurrent, axis=-1),  # (H, 4H), float32
            jnp.concatenate(biases),  # (4H,), float32
        )


@jax.checkpoint
def _apply_gates(cell: jax.Array, gates: jax.Array) -> LSTMCarry:  # (B, H); (B, 4H)
    """Recompute cheap activations in backward instead of saving each gate."""
    i, f, g, o = jnp.split(gates.astype(jnp.float32), 4, axis=-1)  # each (B, H)
    cell = jax.nn.sigmoid(f) * cell + jax.nn.sigmoid(i) * jnp.tanh(g)  # (B, H)
    hidden = jax.nn.sigmoid(o) * jnp.tanh(cell)  # (B, H)
    return cell, hidden  # each (B, H), float32


def initial_carry(num_envs: int, hidden_size: int) -> LSTMCarry:
    """Return (B, H) float32 zero cell and hidden states."""
    chex.assert_scalar_positive(num_envs)
    chex.assert_scalar_positive(hidden_size)
    return (
        jnp.zeros((num_envs, hidden_size), dtype=jnp.float32),  # (B, H)
        jnp.zeros((num_envs, hidden_size), dtype=jnp.float32),  # (B, H)
    )


class LSTM(nn.Module, RecurrentSequenceModel[LSTMCarry]):
    """Time-major LSTM with inputs/outputs in ``dtype`` and float32 carry.

    B is batch size, I is input width, H is ``features``, and T is time.
    Carry is (cell state, hidden state), each (B, H), and is required.
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
        self.cell = _LSTMCell(
            self.features,
            dtype=self.dtype,
            gate_fn=_sigmoid_float32,
            activation_fn=_tanh_float32,
            name="OptimizedLSTMCell_0",
        )

    @nn.nowrap
    def initial_carry(self, num_envs: int) -> LSTMCarry:
        return initial_carry(num_envs, self.features)  # Each (B, H)

    def __call__(
        self,
        x: jax.Array,  # (T, B, I)
        carry: LSTMCarry,  # Each (B, H), float32
        episode_starts: jax.Array,  # (T, B), bool
    ) -> tuple[LSTMCarry, jax.Array]:
        chex.assert_rank(x, 3)
        chex.assert_type(x, float)
        chex.assert_shape(carry, (x.shape[1], self.features))
        chex.assert_type(carry, jnp.float32)
        chex.assert_shape(episode_starts, x.shape[:2])
        chex.assert_type(episode_starts, jnp.bool_)
        for size in x.shape:
            chex.assert_scalar_positive(size)
        weights = self.cell.weights(x, carry[1])
        projected = jnp.matmul(x.astype(self.dtype), weights.input)  # (T, B, 4H)

        def recurrent_step(
            memory: LSTMCarry,  # Each (B, H)
            inputs: tuple[jax.Array, jax.Array],  # (B, 4H), (B,)
        ) -> tuple[LSTMCarry, jax.Array]:
            inputs_x, starts = inputs  # (B, 4H), (B,)
            cell, hidden = jax.tree.map(lambda state: jnp.where(starts[:, None], 0, state), memory)  # each (B, H)
            # Cast inside the scan so its parameter-gradient accumulators stay float32.
            recurrent = jnp.matmul(hidden.astype(self.dtype), weights.recurrent.astype(self.dtype))  # (B, 4H)
            gates = (recurrent + weights.bias.astype(self.dtype)) + inputs_x  # (B, 4H)
            memory = _apply_gates(cell, gates)  # each (B, H)
            return memory, memory[1].astype(self.dtype)  # carry: each (B, H); output: (B, H)

        carry, output = jax.lax.scan(recurrent_step, carry, (projected, episode_starts))  # (B, H) leaves; (T, B, H)
        return carry, output  # Carry: each (B, H); output: (T, B, H)

    def step(
        self,
        x: jax.Array,  # (B, I)
        carry: LSTMCarry,  # Each (B, H), float32
        episode_starts: jax.Array,  # (B,), bool
    ) -> tuple[LSTMCarry, jax.Array]:
        chex.assert_rank(x, 2)
        chex.assert_type(x, float)
        chex.assert_shape(carry, (x.shape[0], self.features))
        chex.assert_type(carry, jnp.float32)
        chex.assert_shape(episode_starts, (x.shape[0],))
        chex.assert_type(episode_starts, jnp.bool_)
        for size in x.shape:
            chex.assert_scalar_positive(size)

        def reset(state: jax.Array) -> jax.Array:  # (B, H) -> (B, H)
            return jnp.where(episode_starts[:, None], 0, state)  # (B, H)

        carry = jax.tree.map(reset, carry)  # Each (B, H)
        carry, output = self.cell(carry, x)  # Carry: each (B, H); output: (B, H)
        return carry, output.astype(self.dtype)  # Carry: each (B, H); output: (B, H)


class LSTMStack(nn.Module, RecurrentSequenceModel[LSTMStackCarry]):
    """Pre-norm LSTM layers with residual connections and optional SwiGLU MLPs.

    Each layer computes ``x += LSTM(RMSNorm(x))`` followed by
    ``x += SwiGLU(RMSNorm(x))`` when ``intermediate_size > 0``. A final
    RMSNorm produces outputs in ``dtype``. Inputs must have width ``features``.
    Parameters, residual additions, and per-layer (cell, hidden) carry stay
    float32. Episode starts reset every layer's carry for that example.
    T is time, B is batch size, H is ``features``, and M is ``intermediate_size``.
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
        return tuple(initial_carry(num_envs, self.features) for _ in range(self.num_layers))  # Leaves: (B, H)

    @nn.compact
    def __call__(
        self,
        x: jax.Array,  # (T, B, H)
        carry: LSTMStackCarry,  # Leaves: (B, H), float32
        episode_starts: jax.Array,  # (T, B), bool
    ) -> tuple[LSTMStackCarry, jax.Array]:
        self._validate_config()
        chex.assert_shape(x, (None, None, self.features))
        chex.assert_type(x, float)
        for size in x.shape[:2]:
            chex.assert_scalar_positive(size)
        if len(carry) != self.num_layers:
            raise ValueError("carry must contain one LSTMCarry per layer")

        x = x.astype(jnp.float32)  # (T, B, H)
        updated: list[LSTMCarry] = []  # Leaves: (B, H)
        mlp_init = nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform")
        residual_init = nn.initializers.variance_scaling(1 / (6 * self.num_layers), "fan_in", "uniform")
        for i, state in enumerate(carry):
            normalized = nn.RMSNorm(epsilon=1e-5, dtype=self.dtype, name=f"norm_mixer_{i}")(x)  # (T, B, H)
            state, mixed = LSTM(self.features, dtype=self.dtype, name=f"mixer_{i}")(
                normalized, state, episode_starts
            )  # State: each (B, H); mixed: (T, B, H)
            updated.append(state)
            x = x + mixed.astype(jnp.float32)  # (T, B, H)
            if self.intermediate_size:
                normalized = nn.RMSNorm(epsilon=1e-5, dtype=self.dtype, name=f"norm_mlp_{i}")(x)  # (T, B, H)
                gate = nn.Dense(
                    self.intermediate_size,
                    use_bias=False,
                    dtype=self.dtype,
                    kernel_init=mlp_init,
                    name=f"mlp_gate_{i}",
                )(normalized)  # (T, B, M)
                value = nn.Dense(
                    self.intermediate_size,
                    use_bias=False,
                    dtype=self.dtype,
                    kernel_init=mlp_init,
                    name=f"mlp_up_{i}",
                )(normalized)  # (T, B, M)
                hidden = nn.silu(gate) * value  # (T, B, M)
                mixed = nn.Dense(
                    self.features,
                    use_bias=False,
                    dtype=self.dtype,
                    kernel_init=residual_init,
                    name=f"mlp_down_{i}",
                )(hidden)  # (T, B, H)
                x = x + mixed.astype(jnp.float32)  # (T, B, H)
        output = nn.RMSNorm(epsilon=1e-5, dtype=self.dtype, name="final_norm")(x)  # (T, B, H)
        return tuple(updated), output  # Carry leaves: (B, H); output: (T, B, H)

    def step(
        self,
        x: jax.Array,  # (B, H)
        carry: LSTMStackCarry,  # Leaves: (B, H), float32
        episode_starts: jax.Array,  # (B,), bool
    ) -> tuple[LSTMStackCarry, jax.Array]:
        carry, output = self(x[None], carry, episode_starts[None])  # Carry leaves: (B, H); output: (1, B, H)
        return carry, output[0]  # Carry leaves: (B, H); output: (B, H)
