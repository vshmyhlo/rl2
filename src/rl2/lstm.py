"""LSTM sequence model with per-example episode resets and float32 memory."""

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from rl2.sequence_model import RecurrentSequenceModel
from rl2.shape_checker import ShapeChecker

type LSTMCarry = tuple[jax.Array, jax.Array]


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
    """

    features: int
    dtype: jax.typing.DTypeLike = jnp.float32

    def setup(self) -> None:
        chex.assert_scalar_positive(self.features)
        # Retain the parameter names used by PPO's original LSTM cell.
        self.cell = nn.OptimizedLSTMCell(self.features, dtype=self.dtype, name="OptimizedLSTMCell_0")

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
