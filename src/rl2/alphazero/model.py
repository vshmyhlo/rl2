"""Residual policy and value network for AlphaZero."""

from typing import Any, Protocol

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

type Parameters = dict[str, Any]


class PolicyValueApplyFn(Protocol):
    """Apply policy/value parameters to batched observations."""

    def __call__(
        self,
        variables: dict[str, Parameters],
        observation: jax.Array,  # (B, H, W, C)
        /,
    ) -> tuple[jax.Array, jax.Array]:  # (B, A), (B,)
        """Return (B, A) logits and (B,) values for (B, H, W, C) observations."""
        ...


class PolicyValueNet(nn.Module):
    num_actions: int
    channels: int = 64
    num_blocks: int = 3
    dtype: jax.typing.DTypeLike | None = None

    @nn.compact
    def __call__(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:  # (B, H, W, C) -> (B, A), (B,)
        """Predict policy logits and values from batched board observations.

        Args:
            observation: (B, H, W, C) board observations, with batch size B, board height H,
                board width W, and observation channels C. Converted to float32.

        Returns:
            logits: (B, num_actions) float32 unmasked policy logits.
            value: (B,) float32 values in [-1, 1], from the player-to-move perspective.
        """
        # Flax Conv accepts unbatched inputs and extra batch axes, but the heads
        # below require exactly one batch axis when flattening board features.
        chex.assert_rank(observation, 4)
        # PGX chess observations are floats; tic-tac-toe observations are bools.
        observation = observation.astype(jnp.float32)  # (B, H, W, C)
        x = nn.relu(nn.Conv(self.channels, (3, 3), dtype=self.dtype)(observation))  # (B, H, W, D)
        for _ in range(self.num_blocks):
            residual = x  # (B, H, W, D)
            x = nn.relu(nn.LayerNorm(dtype=self.dtype)(nn.Conv(self.channels, (3, 3), dtype=self.dtype)(x)))
            x = nn.relu(residual + nn.LayerNorm(dtype=self.dtype)(nn.Conv(self.channels, (3, 3), dtype=self.dtype)(x)))
        policy = nn.relu(nn.Conv(2, (1, 1), dtype=self.dtype)(x)).reshape((x.shape[0], -1))  # (B, 2HW)
        logits = nn.Dense(self.num_actions, dtype=self.dtype)(policy).astype(jnp.float32)  # (B, A)
        value = nn.relu(nn.Conv(1, (1, 1), dtype=self.dtype)(x)).reshape((x.shape[0], -1))  # (B, HW)
        value = nn.Dense(1, dtype=self.dtype)(nn.relu(nn.Dense(self.channels, dtype=self.dtype)(value)))  # (B, 1)
        value = jnp.tanh(value.astype(jnp.float32))[:, 0]  # (B,)
        return logits, value
