"""Residual policy and value network for AlphaZero."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

type Parameters = dict[str, Any]


class PolicyValueNet(nn.Module):
    num_actions: int
    channels: int = 64
    num_blocks: int = 3

    @nn.compact
    def __call__(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Predict policy logits and values from batched board observations.

        Args:
            observation: Shape (B, H, W, C), with batch size B, board height H,
                board width W, and observation channels C. Converted to float32.

        Returns:
            Float32 policy logits of shape (B, num_actions) and values of shape
            (B,) in [-1, 1], from the player-to-move perspective.
        """
        # PGX chess observations are floats; tic-tac-toe observations are bools.
        observation = observation.astype(jnp.float32)
        x = nn.relu(nn.Conv(self.channels, (3, 3))(observation))
        for _ in range(self.num_blocks):
            residual = x
            x = nn.relu(nn.LayerNorm()(nn.Conv(self.channels, (3, 3))(x)))
            x = nn.relu(residual + nn.LayerNorm()(nn.Conv(self.channels, (3, 3))(x)))
        policy = nn.relu(nn.Conv(2, (1, 1))(x)).reshape((x.shape[0], -1))
        logits = nn.Dense(self.num_actions)(policy)
        value = nn.relu(nn.Conv(1, (1, 1))(x)).reshape((x.shape[0], -1))
        value = jnp.tanh(nn.Dense(1)(nn.relu(nn.Dense(self.channels)(value))))[:, 0]
        return logits, value
