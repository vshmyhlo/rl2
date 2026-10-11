"""JAX array interface for Gymnasium vector environments with array observations."""

from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np


class JaxVectorEnv(gym.vector.VectorWrapper):
    """Transfer actions/options to the host and environment outputs to JAX.

    Info dictionaries retain Gymnasium's original values, including non-array
    metadata. Outputs are copied because vector environments may reuse their
    NumPy buffers, including when running with shared memory and ``copy=False``.
    """

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,  # reset_mask, when supplied: (B,)
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Return (B, *O) observations and unchanged info after a full or masked reset."""
        obs, info = self.env.reset(seed=seed, options=jax.device_get(options))  # obs: (B, *O)
        return jnp.array(obs, copy=True), info  # observations: (B, *O)

    def step(
        self,
        actions: jax.Array,  # (B, *A); A is empty for discrete actions
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, Any]]:
        """Advance all environments.

        Args:
            actions: (B, *A) batched actions; A is empty for discrete actions.

        Returns:
            observations: (B, *O) observations with environment-specific dimensions O.
            rewards: (B,) float32 raw rewards.
            terminated: (B,) boolean terminal flags.
            truncated: (B,) boolean time-limit flags.
            info: Gymnasium metadata, passed through unchanged.
        """
        host_actions = np.asarray(jax.device_get(actions))  # (B, *A)
        obs, rewards, terminated, truncated, info = self.env.step(host_actions)  # (B, *O), 3 x (B,), info
        return (
            jnp.array(obs, copy=True),  # (B, *O)
            jnp.array(rewards, dtype=jnp.float32, copy=True),  # (B,)
            jnp.array(terminated, dtype=jnp.bool_, copy=True),  # (B,)
            jnp.array(truncated, dtype=jnp.bool_, copy=True),  # (B,)
            info,
        )
