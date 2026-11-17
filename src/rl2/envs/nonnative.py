"""Shared interface for host-stepped, batched PPO environments."""

from abc import ABC, abstractmethod
from typing import NamedTuple

import jax
import numpy as np
from numpy.typing import NDArray

from rl2.actions import ActionSpace

type Observation = jax.Array | dict[str, Observation] | tuple[Observation, ...]


class NonnativeStep(NamedTuple):
    """Batched transition with same-step autoreset; both end flags may be true."""

    observations: Observation  # leaves: (B, *O), autoreset observations at episode ends
    rewards: jax.Array  # (B,) float32
    terminated: jax.Array  # (B,) bool
    truncated: jax.Array  # (B,) bool
    final_observations: Observation  # leaves: (B, *O), before autoreset


class BaseNonnativeEnv(ABC):
    """Own simulator state and RNGs for B environments stepped outside JAX compilation.

    O denotes each observation leaf's dimensions; image channels are last.
    Episode ends autoreset within step, preserving final observations for bootstrapping.
    """

    num_envs: int
    action_space: ActionSpace

    @abstractmethod
    def reset(self) -> Observation:
        """Return (B, *O) initial observation leaves for all environments."""

    @abstractmethod
    def step(
        self,
        actions: jax.Array,  # (B,) integer IDs or (B, *action_shape) continuous actions
    ) -> NonnativeStep:
        """Return (B, *O) observation leaves and (B,) rewards and end flags."""

    @abstractmethod
    def render(self) -> NDArray[np.uint8]:  # (H, W, 3)
        """Return (H, W, 3) uint8 RGB pixels for a single evaluation environment."""

    @abstractmethod
    def close(self) -> None:
        """Release simulator and rendering resources."""
