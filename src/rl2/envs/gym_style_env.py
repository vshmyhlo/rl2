"""Shared interface for host-stepped, batched PPO environments."""

from abc import ABC, abstractmethod
from typing import Any, NamedTuple

import gymnasium as gym
import jax
import numpy as np
from numpy.typing import NDArray

type Observation[Array] = Array | dict[str, Observation[Array]] | tuple[Observation[Array], ...]


def replace_observation[Obs](
    batch: Obs,  # NumPy leaves: (B, *O)
    observation: Obs,  # NumPy leaves: (*O,)
    *,
    index: int,
) -> None:
    """Replace one batch entry in place, preserving dict and tuple structure."""

    def replace_leaf(
        leaf: NDArray[Any],  # (B, *O)
        replacement: NDArray[Any],  # (*O,)
    ) -> None:
        leaf[index] = replacement

    jax.tree.map(replace_leaf, batch, observation)


class GymStyleStep[Obs](NamedTuple):
    """Batched transition with same-step autoreset; both end flags may be true."""

    observations: Obs  # NumPy leaves: (B, *O), autoreset observations at episode ends
    rewards: NDArray[np.float32]  # (B,) float32
    terminated: NDArray[np.bool_]  # (B,) bool
    truncated: NDArray[np.bool_]  # (B,) bool
    final_observations: Obs  # NumPy leaves: (B, *O), before autoreset


class GymStyleEnv[Obs](ABC):
    """Own simulator state and RNGs for B environments stepped on the host.

    O denotes each observation leaf's dimensions; image channels are last.
    Episode ends autoreset within step, preserving final observations for bootstrapping.
    """

    num_envs: int
    action_space: gym.Space

    @abstractmethod
    def reset(self) -> Obs:
        """Return (B, *O) initial observation leaves for all environments."""

    @abstractmethod
    def step(
        self,
        actions: NDArray[Any],  # (B,) integer IDs or (B, *action_shape) continuous actions
    ) -> GymStyleStep[Obs]:
        """Return (B, *O) observation leaves and (B,) rewards and end flags."""

    @abstractmethod
    def render(self) -> NDArray[np.uint8]:  # (B, H, W, 3)
        """Return (B, H, W, 3) uint8 RGB pixels in environment order."""

    @abstractmethod
    def close(self) -> None:
        """Release simulator and rendering resources."""
