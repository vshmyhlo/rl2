"""Shared interface for host-stepped, batched PPO environments."""

from abc import ABC, abstractmethod
from typing import Any, NamedTuple, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

type Observation[Array] = Array | dict[str, Observation[Array]] | tuple[Observation[Array], ...]


def replace_observation(
    batch: Observation[NDArray[Any]],  # leaves: (B, *O)
    observation: Observation[NDArray[Any]],  # leaves: (*O,)
    *,
    index: int,
) -> None:
    """Replace one batch entry in place, preserving dict and tuple structure."""
    if isinstance(batch, dict):
        observation = cast(dict[str, Observation[NDArray[Any]]], observation)
        for key in batch:
            replace_observation(batch[key], observation[key], index=index)
    elif isinstance(batch, tuple):
        observation = cast(tuple[Observation[NDArray[Any]], ...], observation)
        for leaf, replacement in zip(batch, observation, strict=True):
            replace_observation(leaf, replacement, index=index)
    else:
        batch[index] = cast(NDArray[Any], observation)


class GymStyleStep(NamedTuple):
    """Batched transition with same-step autoreset; both end flags may be true."""

    observations: Observation[NDArray[Any]]  # leaves: (B, *O), autoreset observations at episode ends
    rewards: NDArray[np.float32]  # (B,) float32
    terminated: NDArray[np.bool_]  # (B,) bool
    truncated: NDArray[np.bool_]  # (B,) bool
    final_observations: Observation[NDArray[Any]]  # leaves: (B, *O), before autoreset


class GymStyleEnv(ABC):
    """Own simulator state and RNGs for B environments stepped on the host.

    O denotes each observation leaf's dimensions; image channels are last.
    Episode ends autoreset within step, preserving final observations for bootstrapping.
    """

    num_envs: int
    action_space: gym.Space

    @abstractmethod
    def reset(self) -> Observation[NDArray[Any]]:
        """Return (B, *O) initial observation leaves for all environments."""

    @abstractmethod
    def step(
        self,
        actions: NDArray[Any],  # (B,) integer IDs or (B, *action_shape) continuous actions
    ) -> GymStyleStep:
        """Return (B, *O) observation leaves and (B,) rewards and end flags."""

    @abstractmethod
    def render(self) -> NDArray[np.uint8]:  # (H, W, 3)
        """Return (H, W, 3) uint8 RGB pixels for a single evaluation environment."""

    @abstractmethod
    def close(self) -> None:
        """Release simulator and rendering resources."""
