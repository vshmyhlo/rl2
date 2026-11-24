"""Shared interface for host-stepped, batched PPO environments."""

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any, NamedTuple, TypedDict

import gymnasium as gym
import jax
import numpy as np
from numpy.typing import NDArray


class StopReason(IntEnum):
    ONGOING = 0  # Episode continues; excluded from stop fractions.
    SUCCESS = 1  # Termination: goal reached; takes precedence over a simultaneous collision.
    COLLISION = 2  # Termination: collision ended the episode without success.
    TERMINATED = 3  # Termination: other or unspecified terminal condition.
    TIME_LIMIT = 4  # Truncation: episode horizon reached without a terminal condition.
    SCENE_ROTATION = 5  # Truncation: scene reload interrupted an otherwise ongoing episode.


STOP_REASON_NAMES = {int(reason): reason.name.lower() for reason in StopReason if reason != StopReason.ONGOING}


class RouteSamplingInfo(TypedDict):
    """Starts sampled since the previous step, including resets; N may differ from B."""

    initial_goal_distance_meters: NDArray[np.float64]  # (N,)
    reference_route_length_meters: NDArray[np.float64]  # (N,)
    direct_path_blocked: NDArray[np.bool_]  # (N,)
    difficulty: NDArray[np.str_]  # (N,) easy, medium, or hard


class GymStyleInfo(TypedDict, total=False):
    """Optional host metadata: pre-reset transition measurements and newly sampled starts."""

    stop_reason: NDArray[np.int32]  # (B,) StopReason codes
    goal_distance: NDArray[np.float32]  # (B,) Euclidean distance to the goal in metres
    power_ratio: NDArray[np.float32]  # (B,) mean action power relative to hover
    power_penalty: NDArray[np.float32]  # (B,) nonnegative consumption cost subtracted from reward
    reward_components: dict[str, NDArray[np.float32]]  # (B,) signed contributions to raw reward
    route_sampling: RouteSamplingInfo  # (N,) leaves; omitted when no starts were sampled


def replace_observation[ObservationT](
    batch: ObservationT,  # NumPy leaves: (B, *O)
    observation: ObservationT,  # NumPy leaves: (*O,)
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


class GymStyleStep[ObservationT](NamedTuple):
    """Batched transition with same-step autoreset; both end flags may be true."""

    observations: ObservationT  # NumPy leaves: (B, *O), autoreset observations at episode ends
    rewards: NDArray[np.float32]  # (B,) float32
    terminated: NDArray[np.bool_]  # (B,) bool
    truncated: NDArray[np.bool_]  # (B,) bool
    final_observations: ObservationT  # NumPy leaves: (B, *O), before autoreset
    info: GymStyleInfo | None = None  # optional (B,) transition and (N,) sampled-start metadata


class GymStyleEnv[ObservationT](ABC):
    """Own simulator state and RNGs for B environments stepped on the host.

    O denotes each observation leaf's dimensions; image channels are last.
    Episode ends autoreset within step, preserving final observations for bootstrapping.
    """

    num_envs: int
    action_space: gym.Space

    @abstractmethod
    def reset(self) -> ObservationT:
        """Return (B, *O) initial observation leaves for all environments."""

    @abstractmethod
    def step(
        self,
        actions: NDArray[Any],  # (B,) integer IDs or (B, *action_shape) continuous actions
    ) -> GymStyleStep[ObservationT]:
        """Return (B, *O) observation leaves and (B,) rewards and end flags."""

    @abstractmethod
    def render(self) -> NDArray[np.uint8]:  # (B, H, W, 3)
        """Return (B, H, W, 3) uint8 RGB pixels in environment order."""

    @abstractmethod
    def close(self) -> None:
        """Release simulator and rendering resources."""
