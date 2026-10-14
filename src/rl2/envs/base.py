"""Shared interface for single-environment PPO adapters."""

from abc import ABC, abstractmethod
from typing import NamedTuple, TypedDict

import jax
import numpy as np
from numpy.typing import NDArray

from rl2.actions import ActionSpace
from rl2.envs.observations import Observation


class StepInfo[State](TypedDict):
    """Termination flags and final observation/state before autoreset."""

    terminated: jax.Array  # () bool
    truncated: jax.Array  # () bool
    final_observation: Observation  # leaves: (*O,) adapter-specific dtype
    final_state: State  # leaves: (...)


class EnvStep[State](NamedTuple):
    """Step result with observation/state reset when done."""

    observation: Observation  # leaves: (*O,) adapter-specific dtype
    state: State  # leaves: (...)
    reward: jax.Array  # () floating-point
    done: jax.Array  # () bool, terminated or truncated
    info: StepInfo[State]


class BaseEnv[State, Params](ABC):
    """Explicit state and RNGs, discrete or continuous actions, and same-step autoreset.

    Actions retain native values; observation arrays or pytrees use adapter-specific layouts.
    O denotes each observation leaf's dimensions. Reset and step support
    jax.jit and jax.vmap; rendering may run on the host.
    """

    params: Params
    action_space: ActionSpace

    @abstractmethod
    def reset(
        self,
        key: jax.Array,  # () typed PRNG key
        params: Params,
    ) -> tuple[Observation, State]:  # leaves: (*O,); state leaves: (...)
        """Return (*O,) observation leaves in the adapter's dtype and (...) state leaves."""

    @abstractmethod
    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: State,  # leaves: (...)
        action: jax.Array,  # () integer ID or (*action_shape,) continuous action
        params: Params,
    ) -> EnvStep[State]:
        """Advance and autoreset when terminated or truncated."""

    def step_batch(
        self,
        keys: jax.Array,  # (B,) typed PRNG keys
        state: State,  # leaves: (B, ...)
        actions: jax.Array,  # (B,) integer IDs or (B, *action_shape) continuous actions
        params: Params,
    ) -> EnvStep[State]:
        """Return (B, *O) observations, (B,) scalars and (B, ...) state/info leaves."""
        return jax.vmap(self.step, in_axes=(0, 0, 0, None))(keys, state, actions, params)

    @abstractmethod
    def render(
        self,
        state: State,  # leaves: (...)
    ) -> jax.Array | NDArray[np.uint8]:  # (H, W, 3)
        """Return (H, W, 3) uint8 RGB pixels for a state."""
