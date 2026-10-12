"""Shared interface for single-environment PPO adapters."""

from abc import ABC, abstractmethod
from typing import NamedTuple, TypedDict

import jax
import numpy as np
from numpy.typing import NDArray


class StepInfo[State](TypedDict):
    """Termination flags and final observation/state before autoreset."""

    terminated: jax.Array  # () bool
    truncated: jax.Array  # () bool
    final_observation: jax.Array  # (*O,) uint8
    final_state: State  # leaves: (...)


class EnvStep[State](NamedTuple):
    """Step result with observation/state reset when done."""

    observation: jax.Array  # (*O,) uint8
    state: State  # leaves: (...)
    reward: jax.Array  # () floating-point
    done: jax.Array  # () bool, terminated or truncated
    info: StepInfo[State]


class BaseEnv[State, Params](ABC):
    """Explicit state and RNGs, discrete actions, and same-step autoreset.

    O denotes adapter-specific observation dimensions. Reset and step support
    jax.jit and jax.vmap; rendering may run on the host.
    """

    params: Params
    num_actions: int

    @abstractmethod
    def reset(
        self,
        key: jax.Array,  # () typed PRNG key
        params: Params,
    ) -> tuple[jax.Array, State]:  # (*O,); state leaves: (...)
        """Return (*O,) uint8 observation and (...) state leaves."""

    @abstractmethod
    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: State,  # leaves: (...)
        action: jax.Array,  # () integer action ID
        params: Params,
    ) -> EnvStep[State]:
        """Advance and autoreset when terminated or truncated."""

    @abstractmethod
    def render(
        self,
        state: State,  # leaves: (...)
    ) -> jax.Array | NDArray[np.uint8]:  # (H, W, 3)
        """Return (H, W, 3) uint8 RGB pixels for a state."""
