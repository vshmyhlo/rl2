"""Shared interface and transitions for cooperative multi-agent environments."""

from abc import ABC, abstractmethod
from typing import NamedTuple

import jax
import numpy as np
from numpy.typing import NDArray

from rl2.actions import DiscreteSpace


class AgentObservation(NamedTuple):
    """N agents, O local features, A discrete actions; optional leading batch axes."""

    observations: jax.Array  # (..., N, O) float32
    action_masks: jax.Array  # (..., N, A) bool


class MultiAgentStep[State](NamedTuple):
    observation: AgentObservation  # (N, O); (N, A), after autoreset
    state: State  # leaves: (...)
    rewards: jax.Array  # (N,)
    done: jax.Array  # () environment completion
    agent_dones: jax.Array  # (N,) recurrent reset flags
    terminated: jax.Array  # () task ended
    truncated: jax.Array  # () time limit reached
    final_observation: AgentObservation  # (N, O); (N, A), before autoreset
    final_state: State  # leaves: (...), before autoreset


class BaseMultiAgentEnv[State](ABC):
    """Explicit state and RNGs, discrete actions, and same-step team autoreset.

    Actions and observations retain native values; agent dictionaries are packed
    in a fixed order. Masks expose native availability, or all actions when absent.
    N denotes agents, O local observation features, and A discrete actions.
    Reset, step, and memory_resets support jax.jit and jax.vmap;
    rendering may run on the host.
    """

    num_agents: int
    action_space: DiscreteSpace

    @abstractmethod
    def reset(
        self,
        key: jax.Array,  # () typed PRNG key
    ) -> tuple[AgentObservation, State]:  # (N, O)/(N, A); state leaves: (...)
        """Return (N, O)/(N, A) observations/masks and (...) state leaves."""

    @abstractmethod
    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: State,  # leaves: (...)
        actions: jax.Array,  # (N,) integer action IDs
    ) -> MultiAgentStep[State]:
        """Advance and autoreset the team when terminated or truncated."""

    @abstractmethod
    def memory_resets(
        self,
        state: State,  # leaves: (...)
    ) -> jax.Array:  # (N,) bool
        """Return (N,) bool recurrent reset flags, including for final-state bootstrapping."""

    @abstractmethod
    def render(
        self,
        state: State,  # leaves: (...)
    ) -> jax.Array | NDArray[np.uint8]:  # (H, W, 3)
        """Return (H, W, 3) uint8 RGB pixels for a state."""
