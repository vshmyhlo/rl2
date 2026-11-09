"""Shared interface and transitions for multi-agent environments."""

from abc import ABC, abstractmethod
from typing import NamedTuple

import jax
import numpy as np
from numpy.typing import NDArray

from rl2.actions import DiscreteSpace, MultiDiscreteSpace


class AgentObservation(NamedTuple):
    """N agents, O features, A categories, optional D action components and batch axes."""

    observation: jax.Array  # (..., N, O) float32
    actions_mask: jax.Array  # (..., N, *C, A) bool


class MultiAgentStep[State](NamedTuple):
    """Transition with done = terminated | truncated; both flags may be true.

    Final observation/state are populated on every step, before any autoreset.
    agent_done resets memory for observation; agent_inactive resets memory for
    final_observation when bootstrapping, excluding environment completion alone.
    """

    observation: AgentObservation  # (N, O); (N, *C, A), after autoreset
    state: State  # leaves: (...)
    reward: jax.Array  # (N,) floating-point
    done: jax.Array  # () bool environment completion
    agent_done: jax.Array  # (N,) bool agent inactivity or environment completion
    agent_inactive: jax.Array  # (N,) bool agent inactivity in the final state
    terminated: jax.Array  # () bool task ended
    truncated: jax.Array  # () bool time limit reached
    final_observation: AgentObservation  # (N, O); (N, *C, A), before autoreset
    final_state: State  # leaves: (...), before autoreset


class BaseMultiAgentEnv[State](ABC):
    """Explicit state and RNGs, discrete actions, and same-step environment autoreset.

    Agent dictionaries are packed in a fixed order; adapters document observation
    layouts and any lossless action encoding. N denotes agents, O local features,
    and A categories. C is empty for discrete actions or (D,) for multi-discrete
    actions. Masks expose available actions, including padding for unequal counts.
    Reset and step support caller-applied jax.jit and jax.vmap.
    Adapters do not add JIT boundaries; rendering may run on the host.
    """

    num_agents: int
    action_space: DiscreteSpace | MultiDiscreteSpace

    @abstractmethod
    def reset(
        self,
        key: jax.Array,  # () typed PRNG key
    ) -> tuple[AgentObservation, State]:  # (N, O)/(N, *C, A); state leaves: (...)
        """Return (N, O)/(N, *C, A) observations/masks and (...) state leaves."""

    @abstractmethod
    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: State,  # leaves: (...)
        action: jax.Array,  # (N, *C) integer action IDs
    ) -> MultiAgentStep[State]:
        """Advance and autoreset the environment when terminated or truncated."""

    @abstractmethod
    def render(
        self,
        state: State,  # leaves: (...)
    ) -> jax.Array | NDArray[np.uint8]:  # (H, W, 3)
        """Return (H, W, 3) uint8 RGB pixels for a state."""
