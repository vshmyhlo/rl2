"""Array-based transitions for cooperative multi-agent environments."""

from typing import NamedTuple

import jax


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
