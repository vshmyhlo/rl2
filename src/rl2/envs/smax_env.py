"""SMAX team observations, legal actions, and explicit autoreset boundaries."""

from functools import partial
from typing import Annotated, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario
from jaxmarl.environments.smax.heuristic_enemy_smax_env import State
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.actions import DiscreteSpace


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class SMAXConfig:
    map_name: str = "3m"
    max_steps: Annotated[int, Field(gt=0, strict=True)] = 100

    def __post_init__(self) -> None:
        try:
            map_name_to_scenario(self.map_name)
        except KeyError as error:
            raise ValueError(f"Unknown SMAX map: {self.map_name}") from error


class AgentObservation(NamedTuple):
    """N agents, O local features, A discrete actions; optional leading batch axes."""

    observations: jax.Array  # (..., N, O) float32
    action_masks: jax.Array  # (..., N, A) bool


class SMAXStep(NamedTuple):
    observation: AgentObservation  # (N, O); (N, A), after autoreset
    state: State  # leaves: (...)
    rewards: jax.Array  # (N,) shared team reward
    done: jax.Array  # () environment completion
    agent_dones: jax.Array  # (N,) dead agents or environment completion
    terminated: jax.Array  # () battle ended
    truncated: jax.Array  # () time limit reached
    final_observation: AgentObservation  # (N, O); (N, A), before autoreset
    final_state: State  # leaves: (...), before autoreset


class SMAXEnv:
    def __init__(self, config: SMAXConfig) -> None:
        self.config = config
        self.env = HeuristicEnemySMAX(
            scenario=map_name_to_scenario(config.map_name),
            max_steps=config.max_steps,
            observation_type="unit_list",
            action_type="discrete",
        )
        self.agents = tuple(self.env.agents)
        self.num_agents = self.env.num_agents
        self.action_space = DiscreteSpace(self.env.action_space(self.agents[0]).n)

    def _observation(self, observations: dict[str, jax.Array], state: State) -> AgentObservation:
        """Pack agent observations: (O,) leaves and state: (...) leaves in fixed order."""
        masks = self.env.get_avail_actions(state)  # agent leaves: (A,)
        return AgentObservation(
            jnp.stack([observations[agent] for agent in self.agents]),  # (N, O)
            jnp.stack([masks[agent] for agent in self.agents]).astype(bool),  # (N, A)
        )

    @partial(jax.jit, static_argnames="self")
    def reset(self, key: jax.Array) -> tuple[AgentObservation, State]:  # key: ()
        """Return (N, O)/(N, A) observations/masks and (...) state leaves."""
        observations, state = self.env.reset(key)  # agent leaves: (O,); state leaves: (...)
        return self._observation(observations, state), state

    @partial(jax.jit, static_argnames="self")
    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: State,  # leaves: (...)
        actions: jax.Array,  # (N,) int32
    ) -> SMAXStep:
        step_key, reset_key = jax.random.split(key)  # each ()
        observations, final_state, rewards, dones, _ = self.env.step_env(
            step_key, state, {agent: actions[i] for i, agent in enumerate(self.agents)}
        )  # observations: agent (O,); rewards/dones: agent (); state leaves: (...)
        alive = final_state.state.unit_alive  # (N + enemies,)
        terminated = ~alive[: self.num_agents].any() | ~alive[self.num_agents :].any()  # ()
        truncated = final_state.state.step >= self.config.max_steps  # ()
        done = terminated | truncated  # ()
        final_state = final_state.replace(state=final_state.state.replace(done=done))
        final_observation = self._observation(observations, final_state)
        observation, next_state = jax.lax.cond(
            done, lambda: self.reset(reset_key), lambda: (final_observation, final_state)
        )  # observations: (N, O)/(N, A); state leaves: (...)
        return SMAXStep(
            observation,
            next_state,
            jnp.stack([rewards[agent] for agent in self.agents]),  # (N,)
            done,  # ()
            jnp.stack([dones[agent] for agent in self.agents]) | done,  # (N,)
            terminated,
            truncated,
            final_observation,
            final_state,
        )

    def render(self, state: State) -> NDArray[np.uint8]:  # leaves: (...) -> (H, W, 3)
        figure = Figure(figsize=(5, 5), dpi=80)
        FigureCanvasAgg(figure)
        battle = state.state  # leaves: (...)
        actions = {agent: battle.prev_attack_actions[i] for i, agent in enumerate(self.env.all_agents)}  # scalar leaves
        frame = self.env.init_render(figure.subplots(), (None, battle, actions), 0, int(battle.step))
        return np.asarray(frame.get_array(), dtype=np.uint8)[..., :3].copy()  # (H, W, 3)
