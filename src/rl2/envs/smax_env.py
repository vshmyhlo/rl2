"""SMAX team observations, legal actions, and explicit autoreset boundaries."""

from typing import Annotated, Literal

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
from rl2.envs.multi_agent import AgentObservation, BaseMultiAgentEnv, MultiAgentStep


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class SMAXConfig:
    id: Literal["SMAX"] = "SMAX"
    map_name: str = "3m"
    max_steps: Annotated[int, Field(gt=0, lt=2**31, strict=True)] = 100

    def __post_init__(self) -> None:
        try:
            map_name_to_scenario(self.map_name)
        except KeyError as error:
            raise ValueError(f"Unknown SMAX map: {self.map_name}") from error


class SMAXEnv(BaseMultiAgentEnv[State]):
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

    def _observation(self, observation: dict[str, jax.Array], state: State) -> AgentObservation:
        """Pack agent observation: (O,) leaves and state: (...) leaves in fixed order."""
        actions_mask = self.env.get_avail_actions(state)  # agent leaves: (A,)
        return AgentObservation(
            observation=jnp.stack([observation[agent] for agent in self.agents]),  # (N, O)
            actions_mask=jnp.stack([actions_mask[agent] for agent in self.agents]).astype(bool),  # (N, A)
        )

    def reset(self, key: jax.Array) -> tuple[AgentObservation, State]:  # key: ()
        """Return (N, O)/(N, A) observations/masks and (...) state leaves."""
        observation, state = self.env.reset(key)  # agent leaves: (O,); state leaves: (...)
        return self._observation(observation, state), state

    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: State,  # leaves: (...)
        action: jax.Array,  # (N,) int32
    ) -> MultiAgentStep[State]:
        step_key, reset_key = jax.random.split(key)  # each ()
        observation, final_state, reward, native_done, _ = self.env.step_env(
            step_key, state, {agent: action[i] for i, agent in enumerate(self.agents)}
        )  # observation: agent (O,); reward/native_done: agent (); state leaves: (...)
        alive = final_state.state.unit_alive  # (N + enemies,)
        terminated = ~alive[: self.num_agents].any() | ~alive[self.num_agents :].any()  # ()
        truncated = final_state.state.step >= self.config.max_steps  # ()
        done = terminated | truncated  # ()
        final_state = final_state.replace(state=final_state.state.replace(done=done))
        final_observation = self._observation(observation, final_state)  # (N, O)/(N, A)
        observation, next_state = jax.lax.cond(
            done, lambda: self.reset(reset_key), lambda: (final_observation, final_state)
        )  # observation: (N, O)/(N, A); state leaves: (...)
        return MultiAgentStep(
            observation=observation,  # (N, O)/(N, A)
            state=next_state,  # leaves: (...)
            reward=jnp.stack([reward[agent] for agent in self.agents]),  # (N,)
            done=done,  # ()
            agent_done=jnp.stack([native_done[agent] for agent in self.agents]) | done,  # (N,)
            terminated=terminated,  # ()
            truncated=truncated,  # ()
            final_observation=final_observation,  # (N, O)/(N, A)
            final_state=final_state,  # leaves: (...)
        )

    def memory_reset(self, state: State) -> jax.Array:  # state leaves: (...) -> (N,)
        return ~state.state.unit_alive[: self.num_agents]  # (N,)

    def render(self, state: State) -> NDArray[np.uint8]:  # leaves: (...) -> (H, W, 3)
        figure = Figure(figsize=(5, 5), dpi=80)
        FigureCanvasAgg(figure)
        battle = state.state  # leaves: (...)
        action = {agent: battle.prev_attack_actions[i] for i, agent in enumerate(self.env.all_agents)}  # scalar leaves
        frame = self.env.init_render(figure.subplots(), (None, battle, action), 0, int(battle.step))
        return np.asarray(frame.get_array(), dtype=np.uint8)[..., :3].copy()  # (H, W, 3)
