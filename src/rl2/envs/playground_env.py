"""MuJoCo Playground Panda manipulation with vector observations for PPO."""

from functools import partial
from pathlib import Path
from typing import Annotated, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda.pick import PandaPickCube, default_config
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.actions import ContinuousSpace
from rl2.envs.base import BaseEnv, EnvStep


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class PlaygroundConfig:
    id: Literal["MuJoCo-Playground"] = "MuJoCo-Playground"
    task: Literal["PandaPickCube"] = "PandaPickCube"
    max_steps: Annotated[int, Field(gt=0, lt=2**31, strict=True)] = 150


@struct.dataclass
class PlaygroundParams:
    max_steps: int = 150


@struct.dataclass
class PlaygroundState:
    physics: mjx_env.State  # leaves: (...)
    step_count: jax.Array  # () int32


class PlaygroundEnv(BaseEnv[PlaygroundState, PlaygroundParams]):
    def __init__(self, config: PlaygroundConfig) -> None:
        assets = Path(mjx_env.MENAGERIE_PATH) / "franka_emika_panda/mjx_panda.xml"
        if not assets.exists():
            raise FileNotFoundError("Panda assets missing; run: uv run python -m rl2.envs.playground_assets")
        settings = default_config()
        settings.impl = "jax"
        settings.episode_length = config.max_steps
        # Construct directly: the registry otherwise clones the entire Menagerie repository.
        self.env = PandaPickCube(config=settings)
        self.params = PlaygroundParams(max_steps=config.max_steps)
        size = self.env.action_size
        self.action_space = ContinuousSpace(shape=(size,), low=(-1.0,) * size, high=(1.0,) * size)

    @partial(jax.jit, static_argnames="self")
    def reset(self, key: jax.Array, params: PlaygroundParams) -> tuple[jax.Array, PlaygroundState]:  # key: ()
        """Return (66,) float32 observation and (...) physics state leaves."""
        physics = self.env.reset(key)  # leaves: (...)
        state = PlaygroundState(physics=physics, step_count=jnp.array(0, jnp.int32))
        return cast(jax.Array, physics.obs), state  # (66,); state leaves: (...)

    @partial(jax.jit, static_argnames="self")
    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: PlaygroundState,  # leaves: (...)
        action: jax.Array,  # (8,) continuous controls
        params: PlaygroundParams,
    ) -> EnvStep[PlaygroundState]:
        """Advance physics and autoreset, retaining (66,) final_observation for bootstrapping."""
        physics = self.env.step(state.physics, action)  # leaves: (...)
        final_state = state.replace(physics=physics, step_count=state.step_count + 1)
        final_observation = cast(jax.Array, physics.obs)  # (66,)
        terminated = physics.done.astype(jnp.bool_)  # ()
        truncated = final_state.step_count >= params.max_steps  # ()
        done = terminated | truncated  # ()

        def reset_done(_: None) -> tuple[jax.Array, PlaygroundState]:
            return self.reset(key, params)  # (66,); state leaves: (...)

        def continue_episode(_: None) -> tuple[jax.Array, PlaygroundState]:
            return final_observation, final_state  # (66,); state leaves: (...)

        observation, next_state = jax.lax.cond(done, reset_done, continue_episode, None)  # (66,); leaves: (...)
        return EnvStep(
            observation=observation,  # (66,)
            state=next_state,  # leaves: (...)
            reward=physics.reward,  # ()
            done=done,  # ()
            info={
                "terminated": terminated,  # ()
                "truncated": truncated,  # ()
                "final_observation": final_observation,  # (66,)
                "final_state": final_state,  # leaves: (...)
            },
        )

    def render(self, state: PlaygroundState) -> NDArray[np.uint8]:  # leaves: (...) -> (240, 320, 3)
        return self.env.render([state.physics], height=240, width=320)[0]  # (240, 320, 3)
