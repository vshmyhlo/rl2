"""MuJoCo Playground control, locomotion, and manipulation with vector observations."""

from functools import partial
from pathlib import Path
from typing import Annotated, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import cartpole
from mujoco_playground._src.locomotion.go1 import joystick
from mujoco_playground._src.manipulation.franka_emika_panda import pick
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.actions import ContinuousSpace
from rl2.envs.base import BaseEnv, EnvStep


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class PlaygroundConfig:
    id: Literal["MuJoCo-Playground"] = "MuJoCo-Playground"
    task: Literal["CartpoleSwingup", "Go1JoystickFlatTerrain", "PandaPickCube"] = "PandaPickCube"
    max_steps: Annotated[int, Field(gt=0, lt=2**31, strict=True)] = 150


@struct.dataclass
class PlaygroundParams:
    max_steps: int = 150


@struct.dataclass
class PlaygroundState:
    physics: mjx_env.State  # leaves: (...)
    step_count: jax.Array  # () int32


class PlaygroundEnv(BaseEnv[PlaygroundState, PlaygroundParams]):
    """O and A are task-specific observation and action dimensions.

    Go1 exposes its noisy (48,) policy state to both heads; privileged state is
    retained only in physics, not fed to the shared policy/value encoder.
    """

    def __init__(self, config: PlaygroundConfig) -> None:
        self.task = config.task
        if config.task == "CartpoleSwingup":
            settings = cartpole.default_config()
            constructor = partial(cartpole.Balance, swing_up=True, sparse=False)
        elif config.task == "Go1JoystickFlatTerrain":
            assets = Path(mjx_env.MENAGERIE_PATH) / "unitree_go1/assets/trunk.stl"
            if not assets.exists():
                raise FileNotFoundError(
                    "Go1 assets missing; run: uv run python -m rl2.envs.playground_assets --robot go1"
                )
            settings = joystick.default_config()
            constructor = partial(joystick.Joystick, task="flat_terrain")
        else:
            assets = Path(mjx_env.MENAGERIE_PATH) / "franka_emika_panda/mjx_panda.xml"
            if not assets.exists():
                raise FileNotFoundError("Panda assets missing; run: uv run python -m rl2.envs.playground_assets")
            settings = pick.default_config()
            constructor = pick.PandaPickCube
        settings.impl = "jax"
        settings.episode_length = config.max_steps
        # Construct directly: the registry otherwise clones the entire Menagerie repository.
        # MJX loads onto one device; defer the training mesh to batched simulation.
        with jax.set_mesh(None):
            self.env = constructor(config=settings)
        self.params = PlaygroundParams(max_steps=config.max_steps)
        size = self.env.action_size
        self.action_space = ContinuousSpace(shape=(size,), low=(-1.0,) * size, high=(1.0,) * size)

    def _observation(self, physics: mjx_env.State) -> jax.Array:  # physics leaves: (...) -> (O,)
        if self.task == "Go1JoystickFlatTerrain":
            return cast(dict[str, jax.Array], physics.obs)["state"]  # (48,)
        return cast(jax.Array, physics.obs)  # (O,)

    def reset(self, key: jax.Array, params: PlaygroundParams) -> tuple[jax.Array, PlaygroundState]:  # key: ()
        """Return (O,) float32 observation and (...) physics state leaves."""
        physics = self.env.reset(key)  # leaves: (...)
        state = PlaygroundState(physics=physics, step_count=jnp.array(0, jnp.int32))
        return self._observation(physics), state  # (O,); state leaves: (...)

    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: PlaygroundState,  # leaves: (...)
        action: jax.Array,  # (A,) continuous controls
        params: PlaygroundParams,
    ) -> EnvStep[PlaygroundState]:
        """Advance physics and autoreset, retaining (O,) final_observation for bootstrapping."""
        physics = self.env.step(state.physics, action)  # leaves: (...)
        final_state = state.replace(physics=physics, step_count=state.step_count + 1)
        final_observation = self._observation(physics)  # (O,)
        terminated = physics.done.astype(jnp.bool_)  # ()
        truncated = final_state.step_count >= params.max_steps  # ()
        done = terminated | truncated  # ()

        def reset_done(_: None) -> tuple[jax.Array, PlaygroundState]:
            return self.reset(key, params)  # (O,); state leaves: (...)

        def continue_episode(_: None) -> tuple[jax.Array, PlaygroundState]:
            return final_observation, final_state  # (O,); state leaves: (...)

        observation, next_state = jax.lax.cond(done, reset_done, continue_episode, None)  # (O,); leaves: (...)
        return EnvStep(
            observation=observation,  # (O,)
            state=next_state,  # leaves: (...)
            reward=physics.reward,  # ()
            done=done,  # ()
            info={
                "terminated": terminated,  # ()
                "truncated": truncated,  # ()
                "final_observation": final_observation,  # (O,)
                "final_state": final_state,  # leaves: (...)
            },
        )

    def render(self, state: PlaygroundState) -> NDArray[np.uint8]:  # leaves: (...) -> (240, 320, 3)
        return self.env.render([state.physics], height=240, width=320)[0]  # (240, 320, 3)
