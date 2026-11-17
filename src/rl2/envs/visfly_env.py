"""Host-stepped VisFly batches with tensor snapshots and timeout bootstrapping."""

from contextlib import ExitStack, closing
from copy import deepcopy
from dataclasses import field, replace
from typing import Annotated, Any, Literal, TypedDict

import cv2
import jax
import numpy as np
from gymnasium.envs.registration import load_env_creator
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.envs.gym_style_env import (
    GymStyleEnv,
    GymStyleStep,
    replace_observation,
)


class RawVisFlyObservation[Array](TypedDict):
    """NavigationEnv fields before conversion; *B is (B,) or () for terminal observations.

    S = 12 with Euler orientation or 13 with quaternion orientation (the default).
    """

    state: Array  # (*B, S): position, orientation, linear velocity, angular velocity
    target: Array  # (*B, 3): target position
    rgb: Array  # (*B, 3, H, W), or (*B, H, W, 3) if channels_first=False; uint8 pixels


class VisFlyObservation[Array](TypedDict):
    """Channels-last NavigationEnv fields; *B denotes batch/sequence dimensions or () for terminals."""

    state: Array  # (*B, S): same ordering and dtype as RawVisFlyObservation
    target: Array  # (*B, 3): target position
    rgb: Array  # (*B, H, W, 3): uint8 pixels


def to_numpy(value: Any) -> NDArray[Any]:
    """Return (*S,) owned host data from a VisFly tensor or NumPy value."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # (*S,)
    return np.array(value, copy=True)  # (*S,)


def convert_observation(
    value: RawVisFlyObservation[Any],  # state: (*B, S), target: (*B, 3), rgb: (*B, 3, H, W)
    channels_first: bool,
) -> VisFlyObservation[NDArray[Any]]:
    """Return (*B, S) state, (*B, 3) target, and (*B, H, W, 3) RGB.

    Detach tensors, move them to CPU, and copy all leaves into NumPy snapshots.
    Only RGB changes layout: (*B, 3, H, W) -> (*B, H, W, 3). With
    channels_first=False, input RGB must already be channels-last.
    Values and dtypes are preserved; no normalization or clipping is applied.
    """
    # VisFly's TensorDict subclass is not a registered JAX pytree.
    snapshot = jax.tree.map(to_numpy, dict(value))  # leaves: (*B, *O)
    rgb = snapshot["rgb"]  # (*B, 3, H, W) or (*B, H, W, 3)
    return VisFlyObservation(
        state=snapshot["state"],  # (*B, S)
        target=snapshot["target"],  # (*B, 3)
        rgb=np.moveaxis(rgb, -3, -1) if channels_first else rgb,  # (*B, H, W, 3)
    )


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class VisFlyConfig:
    id: Literal["VisFly"] = "VisFly"
    entry_point: str = "rl2.envs.visfly.navigation:RGBNavigationEnv"  # Must implement RawVisFlyObservation.
    num_scenes: Annotated[int, Field(gt=0)] = 1
    scene_rotation_env_steps: Annotated[int, Field(gt=0)] | None = None  # Across all agents; checked per vector step.
    kwargs: dict[str, Any] = field(default_factory=dict, hash=False)
    channels_first: bool = True  # Backend RGB is CHW; False accepts already-HWC RGB.


class VisFlyEnv(GymStyleEnv[VisFlyObservation[NDArray[Any]]]):
    """Own a VisFly autoresetting batch, seeded at construction."""

    def __init__(self, config: VisFlyConfig, num_envs: int, seed: int, *, evaluation: bool = False) -> None:
        if evaluation:
            config = replace(config, num_scenes=1)
        self.config = config
        self.num_envs = num_envs
        self._scene_env_steps = 0
        if num_envs % config.num_scenes:
            raise ValueError("num_envs must be divisible by env.num_scenes")
        reserved = {"num_scene", "num_agent_per_scene", "seed", "requires_grad"} & config.kwargs.keys()
        if reserved:
            raise ValueError(f"VisFly kwargs are managed by the trainer: {sorted(reserved)}")
        creator = load_env_creator(config.entry_point)
        kwargs = deepcopy(config.kwargs)
        if config.scene_rotation_env_steps is not None:
            kwargs.setdefault("random_kwargs", {})["scene_iter"] = True
        with ExitStack() as resources:
            self.backend = resources.enter_context(
                closing(
                    creator(
                        num_scene=config.num_scenes,
                        num_agent_per_scene=num_envs // config.num_scenes,
                        seed=seed,
                        requires_grad=False,
                        **kwargs,
                    )
                )
            )
            self.action_space = self.backend.action_space
            if self.backend.num_envs != num_envs:
                raise ValueError("VisFly environment count does not match num_envs")
            resources.pop_all()

    def reset(self) -> VisFlyObservation[NDArray[Any]]:
        """Return (B, S) state, (B, 3) target, and (B, H, W, 3) RGB."""
        self._scene_env_steps = 0
        return convert_observation(self.backend.reset(), self.config.channels_first)

    def step(self, actions: NDArray[Any]) -> GymStyleStep[VisFlyObservation[NDArray[Any]]]:  # (B, *action_shape)
        host_actions = np.array(actions, copy=True)  # (B, *action_shape)
        observations, rewards, done, infos = self.backend.step(host_actions)
        done = to_numpy(done).astype(bool)  # (B,)
        truncated = np.array(
            [done[i] and info.get("TimeLimit.truncated", False) for i, info in enumerate(infos)]
        )  # (B,)
        # VisFly can terminate on the same step as its horizon; termination wins.
        terminated = np.array(
            [done[i] and info.get("episode_done", not truncated[i]) for i, info in enumerate(infos)]
        )  # (B,)
        observations = convert_observation(observations, self.config.channels_first)  # rgb: (B, H, W, 3)
        final = jax.tree.map(to_numpy, observations)  # leaves: (B, *O)
        for index in np.flatnonzero(done):
            if truncated[index] and not terminated[index] and "terminal_observation" not in infos[index]:
                raise ValueError("VisFly timeouts require terminal_observation")
            if "terminal_observation" in infos[index]:
                terminal = convert_observation(
                    infos[index]["terminal_observation"], self.config.channels_first
                )  # state: (S,), target: (3,), rgb: (H, W, 3)
                replace_observation(final, terminal, index=int(index))
        self._scene_env_steps += self.num_envs
        interval = self.config.scene_rotation_env_steps
        if interval is not None and self._scene_env_steps >= interval:
            # End ongoing episodes at the old scene state, then reload all scenes together.
            truncated |= ~done  # (B,)
            observations = self.reset()  # leaves: (B, *O)
        return GymStyleStep(
            observations=observations,  # state: (B, S), target: (B, 3), rgb: (B, H, W, 3)
            rewards=to_numpy(rewards).astype(np.float32, copy=False),  # (B,)
            terminated=to_numpy(terminated).astype(bool, copy=False),  # (B,)
            truncated=to_numpy(truncated).astype(bool, copy=False),  # (B,)
            final_observations=final,  # state: (B, S), target: (B, 3), rgb: (B, H, W, 3)
        )

    def render(self) -> NDArray[np.uint8]:  # (B, H, W_scene + W_agent, 3)
        """Return (B, H, W_scene + W_agent, 3) uint8 scene/agent views, left/right.

        Scale agent RGB to the scene height with nearest-neighbor sampling, preserving aspect ratio.
        """
        frames = to_numpy(self.backend.render())  # (S, H, W, C) or (H, W, C); S: scenes
        if frames.ndim == 3:
            frames = frames[None]  # (1, H, W, C)
        if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
            raise ValueError("Rendering requires RGB arrays; configure VisFly render settings")
        if frames.shape[0] != self.config.num_scenes:
            raise ValueError("VisFly rendering requires one RGB array per scene")
        frames = frames[..., :3].astype(np.uint8)  # (S, H, W, 3)
        frames = np.repeat(frames, self.num_envs // self.config.num_scenes, axis=0)  # (B, H, W_scene, 3)
        rgb = to_numpy(self.backend.get_observation()["rgb"])  # (B, 3, H_agent, W_agent) or (B, H_agent, W_agent, 3)
        if self.config.channels_first:
            rgb = np.moveaxis(rgb, 1, -1)  # (B, H_agent, W_agent, 3)
        height = frames.shape[1]
        width = max(1, round(rgb.shape[2] * height / rgb.shape[1]))
        agent_frames = np.stack(
            [cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST) for frame in rgb]
        )  # (B, H, W_agent, 3)
        return np.concatenate((frames, agent_frames), axis=2)  # (B, H, W_scene + W_agent, 3)

    def close(self) -> None:
        self.backend.close()
