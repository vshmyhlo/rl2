"""Host-stepped VisFly batches with tensor snapshots and timeout bootstrapping.

Omit goal distances if a backend omits any completed episode's terminal observation.
"""

import random
from collections.abc import Generator
from contextlib import ExitStack, closing, contextmanager
from copy import deepcopy
from dataclasses import field, replace
from typing import Annotated, Any, Literal, NotRequired, TypedDict

import cv2
import jax
import numpy as np
import torch
from gymnasium.envs.registration import load_env_creator
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.envs.gym_style_env import (
    GymStyleEnv,
    GymStyleInfo,
    GymStyleStep,
    StopReason,
    replace_observation,
)
from rl2.envs.visfly.render import RenderContext, depth_panel, diagnostics_panel


class RawVisFlyObservation[Array](TypedDict):
    """NavigationEnv fields before conversion; *B is (B,) or () for terminal observations.

    S = 13 for RGBNavigationEnv; custom backends may use 12 with Euler orientation.
    """

    state: Array  # (*B, S): position, orientation, linear velocity, angular velocity
    target: Array  # (*B, 3): target position
    rgb: Array  # (*B, 3, H, W), or (*B, H, W, 3) if channels_first=False; uint8 pixels
    depth: NotRequired[Array]  # (*B, 1, H, W), or (*B, H, W, 1); float32 metres, when configured


class VisFlyObservation[Array](TypedDict):
    """Channels-last NavigationEnv fields; *B denotes batch/sequence dimensions or () for terminals."""

    state: Array  # (*B, S): same ordering and dtype as RawVisFlyObservation
    target: Array  # (*B, 3): target position
    rgb: Array  # (*B, H, W, 3): uint8 pixels
    depth: NotRequired[Array]  # (*B, H, W, 1): float32 metres, when configured


def to_numpy(value: Any) -> NDArray[Any]:
    """Return (*S,) owned host data from a VisFly tensor or NumPy value."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # (*S,)
    return np.array(value, copy=True)  # (*S,)


def convert_observation(
    value: RawVisFlyObservation[Any],  # state: (*B, S), target: (*B, 3), images: (*B, C, H, W) or (*B, H, W, C)
    channels_first: bool,
) -> VisFlyObservation[NDArray[Any]]:
    """Return (*B, S) state, (*B, 3) target, (*B, H, W, 3) RGB, and optional (*B, H, W, 1) depth.

    Detach tensors, move them to CPU, and copy all leaves into NumPy snapshots.
    Images change layout: (*B, C, H, W) -> (*B, H, W, C). With
    channels_first=False, input images must already be channels-last.
    Values and dtypes are preserved; no normalization or clipping is applied.
    """
    # VisFly's TensorDict subclass is not a registered JAX pytree.
    snapshot = jax.tree.map(to_numpy, dict(value))  # leaves: (*B, *O)
    rgb = snapshot["rgb"]  # (*B, 3, H, W) or (*B, H, W, 3)
    observation = VisFlyObservation(
        state=snapshot["state"],  # (*B, S)
        target=snapshot["target"],  # (*B, 3)
        rgb=np.moveaxis(rgb, -3, -1) if channels_first else rgb,  # (*B, H, W, 3)
    )
    if "depth" in snapshot:
        depth = snapshot["depth"]  # (*B, 1, H, W) or (*B, H, W, 1)
        observation["depth"] = np.moveaxis(depth, -3, -1) if channels_first else depth  # (*B, H, W, 1)
    return observation


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class VisFlyConfig:
    id: Literal["VisFly"] = "VisFly"
    entry_point: str = "rl2.envs.visfly.navigation:RGBNavigationEnv"  # Must implement RawVisFlyObservation.
    num_scenes: Annotated[int, Field(gt=0)] = 1
    scene_rotation_env_steps: Annotated[int, Field(gt=0)] | None = None  # Across all agents; checked per vector step.
    kwargs: dict[str, Any] = field(default_factory=dict, hash=False)
    channels_first: bool = True  # Backend images are CHW; False accepts already-HWC images.


class VisFlyEnv(GymStyleEnv[VisFlyObservation[NDArray[Any]]]):
    """Own a VisFly batch and its Python/CPU Torch RNGs; backend calls must run serially."""

    @contextmanager
    def _use_rng(self) -> Generator[None]:
        """Isolate scene-loader shuffling and CPU sampling, including autoresets."""
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self._rng_state)
            python_rng = random.getstate()
            random.setstate(self._python_rng_state)
            try:
                yield
            finally:
                self._rng_state = torch.get_rng_state()  # (R,) serialized CPU RNG state
                self._python_rng_state = random.getstate()
                random.setstate(python_rng)

    def __init__(self, config: VisFlyConfig, num_envs: int, seed: int, *, evaluation: bool = False) -> None:
        if evaluation:
            config = replace(config, num_scenes=1)
        self.config = config
        self.num_envs = num_envs
        self._scene_env_steps = 0
        self._contexts = [RenderContext(step=0, episode_return=0.0) for _ in range(num_envs)]
        self._last_contexts = self._contexts
        if num_envs % config.num_scenes:
            raise ValueError("num_envs must be divisible by env.num_scenes")
        reserved = {"num_scene", "num_agent_per_scene", "seed", "requires_grad"} & config.kwargs.keys()
        if reserved:
            raise ValueError(f"VisFly kwargs are managed by the trainer: {sorted(reserved)}")
        kwargs = deepcopy(config.kwargs)
        if config.scene_rotation_env_steps is not None:
            kwargs.setdefault("random_kwargs", {})["scene_iter"] = True
        self._rng_state = torch.Generator().manual_seed(seed).get_state()  # (R,) serialized CPU RNG state
        self._python_rng_state = random.Random(seed).getstate()
        with self._use_rng(), ExitStack() as resources:
            creator = load_env_creator(config.entry_point)  # Upstream imports initialize a Torch model.
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
        """Return (B, S) state, (B, 3) target, (B, H, W, 3) RGB, and optional (B, H, W, 1) depth."""
        self._scene_env_steps = 0
        self._contexts = [RenderContext(step=0, episode_return=0.0) for _ in range(self.num_envs)]
        self._last_contexts = self._contexts
        with self._use_rng():
            return convert_observation(self.backend.reset(), self.config.channels_first)

    def step(self, actions: NDArray[Any]) -> GymStyleStep[VisFlyObservation[NDArray[Any]]]:  # (B, *action_shape)
        host_actions = np.array(actions, copy=True)  # (B, *action_shape)
        with self._use_rng():
            observations, rewards, done, infos = self.backend.step(host_actions)
        rewards = to_numpy(rewards).astype(np.float32, copy=False)  # (B,)
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
        stop_reasons = np.full(self.num_envs, StopReason.ONGOING, dtype=np.int32)  # (B,)
        has_final_observations = True
        for index in np.flatnonzero(done):
            info = infos[index]
            # Success wins simultaneous collisions; termination wins simultaneous timeouts.
            if terminated[index]:
                if info.get("is_success", False):
                    reason = StopReason.SUCCESS
                elif info.get("episode", {}).get("extra", {}).get("collision", False):
                    reason = StopReason.COLLISION
                else:
                    reason = StopReason.TERMINATED
            else:
                reason = StopReason.TIME_LIMIT
            stop_reasons[index] = reason
            if truncated[index] and not terminated[index] and "terminal_observation" not in infos[index]:
                raise ValueError("VisFly timeouts require terminal_observation")
            if "terminal_observation" in infos[index]:
                terminal = convert_observation(
                    infos[index]["terminal_observation"], self.config.channels_first
                )  # state: (S,), target: (3,), rgb: (H, W, 3)
                replace_observation(final, terminal, index=int(index))
            else:
                has_final_observations = False
        contexts = self._contexts  # Preserve episode totals across scene resets.
        self._scene_env_steps += self.num_envs
        interval = self.config.scene_rotation_env_steps
        if interval is not None and self._scene_env_steps >= interval:
            # End ongoing episodes at the old scene state, then reload all scenes together.
            stop_reasons[~done] = StopReason.SCENE_ROTATION
            truncated |= ~done  # (B,)
            observations = self.reset()  # leaves: (B, *O)
        self._last_contexts = [
            RenderContext(
                step=context.step + 1,
                episode_return=context.episode_return + float(rewards[index]),
                reward=float(rewards[index]),
                action=host_actions[index],  # (A,)
                stop_reason=int(stop_reasons[index]) if terminated[index] or truncated[index] else None,
            )
            for index, context in enumerate(contexts)
        ]
        self._contexts = [
            RenderContext(step=0, episode_return=0.0) if terminated[index] or truncated[index] else context
            for index, context in enumerate(self._last_contexts)
        ]
        info = GymStyleInfo(stop_reason=stop_reasons)  # leaves: (B,)
        if has_final_observations:
            info["goal_distance"] = np.linalg.norm(final["target"] - final["state"][:, :3], axis=-1).astype(
                np.float32
            )  # (B,)
        return GymStyleStep(
            observations=observations,  # state: (B, S), target: (B, 3), rgb: (B, H, W, 3)
            rewards=rewards,  # (B,)
            terminated=terminated,  # (B,)
            truncated=truncated,  # (B,)
            final_observations=final,  # state: (B, S), target: (B, 3), rgb: (B, H, W, 3)
            info=info,
        )

    def _diagnostics_panel(
        self,
        observation: VisFlyObservation[NDArray[Any]],  # leaves: (B, *O)
        *,
        index: int,
        height: int,
        context: RenderContext,
    ) -> NDArray[np.uint8]:  # (H, W_debug, 3)
        return diagnostics_panel(
            observation["state"][index],  # (S,)
            observation["target"][index],  # (3,)
            height=height,
            action_low=self.action_space.low,  # (A,)
            action_high=self.action_space.high,  # (A,)
            action_type=self.config.kwargs.get("dynamics_kwargs", {}).get("action_type", "bodyrate"),
            context=context,
            max_episode_steps=self.config.kwargs.get("max_episode_steps"),
        )  # (H, W_debug, 3)

    def render(self) -> NDArray[np.uint8]:  # (B, H, W_scene + W_agent + W_depth + W_debug, 3)
        """Return (B, H, W_scene + W_agent + W_depth + W_debug, 3) scene/RGB/depth/diagnostics.

        Scale camera views to scene height with nearest-neighbor sampling, preserving aspect ratio.
        W_depth is zero without depth; otherwise show a fixed 0–20 m inferno colour scale.
        W_debug is round(1.5 * H). Diagnostics include per-agent episode and last-action details.
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
        observation = convert_observation(self.backend.get_observation(), self.config.channels_first)  # leaves: (B, *O)
        rgb = observation["rgb"]  # (B, H_agent, W_agent, 3)
        height = frames.shape[1]
        width = max(1, round(rgb.shape[2] * height / rgb.shape[1]))
        agent_frames = np.stack(
            [cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST) for frame in rgb]
        )  # (B, H, W_agent, 3)
        views = [frames, agent_frames]  # each (B, H, W_view, 3)
        if "depth" in observation:
            views.append(
                np.stack([depth_panel(depth, height=height) for depth in observation["depth"]])
            )  # (B, H, W_depth, 3)
        panels = np.stack(
            [
                self._diagnostics_panel(
                    observation,
                    index=index,
                    height=height,
                    context=self._contexts[index],
                )
                for index in range(self.num_envs)
            ]
        )  # (B, H, W_debug, 3)
        return np.concatenate((*views, panels), axis=2)  # (B, H, W_scene + W_agent + W_depth + W_debug, 3)

    def render_outcome(
        self,
        frame: NDArray[np.uint8],  # (H, W_scene + W_agent + W_depth + W_debug, 3)
        observation: VisFlyObservation[NDArray[Any]],  # leaves: (B, *O), before autoreset
        *,
        index: int = 0,
    ) -> NDArray[np.uint8]:  # (H, W_scene + W_agent + W_depth + W_debug, 3)
        """Return (H, W, 3) diagnostics for the latest transition with labeled last pre-action views."""
        panel = self._diagnostics_panel(
            observation, index=index, height=frame.shape[0], context=self._last_contexts[index]
        )  # (H, W_debug, 3)
        result = frame.copy()  # (H, W, 3)
        result[:, -panel.shape[1] :] = panel
        return result  # (H, W, 3)

    def close(self) -> None:
        self.backend.close()
