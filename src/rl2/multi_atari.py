"""Raw Atari game mixture, registered as ``rl2/MultiAtari-v0``.

The default pool is Atari-57. All of these games share native 210x160 RGB
observations. Preprocessing, resizing, and frame stacking belong outside this
environment, in ``ppo.make_env``. Custom pools must share a native screen size.

Each reset samples uniformly with replacement; rewards keep their native scale.
Game identity is provided in info, not added to the policy's observation.
"""

from collections.abc import Sequence
from typing import Any, ClassVar, SupportsFloat

import ale_py
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from rl2.atari_scores import get_reference_scores

type Observation = NDArray[np.uint8]
type Action = int | np.integer[Any]

ENV_ID = "rl2/MultiAtari-v0"
ACTION_MEANINGS = tuple(ale_py.Action(action).name for action in range(18))


class FullAtariActionSpace(gym.ActionWrapper):
    """Map the 18 console actions to a game's local legal-action indices.

    ALE's full legal set still excludes FIRE for Skiing. Treat fire as an
    ineffective button there, preserving the direction of compound actions.
    """

    def __init__(self, env: gym.Env[Observation, Action]) -> None:
        super().__init__(env)
        local_actions = {meaning: index for index, meaning in enumerate(env.unwrapped.get_action_meanings())}
        mapping: list[int] = []
        for meaning in ACTION_MEANINGS:
            local_meaning = meaning if meaning in local_actions else meaning.removesuffix("FIRE") or "NOOP"
            if local_meaning not in local_actions:
                raise ValueError(f"Cannot map Atari action {meaning} to legal actions {tuple(local_actions)}")
            mapping.append(local_actions[local_meaning])
        self._action_map = tuple(mapping)
        self.action_space = gym.spaces.Discrete(len(ACTION_MEANINGS))

    def action(self, action: Action) -> int:
        if not self.action_space.contains(action):
            raise gym.error.InvalidAction(f"Expected an Atari action in [0, 17], got {action!r}")
        return self._action_map[int(action)]

    def get_action_meanings(self) -> list[str]:
        return list(ACTION_MEANINGS)


class MultiAtariEnv(gym.Env[Observation, Action]):
    """Switch raw emulators at reset while preserving the full 18-action mapping.

    Exposes ``ale``, ``_frameskip``, and ``get_action_meanings`` so Gymnasium's
    AtariPreprocessing can wrap the mixture and always read the active emulator.
    Only one emulator is kept open. Explicit reset may abandon an episode.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        env_ids: Sequence[str] | None = None,
        *,
        frameskip: int = 1,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        gym.register_envs(ale_py)
        if env_ids is None:
            env_ids = sorted(env_id for env_id in gym.registry if get_reference_scores(env_id) is not None)
        if isinstance(env_ids, str) or not env_ids:
            raise ValueError("env_ids must be a nonempty sequence of ALE/<Game>-v5 IDs")
        if any(not env_id.startswith("ALE/") or not env_id.endswith("-v5") for env_id in env_ids):
            raise ValueError("Use explicit ALE/<Game>-v5 environment IDs")
        if len(set(env_ids)) != len(env_ids):
            raise ValueError("env_ids must be unique for uniform game sampling")
        if type(frameskip) is not int or frameskip != 1:
            raise ValueError("Use frameskip=1; apply action repeat with AtariPreprocessing outside the mixture")
        self.env_ids = tuple(env_ids)
        self.render_mode = render_mode
        self._frameskip = frameskip
        # Wrappers need the raw spaces and ALE interface before the first reset.
        self.env_id: str | None = self.env_ids[0]
        self.env: gym.Env[Observation, Action] | None = self._make_env(self.env_id)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = dict(self.env.metadata)
        self._needs_reset = True

    def _make_env(self, env_id: str) -> gym.Env[Observation, Action]:
        env = gym.make(
            env_id,
            obs_type="rgb",
            frameskip=self._frameskip,
            full_action_space=True,
            repeat_action_probability=0.25,
            render_mode=self.render_mode,
        )
        try:
            return FullAtariActionSpace(env)
        except Exception:
            env.close()
            raise

    @property
    def ale(self) -> ale_py.ALEInterface:
        if self.env is None:
            raise gym.error.ResetNeeded("Call reset() to open an emulator")
        return self.env.unwrapped.ale

    def get_action_meanings(self) -> list[str]:
        return list(ACTION_MEANINGS)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self._needs_reset = True
        env_id = self.env_ids[int(self.np_random.integers(len(self.env_ids)))]
        episode_seed = int(self.np_random.integers(2**31))
        if self.env is None or env_id != self.env_id:
            self.close()
            env = self._make_env(env_id)
            if env.observation_space != self.observation_space or env.action_space != self.action_space:
                env.close()
                raise ValueError(f"{env_id} does not share the pool's observation/action spaces")
            self.env = env
            self.env_id = env_id
            self.metadata.update(env.metadata)
        observation, info = self.env.reset(seed=episode_seed, options=options)
        self._needs_reset = False
        return observation, {**info, "env_id": self.env_id}

    def step(self, action: Action) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.env is None or self._needs_reset:
            raise gym.error.ResetNeeded("Call reset() before stepping a new episode")
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._needs_reset = terminated or truncated
        return observation, reward, terminated, truncated, {**info, "env_id": self.env_id}

    def render(self) -> NDArray[np.uint8] | None:
        if self.env is None:
            raise gym.error.ResetNeeded("Call reset() before rendering")
        return self.env.render()

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
        self.env_id = None
        self._needs_reset = True


def register_envs() -> None:
    """Register the mixture in each process, including spawned PPO workers."""
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point="rl2.multi_atari:MultiAtariEnv")
