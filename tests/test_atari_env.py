from typing import Any
from unittest.mock import Mock, patch

import cv2
import gymnasium as gym
import numpy as np
import pytest

from rl2 import atari_env
from rl2.atari_env import make_env


@pytest.mark.parametrize("stacked", (False, True))
def test_resize_only_preserves_emulator_transitions(stacked: bool) -> None:
    raw = make_env("ALE/SpaceInvaders-v5", frame_stack=stacked)
    resized = make_env("ALE/SpaceInvaders-v5", frame_stack=stacked, observation_size=84)
    try:
        original, raw_info = raw.reset(seed=7)
        observation, resized_info = resized.reset(seed=7)
        assert raw_info == resized_info
        for action in (None, 0, 1, 2, 3, 0):
            if action is not None:
                original, *raw_transition = raw.step(action)
                observation, *resized_transition = resized.step(action)
                assert raw_transition == resized_transition
            expected = np.stack([cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA) for frame in original])
            np.testing.assert_array_equal(observation, expected)
            assert observation.dtype == np.uint8
            assert resized.observation_space.contains(observation)
            assert raw.unwrapped.ale.getFrameNumber() == resized.unwrapped.ale.getFrameNumber()
    finally:
        raw.close()
        resized.close()


def test_make_env_closes_emulator_when_wrapping_fails() -> None:
    env = Mock()
    with (
        patch.object(atari_env.gym, "make", return_value=env),
        patch.object(atari_env, "AtariPreprocessing", side_effect=ValueError("invalid preprocessing")),
        pytest.raises(ValueError, match="invalid preprocessing"),
    ):
        make_env("ALE/Pong-v5", atari_preprocessing=True)
    env.close.assert_called_once()


@pytest.mark.parametrize("frame_limit", (101, 102, 103, 104))
def test_atari_timeout_observation_is_current(frame_limit: int) -> None:
    # End on each offset within action repeat, including before pooling starts.
    gym_make = gym.make

    def limited_env(*args: Any, **kwargs: Any) -> gym.Env:
        return gym_make(*args, **kwargs, max_num_frames_per_episode=frame_limit)

    with patch("rl2.atari_env.gym.make", new=limited_env):
        env = make_env("ALE/Pong-v5", atari_preprocessing=True)
    try:
        env.env.noop_max = 0
        obs, _ = env.reset(seed=1)
        for _ in range(26):
            obs, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        assert truncated
        assert not terminated
        expected = cv2.resize(env.unwrapped.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(obs[-1], expected)
    finally:
        env.close()


class SyntheticAtariEnv(gym.Env):
    """Expose a distinct screen on each raw step, including the final step."""

    def __init__(self, end_step: int, terminated: bool) -> None:
        self.observation_space = gym.spaces.Box(0, 255, (210, 160, 3), np.uint8)
        self.action_space = gym.spaces.Discrete(18)
        self._frameskip = 1
        self.ale = self
        self.end_step = end_step
        self.terminated = terminated
        self.steps = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(self.observation_space.shape, np.uint8), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.steps += 1
        done = self.steps >= self.end_step
        return (
            np.full(self.observation_space.shape, self.steps, np.uint8),
            0.0,
            done and self.terminated,
            done and not self.terminated,
            {},
        )

    def lives(self) -> int:
        return 1

    def getScreenGrayscale(self, output: np.ndarray) -> None:
        output.fill(self.steps)

    def getScreenRGB(self, output: np.ndarray) -> None:
        output.fill(self.steps)


@pytest.mark.parametrize(
    "end_step,terminated,grayscale", [(1, False, False), (2, True, True), (3, False, True), (4, True, False)]
)
def test_preprocessing_returns_final_screen(end_step: int, terminated: bool, grayscale: bool) -> None:
    with atari_env.AtariPreprocessing(
        SyntheticAtariEnv(end_step, terminated), noop_max=0, grayscale_obs=grayscale
    ) as env:
        env.reset(seed=0)
        obs, _, actual_terminated, actual_truncated, _ = env.step(0)
        assert actual_terminated == terminated
        assert actual_truncated == (not terminated)
        np.testing.assert_array_equal(obs, np.full(obs.shape, end_step, np.uint8))
