"""Gymnasium Atari helpers for the world-model trainer and Atari evaluation."""

from typing import Any, SupportsFloat

import ale_py
import chex
import gymnasium as gym
from numpy import integer, uint8
from numpy.typing import NDArray

from rl2.multi_atari import register_envs


class AtariPreprocessing(gym.wrappers.AtariPreprocessing):
    def step(
        self,
        action: int | integer[Any],
    ) -> tuple[NDArray[Any], SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)  # obs: (H, W[, C])
        if terminated or truncated:
            # Gymnasium breaks action repeat before capturing the final screen.
            # Return the actual final screen, including for timeout bootstrapping.
            capture = self.ale.getScreenGrayscale if self.grayscale_obs else self.ale.getScreenRGB
            capture(self.obs_buffer[0])
            self.obs_buffer[1].fill(0)
            obs = self._get_obs()  # (H, W[, C])
        return obs, reward, terminated, truncated, info  # obs: (H, W[, C])


def make_env(
    env_id: str,
    render_mode: str | None = None,
    frame_stack: bool = False,
    atari_preprocessing: bool = False,
    observation_size: int | None = None,
    grayscale_obs: bool = True,
) -> gym.Env[NDArray[uint8], int | integer[Any]]:
    chex.assert_type(grayscale_obs, bool)
    if observation_size is not None and (type(observation_size) is not int or observation_size < 1):
        raise ValueError("observation_size must be a positive integer or null")
    gym.register_envs(ale_py)
    register_envs()
    env = gym.make(env_id, frameskip=1, render_mode=render_mode)
    try:
        if atari_preprocessing:
            env = AtariPreprocessing(env, grayscale_obs=grayscale_obs)
        if observation_size is not None:
            env = gym.wrappers.ResizeObservation(env, (observation_size, observation_size))
        if frame_stack:
            return gym.wrappers.FrameStackObservation(env, stack_size=4)
        return gym.wrappers.ReshapeObservation(env, (1, *env.observation_space.shape))
    except BaseException:
        env.close()
        raise
