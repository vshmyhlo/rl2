"""VisFly tensor snapshots and scene configuration."""

from contextlib import closing
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl2.envs.visfly_env import VisFlyConfig, VisFlyEnv, host_observation


def test_nested_tensor_observations_are_detached_snapshots() -> None:
    state = torch.ones((2, 3), requires_grad=True)  # (B, O)
    image = np.zeros((2, 1, 3, 4), dtype=np.uint8)  # (B, C, H, W)
    snapshot = host_observation({"sensors": (state, {"image": image})})  # leaves: (B, *O)
    vectors, sensors = snapshot["sensors"]  # (B, O); {image: (B, C, H, W)}
    assert vectors.shape == (2, 3)
    assert vectors.dtype == np.float32
    assert sensors["image"].shape == (2, 1, 3, 4)
    assert sensors["image"].dtype == np.uint8
    with torch.no_grad():
        state.zero_()
    image[:] = 1
    np.testing.assert_array_equal(vectors, 1)
    np.testing.assert_array_equal(sensors["image"], 0)


def test_invalid_scene_count_fails_before_loading_backend() -> None:
    with (
        patch("rl2.envs.visfly_env.load_env_creator") as loader,
        pytest.raises(ValueError, match="num_scenes"),
    ):
        VisFlyEnv(config=VisFlyConfig(num_scenes=3), num_envs=2, seed=0)
    loader.assert_not_called()


def test_evaluation_uses_one_scene() -> None:
    backend = Mock(num_envs=1, action_space=gym.spaces.Discrete(2))
    config = VisFlyConfig(num_scenes=2)
    with (
        patch("rl2.envs.visfly_env.load_env_creator", return_value=Mock(return_value=backend)) as loader,
        closing(VisFlyEnv(config=config, num_envs=1, seed=17, evaluation=True)) as env,
    ):
        assert env.config.num_scenes == 1
        assert config.num_scenes == 2
        loader.return_value.assert_called_once_with(num_scene=1, num_agent_per_scene=1, seed=17, requires_grad=False)
    backend.close.assert_called_once_with()


def test_visfly_count_mismatch_closes_backend() -> None:
    backend = Mock(num_envs=1, action_space=gym.spaces.Discrete(2))
    with (
        patch("rl2.envs.visfly_env.load_env_creator", return_value=Mock(return_value=backend)),
        pytest.raises(ValueError, match="environment count"),
    ):
        VisFlyEnv(config=VisFlyConfig(), num_envs=2, seed=0)
    backend.close.assert_called_once_with()
