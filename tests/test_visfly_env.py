"""VisFly tensor snapshots and scene configuration."""

from contextlib import closing
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl2.envs.gym_style_env import replace_observation
from rl2.envs.visfly_env import VisFlyConfig, VisFlyEnv, convert_observation, host_observation


def test_nested_numpy_observations_preserve_layout_and_snapshots() -> None:
    image = np.arange(24, dtype=np.uint8).reshape(2, 1, 3, 4)  # (B, C, H, W)
    state = np.ones((2, 3), dtype=np.float64)  # (B, O)
    initial = host_observation({"sensors": (image, {"state": state})})  # nested (B, *O) leaves
    final = host_observation(initial)  # nested (B, *O) leaves
    terminal = {"sensors": (np.full((1, 3, 4), 99, dtype=np.uint8), {"state": np.zeros(3)})}  # (*O,) leaves
    replace_observation(final, terminal, index=1)
    converted = convert_observation(final, channels_first=True)  # nested (B, *O) leaves
    pixels, vectors = converted["sensors"]  # (B, H, W, C); {state: (B, O)}
    assert pixels.shape == (2, 3, 4, 1)
    assert pixels.dtype == np.uint8
    assert vectors["state"].shape == (2, 3)
    assert vectors["state"].dtype == np.float64
    np.testing.assert_array_equal(pixels[0, ..., 0], image[0, 0])
    np.testing.assert_array_equal(pixels[1], 99)
    np.testing.assert_array_equal(vectors["state"], [[1, 1, 1], [0, 0, 0]])
    image[:] = 0
    state[:] = 0
    np.testing.assert_array_equal(initial["sensors"][0], np.arange(24).reshape(2, 1, 3, 4))
    np.testing.assert_array_equal(initial["sensors"][1]["state"], 1)
    unchanged = convert_observation(initial, channels_first=False)  # nested (B, *O) leaves
    assert unchanged["sensors"][0].shape == (2, 1, 3, 4)


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


@pytest.mark.parametrize(
    ("num_envs", "num_scenes", "unbatched"),
    [
        pytest.param(1, 1, True, id="single_image"),
        pytest.param(1, 1, False, id="singleton_batch"),
        pytest.param(4, 2, False, id="shared_scene_cameras"),
    ],
)
def test_visfly_render_preserves_batch_and_scene_order(num_envs: int, num_scenes: int, unbatched: bool) -> None:
    pixels = torch.arange(num_scenes * 2 * 3 * 4).reshape(num_scenes, 2, 3, 4)  # (S, H, W, RGBA)
    backend = Mock(num_envs=num_envs, action_space=gym.spaces.Discrete(2))
    backend.render.return_value = pixels[0] if unbatched else pixels
    with (
        patch("rl2.envs.visfly_env.load_env_creator", return_value=Mock(return_value=backend)),
        closing(VisFlyEnv(config=VisFlyConfig(num_scenes=num_scenes), num_envs=num_envs, seed=0)) as env,
    ):
        frames = env.render()  # (B, H, W, 3)
        assert frames.shape == (num_envs, 2, 3, 3)
        assert frames.dtype == np.uint8
        for index in range(num_envs):
            scene_index = index // (num_envs // num_scenes)
            np.testing.assert_array_equal(frames[index], pixels[scene_index, ..., :3].numpy())
        snapshot = frames.copy()  # (B, H, W, 3)
        pixels.zero_()
        np.testing.assert_array_equal(frames, snapshot)


@pytest.mark.parametrize(
    ("frame_shape", "message"),
    [
        pytest.param((2, 3), "requires RGB arrays", id="missing_channels"),
        pytest.param((1, 2, 3, 2), "requires RGB arrays", id="invalid_channels"),
        pytest.param((2, 2, 3, 3), "one RGB array per scene", id="scene_count_mismatch"),
    ],
)
def test_visfly_render_rejects_invalid_images(frame_shape: tuple[int, ...], message: str) -> None:
    backend = Mock(num_envs=1, action_space=gym.spaces.Discrete(2))
    backend.render.return_value = np.zeros(frame_shape)
    with (
        patch("rl2.envs.visfly_env.load_env_creator", return_value=Mock(return_value=backend)),
        closing(VisFlyEnv(config=VisFlyConfig(), num_envs=1, seed=0)) as env,
        pytest.raises(ValueError, match=message),
    ):
        env.render()
