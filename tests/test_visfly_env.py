"""VisFly tensor snapshots and scene configuration."""

from contextlib import closing
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl2.envs.visfly_env import RawVisFlyObservation, VisFlyConfig, VisFlyEnv, convert_observation


@pytest.mark.parametrize(
    ("batch_shape", "channels_first"),
    [
        pytest.param((2,), True, id="batch"),
        pytest.param((1,), True, id="singleton_batch"),
        pytest.param((), True, id="terminal"),
        pytest.param((2,), False, id="already_channels_last"),
    ],
)
def test_navigation_observation_layout_and_snapshots(batch_shape: tuple[int, ...], channels_first: bool) -> None:
    state = np.ones((*batch_shape, 13), dtype=np.float64)  # (*B, S)
    target = np.full((*batch_shape, 3), 9, dtype=np.float32)  # (*B, 3)
    rgb_shape = (*batch_shape, 3, 2, 4)
    rgb = np.arange(np.prod(rgb_shape), dtype=np.uint8).reshape(rgb_shape)  # (*B, 3, H, W)
    expected_rgb = np.moveaxis(rgb, -3, -1).copy()  # (*B, H, W, 3)
    raw = RawVisFlyObservation(
        state=state,
        target=target,
        rgb=rgb if channels_first else np.moveaxis(rgb, -3, -1),
    )
    converted = convert_observation(raw, channels_first=channels_first)
    assert set(converted) == {"state", "target", "rgb"}
    assert converted["state"].shape == (*batch_shape, 13)
    assert converted["target"].shape == (*batch_shape, 3)
    assert converted["rgb"].shape == (*batch_shape, 2, 4, 3)
    assert converted["state"].dtype == np.float64
    assert converted["target"].dtype == np.float32
    assert converted["rgb"].dtype == np.uint8
    state[:] = 0
    target[:] = 0
    rgb[:] = 0
    np.testing.assert_array_equal(converted["state"], 1)
    np.testing.assert_array_equal(converted["target"], 9)
    np.testing.assert_array_equal(converted["rgb"], expected_rgb)


def test_tensor_observations_are_detached_snapshots() -> None:
    state = torch.ones((2, 12), requires_grad=True)  # (B, S): Euler orientation
    target = torch.full((2, 3), 9.0, requires_grad=True)  # (B, 3)
    rgb = torch.full((2, 3, 2, 4), 255, dtype=torch.uint8)  # (B, 3, H, W)
    snapshot = convert_observation(RawVisFlyObservation(state=state, target=target, rgb=rgb), channels_first=True)
    assert set(snapshot) == {"state", "target", "rgb"}
    assert snapshot["rgb"].shape == (2, 2, 4, 3)
    assert snapshot["rgb"].dtype == np.uint8
    assert snapshot["state"].shape == (2, 12)
    assert snapshot["target"].shape == (2, 3)
    assert snapshot["state"].dtype == snapshot["target"].dtype == np.float32
    with torch.no_grad():
        state.zero_()
        target.zero_()
        rgb.zero_()
    np.testing.assert_array_equal(snapshot["state"], 1)
    np.testing.assert_array_equal(snapshot["target"], 9)
    np.testing.assert_array_equal(snapshot["rgb"], 255)


def test_visfly_dict_subclass_observations() -> None:
    class TensorDict(dict[str, torch.Tensor]):
        """VisFly's container is not registered as a JAX pytree."""

    state = torch.ones((2, 13))  # (B, S)
    observation = TensorDict(
        state=state,  # (B, S)
        target=torch.zeros((2, 3)),  # (B, 3)
        rgb=torch.zeros((2, 3, 2, 4), dtype=torch.uint8),  # (B, 3, H, W)
    )
    backend = Mock(num_envs=2, action_space=gym.spaces.Box(low=-1, high=1, shape=(4,)))
    backend.reset.return_value = observation
    backend.step.return_value = (
        observation,
        torch.zeros(2),  # (B,)
        torch.tensor([True, False]),  # (B,)
        [
            {
                "TimeLimit.truncated": True,
                "terminal_observation": TensorDict(
                    state=torch.full((13,), 2.0),  # (S,)
                    target=torch.full((3,), 9.0),  # (3,)
                    rgb=torch.arange(24, dtype=torch.uint8).reshape(3, 2, 4),  # (3, H, W)
                ),
            },
            {},
        ],
    )
    with (
        patch("rl2.envs.visfly_env.load_env_creator", return_value=Mock(return_value=backend)),
        closing(VisFlyEnv(config=VisFlyConfig(), num_envs=2, seed=0)) as env,
    ):
        initial = env.reset()  # leaves: (B, O)
        assert isinstance(initial, dict)
        assert initial["state"].shape == (2, 13)
        result = env.step(np.zeros((2, 4), dtype=np.float32))
        assert result.observations["state"].shape == (2, 13)
        np.testing.assert_array_equal(result.final_observations["state"][0], 2)
        np.testing.assert_array_equal(result.final_observations["state"][1], 1)
        np.testing.assert_array_equal(result.final_observations["target"], [[9, 9, 9], [0, 0, 0]])
        assert initial["rgb"].shape == result.final_observations["rgb"].shape == (2, 2, 4, 3)
        np.testing.assert_array_equal(
            result.final_observations["rgb"][0], np.moveaxis(np.arange(24).reshape(3, 2, 4), 0, -1)
        )
        np.testing.assert_array_equal(result.final_observations["rgb"][1], 0)
        np.testing.assert_array_equal(result.observations["rgb"], 0)
        state.zero_()
        np.testing.assert_array_equal(initial["state"], 1)
        np.testing.assert_array_equal(result.observations["state"], 1)


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


def test_scene_rotation_truncates_ongoing_episodes_with_old_scene_observations() -> None:
    def observation(value: int) -> RawVisFlyObservation[np.ndarray]:
        return RawVisFlyObservation(
            state=np.full((4, 13), value, dtype=np.float32),  # (B, S)
            target=np.full((4, 3), value, dtype=np.float32),  # (B, 3)
            rgb=np.full((4, 3, 2, 4), value, dtype=np.uint8),  # (B, 3, H, W)
        )

    backend = Mock(num_envs=4, action_space=gym.spaces.Box(low=-1, high=1, shape=(4,)))
    backend.reset.side_effect = [observation(0), observation(100), observation(200)]
    terminal = {key: value[0] for key, value in observation(10).items()}  # leaves: (*O,)
    backend.step.return_value = (
        observation(1),
        np.ones(4),  # (B,)
        np.array([False, True, True, False]),  # (B,)
        [
            {},
            {"episode_done": True, "terminal_observation": terminal},
            {"TimeLimit.truncated": True, "terminal_observation": terminal},
            {},
        ],
    )
    config = VisFlyConfig(scene_rotation_env_steps=7, kwargs={"random_kwargs": {"scene_iter": False}})
    with (
        patch("rl2.envs.visfly_env.load_env_creator", return_value=Mock(return_value=backend)) as loader,
        closing(VisFlyEnv(config=config, num_envs=4, seed=0)) as env,
    ):
        env.reset()
        first = env.step(np.zeros((4, 4)))
        np.testing.assert_array_equal(first.truncated, [False, False, True, False])
        assert backend.reset.call_count == 1
        rotated = env.step(np.zeros((4, 4)))
        np.testing.assert_array_equal(rotated.terminated, [False, True, False, False])
        np.testing.assert_array_equal(rotated.truncated, [True, False, True, True])
        np.testing.assert_array_equal(rotated.observations["state"], 100)
        np.testing.assert_array_equal(rotated.final_observations["state"][:, 0], [1, 10, 10, 1])
        np.testing.assert_array_equal(rotated.final_observations["target"][:, 0], [1, 10, 10, 1])
        np.testing.assert_array_equal(rotated.final_observations["rgb"][:, 0, 0, 0], [1, 10, 10, 1])
        env.step(np.zeros((4, 4)))
        assert backend.reset.call_count == 2
        env.reset()
        env.step(np.zeros((4, 4)))
        assert backend.reset.call_count == 3  # Explicit resets also restart the rotation counter.
        assert loader.return_value.call_args.kwargs["random_kwargs"]["scene_iter"] is True
        assert config.kwargs["random_kwargs"]["scene_iter"] is False


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
