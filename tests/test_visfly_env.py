"""VisFly tensor snapshots and scene configuration."""

import random
from contextlib import closing
from typing import Any, NamedTuple
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl2.envs.gym_style_env import StopReason
from rl2.envs.visfly.render import depth_panel
from rl2.envs.visfly_env import RawVisFlyObservation, VisFlyConfig, VisFlyEnv, convert_observation


class RandomTransition(NamedTuple):
    observation: RawVisFlyObservation[torch.Tensor]  # leaves: (B, *O)
    rewards: torch.Tensor  # (B,)
    done: torch.Tensor  # (B,)
    infos: list[dict[str, Any]]


def random_backend(*, seed: int, **kwargs: Any) -> Mock:
    assert torch.is_inference_mode_enabled()
    torch.manual_seed(seed)  # Match upstream construction's global seeding and draws.
    torch.rand(1)
    random.random()  # The upstream scene/object loaders also consume Python's global RNG.
    backend = Mock(num_envs=1, action_space=gym.spaces.Box(low=-1, high=1, shape=(4,)))

    def observation() -> RawVisFlyObservation[torch.Tensor]:
        assert torch.is_inference_mode_enabled()
        return RawVisFlyObservation(
            state=torch.rand((1, 13)),  # (B, S)
            target=torch.rand((1, 3)) + random.random(),  # (B, 3)
            rgb=torch.zeros((1, 3, 2, 4), dtype=torch.uint8),  # (B, C, H, W)
        )

    def step(actions: np.ndarray) -> RandomTransition:  # (B, A)
        assert torch.is_inference_mode_enabled()
        # An autoreset also consumes the backend's global RNG inside step().
        terminal = observation()  # leaves: (B, *O)
        return RandomTransition(
            observation=observation(),
            rewards=torch.rand(1),  # (B,)
            done=torch.ones(1, dtype=torch.bool),  # (B,)
            infos=[{"terminal_observation": {key: value[0] for key, value in terminal.items()}}],
        )

    backend.reset.side_effect = observation
    backend.step.side_effect = step
    return backend


def test_rng_streams_are_independent_of_evaluation_and_global_draws() -> None:
    config = VisFlyConfig(scene_rotation_env_steps=1)
    actions = np.zeros((1, 4), dtype=np.float32)  # (B, A)
    with (
        torch.random.fork_rng(devices=[]),
        patch("rl2.envs.visfly_env.load_env_creator", return_value=random_backend),
    ):
        initial_rng = torch.get_rng_state()  # (R,)
        initial_python_rng = random.getstate()
        with closing(VisFlyEnv(config=config, num_envs=1, seed=7)) as baseline:
            expected_reset = baseline.reset()  # leaves: (B, *O)
            expected_steps = [baseline.step(actions), baseline.step(actions)]
        torch.testing.assert_close(torch.get_rng_state(), initial_rng)
        assert random.getstate() == initial_python_rng
        assert not torch.is_inference_mode_enabled()

        with closing(VisFlyEnv(config=config, num_envs=1, seed=7)) as training:
            actual_reset = training.reset()  # leaves: (B, *O)
            for key in expected_reset:
                np.testing.assert_array_equal(actual_reset[key], expected_reset[key])
            for expected in expected_steps:
                with closing(VisFlyEnv(config=config, num_envs=1, seed=19, evaluation=True)) as evaluation:
                    evaluation.reset()
                    evaluation.step(actions)
                torch.rand(3)
                random.random()
                global_rng = torch.get_rng_state()  # (R,)
                global_python_rng = random.getstate()
                actual = training.step(actions)
                for key in expected.observations:
                    np.testing.assert_array_equal(actual.observations[key], expected.observations[key])
                    np.testing.assert_array_equal(actual.final_observations[key], expected.final_observations[key])
                np.testing.assert_array_equal(actual.rewards, expected.rewards)
                torch.testing.assert_close(torch.get_rng_state(), global_rng)
                assert random.getstate() == global_python_rng


@pytest.mark.parametrize("operation", ["load", "construct", "reset", "step"])
def test_backend_errors_restore_global_rng(operation: str) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        torch.rand(3)
        random.random()
        raise RuntimeError("backend failed")

    with (
        torch.random.fork_rng(devices=[]),
        patch(
            "rl2.envs.visfly_env.load_env_creator",
            side_effect=fail if operation == "load" else None,
            return_value=fail if operation == "construct" else random_backend,
        ),
    ):
        initial_rng = torch.get_rng_state()  # (R,)
        initial_python_rng = random.getstate()
        with (
            pytest.raises(RuntimeError, match="backend failed"),
            closing(VisFlyEnv(config=VisFlyConfig(), num_envs=1, seed=7)) as env,
        ):
            if operation == "reset":
                env.backend.reset.side_effect = fail
                env.reset()
            else:
                env.backend.step.side_effect = fail
                env.step(np.zeros((1, 4), dtype=np.float32))
        torch.testing.assert_close(torch.get_rng_state(), initial_rng)
        assert random.getstate() == initial_python_rng


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
    depth = np.arange(np.prod((*batch_shape, 1, 2, 4)), dtype=np.float32).reshape(
        (*batch_shape, 1, 2, 4)
    )  # (*B, 1, H, W)
    expected_depth = np.moveaxis(depth, -3, -1).copy()  # (*B, H, W, 1)
    raw = RawVisFlyObservation(
        state=state,
        target=target,
        rgb=rgb if channels_first else np.moveaxis(rgb, -3, -1),
        depth=depth if channels_first else np.moveaxis(depth, -3, -1),  # (*B, 1, H, W) or (*B, H, W, 1)
    )
    converted = convert_observation(raw, channels_first=channels_first)
    assert set(converted) == {"state", "target", "rgb", "depth"}
    assert converted["state"].shape == (*batch_shape, 13)
    assert converted["target"].shape == (*batch_shape, 3)
    assert converted["rgb"].shape == (*batch_shape, 2, 4, 3)
    assert converted["state"].dtype == np.float64
    assert converted["target"].dtype == np.float32
    assert converted["rgb"].dtype == np.uint8
    assert converted["depth"].shape == (*batch_shape, 2, 4, 1)
    assert converted["depth"].dtype == np.float32
    state[:] = 0
    target[:] = 0
    rgb[:] = 0
    depth[:] = 0
    np.testing.assert_array_equal(converted["state"], 1)
    np.testing.assert_array_equal(converted["target"], 9)
    np.testing.assert_array_equal(converted["rgb"], expected_rgb)
    np.testing.assert_array_equal(converted["depth"], expected_depth)


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
        depth=torch.full((2, 1, 2, 4), 20.0),  # (B, 1, H, W)
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
                    depth=torch.full((1, 2, 4), 3.0),  # (1, H, W)
                ),
            },
            {},
        ],
    )
    with (
        patch("rl2.envs.visfly_env.load_env_creator", return_value=Mock(return_value=backend)),
        closing(VisFlyEnv(config=VisFlyConfig(scene_rotation_env_steps=2), num_envs=2, seed=0)) as env,
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
        assert initial["depth"].shape == result.final_observations["depth"].shape == (2, 2, 4, 1)
        np.testing.assert_array_equal(result.final_observations["depth"][0], 3)
        np.testing.assert_array_equal(result.final_observations["depth"][1], 20)
        np.testing.assert_array_equal(result.observations["depth"], 20)
        np.testing.assert_array_equal(result.truncated, [True, True])
        observation["depth"].zero_()
        np.testing.assert_array_equal(initial["depth"], 20)
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
            target=np.zeros((4, 3), dtype=np.float32),  # (B, 3)
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
        np.testing.assert_array_equal(first.info["stop_reason"], [0, StopReason.TERMINATED, StopReason.TIME_LIMIT, 0])
        assert backend.reset.call_count == 1
        rotated = env.step(np.zeros((4, 4)))
        np.testing.assert_array_equal(rotated.terminated, [False, True, False, False])
        np.testing.assert_array_equal(rotated.truncated, [True, False, True, True])
        np.testing.assert_array_equal(
            rotated.info["stop_reason"],
            [StopReason.SCENE_ROTATION, StopReason.TERMINATED, StopReason.TIME_LIMIT, StopReason.SCENE_ROTATION],
        )
        np.testing.assert_array_equal(rotated.observations["state"], 100)
        np.testing.assert_array_equal(rotated.final_observations["state"][:, 0], [1, 10, 10, 1])
        np.testing.assert_array_equal(rotated.final_observations["target"], 0)
        np.testing.assert_allclose(rotated.info["goal_distance"], np.sqrt(3) * np.array([1, 10, 10, 1]))
        np.testing.assert_array_equal(rotated.final_observations["rgb"][:, 0, 0, 0], [1, 10, 10, 1])
        env.step(np.zeros((4, 4)))
        assert backend.reset.call_count == 2
        env.reset()
        env.step(np.zeros((4, 4)))
        assert backend.reset.call_count == 3  # Explicit resets also restart the rotation counter.
        assert loader.return_value.call_args.kwargs["random_kwargs"]["scene_iter"] is True
        assert config.kwargs["random_kwargs"]["scene_iter"] is False


def test_visfly_stop_reasons_preserve_terminal_info_and_precedence() -> None:
    observation = RawVisFlyObservation(
        state=np.zeros((5, 13), dtype=np.float32),  # (B, S)
        target=np.zeros((5, 3), dtype=np.float32),  # (B, 3)
        rgb=np.zeros((5, 3, 2, 4), dtype=np.uint8),  # (B, C, H, W)
    )
    terminal = {key: value[0].copy() for key, value in observation.items()}  # leaves: (*O,)
    backend = Mock(num_envs=5, action_space=gym.spaces.Box(low=-1, high=1, shape=(4,)))
    backend.step.return_value = (
        observation,
        np.zeros(5, dtype=np.float32),  # (B,)
        np.array([True, True, True, True, False]),  # (B,)
        [
            {
                "is_success": True,
                "episode_done": True,
                "TimeLimit.truncated": True,
                "episode": {"extra": {"collision": True}},
            },
            {
                "is_success": False,
                "episode_done": True,
                "TimeLimit.truncated": True,
                "episode": {"extra": {"collision": True}},
            },
            {"TimeLimit.truncated": True, "terminal_observation": terminal},
            {"episode_done": True},
            {"is_success": True, "episode": {"extra": {"collision": True}}},  # Stale info on a live episode.
        ],
    )
    with (
        patch("rl2.envs.visfly_env.load_env_creator", return_value=Mock(return_value=backend)),
        closing(VisFlyEnv(config=VisFlyConfig(), num_envs=5, seed=0)) as env,
    ):
        result = env.step(np.zeros((5, 4), dtype=np.float32))
    assert result.info["stop_reason"] is not None
    assert result.info["stop_reason"].shape == (5,)
    assert result.info["stop_reason"].dtype == np.int32
    assert "goal_distance" not in result.info  # Some terminations lack a terminal observation.
    np.testing.assert_array_equal(
        result.info["stop_reason"],
        [StopReason.SUCCESS, StopReason.COLLISION, StopReason.TIME_LIMIT, StopReason.TERMINATED, StopReason.ONGOING],
    )
    np.testing.assert_array_equal(
        result.info["stop_reason"] != StopReason.ONGOING, result.terminated | result.truncated
    )


@pytest.mark.parametrize(
    ("num_envs", "num_scenes", "unbatched", "channels_first"),
    [
        pytest.param(1, 1, True, True, id="single_image"),
        pytest.param(1, 1, False, False, id="singleton_batch_channels_last"),
        pytest.param(4, 2, False, True, id="shared_scene_cameras"),
    ],
)
def test_visfly_render_preserves_batch_and_scene_order(
    num_envs: int, num_scenes: int, unbatched: bool, channels_first: bool
) -> None:
    pixels = torch.arange(num_scenes * 2 * 3 * 4).reshape(num_scenes, 2, 3, 4)  # (S, H, W, RGBA)
    rgb = np.arange(num_envs * 1 * 2 * 3, dtype=np.uint8).reshape(num_envs, 1, 2, 3)  # (B, H_agent, W_agent, 3)
    state = np.zeros((num_envs, 13), dtype=np.float32)  # (B, S)
    state[:, 3] = 1  # Identity wxyz quaternion.
    backend = Mock(num_envs=num_envs, action_space=gym.spaces.Box(low=-1, high=1, shape=(4,)))
    backend.render.return_value = pixels[0] if unbatched else pixels
    backend.get_observation.return_value = RawVisFlyObservation(
        state=state,  # (B, S)
        target=np.zeros((num_envs, 3), dtype=np.float32),  # (B, 3)
        rgb=np.moveaxis(rgb, -1, 1) if channels_first else rgb,  # (B, 3, H, W) or (B, H, W, 3)
    )
    if num_scenes > 1:
        backend.get_observation.return_value["depth"] = np.arange(num_envs * 2, dtype=np.float32).reshape(
            num_envs, 1, 1, 2
        )  # (B, 1, H_agent, W_agent)
    with (
        patch("rl2.envs.visfly_env.load_env_creator", return_value=Mock(return_value=backend)),
        closing(
            VisFlyEnv(
                config=VisFlyConfig(num_scenes=num_scenes, channels_first=channels_first), num_envs=num_envs, seed=0
            )
        ) as env,
    ):
        frames = env.render()  # (B, H, W, 3)
        debug_start = 11 if num_scenes > 1 else 7
        assert frames.shape == (num_envs, 2, debug_start + 3, 3)
        assert frames.dtype == np.uint8
        for index in range(num_envs):
            scene_index = index // (num_envs // num_scenes)
            np.testing.assert_array_equal(frames[index, :, :3], pixels[scene_index, ..., :3].numpy())
        np.testing.assert_array_equal(frames[:, :, 3:7], rgb.repeat(2, axis=1).repeat(2, axis=2))
        if num_scenes > 1:
            for index in range(num_envs):
                depth = backend.get_observation.return_value["depth"][index].transpose(1, 2, 0)  # (H_agent, W_agent, 1)
                np.testing.assert_array_equal(frames[index, :, 7:11], depth_panel(depth, height=2))
        assert frames[:, :, debug_start:].any()  # Diagnostics occupy the final panel.
        snapshot = frames.copy()  # (B, H, W, 3)
        pixels.zero_()
        rgb[:] = 0
        np.testing.assert_array_equal(frames, snapshot)
        refreshed = env.render()  # (B, H, W, 3)
        np.testing.assert_array_equal(refreshed[:, :, :7], 0)  # Read the latest scene and sensor pixels.
        np.testing.assert_array_equal(refreshed[:, :, 7:], snapshot[:, :, 7:])


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
