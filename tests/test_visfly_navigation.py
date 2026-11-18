"""RGB backend behavior without loading Habitat's native renderer."""

import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from rl2.envs import visfly_env


def habitat_to_std(corners: NDArray[Any]) -> tuple[torch.Tensor, None]:  # (2, 3) -> (2, 3)
    positions = torch.as_tensor(corners, dtype=torch.float32)[:, [2, 0, 1]] * torch.tensor([-1.0, -1.0, 1.0])  # (2, 3)
    return positions, None


def collision_free(positions: torch.Tensor, **kwargs: Any) -> NDArray[np.bool_]:  # (N, 3) -> (N,)
    return np.zeros(len(positions), dtype=bool)  # (N,)


class GeneratedState(NamedTuple):
    position: torch.Tensor  # (N, 3)
    angles: torch.Tensor  # (N, 3)
    velocity: torch.Tensor  # (N, 3)
    angular_velocity: torch.Tensor  # (N, 3)


class GeneratorStub:
    device = torch.device("cpu")

    def generate(self, num: int, **kwargs: Any) -> GeneratedState:
        return GeneratedState(
            position=torch.zeros((num, 3)),  # (N, 3)
            angles=torch.full((num, 3), 0.1),  # (N, 3)
            velocity=torch.full((num, 3), 0.2),  # (N, 3)
            angular_velocity=torch.full((num, 3), 0.3),  # (N, 3)
        )


class NavigationStub:
    def __init__(self, *, visual: bool, sensor_kwargs: list[dict[str, Any]], **kwargs: Any) -> None:
        self.kwargs = {"visual": visual, "sensor_kwargs": sensor_kwargs, **kwargs}
        self.device = torch.device("cpu")
        self.state = torch.ones((2, 13))  # (B, S)
        self.target = torch.zeros((2, 3))  # (B, 3)
        self.position = torch.ones((2, 3))  # (B, 3)
        self.num_envs = 2
        self.num_agent_per_scene = 1
        self.base_reward = torch.tensor([-0.25, 3.0])  # (B,)
        self.envs = SimpleNamespace(
            sceneManager=SimpleNamespace(
                _scene_loader=SimpleNamespace(paths=["scene_c", "scene_a", "scene_b"], shuffle=True),
                _scene_bounds=[SimpleNamespace(min=[-10, 0, -10], max=[0, 4, 0])] * 2,
                get_point_is_collision=collision_free,
            ),
            stateGenerators=[GeneratorStub(), GeneratorStub()],
        )
        height, width = sensor_kwargs[0]["resolution"]
        self.sensor_obs = {"color": np.full((2, 3, height, width), 255, dtype=np.uint8)}  # (B, 3, H, W)
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(13,)),
                "target": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "color": gym.spaces.Box(low=0, high=255, shape=(3, height, width), dtype=np.uint8),
            }
        )

    def get_reward(self, predicted_obs: dict[str, torch.Tensor] | None = None) -> torch.Tensor:  # (B,)
        return self.base_reward  # (B,)

    def get_full_observation(self) -> dict[str, torch.Tensor]:
        return self.get_observation()  # leaves: (B, *O)

    def reset(self, state: Any = None, **kwargs: Any) -> dict[str, torch.Tensor]:
        return self.get_full_observation()  # leaves: (B, *O)

    def reset_agent_by_id(
        self, agent_indices: torch.Tensor | None = None, state: Any = None, reset_obs: Any = None
    ) -> dict[str, torch.Tensor]:
        return self.get_full_observation()  # leaves: (B, *O)


@pytest.fixture(scope="module")
def navigation_class() -> type[Any]:
    modules = {
        "habitat_sim": SimpleNamespace(SensorType=SimpleNamespace(COLOR=object())),
        "VisFly.envs.NavigationEnv": SimpleNamespace(NavigationEnv=NavigationStub),
        "VisFly.utils.type": SimpleNamespace(TensorDict=dict),
        "VisFly.utils.common": SimpleNamespace(habitat_to_std=habitat_to_std),
        "VisFly.utils.maths": SimpleNamespace(Quaternion=Mock()),
    }
    with patch.dict("sys.modules", modules):
        namespace = runpy.run_path(str(Path(visfly_env.__file__).parent / "visfly" / "navigation.py"))
    return namespace["RGBNavigationEnv"]


def test_navigation_failure_penalty_is_opt_in(navigation_class: type[Any]) -> None:
    env = navigation_class()
    reward = env.get_reward()  # (B,)
    assert reward.shape == (2,)
    torch.testing.assert_close(reward, env.base_reward)


def test_navigation_failure_penalty_terminal_events(navigation_class: type[Any]) -> None:
    env = navigation_class(failure_penalty=20.0)
    # Collision, bounds, both, other failure, success+failure, timeout, ongoing, already ended.
    env.base_reward = torch.tensor([-0.1, -0.2, -0.3, -0.4, 3.0, -0.5, -0.6, -0.7])  # (B,)
    env.is_collision = torch.tensor([True, False, True, False, True, False, False, True])  # (B,)
    env.is_out_bounds = torch.tensor([False, True, True, False, True, False, False, False])  # (B,)
    env._failure = torch.tensor([False, False, False, True, False, False, False, False])  # (B,)
    env._success = torch.tensor([False, False, False, False, True, False, False, False])  # (B,)
    env._episode_done = torch.tensor([False, False, False, False, False, False, False, True])  # (B,)
    env._step_count = torch.tensor([1, 1, 500, 1, 1, 500, 1, 2])  # (B,)
    env.max_episode_steps = 500
    env.is_collision_reset = True
    reward = env.get_reward()  # (B,)
    assert reward.shape == (8,)
    torch.testing.assert_close(reward, torch.tensor([-20.1, -20.2, -20.3, -20.4, 3.0, -0.5, -0.6, -0.7]))
    torch.testing.assert_close(env.base_reward, torch.tensor([-0.1, -0.2, -0.3, -0.4, 3.0, -0.5, -0.6, -0.7]))


def test_navigation_nonterminal_collision_has_no_penalty(navigation_class: type[Any]) -> None:
    env = navigation_class(failure_penalty=20.0)
    env.base_reward = torch.tensor([-0.25])  # (B,): singleton evaluation batch
    env.is_collision = torch.tensor([True])  # (B,)
    env.is_out_bounds = torch.tensor([False])  # (B,)
    env._failure = torch.tensor([False])  # (B,)
    env._success = torch.tensor([False])  # (B,)
    env._episode_done = torch.tensor([False])  # (B,)
    env.is_collision_reset = False
    reward = env.get_reward()  # (B,)
    assert reward.shape == (1,)
    torch.testing.assert_close(reward, env.base_reward)


@pytest.mark.parametrize("penalty", [-1.0, float("nan"), float("inf")], ids=["negative", "nan", "infinite"])
def test_navigation_rejects_invalid_failure_penalty(navigation_class: type[Any], penalty: float) -> None:
    with pytest.raises(ValueError, match="failure_penalty"):
        navigation_class(failure_penalty=penalty)


def test_scene_bounds_randomization_and_seeded_scene_order(navigation_class: type[Any]) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        env = navigation_class(scene_randomization=True, seed=7)
        same = navigation_class(scene_randomization=True, seed=7)
        assert env.envs.sceneManager._scene_loader.paths == same.envs.sceneManager._scene_loader.paths
        assert set(env.envs.sceneManager._scene_loader.paths) == {"scene_a", "scene_b", "scene_c"}
        # The wrapper reads bounds after each scene load rather than caching the first layout.
        randomizer = env.envs.stateGenerators[0]
        quaternion = torch.tensor([[1.0], [0.0], [0.0], [0.0]])  # (4, N)
        with patch.dict(
            randomizer.safe_generate.__globals__,
            Quaternion=SimpleNamespace(
                from_euler=Mock(return_value=SimpleNamespace(toTensor=Mock(return_value=quaternion)))
            ),
        ):
            state = randomizer.safe_generate()
            assert state.position.shape == state.velocity.shape == state.angular_velocity.shape == (1, 3)
            assert state.orientation.shape == (1, 4)
            assert torch.all((state.position[0, :2] >= 0.3) & (state.position[0, :2] <= 9.7))
            assert 0.5 <= state.position[0, 2] <= 2.5
            torch.testing.assert_close(state.velocity, torch.full((1, 3), 0.2))
            torch.testing.assert_close(state.angular_velocity, torch.full((1, 3), 0.3))
            env.envs.sceneManager._scene_bounds[0] = SimpleNamespace(min=[-20, 0, -20], max=[-10, 4, -10])
            moved = randomizer.safe_generate()
            assert torch.all((moved.position[0, :2] >= 10.3) & (moved.position[0, :2] <= 19.7))


def test_reset_goals_preserve_other_agents_and_terminal_targets(navigation_class: type[Any]) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        env = navigation_class(scene_randomization=True)
        initial = env.reset()  # leaves: (B, *O)
        targets = initial["target"].clone()  # (B, 3)
        distances = (targets - env.position).norm(dim=-1)  # (B,)
        assert torch.all((distances >= 2.0) & (distances <= 8.0))
        refreshed = env.reset_agent_by_id(torch.tensor([1]))  # leaves: (B, *O)
        torch.testing.assert_close(initial["target"], targets)
        torch.testing.assert_close(refreshed["target"][0], targets[0])
        assert not torch.equal(refreshed["target"][1], targets[1])


def test_rgb_backend_defaults_and_observation_space(navigation_class: type[Any]) -> None:
    env = navigation_class(num_scene=2, seed=7)
    assert env.kwargs["visual"] is True
    assert env.kwargs["num_scene"] == 2
    assert env.kwargs["seed"] == 7
    observation = env.get_observation()  # leaves: (B, *O)
    assert set(observation) == set(env.observation_space.spaces) == {"state", "target", "rgb"}
    assert observation["rgb"].shape == (2, 3, 64, 64)
    assert observation["rgb"].dtype == torch.uint8
    assert env.observation_space.contains({key: value[0].numpy() for key, value in observation.items()})


def test_rgb_terminal_image_survives_sensor_buffer_reuse(navigation_class: type[Any]) -> None:
    env = navigation_class(sensor_kwargs=[{"sensor_type": "COLOR", "uuid": "color", "resolution": [2, 4]}])
    observation = env.get_observation()  # leaves: (B, *O)
    terminal = observation["rgb"][0]  # (3, H, W): upstream retains a view before autoreset
    env.sensor_obs["color"][0] = 0
    after_reset = env.get_observation()  # leaves: (B, *O)
    assert after_reset["rgb"].shape == (2, 3, 2, 4)
    np.testing.assert_array_equal(terminal.numpy(), 255)
    np.testing.assert_array_equal(after_reset["rgb"][0].numpy(), 0)
    np.testing.assert_array_equal(after_reset["rgb"][1].numpy(), 255)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"visual": False}, id="nonvisual"),
        pytest.param({"sensor_kwargs": []}, id="no_camera"),
        pytest.param({"sensor_kwargs": [{"sensor_type": "DEPTH", "uuid": "color"}]}, id="depth"),
        pytest.param({"sensor_kwargs": [{"sensor_type": "COLOR", "uuid": "rgb"}]}, id="invalid_uuid"),
        pytest.param({"sensor_kwargs": [{"sensor_type": "COLOR", "uuid": "color"}] * 2}, id="multiple_cameras"),
    ],
)
def test_rgb_backend_rejects_incompatible_sensors(navigation_class: type[Any], kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="RGBNavigationEnv requires"):
        navigation_class(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"spawn_clearance": 0}, id="zero_clearance"),
        pytest.param({"flight_height": (2.0, 1.0)}, id="reversed_height"),
        pytest.param({"goal_distance": (0.1, 2.0)}, id="goal_inside_success_radius"),
        pytest.param({"goal_distance": (3.0, 2.0)}, id="reversed_goal_distance"),
    ],
)
def test_scene_randomization_rejects_invalid_ranges(navigation_class: type[Any], kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        navigation_class(scene_randomization=True, **kwargs)
