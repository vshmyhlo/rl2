"""RGB/RGB-D backend behavior without loading Habitat's native renderer."""

import runpy
from contextlib import closing
from dataclasses import asdict, replace
from functools import partial
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, NamedTuple
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from rl2.envs import visfly_env
from rl2.envs.visfly.routes import Route, RouteConfig
from rl2.envs.visfly.scenarios import Scenario


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
        scene_kwargs = kwargs.get("scene_kwargs", {})
        root = scene_kwargs.get("path", ".")
        paths = (
            [str(path) for path in Path(root).rglob("*.scene_instance.json")]
            if scene_kwargs
            else ["scene_c", "scene_a", "scene_b"]
        )
        self.envs = SimpleNamespace(
            dynamics=Mock(),
            uav_radius=0.1,
            sceneManager=SimpleNamespace(
                scenes=[None, None],
                root_path=root,
                _scene_loader=SimpleNamespace(paths=paths, shuffle=scene_kwargs.get("shuffle", True)),
                _load_scene=Mock(),
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
        if any(sensor["uuid"] == "depth" for sensor in sensor_kwargs):
            self.sensor_obs["depth"] = np.full((2, 1, height, width), 7, dtype=np.float32)  # (B, 1, H, W)
            self.observation_space.spaces["depth"] = gym.spaces.Box(
                low=0, high=np.inf, shape=(1, height, width), dtype=np.float32
            )

    def get_reward(self, predicted_obs: dict[str, torch.Tensor] | None = None) -> torch.Tensor:  # (B,)
        return self.base_reward  # (B,)

    def get_full_observation(self) -> dict[str, torch.Tensor]:
        self._observations = self.get_observation()  # leaves: (B, *O)
        return self._observations  # leaves: (B, *O)

    def reset(self, state: Any = None, **kwargs: Any) -> dict[str, torch.Tensor]:
        return self.get_full_observation()  # leaves: (B, *O)

    def reset_agent_by_id(
        self, agent_indices: torch.Tensor | None = None, state: Any = None, reset_obs: Any = None
    ) -> dict[str, torch.Tensor]:
        return self.get_full_observation()  # leaves: (B, *O)


@pytest.fixture(scope="module")
def navigation_module() -> SimpleNamespace:
    modules = {
        "habitat_sim": SimpleNamespace(SensorType=SimpleNamespace(COLOR=object(), DEPTH=object())),
        "VisFly.envs.NavigationEnv": SimpleNamespace(NavigationEnv=NavigationStub),
        "VisFly.utils.type": SimpleNamespace(TensorDict=dict),
        "VisFly.utils.common": SimpleNamespace(habitat_to_std=habitat_to_std),
        "VisFly.utils.maths": SimpleNamespace(Quaternion=Mock()),
    }
    with patch.dict("sys.modules", modules):
        namespace = runpy.run_path(str(Path(visfly_env.__file__).parent / "visfly" / "navigation.py"))
    return SimpleNamespace(**namespace)


@pytest.fixture(scope="module")
def navigation_class(navigation_module: SimpleNamespace) -> type[Any]:
    return navigation_module.RGBNavigationEnv


def test_navigation_failure_penalty_is_opt_in(navigation_class: type[Any]) -> None:
    env = navigation_class()
    reward = env.get_reward()  # (B,)
    assert reward.shape == (2,)
    torch.testing.assert_close(reward, env.base_reward)


def test_navigation_rejects_euler_before_initializing_backend(navigation_class: type[Any]) -> None:
    with (
        patch.object(NavigationStub, "__init__", return_value=None) as initialize,
        pytest.raises(ValueError, match="requires ori_output_type='quaternion'"),
    ):
        navigation_class(dynamics_kwargs={"ori_output_type": "euler"})
    initialize.assert_not_called()


@pytest.mark.parametrize(
    ("split", "randomization", "shuffle"),
    [("train", True, True), ("test", False, False), ("test", True, True)],
)
def test_navigation_scene_split(
    navigation_class: type[Any], tmp_path: Path, split: Literal["train", "test"], randomization: bool, shuffle: bool
) -> None:
    # Ancestors outside the configured root must not affect membership.
    root = tmp_path / "dataset_test" / "scenes"
    train = ["apt.scene_instance.json", "latest/room.scene_instance.json", "train/nested/room.scene_instance.json"]
    test = ["garage_test/room.scene_instance.json", "test/nested/room.scene_instance.json"]
    for name in train + test:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    kwargs = {
        "scene_kwargs": {"path": str(root), "shuffle": shuffle},
        "scene_split": split,
        "scene_randomization": randomization,
        "seed": 7,
    }
    env = navigation_class(**kwargs)
    loader = env.envs.sceneManager._scene_loader
    expected = {str(root / name) for name in (train if split == "train" else test)}
    assert set(loader.paths) == expected
    assert {loader.paths[index] for index in loader._select_index} == expected
    assert loader.paths == navigation_class(**kwargs).envs.sceneManager._scene_loader.paths
    if not shuffle:
        assert loader.paths == sorted(expected)
    assert "scene_split" not in env.kwargs


def test_navigation_scene_split_defaults_to_train(navigation_class: type[Any], tmp_path: Path) -> None:
    train_path = tmp_path / "room.scene_instance.json"
    train_path.write_text("{}")
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "room.scene_instance.json").write_text("{}")

    env = navigation_class(scene_kwargs={"path": str(tmp_path)})
    loader = env.envs.sceneManager._scene_loader
    assert loader.paths == [str(train_path)]
    assert loader._select_index == [0]


@pytest.mark.parametrize(("split", "directory"), [("train", "garage_test"), ("test", "garage")])
def test_navigation_rejects_empty_scene_split(
    navigation_class: type[Any], tmp_path: Path, split: Literal["train", "test"], directory: str
) -> None:
    root = tmp_path / directory
    root.mkdir()
    (root / "room.scene_instance.json").write_text("{}")
    with (
        patch.object(navigation_class, "close", autospec=True, side_effect=navigation_class.close) as close,
        pytest.raises(ValueError, match=f"No scenes in the {split} split"),
    ):
        navigation_class(scene_kwargs={"path": str(root)}, scene_split=split)
    close.assert_called_once()
    close.call_args.args[0].envs.dynamics.close.assert_called_once_with()


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


@pytest.mark.parametrize(
    ("scene_min", "scene_max", "height", "expected_lower", "expected_upper"),
    [
        pytest.param(
            [-110, -1, -110],
            [110, 25, 110],
            (0.5, 23.0),
            [-100, -100, 0],
            [100, 100, 20],
            id="all_dynamics_limits",
        ),
        pytest.param(
            [-10, -1, -10],
            [0, 4, 0],
            (0.5, 2.5),
            [0.3, 0.3, 0],
            [9.7, 9.7, 1.5],
            id="height_stays_relative_to_scene_floor",
        ),
    ],
)
def test_sampling_bounds_intersect_dynamics_limits(
    navigation_module: SimpleNamespace,
    scene_min: list[float],
    scene_max: list[float],
    height: tuple[float, float],
    expected_lower: list[float],
    expected_upper: list[float],
) -> None:
    bounds = navigation_module.sampling_bounds(
        SimpleNamespace(min=scene_min, max=scene_max), clearance=0.3, height=height
    )
    assert bounds.lower.shape == bounds.upper.shape == (3,)
    torch.testing.assert_close(bounds.lower, torch.tensor(expected_lower, dtype=torch.float32))
    torch.testing.assert_close(bounds.upper, torch.tensor(expected_upper, dtype=torch.float32))


@pytest.mark.parametrize("operation", ["spawn", "target"])
def test_randomization_rejects_scenes_outside_dynamics_limits(navigation_class: type[Any], operation: str) -> None:
    env = navigation_class(scene_randomization=True)
    manager = env.envs.sceneManager
    manager._scene_bounds[0] = SimpleNamespace(min=[-10, -5, -10], max=[0, -1, 0])
    manager.get_point_is_collision = Mock(side_effect=collision_free)
    with pytest.raises(ValueError, match="no sampling volume"):
        if operation == "spawn":
            env.envs.stateGenerators[0].safe_generate()
        else:
            env.reset()
    manager.get_point_is_collision.assert_not_called()


def scene_stub(scene_min: list[float], scene_max: list[float]) -> Mock:
    scene = Mock()
    scene.get_active_scene_graph.return_value.get_root_node.return_value.cumulative_bb = SimpleNamespace(
        min=scene_min, max=scene_max
    )
    return scene


@pytest.mark.parametrize(
    ("scene_min", "scene_max"),
    [
        pytest.param([-10, -5, -10], [0, -1, 0], id="below_dynamics_floor"),
        pytest.param([-0.5, 0, -10], [0, 4, 0], id="narrower_than_clearance"),
        pytest.param([-10, 0, -10], [0, 0.5, 0], id="below_minimum_flight_height"),
    ],
)
def test_scene_loading_skips_and_caches_empty_bounds(
    navigation_class: type[Any], caplog: pytest.LogCaptureFixture, scene_min: list[float], scene_max: list[float]
) -> None:
    env = navigation_class(scene_randomization=True)
    manager = env.envs.sceneManager
    bad = scene_stub(scene_min, scene_max)
    good = scene_stub([-10, 0, -10], [0, 4, 0])
    env._load_unchecked_scene.side_effect = [bad, good, good]
    manager._scene_loader.paths = ["bad", "good"]
    manager._scene_loader.next = Mock(side_effect=[["bad"], ["good"], ["good"]])

    assert manager._load_scene("bad") is good
    bad.close.assert_called_once_with()
    good.close.assert_not_called()
    assert "bad: no sampling volume" in caplog.text
    assert "lower=" in caplog.text and "upper=" in caplog.text
    assert "clearance=0.3" in caplog.text and "flight_height=(0.5, 2.5)" in caplog.text
    assert manager._load_scene("bad") is good
    assert [call.args[0] for call in env._load_unchecked_scene.call_args_list] == ["bad", "good", "good"]


def test_scene_loading_stops_when_inventory_is_incompatible(navigation_class: type[Any]) -> None:
    env = navigation_class(scene_randomization=True)
    manager = env.envs.sceneManager
    scenes = [scene_stub([-10, -5, -10], [0, -1, 0]) for _ in range(2)]
    env._load_unchecked_scene.side_effect = scenes
    manager._scene_loader.paths = ["bad_a", "bad_b"]
    manager._scene_loader.next = Mock(side_effect=[["bad_a"], ["bad_b"]])
    with pytest.raises(ValueError, match="No compatible VisFly scenes remain.*bad_a.*bad_b"):
        manager._load_scene("bad_a")
    assert env._load_unchecked_scene.call_count == 2
    for scene in scenes:
        scene.close.assert_called_once_with()


def test_scene_loading_preserves_unrelated_errors(navigation_class: type[Any]) -> None:
    env = navigation_class(scene_randomization=True)
    env._load_unchecked_scene.side_effect = RuntimeError("renderer failed")
    with pytest.raises(RuntimeError, match="renderer failed"):
        env.envs.sceneManager._load_scene("broken")
    assert not env._rejected_scenes


def test_scene_validation_is_disabled_without_randomization(navigation_class: type[Any]) -> None:
    env = navigation_class()
    assert isinstance(env.envs.sceneManager._load_scene, Mock)


def test_reset_replaces_invalid_scene_during_rotation(navigation_class: type[Any]) -> None:
    paths = cycle(["good", "good", "bad", "good"])

    def next_paths(count: int) -> list[str]:
        return [next(paths) for _ in range(count)]

    def load_scene(path: str) -> Mock:
        if path == "bad":
            return scene_stub([-10, -5, -10], [0, -1, 0])
        return scene_stub([-10, 0, -10], [0, 4, 0])

    raw_load = Mock(side_effect=load_scene)
    initialize = NavigationStub.__init__

    def init(self: Any, **kwargs: Any) -> None:
        initialize(self, **kwargs)
        self.envs.sceneManager._load_scene = raw_load

    with patch.object(NavigationStub, "__init__", init):
        env = navigation_class(scene_randomization=True)
    manager = env.envs.sceneManager
    manager._scene_loader.paths = ["good", "bad"]

    def reset(self: Any, state: Any = None, **kwargs: Any) -> dict[str, torch.Tensor]:
        # Match upstream batch loading and reverse-order context cleanup.
        selected = manager._scene_loader.next(2)
        for scene in reversed(manager.scenes):
            if scene is not None:
                scene.close()
        for index, path in enumerate(selected):
            scene = manager._load_scene(path)
            manager.scenes[index] = scene
            manager._scene_bounds[index] = scene.get_active_scene_graph().get_root_node().cumulative_bb
        return self.get_full_observation()  # leaves: (B, *O)

    manager._scene_loader.next = next_paths
    with patch.object(NavigationStub, "reset", reset), torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        for _ in range(3):
            observations = env.reset()  # leaves: (B, *O)
            assert observations["target"].shape == (2, 3)
            assert observations["rgb"].shape == (2, 3, 64, 64)
            distances = (observations["target"] - env.position).norm(dim=-1)  # (B,)
            assert torch.all((distances >= 2) & (distances <= 8))
    assert list(env._rejected_scenes) == ["bad"]
    assert sum(call.args[0] == "bad" for call in raw_load.call_args_list) == 1


def test_spawn_clearance_accepts_drone_radius(navigation_class: type[Any]) -> None:
    env = navigation_class(scene_randomization=True, spawn_clearance=0.1)
    assert env.spawn_clearance == env.envs.uav_radius


def test_spawn_clearance_rejects_less_than_drone_radius(navigation_class: type[Any]) -> None:
    with (
        patch.object(navigation_class, "close", autospec=True, side_effect=navigation_class.close) as close,
        pytest.raises(ValueError, match="spawn_clearance.*drone collision radius"),
    ):
        navigation_class(scene_randomization=True, spawn_clearance=0.05)
    close.assert_called_once()
    close.call_args.args[0].envs.dynamics.close.assert_called_once_with()


@pytest.mark.parametrize("loaded_count", [pytest.param(2, id="partial_reset"), pytest.param(3, id="full_reset")])
def test_close_preserves_reset_error_with_shared_context(navigation_class: type[Any], loaded_count: int) -> None:
    env = navigation_class()
    active = set(range(loaded_count))

    def close_scene(index: int) -> None:
        assert 0 in active, "GL::Context::current(): no current context"
        active.remove(index)

    scenes = [Mock(close=Mock(side_effect=partial(close_scene, index))) for index in range(loaded_count)]
    env.envs.sceneManager.scenes = scenes + [None] * (3 - loaded_count)
    with pytest.raises(ValueError, match="reset failed"), closing(env):
        raise ValueError("reset failed")
    assert not active
    assert env.envs.sceneManager.scenes == [None] * 3
    env.envs.dynamics.close.assert_called_once_with()
    env.close()
    for scene in scenes:
        scene.close.assert_called_once_with()


def test_reset_goals_preserve_other_agents_and_terminal_targets(navigation_class: type[Any]) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        env = navigation_class(scene_randomization=True)
        initial = env.reset()  # leaves: (B, *O)
        targets = initial["target"].clone()  # (B, 3)
        distances = (targets - env.position).norm(dim=-1)  # (B,)
        assert torch.all((distances >= 2.0) & (distances <= 8.0))
        refreshed = env.reset_agent_by_id(torch.tensor([1]))  # leaves: (B, *O)
        assert refreshed is env._observations  # Upstream step returns this cached observation after autoreset.
        torch.testing.assert_close(initial["target"], targets)
        torch.testing.assert_close(refreshed["target"][0], targets[0])
        assert not torch.equal(refreshed["target"][1], targets[1])
        torch.testing.assert_close(refreshed["target"], env.target)


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
    env = navigation_class(
        sensor_kwargs=[
            {"sensor_type": "COLOR", "uuid": "color", "resolution": [2, 4]},
            {"sensor_type": "DEPTH", "uuid": "depth", "resolution": [2, 4]},
        ]
    )
    observation = env.get_observation()  # leaves: (B, *O)
    assert set(observation) == set(env.observation_space.spaces) == {"state", "target", "rgb", "depth"}
    assert observation["depth"].dtype == torch.float32
    assert env.observation_space.contains({key: value[0].numpy() for key, value in observation.items()})
    terminal = observation["rgb"][0]  # (3, H, W): upstream retains a view before autoreset
    terminal_depth = observation["depth"][0]  # (1, H, W)
    env.sensor_obs["color"][0] = 0
    env.sensor_obs["depth"][0] = 20
    after_reset = env.get_observation()  # leaves: (B, *O)
    assert after_reset["rgb"].shape == (2, 3, 2, 4)
    np.testing.assert_array_equal(terminal.numpy(), 255)
    np.testing.assert_array_equal(after_reset["rgb"][0].numpy(), 0)
    np.testing.assert_array_equal(after_reset["rgb"][1].numpy(), 255)
    assert after_reset["depth"].shape == (2, 1, 2, 4)
    np.testing.assert_array_equal(terminal_depth.numpy(), 7)
    np.testing.assert_array_equal(after_reset["depth"][0].numpy(), 20)
    np.testing.assert_array_equal(after_reset["depth"][1].numpy(), 7)


@pytest.mark.parametrize("setting", ["resolution", "position", "orientation"])
def test_rgbd_requires_aligned_cameras(navigation_class: type[Any], setting: str) -> None:
    depth: dict[str, Any] = {"sensor_type": "DEPTH", "uuid": "depth", "resolution": [2, 4]}
    depth[setting] = [4, 2] if setting == "resolution" else [1, 0, 0]
    with pytest.raises(ValueError, match=f"matching RGB/depth {setting}"):
        navigation_class(
            sensor_kwargs=[
                {"sensor_type": "COLOR", "uuid": "color", "resolution": [2, 4]},
                depth,
            ]
        )


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


def test_fixed_scenario_pins_scene_state_and_partial_reset_targets(navigation_class: type[Any], tmp_path: Path) -> None:
    root = tmp_path / "test"
    root.mkdir()
    for name in ("a", "b"):
        (root / f"{name}.scene_instance.json").write_text("{}")
    route = Route(
        points=((1.0, 1.0, 1.0), (8.0, 8.0, 1.0)),
        direct_blocked=False,
        constrained=False,
        long=True,
        detour_threshold=1.25,
    )
    scenario = Scenario(
        scene="b.scene_instance.json",
        category="long_traversal",
        route=route,
        orientation=(1.0, 0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
        max_episode_steps=1000,
        seed=3,
    )
    env = navigation_class(
        scene_randomization=True, scene_split="test", scenario=asdict(scenario), scene_kwargs={"path": str(root)}
    )
    assert env.envs.sceneManager._scene_loader.paths == [str(root / scenario.scene)]
    for generator in env.envs.stateGenerators:
        sampled = generator.safe_generate(num=2)
        assert sampled.position.shape == sampled.velocity.shape == sampled.angular_velocity.shape == (2, 3)
        assert sampled.orientation.shape == (2, 4)
        torch.testing.assert_close(sampled.position, torch.tensor([route.points[0]] * 2))
        torch.testing.assert_close(sampled.orientation, torch.tensor([scenario.orientation] * 2))
        assert sampled.velocity.count_nonzero() == 0
    env._randomize_targets(torch.arange(2))
    previous = env.target  # (B, 3)
    env.envs.stateGenerators[1].safe_generate()
    env._randomize_targets(torch.tensor([1]))
    assert previous.data_ptr() != env.target.data_ptr()
    torch.testing.assert_close(env.target, torch.tensor([route.points[-1]] * 2))
    # Replay rejects stale geometry rather than silently selecting another route.
    env.envs.sceneManager.get_point_is_collision = Mock(return_value=np.ones(1000, dtype=bool))
    bad = replace(scenario, route=replace(route, points=((-1.0, 1.0, 1.0), (8.0, 8.0, 1.0))))
    env.envs.stateGenerators[0].scenario = bad
    with pytest.raises(ValueError, match="Saved evaluation route"):
        env.envs.stateGenerators[0].safe_generate()


def test_route_catalogues_refresh_on_scene_rotation_and_metrics_drain(navigation_class: type[Any]) -> None:
    config = RouteConfig(spacing=3, sources=2, goals_per_source=3, jitter=0)
    env = navigation_class(scene_randomization=True, route_sampling=asdict(config))
    manager = env.envs.sceneManager
    manager.scenes = [object(), object()]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4)
        first = env.route_catalogue(0, config)
        assert first is env.route_catalogue(0, config)
        sampled = env._sample_route(0)
        assert sampled in first or tuple(reversed(sampled.points)) in [route.points for route in first]
        manager.scenes[0] = object()
        manager._scene_bounds[0] = SimpleNamespace(min=[-20, 0, -20], max=[-10, 4, -10])
        second = env.route_catalogue(0, config)
    assert second is not first
    assert all(route.points[0][0] >= 10 for route in second)
    metrics = env.sampling_metrics()
    assert metrics["rollout/sampled_episode_count"] == 1
    assert metrics["rollout/direct_path_blocked_fraction"] == 0
    assert metrics["rollout/difficulty_fraction/hard"] == 0
    assert sum(metrics[f"rollout/difficulty_fraction/{level}"] for level in ("easy", "medium", "hard")) == 1
    assert env.sampling_metrics() == {}
