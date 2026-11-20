"""Real headless rendering checks; run inside the --visfly image on an NVIDIA GPU."""

import os
from contextlib import closing
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import jax
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from numpy.typing import NDArray

from rl2.envs.visfly.catalogues import build_catalogue, planning_seed
from rl2.envs.visfly.collision import SceneCollision, collision_points
from rl2.envs.visfly.generate_manifest import generate_manifest
from rl2.envs.visfly.planning import build_roadmap, build_routes, sample_route, segments_clear
from rl2.envs.visfly.routes import Route, RouteConfig
from rl2.envs.visfly.sampling import sampling_bounds
from rl2.envs.visfly.scenarios import EvaluationConfig
from rl2.envs.visfly.sensors import SharedRgbdRenderer
from rl2.envs.visfly_env import evaluation_manifest

# Exercise the trainer's native-library import order before constructing VisFly.
from rl2.train_ppo_gym import VisFlyConfig, VisFlyEnv

pytestmark = pytest.mark.skipif(os.environ.get("RL2_VISFLY_TESTS") != "1", reason="requires the VisFly GPU image")


class PlanningRecord(NamedTuple):
    reset: dict[str, NDArray[Any]]  # leaves: (B, *O)
    step: dict[str, NDArray[Any]]  # leaves: (B, *O)
    routes: list[list[Route]]


def test_parallel_prefetch_matches_serial_planning_and_live_geometry() -> None:
    root = Path(os.environ["VISFLY_SCENE_PATH"])
    base = VisFlyConfig(
        num_scenes=2,
        scene_rotation_env_steps=128,
        route_sampling=RouteConfig(spacing=8, sources=3, goals_per_source=3, min_distance=1),
        kwargs={
            "device": "cpu",
            "scene_randomization": True,
            "max_episode_steps": 1,
            "scene_kwargs": {"path": str(root), "scene_type": "json"},
            "sensor_kwargs": [
                {"sensor_type": "COLOR", "uuid": "color", "resolution": [16, 16]},
                {"sensor_type": "DEPTH", "uuid": "depth", "resolution": [16, 16]},
            ],
        },
    )
    runs = []
    for workers, prefetch in ((1, False), (2, True)):
        config = replace(base, kwargs={**base.kwargs, "planning_workers": workers, "planning_prefetch": prefetch})
        records = []
        with closing(VisFlyEnv(config=config, num_envs=4, seed=7)) as env:
            manager = env.backend.envs.sceneManager
            loader = manager._scene_loader
            loader.paths = [
                str(root / scene)
                for scene in (
                    "garage_empty/garage_empty_0.scene_instance.json",
                    "box30_high_low/box30_high_low_8.scene_instance.json",
                )
            ]
            loader._select_index = [0, 1]
            loader._current_index = 0
            for generation in range(2):
                observations = env.reset()  # leaves: (4, *O)
                assert observations["state"].shape == (4, 13)
                assert observations["target"].shape == (4, 3)
                assert observations["rgb"].shape == (4, 16, 16, 3)
                assert observations["depth"].shape == (4, 16, 16, 1)
                catalogues = [env.backend._route_catalogues[index] for index in range(2)]
                if generation == 0:
                    for index, catalogue in enumerate(catalogues):
                        # _load_planned_scenes already advanced the generation counter.
                        task = replace(
                            env.backend._planning_task(index, loader.paths[index]),
                            seed=planning_seed(7, generation, index),
                        )
                        bounds = sampling_bounds(manager._scene_bounds[index], 0.3, (0.5, 2.5))
                        reference = build_catalogue(task, bounds, SceneCollision(scene=manager.scenes[index]))
                        assert catalogue.routes == reference.routes
                        assert catalogue.long_threshold == reference.long_threshold
                transition = env.step(np.zeros((4, 4), dtype=np.float32))  # (B, A)
                assert transition.terminated.shape == (4,) and transition.truncated.all()
                records.append(
                    PlanningRecord(
                        reset=observations,  # leaves: (4, *O)
                        step=transition.observations,  # leaves: (4, *O)
                        routes=[catalogue.routes for catalogue in catalogues],
                    )
                )
                if prefetch:
                    pending = env.backend._planner._pending
                    assert len(pending) == 2
                    for future in pending.values():
                        future.result()
            assert records[0].routes != records[1].routes, "revisiting a scene must use a fresh catalogue"
        assert env.backend._planner is None
        runs.append(records)
    for serial, parallel in zip(*runs, strict=True):
        for serial_obs, parallel_obs in ((serial.reset, parallel.reset), (serial.step, parallel.step)):
            for key in serial_obs:
                np.testing.assert_array_equal(serial_obs[key], parallel_obs[key])
        assert serial.routes == parallel.routes


def test_shared_rgbd_matches_separate_passes_across_reload_and_video() -> None:
    config = VisFlyConfig(
        num_scenes=2,
        scene_rotation_env_steps=128,
        kwargs={
            "device": "cpu",
            "scene_randomization": True,
            "max_episode_steps": 1,
            "scene_kwargs": {
                "path": os.environ["VISFLY_SCENE_PATH"],
                "scene_type": "json",
                "render_settings": {"sensor_type": "color", "mode": "follow", "view": "near", "resolution": [32, 32]},
            },
            "sensor_kwargs": [
                {"sensor_type": "COLOR", "uuid": "color", "resolution": [32, 32]},
                {"sensor_type": "DEPTH", "uuid": "depth", "resolution": [32, 32]},
            ],
        },
    )
    with closing(VisFlyEnv(config=config, num_envs=4, seed=7)) as env:
        manager = env.backend.envs.sceneManager
        loader = manager._scene_loader
        loader.paths = [
            str(Path(os.environ["VISFLY_SCENE_PATH"]) / scene)
            for scene in (
                "garage_empty/garage_empty_0.scene_instance.json",
                "v3_sc0_staging_08.scene_instance.json",
                "box30_high_low/box30_high_low_8.scene_instance.json",
                "box30_wall_low/box30_wall_low_45.scene_instance.json",
            )
        ]
        loader._select_index = list(range(4))
        renderer = manager.get_observation
        assert isinstance(renderer, SharedRgbdRenderer)
        previous_scenes = tuple(manager.scenes)
        for _ in range(2):
            env.reset()
            assert all(before is not after for before, after in zip(previous_scenes, manager.scenes, strict=True))
            previous_scenes = tuple(manager.scenes)
            assert renderer._groups, "test image must include RGB_DEPTH support"
            for _ in range(2):
                renderer._restore_targets()
                reference = [
                    {key: value.copy() for key, value in obs.items()} for obs in renderer._original()
                ]  # leaves: (H, W, C) or (H, W)
                renderer._prepare()
                images = manager.get_observation()
                for expected, actual in zip(reference, images, strict=True):
                    for key in ("color", "depth"):
                        np.testing.assert_array_equal(actual[key], expected[key])
                selected = manager.get_observation(indices=torch.tensor([3, 0]))
                for image, index in zip(selected, (3, 0), strict=True):
                    np.testing.assert_array_equal(image["depth"], reference[index]["depth"])
                assert manager.get_observation(indices=[]) == []
                step = env.step(np.zeros((4, 4), dtype=np.float32))  # (B, A)
                assert step.truncated.all()
                assert step.observations["depth"].shape == (4, 32, 32, 1)
        frame = env.render()  # (B, H, W, 3)
        assert frame.shape[0] == 4 and not renderer._groups
        separate = renderer._original()
        after_video = manager.get_observation()
        for expected, actual in zip(separate, after_video, strict=True):
            np.testing.assert_array_equal(actual["depth"], expected["depth"])


def test_native_collision_and_route_parity() -> None:
    root = Path(os.environ["VISFLY_SCENE_PATH"])
    config = VisFlyConfig(
        route_sampling=None,
        kwargs={
            "device": "cpu",
            "visual": True,
            "scene_randomization": True,
            "scene_kwargs": {"path": str(root), "scene_type": "json"},
            "scene_path": str(root / "box30_0.01tree/box30_0.01tree_0.scene_instance.json"),
            "sensor_kwargs": [{"sensor_type": "COLOR", "uuid": "color", "resolution": [16, 16]}],
        },
    )
    with closing(VisFlyEnv(config=config, num_envs=1, seed=7)) as env:
        env.reset()
        manager = env.backend.envs.sceneManager
        scene = manager.scenes[0]
        reference = partial(collision_points, scene)
        native = SceneCollision(scene=scene)
        bounds = manager._scene_bounds[0]
        corners = np.array([bounds.min, bounds.max])[:, [2, 0, 1]] * [-1, -1, 1]  # (2, 3)
        lower = torch.tensor(corners.min(axis=0) + 0.4, dtype=torch.float32)  # (3,)
        upper = torch.tensor(corners.max(axis=0) - 0.4, dtype=torch.float32)  # (3,)
        lower[2], upper[2] = lower[2] + 0.5, lower[2] + 1.5
        rng = np.random.default_rng(7)
        points = rng.uniform(lower.numpy() - 1, upper.numpy() + 1, (256, 3)).astype(np.float32)  # (N, 3)
        for batch in (points, points[:1], points[:0], points[::2]):
            actual = native(batch, 0.3)  # (N,)
            assert actual.shape == (len(batch),) and actual.dtype == np.bool_
            np.testing.assert_array_equal(actual, reference(batch, 0.3))
        for point, expected in zip(points, reference(points, 0.3), strict=True):
            assert scene.route_point_collides(tuple(point), 0.3) == expected
        with pytest.raises(ValueError, match="shape"):
            scene.route_points_collide(points[:, :2].copy(), 0.3)
        with pytest.raises(TypeError):
            scene.route_points_collide(points.astype(np.float64), 0.3)
        starts, ends = torch.from_numpy(points[:32]), torch.from_numpy(points[32:64])  # each (E, 3)
        torch.testing.assert_close(
            segments_clear(starts, ends, native, 0.3, 0.2), segments_clear(starts, ends, reference, 0.3, 0.2)
        )
        route_config = RouteConfig(spacing=4, sources=3, goals_per_source=4)
        graphs, catalogues, sampled, random_states = [], [], [], []
        for collision in (reference, native):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(7)
                graph = build_roadmap(lower, upper, collision, 0.3, route_config)
                routes = build_routes(graph, collision, 0.3, route_config)
                graphs.append(graph)
                catalogues.append(routes)
                sampled.append(
                    sample_route(
                        routes,
                        collision,
                        0.3,
                        route_config,
                        lower,
                        upper,
                        long_threshold=route_config.long_fraction * graph.extent,
                    )
                )
                random_states.append(torch.get_rng_state())  # (R,)
        torch.testing.assert_close(graphs[0].points, graphs[1].points, rtol=0, atol=0)
        assert (graphs[0].edges != graphs[1].edges).nnz == 0
        np.testing.assert_array_equal(graphs[0].wide_components, graphs[1].wide_components)
        assert catalogues[0] == catalogues[1]
        assert sampled[0] == sampled[1]
        torch.testing.assert_close(random_states[0], random_states[1], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("num_envs", "num_scenes", "scenes"),
    [
        pytest.param(
            4,
            2,
            ("garage_empty/garage_empty_0.scene_instance.json", "v3_sc0_staging_08.scene_instance.json"),
            id="shared_context_rotation",
        ),
        pytest.param(1, 1, ("v3_sc0_staging_08.scene_instance.json",), id="articulated-evaluation"),
    ],
)
def test_rendered_reset_step_and_autoreset(num_envs: int, num_scenes: int, scenes: tuple[str, ...]) -> None:
    config = VisFlyConfig(
        num_scenes=num_scenes,
        scene_rotation_env_steps=num_envs,
        kwargs={
            "device": "cpu",
            "visual": True,
            "max_episode_steps": 1,
            "scene_randomization": True,
            "scene_kwargs": {"path": os.environ["VISFLY_SCENE_PATH"], "scene_type": "json"},
            "dynamics_kwargs": {"action_type": "bodyrate", "dt": 0.03, "ctrl_dt": 0.03},
            "sensor_kwargs": [
                {"sensor_type": "COLOR", "uuid": "color", "resolution": [16, 16]},
                {"sensor_type": "DEPTH", "uuid": "depth", "resolution": [16, 16]},
            ],
        },
    )
    with closing(VisFlyEnv(config=config, num_envs=num_envs, seed=1, evaluation=num_envs == 1)) as env:
        loader = env.backend.envs.sceneManager._scene_loader
        loader.paths = [str(Path(os.environ["VISFLY_SCENE_PATH"]) / scene) for scene in scenes]
        loader._select_index = list(range(len(scenes)))
        observation = env.reset()  # leaves: (B, *O)
        assert isinstance(observation, dict)
        assert set(observation) == {"state", "target", "rgb", "depth"}
        assert observation["rgb"].dtype == np.uint8
        assert observation["rgb"].shape == (num_envs, 16, 16, 3)
        assert observation["state"].shape == (num_envs, 13)
        assert observation["target"].shape == (num_envs, 3)
        rgb = observation["rgb"].copy()  # (B, H, W, 3)
        depth = observation["depth"].copy()  # (B, H, W, 1)
        assert depth.shape == (num_envs, 16, 16, 1)
        assert depth.dtype == np.float32
        assert (depth > 0).all()
        actions = np.zeros((num_envs, *env.action_space.shape), dtype=np.float32)  # (B, A)
        result = env.step(actions)
        assert result.rewards.shape == result.terminated.shape == result.truncated.shape == (num_envs,)
        assert result.truncated.all()
        assert not result.terminated.any()
        assert result.final_observations["rgb"].shape == rgb.shape
        assert result.observations["rgb"].shape == rgb.shape
        np.testing.assert_array_equal(observation["rgb"], rgb)
        np.testing.assert_array_equal(observation["depth"], depth)
        assert result.final_observations["depth"].shape == result.observations["depth"].shape == depth.shape
        assert (rgb > 0).any()
        assert env.step(actions).truncated.all()


def test_generated_manifest_replays_real_scene_after_autoreset(tmp_path: Path) -> None:
    root = tmp_path / "datasets" / "visfly-beta" / "test"
    root.mkdir(parents=True)
    source = Path(os.environ["VISFLY_SCENE_PATH"]) / "box30_test/box30_test_0.scene_instance.json"
    (root / "room.scene_instance.json").write_bytes(source.read_bytes())
    (root / "alias.scene_instance.json").write_bytes(source.read_bytes())
    wall = Path(os.environ["VISFLY_SCENE_PATH"]) / "box30_wall_high_test/box30_wall_high_test_0.scene_instance.json"
    (root / "wall.scene_instance.json").write_bytes(wall.read_bytes())
    config = VisFlyConfig(
        evaluation=EvaluationConfig(
            manifest_path=str(tmp_path / "manifest.json"), routes=RouteConfig(spacing=6, sources=4, goals_per_source=4)
        ),
        kwargs={
            "device": "cpu",
            "visual": True,
            "scene_randomization": True,
            "scene_split": "test",
            "scene_kwargs": {
                "path": str(root),
                "scene_type": "json",
                "render_settings": {"sensor_type": "color", "mode": "follow", "view": "near", "resolution": [32, 32]},
            },
            "dynamics_kwargs": {"action_type": "bodyrate", "dt": 0.03, "ctrl_dt": 0.03},
            "sensor_kwargs": [{"sensor_type": "COLOR", "uuid": "color", "resolution": [16, 16]}],
        },
    )
    manifest = generate_manifest(config, seed=7, workers=2)
    assert evaluation_manifest(config) == manifest
    assert len(manifest.coverage) == 2
    scenario = replace(manifest.scenarios[0], max_episode_steps=1)
    config = replace(config, kwargs={**config.kwargs, "scenario": asdict(scenario), "max_episode_steps": 1})
    with closing(VisFlyEnv(config=config, num_envs=1, seed=scenario.seed, evaluation=True)) as env:
        initial = env.reset()  # leaves: (1, *O)
        np.testing.assert_allclose(initial["state"][0, :3], scenario.route.points[0])
        np.testing.assert_allclose(initial["state"][0, 3:7], scenario.orientation)
        np.testing.assert_allclose(initial["target"][0], scenario.route.points[-1])
        transition = env.step(np.zeros((1, 4), dtype=np.float32))
        assert (transition.terminated | transition.truncated).all()
        assert transition.observations["rgb"].shape == (1, 16, 16, 3)
        np.testing.assert_allclose(transition.observations["state"][0, :3], scenario.route.points[0])
        np.testing.assert_allclose(transition.final_observations["target"][0], scenario.route.points[-1])
        scene = env.backend.envs.sceneManager.scenes[0]
        other = replace(
            scenario,
            seed=18,
            max_episode_steps=2,
            route=replace(scenario.route, points=tuple(reversed(scenario.route.points))),
        )
        env.configure_scenario(other)
        reused = env.reset()  # leaves: (1, *O)
        replay_frame = env.render()  # (1, H, W, 3)
        assert env.backend.envs.sceneManager.scenes[0] is scene
        assert env.config.kwargs["max_episode_steps"] == 2
        assert not env.step(np.zeros((1, 4), dtype=np.float32)).truncated.any()
        assert env.step(np.zeros((1, 4), dtype=np.float32)).truncated.all()
        alternate = replace(
            other, scene="room.scene_instance.json" if other.scene.startswith("alias") else "alias.scene_instance.json"
        )
        env.configure_scenario(alternate)
        switched = env.reset()  # leaves: (1, *O)
        assert env.backend.envs.sceneManager.scenes[0] is not scene
        for name, value in reused.items():
            np.testing.assert_array_equal(switched[name], value)
    fresh_config = replace(config, kwargs={**config.kwargs, "scenario": asdict(other), "max_episode_steps": 2})
    with closing(VisFlyEnv(config=fresh_config, num_envs=1, seed=other.seed, evaluation=True)) as fresh:
        initial = fresh.reset()  # leaves: (1, *O)
        for name, value in reused.items():
            np.testing.assert_array_equal(initial[name], value)
        np.testing.assert_array_equal(fresh.render(), replay_frame)


@pytest.mark.parametrize(("num_envs", "delay"), [(1, True), (4, False)], ids=["evaluation_delays", "batch_no_delays"])
def test_compiled_physics_matches_upstream_through_autoresets(num_envs: int, delay: bool) -> None:
    kwargs = {
        "device": "cpu",
        "visual": True,
        "scene_randomization": True,
        "max_episode_steps": 4,
        "scene_kwargs": {"path": os.environ["VISFLY_SCENE_PATH"], "scene_type": "json", "shuffle": False},
        "scene_path": str(Path(os.environ["VISFLY_SCENE_PATH"]) / "garage_empty/garage_empty_0.scene_instance.json"),
        "dynamics_kwargs": {
            "action_type": "bodyrate",
            "dt": 0.0025,
            "ctrl_dt": 0.02,
            "ctrl_delay": delay,
            "comm_delay": 0.06 if delay else 0.0,
            "wind_settings": [0.1, 0.0, 0.0],
        },
        "sensor_kwargs": [{"sensor_type": "COLOR", "uuid": "color", "resolution": [16, 16]}],
    }
    actions = np.random.default_rng(7).uniform(-0.1, 0.1, size=(10, num_envs, 4)).astype(np.float32)  # (T, B, A)
    traces = []
    for compiled in (False, True):
        config = VisFlyConfig(kwargs={**kwargs, "compiled_dynamics": compiled})
        with (
            jax.set_mesh(Mesh(devices=np.array(jax.devices()), axis_names=("trainer",))),
            closing(VisFlyEnv(config=config, num_envs=num_envs, seed=19)) as env,
        ):
            env.reset()
            traces.append([env.step(action) for action in actions])
    for expected, actual in zip(*traces, strict=True):
        assert actual.observations["rgb"].shape == (num_envs, 16, 16, 3)
        for field in ("observations", "final_observations"):
            np.testing.assert_allclose(
                getattr(actual, field)["state"], getattr(expected, field)["state"], atol=2e-5, rtol=2e-5
            )
            np.testing.assert_array_equal(getattr(actual, field)["target"], getattr(expected, field)["target"])
        np.testing.assert_allclose(actual.rewards, expected.rewards, atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(actual.terminated, expected.terminated)
        np.testing.assert_array_equal(actual.truncated, expected.truncated)
