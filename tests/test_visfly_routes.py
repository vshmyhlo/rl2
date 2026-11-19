"""Geometry, difficulty selection, and fixed evaluation scenario contracts."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from pydantic import TypeAdapter

from rl2.envs.visfly.routes import Route, RouteConfig, build_roadmap, build_routes, sample_route, segments_clear
from rl2.envs.visfly.scenarios import EvaluationConfig, Manifest, SceneCoverage, select_scenarios, unique_scenes


def empty_scene(points: torch.Tensor, radius: float) -> NDArray[np.bool_]:  # (N, 3) -> (N,)
    return np.zeros(len(points), dtype=bool)  # (N,)


def doorway(points: torch.Tensor, radius: float) -> NDArray[np.bool_]:  # (N, 3) -> (N,)
    # A wall across all heights, with an 0.8 m opening centered at y=2.5.
    return (
        (abs(points[:, 0] - 3) < radius + 0.1) & ((points[:, 1] < 2.1 + radius) | (points[:, 1] > 2.9 - radius))
    ).numpy()  # (N,)


def closed_wall(points: torch.Tensor, radius: float) -> NDArray[np.bool_]:  # (N, 3) -> (N,)
    return (abs(points[:, 0] - 3) < radius + 0.01).numpy()  # (N,)


@pytest.fixture(scope="module")
def door_routes() -> list[Route]:
    config = RouteConfig(spacing=0.5, sources=12, goals_per_source=6, min_distance=1, passage_margin=0.4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4)
        graph = build_roadmap(torch.tensor([0.0, 0.0, 1.0]), torch.tensor([6.0, 5.0, 1.5]), doorway, 0.1, config)
        assert graph.points.ndim == 2 and graph.points.shape[1] == 3
        return build_routes(graph, doorway, 0.1, config)


def test_routes_require_feasible_clearance_checked_paths(door_routes: list[Route]) -> None:
    assert any(route.direct_blocked for route in door_routes)
    assert any(route.constrained for route in door_routes)
    assert any("substantial_detour" in route.categories for route in door_routes)
    for route in door_routes:
        points = torch.tensor(route.points)  # (P, 3)
        assert points.shape[1:] == (3,)
        assert segments_clear(points[:-1], points[1:], doorway, 0.1, 0.2).all()
        assert route.length >= route.initial_distance - 1e-6


def test_disconnected_regions_never_form_routes() -> None:
    config = RouteConfig(spacing=1, sources=6, goals_per_source=3, min_distance=1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        graph = build_roadmap(torch.tensor([0.0, 0.0, 1.0]), torch.tensor([6.0, 2.0, 2.0]), closed_wall, 0.1, config)
        routes = build_routes(graph, closed_wall, 0.1, config)
    assert routes
    assert all((route.points[0][0] < 3) == (route.points[-1][0] < 3) for route in routes)


def test_swept_clearance_catches_thin_obstacles_between_probes() -> None:
    starts = torch.tensor([[2.0, 0.0, 1.0], [0.0, 0.0, 1.0]])  # (2, 3)
    ends = torch.tensor([[4.0, 0.0, 1.0], [1.0, 0.0, 1.0]])  # (2, 3)
    result = segments_clear(starts, ends, closed_wall, radius=0.1, step=0.8)  # (2,)
    assert result.shape == (2,)
    assert result.tolist() == [False, True]
    empty = segments_clear(starts[:0], ends[:0], closed_wall, radius=0.1, step=0.8)  # (0,)
    assert empty.shape == (0,)


def test_open_room_paths_are_smoothed_and_not_labeled_detours() -> None:
    config = RouteConfig(spacing=1, sources=2, goals_per_source=3, min_distance=1)
    graph = build_roadmap(torch.tensor([0.0, 0.0, 1.0]), torch.tensor([3.0, 3.0, 2.0]), empty_scene, 0.1, config)
    assert graph.edges.shape == (len(graph.points), len(graph.points))
    assert (graph.edges != graph.edges.T).nnz == 0
    edges = graph.edges.tocoo()  # (N, N), stored edges only
    lengths = (graph.points[edges.row] - graph.points[edges.col]).norm(dim=-1)  # (E,)
    np.testing.assert_allclose(edges.data, lengths.numpy())
    routes = build_routes(graph, empty_scene, 0.1, config)
    assert routes
    assert all(len(route.points) == 2 and not route.direct_blocked and not route.constrained for route in routes)
    assert all(route.length == route.initial_distance for route in routes)


@pytest.mark.parametrize("free_node", [False, True], ids=["no_free_space", "isolated_node"])
def test_roadmap_without_edges_has_no_routes(free_node: bool) -> None:
    def collision(points: torch.Tensor, radius: float) -> NDArray[np.bool_]:  # (N, 3) -> (N,)
        return (points.ne(0).any(dim=-1) | (not free_node)).numpy()  # (N,)

    config = RouteConfig(spacing=1)
    graph = build_roadmap(torch.zeros(3), torch.ones(3), collision, 0.1, config)
    count = int(free_node)
    assert graph.points.shape == (count, 3)
    assert graph.edges.shape == (count, count)
    assert graph.edges.nnz == 0
    assert graph.wide_components.shape == (count,)
    assert graph.wide_components.tolist() == ([0] if free_node else [])
    assert build_routes(graph, collision, 0.1, config) == []


def test_default_route_budget_supports_large_scene() -> None:
    # The failed training reset needed 81 * 81 * 4 nodes at 0.75 m spacing.
    lower = torch.tensor([-30.0, -30.0, 0.5])  # (3,)
    upper = torch.tensor([30.0, 30.0, 2.5])  # (3,)
    graph = build_roadmap(lower, upper, empty_scene, 0.3, RouteConfig())
    assert graph.points.shape == (26244, 3)
    assert graph.edges.shape == (26244, 26244)
    assert graph.wide_components.shape == (26244,)
    torch.testing.assert_close(graph.points.amin(dim=0), lower)
    torch.testing.assert_close(graph.points.amax(dim=0), upper)


def test_route_grid_budget_and_empty_bounds() -> None:
    config = RouteConfig(max_nodes=8)
    with pytest.raises(ValueError, match="nodes"):
        build_roadmap(torch.zeros(3), torch.ones(3) * 10, empty_scene, 0.1, config)
    with pytest.raises(ValueError, match="sampling volume"):
        build_roadmap(torch.ones(3), torch.zeros(3), empty_scene, 0.1, config)


def test_difficulty_sampling_preserves_hard_routes_and_jitters(door_routes: list[Route]) -> None:
    config = RouteConfig(weights=(0, 0, 1), min_distance=1, jitter=0.1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8)
        sampled = sample_route(
            door_routes, doorway, 0.1, config, torch.tensor([0.0, 0.0, 1.0]), torch.tensor([6.0, 5.0, 1.5])
        )
    assert sampled.difficulty == "hard"
    points = torch.tensor(sampled.points)  # (P, 3)
    assert segments_clear(points[:-1], points[1:], doorway, 0.1, config.probe_step).all()
    assert not segments_clear(points[:1], points[-1:], doorway, 0.1, config.probe_step).item()


def test_missing_difficulties_renormalize_without_fake_hard_cases() -> None:
    route = Route(
        points=((0.0, 0.0, 1.0), (3.0, 0.0, 1.0)),
        direct_blocked=False,
        constrained=False,
        long=True,
        detour_threshold=1.25,
    )
    config = RouteConfig(jitter=0)
    selected = sample_route([route], empty_scene, 0.1, config, torch.zeros(3), torch.ones(3) * 4)
    assert selected.difficulty == "medium"
    with pytest.raises(ValueError, match="positively weighted"):
        sample_route([route], empty_scene, 0.1, replace(config, weights=(0, 0, 1)), torch.zeros(3), torch.ones(3) * 4)


def test_evaluation_categories_horizons_and_serialization(door_routes: list[Route], tmp_path: Path) -> None:
    config = EvaluationConfig(manifest_path=str(tmp_path / "eval.json"), min_episode_steps=1)
    scenarios = select_scenarios("test/room.json", door_routes, config, seed=7, control_dt=0.02)
    assert len(scenarios) == 4
    assert len({scenario.route.points for scenario in scenarios}) > 1
    for scenario in scenarios:
        assert scenario.category in scenario.route.categories
        assert (
            scenario.max_episode_steps * 0.02
            >= scenario.route.length / config.reference_speed + config.maneuver_seconds
        )
        assert sum(value * value for value in scenario.orientation) == pytest.approx(1)
    manifest = Manifest(
        scenarios=scenarios,
        coverage=(SceneCoverage(scene="test/room.json", aliases=(), unavailable=()),),
        seed=7,
        generation=config,
        control_dt=0.02,
    )
    assert TypeAdapter(Manifest).validate_json(json.dumps(asdict(manifest))) == manifest
    with pytest.raises(ValueError, match="coverage"):
        replace(
            manifest, coverage=(SceneCoverage(scene="test/room.json", aliases=(), unavailable=("long_traversal",)),)
        )


def test_unique_scene_layouts_preserve_obstacle_variants(tmp_path: Path) -> None:
    paths = []
    for name, contents in [
        ("a", {"stage": "room", "objects": [1]}),
        ("b", {"objects": [1], "stage": "room"}),
        ("c", {"stage": "room", "objects": [2]}),
    ]:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(contents))
        paths.append(str(path))
    assert unique_scenes(paths, tmp_path) == [("a.json", ("b.json",)), ("c.json", ())]


@pytest.mark.parametrize("weights", [(0.0, 0.0, 0.0), (-1.0, 1.0, 1.0)], ids=["empty", "negative"])
def test_route_weights_validation(weights: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError, match="weights"):
        RouteConfig(weights=weights)
