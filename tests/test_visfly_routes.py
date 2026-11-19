"""Geometry, difficulty selection, and fixed evaluation scenario contracts."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from pydantic import TypeAdapter

from rl2.envs.visfly.planning import (
    build_roadmap,
    build_routes,
    prm_edges,
    sample_route,
    segments_clear,
    shortcut_path,
    space_information,
)
from rl2.envs.visfly.routes import Route, RouteConfig
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
def test_roadmap_without_edges_has_no_routes(free_node: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    def collision(points: torch.Tensor, radius: float) -> NDArray[np.bool_]:  # (N, 3) -> (N,)
        return (points.ne(0).any(dim=-1) | (not free_node)).numpy()  # (N,)

    samples = torch.arange(24).reshape(8, 3) / 24  # (8, 3)
    samples[0] = 0
    monkeypatch.setattr(torch, "rand", Mock(return_value=samples))
    config = RouteConfig(spacing=1)
    graph = build_roadmap(torch.zeros(3), torch.ones(3), collision, 0.1, config)
    count = int(free_node)
    assert graph.points.shape == (count, 3)
    assert graph.edges.shape == (count, count)
    assert graph.edges.nnz == 0
    assert graph.wide_components.shape == (count,)
    assert graph.wide_components.tolist() == ([0] if free_node else [])
    assert build_routes(graph, collision, 0.1, config) == []


def test_default_route_budget_supports_large_scene(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test the failed reset's volume without constructing a large graph in a unit test.
    edges = Mock(return_value=torch.empty((0, 2), dtype=torch.int64))
    monkeypatch.setattr("rl2.envs.visfly.planning.prm_edges", edges)
    lower = torch.tensor([-30.0, -30.0, 0.5])  # (3,)
    upper = torch.tensor([30.0, 30.0, 2.5])  # (3,)
    graph = build_roadmap(lower, upper, empty_scene, 0.3, RouteConfig())
    assert graph.points.shape == (26244, 3)
    assert graph.edges.shape == (26244, 26244)
    assert graph.wide_components.shape == (26244,)
    assert (graph.points >= lower).all() and (graph.points <= upper).all()
    assert edges.call_args.args[0] is graph.points


def test_route_sample_budget_and_empty_bounds() -> None:
    config = RouteConfig(max_nodes=8)
    with pytest.raises(ValueError, match="samples"):
        build_roadmap(torch.zeros(3), torch.ones(3) * 10, empty_scene, 0.1, config)
    with pytest.raises(ValueError, match="sampling volume"):
        build_roadmap(torch.ones(3), torch.zeros(3), empty_scene, 0.1, config)


def test_difficulty_sampling_preserves_hard_routes_and_jitters(door_routes: list[Route]) -> None:
    config = RouteConfig(weights=(0, 0, 1), min_distance=1, jitter=0.1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8)
        sampled = sample_route(
            door_routes,
            doorway,
            0.1,
            config,
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor([6.0, 5.0, 1.5]),
            long_threshold=config.long_fraction * (6**2 + 5**2 + 0.5**2) ** 0.5,
        )
    assert sampled.difficulty == "hard"
    points = torch.tensor(sampled.points)  # (P, 3)
    assert segments_clear(points[:-1], points[1:], doorway, 0.1, config.probe_step).all()
    assert not segments_clear(points[:1], points[-1:], doorway, 0.1, config.probe_step).item()


@pytest.mark.parametrize(
    ("distance", "offset", "accepted"),
    [
        pytest.param(5.9, 0.4, False, id="easy_to_medium"),
        pytest.param(6.1, -0.4, False, id="medium_to_easy"),
        pytest.param(5.75, 0.25, False, id="easy_reaches_threshold"),
        pytest.param(6.25, -0.25, True, id="medium_stays_at_threshold"),
        pytest.param(5.5, 0.25, True, id="easy_stays_below_threshold"),
    ],
)
def test_jitter_preserves_long_route_threshold(
    distance: float, offset: float, accepted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    route = Route(
        points=((1.0, 1.0, 1.0), (1.0 + distance, 1.0, 1.0)),
        direct_blocked=False,
        constrained=False,
        long=distance >= 6.0,
        detour_threshold=1.25,
    )
    config = RouteConfig(jitter=0.5)
    # Fix route direction and move only the goal along x.
    draws = torch.tensor([[0.5, 0.5, 0.5], [0.5 + offset, 0.5, 0.5]])  # (2, 3)
    monkeypatch.setattr(torch, "randint", Mock(return_value=torch.tensor(0)))
    monkeypatch.setattr(torch, "rand", Mock(return_value=draws))
    sampled = sample_route(
        [route], empty_scene, 0.1, config, torch.zeros(3), torch.full((3,), 10.0), long_threshold=6.0
    )
    assert torch.tensor(sampled.points).shape == (2, 3)
    assert sampled.initial_distance == pytest.approx(distance + offset if accepted else distance)
    assert sampled.long == (sampled.initial_distance >= 6.0)
    assert sampled.difficulty == route.difficulty


def test_missing_difficulties_renormalize_without_fake_hard_cases() -> None:
    route = Route(
        points=((0.0, 0.0, 1.0), (3.0, 0.0, 1.0)),
        direct_blocked=False,
        constrained=False,
        long=True,
        detour_threshold=1.25,
    )
    config = RouteConfig(jitter=0)
    selected = sample_route([route], empty_scene, 0.1, config, torch.zeros(3), torch.ones(3) * 4, long_threshold=2.0)
    assert selected.difficulty == "medium"
    with pytest.raises(ValueError, match="positively weighted"):
        sample_route(
            [route],
            empty_scene,
            0.1,
            replace(config, weights=(0, 0, 1)),
            torch.zeros(3),
            torch.ones(3) * 4,
            long_threshold=2.0,
        )


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
    encoded = asdict(manifest)
    assert TypeAdapter(Manifest).validate_json(json.dumps(encoded)) == manifest
    del encoded["generation"]["routes"]["neighbors"]
    assert TypeAdapter(Manifest).validate_json(json.dumps(encoded)) == manifest
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


def test_prm_export_preserves_indices_with_isolated_samples() -> None:
    points = torch.tensor([[-1.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])  # (4, 3)

    def wall(positions: torch.Tensor, radius: float) -> NDArray[np.bool_]:  # (N, 3) -> (N,)
        return (positions[:, 0].abs() < radius).numpy()  # (N,)

    info = space_information(torch.tensor([-2.0, -1, -1]), torch.tensor([4.0, 1, 1]), wall, 0.1, 0.2)
    edges = prm_edges(points, info, neighbors=3)  # (E, 2)
    assert edges.shape == (3, 2)
    assert {tuple(sorted(pair)) for pair in edges.tolist()} == {(1, 2), (1, 3), (2, 3)}


def test_catalogue_is_repeatable_after_unrelated_planning() -> None:
    config = RouteConfig(spacing=1, sources=6, goals_per_source=3, min_distance=1)
    lower, upper = torch.zeros(3), torch.tensor([6.0, 5.0, 1.0])  # each (3,)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        first = build_roadmap(lower, upper, doorway, 0.1, config)
        routes = build_routes(first, doorway, 0.1, config)
        build_roadmap(lower, upper, doorway, 0.1, config)
        torch.manual_seed(17)
        second = build_roadmap(lower, upper, doorway, 0.1, config)
        repeated = build_routes(second, doorway, 0.1, config)
    torch.testing.assert_close(first.points, second.points, rtol=0, atol=0)
    assert (first.edges != second.edges).nnz == 0
    assert any(route.direct_blocked for route in routes)
    assert routes == repeated


def test_segment_validation_checks_invalid_starts_and_zero_length() -> None:
    starts = torch.tensor([[3.0, 0, 1], [2.0, 0, 1], [3.0, 0, 1]])  # (3, 3)
    ends = torch.tensor([[2.0, 0, 1], [2.0, 0, 1], [3.0, 0, 1]])  # (3, 3)
    clear = segments_clear(starts, ends, closed_wall, radius=0.1, step=0.2)  # (3,)
    assert clear.shape == (3,)
    assert clear.tolist() == [False, True, False]


def test_shortcut_preserves_clearance_when_inserting_vertices() -> None:
    def obstacle(points: torch.Tensor, radius: float) -> NDArray[np.bool_]:  # (N, 3) -> (N,)
        return (points.norm(dim=-1) < radius).numpy()  # (N,)

    # The original edge clears the swept sphere, but a new midpoint at x=0 lacks
    # the additional half-probe margin required for replay validation.
    points = torch.tensor([[-1.6, 0.17, 0], [1.6, 0.17, 0], [1.6, -0.17, 0]])  # (3, 3)
    assert segments_clear(points[:-1], points[1:], obstacle, radius=0.05, step=0.3).all()
    info = space_information(torch.tensor([-2.0, -1, -1]), torch.tensor([2.0, 1, 1]), obstacle, 0.05, 0.3)
    result = torch.tensor(shortcut_path(points, info))  # (P, 3)
    assert result.shape[1:] == (3,)
    torch.testing.assert_close(result[[0, -1]], points[[0, -1]])
    assert segments_clear(result[:-1], result[1:], obstacle, radius=0.05, step=0.3).all()
