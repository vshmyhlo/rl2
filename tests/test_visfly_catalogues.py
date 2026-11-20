"""Seed isolation, queued work reuse, and planner lifecycle."""

from concurrent.futures import Future
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from rl2.envs.visfly.catalogues import CataloguePlanner, PlannedCatalogue, PlanningTask, build_catalogue, planning_seed
from rl2.envs.visfly.routes import RouteConfig
from rl2.envs.visfly.sampling import SamplingBounds


def free_space(points: NDArray[np.float32], radius: float) -> NDArray[np.bool_]:  # (N, 3) -> (N,)
    return np.zeros(len(points), dtype=bool)  # (N,)


def task() -> PlanningTask:
    return PlanningTask(
        scene_path="scene.json",
        dataset_path="dataset.json",
        seed=planning_seed(7, 0, 0),
        clearance=0.1,
        height=(0.5, 2.5),
        config=RouteConfig(spacing=1, sources=2, goals_per_source=3, min_distance=1),
        scene_seed=7,
    )


def test_catalogue_seed_isolated_and_fresh_per_visit() -> None:
    bounds = SamplingBounds(lower=torch.tensor([0.0, 0.0, 1.0]), upper=torch.tensor([4.0, 3.0, 2.0]))  # each (3,)
    original = torch.get_rng_state().clone()  # (R,)
    first = build_catalogue(task(), bounds, free_space)
    again = build_catalogue(task(), bounds, free_space)
    fresh = build_catalogue(replace(task(), seed=planning_seed(7, 1, 0)), bounds, free_space)
    assert first == again and first.routes
    assert first.routes != fresh.routes
    assert all(np.asarray(route.points).shape[1:] == (3,) for route in first.routes)
    torch.testing.assert_close(torch.get_rng_state(), original)
    assert planning_seed(7, 0, 0) != planning_seed(7, 0, 1)


def test_planner_reuses_prefetch_and_ignores_completion_order() -> None:
    first, second = task(), replace(task(), scene_path="second.json")
    a: Future[PlannedCatalogue] = Future()
    b: Future[PlannedCatalogue] = Future()
    results = [PlannedCatalogue(routes=[], long_threshold=1), PlannedCatalogue(routes=[], long_threshold=2)]
    with patch("rl2.envs.visfly.catalogues.ProcessPoolExecutor") as create:
        pool = create.return_value
        pool.submit.side_effect = [a, b]
        planner = CataloguePlanner(workers=2, worker=Mock())
        planner.submit([first, second])
        planner.submit([first, second])
        b.set_result(results[1])
        a.set_result(results[0])
        assert planner.take(first) == results[0]
        assert planner.take(second) == results[1]
        assert pool.submit.call_count == 2
        planner.close()
        pool.shutdown.assert_called_once_with(wait=True, cancel_futures=True)


def test_planner_propagates_failure_and_cancels_unused_work() -> None:
    failure: Future[PlannedCatalogue] = Future()
    unused: Future[PlannedCatalogue] = Future()
    failure.set_exception(RuntimeError("invalid geometry"))
    with patch("rl2.envs.visfly.catalogues.ProcessPoolExecutor") as create:
        create.return_value.submit.side_effect = [failure, unused]
        planner = CataloguePlanner(workers=1, worker=Mock())
        planner.submit([task(), replace(task(), scene_path="unused.json")])
        with pytest.raises(RuntimeError, match="invalid geometry"):
            planner.take(task())
        planner.close()
        assert unused.cancelled()


def test_planner_submits_unpredicted_scene() -> None:
    future: Future[PlannedCatalogue] = Future()
    result = PlannedCatalogue(routes=[], long_threshold=3)
    future.set_result(result)
    with patch("rl2.envs.visfly.catalogues.ProcessPoolExecutor") as create:
        create.return_value.submit.return_value = future
        planner = CataloguePlanner(workers=1, worker=Mock())
        assert planner.take(task()) == result
        create.return_value.submit.assert_called_once()
        planner.close()
