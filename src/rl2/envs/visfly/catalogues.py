"""Bounded background planning with independent seeds for each scene visit."""

import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from hashlib import sha256

import torch

from rl2.envs.visfly.planning import CollisionQuery, build_roadmap, build_routes
from rl2.envs.visfly.routes import Route, RouteConfig
from rl2.envs.visfly.sampling import SamplingBounds


@dataclass(frozen=True)
class PlanningTask:
    scene_path: str
    dataset_path: str
    seed: int
    clearance: float
    height: tuple[float, float]
    config: RouteConfig
    scene_seed: int


@dataclass
class PlannedCatalogue:
    routes: list[Route]
    long_threshold: float


def planning_seed(seed: int, generation: int, scene_id: int) -> int:
    return int.from_bytes(sha256(f"{seed}:{generation}:{scene_id}".encode()).digest()[:8], "little")


def build_catalogue(task: PlanningTask, bounds: SamplingBounds, collision: CollisionQuery) -> PlannedCatalogue:
    """Plan fresh training routes without consuming the caller's episode RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(task.seed).get_state())
        graph = build_roadmap(
            bounds.lower, bounds.upper, collision, task.clearance, task.config, classify_passages=False
        )
        routes = build_routes(graph, collision, task.clearance, task.config, shortcut=False)
    return PlannedCatalogue(routes=routes, long_threshold=task.config.long_fraction * graph.extent)


def initialize_planner() -> None:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    torch.set_num_threads(1)


class CataloguePlanner:
    """Own one scene batch of futures; consume results in scene order."""

    def __init__(self, workers: int, worker: Callable[[PlanningTask], PlannedCatalogue]) -> None:
        self._worker = worker
        self._pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_planner,
        )
        self._pending: dict[PlanningTask, Future[PlannedCatalogue]] = {}

    def submit(self, tasks: list[PlanningTask]) -> None:
        for task in tasks:
            if task not in self._pending:
                self._pending[task] = self._pool.submit(self._worker, task)

    def take(self, task: PlanningTask) -> PlannedCatalogue:
        self.submit([task])
        return self._pending.pop(task).result()

    def discard(self) -> None:
        for future in self._pending.values():
            future.cancel()
        self._pending.clear()

    def close(self) -> None:
        self.discard()
        self._pool.shutdown(wait=True, cancel_futures=True)
