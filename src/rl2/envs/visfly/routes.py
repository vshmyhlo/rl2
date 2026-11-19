"""Bounded host-side flight roadmaps using VisFly's CPU collision queries.

The grid can miss passages; a missing route is not proof of impossibility.
SciPy supplies sparse graph connectivity and shortest paths. Reference lengths
describe smoothed feasible paths, not exact continuous-space shortest paths.
"""

from collections.abc import Callable
from dataclasses import dataclass as std_dataclass
from dataclasses import replace
from math import ceil, dist, prod
from typing import Annotated, Literal, Self

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

type Point = tuple[float, float, float]
type Category = Literal["long_traversal", "obstacle_avoidance", "substantial_detour", "constrained_passage"]
type Difficulty = Literal["easy", "medium", "hard"]
type CollisionQuery = Callable[[torch.Tensor, float], NDArray[np.bool_]]  # (N, 3), radius -> (N,)
CATEGORIES: tuple[Category, ...] = ("long_traversal", "obstacle_avoidance", "substantial_detour", "constrained_passage")
DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")


@dataclass(frozen=True, config=ConfigDict(extra="forbid", allow_inf_nan=False))
class RouteConfig:
    spacing: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.75
    probe_step: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.2
    max_nodes: Annotated[int, Field(ge=8)] = 32768
    sources: Annotated[int, Field(gt=0)] = 24
    goals_per_source: Annotated[int, Field(ge=3)] = 12
    min_distance: Annotated[float, Field(gt=0.5, allow_inf_nan=False)] = 2.0
    long_fraction: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 0.6
    detour_ratio: Annotated[float, Field(gt=1, allow_inf_nan=False)] = 1.25
    passage_margin: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.3
    weights: tuple[float, float, float] = (0.2, 0.4, 0.4)  # easy, medium, hard
    jitter: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.2

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        if any(not 0 <= weight < float("inf") for weight in self.weights) or sum(self.weights) <= 0:
            raise ValueError("route weights must be nonnegative with a positive sum")
        return self


@dataclass(frozen=True, config=ConfigDict(extra="forbid", allow_inf_nan=False))
class Route:
    points: Annotated[tuple[Point, ...], Field(min_length=2)]  # (P, 3) collision-checked polyline
    direct_blocked: bool
    constrained: bool
    long: bool
    detour_threshold: float

    @property
    def initial_distance(self) -> float:
        return dist(self.points[0], self.points[-1])

    @property
    def length(self) -> float:
        return sum(dist(a, b) for a, b in zip(self.points, self.points[1:]))

    @property
    def categories(self) -> tuple[Category, ...]:
        flags = (
            self.long,
            self.direct_blocked,
            self.direct_blocked and self.length >= self.detour_threshold * self.initial_distance,
            self.constrained,
        )
        return tuple(category for category, flag in zip(CATEGORIES, flags) if flag)

    @property
    def difficulty(self) -> Difficulty:
        if self.direct_blocked:
            return "hard"
        return "medium" if self.long else "easy"


def segments_clear(
    starts: torch.Tensor,  # (N, 3)
    ends: torch.Tensor,  # (N, 3)
    collision: CollisionQuery,
    radius: float,
    step: float,
) -> torch.Tensor:  # (N,)
    """Check swept spheres; half-step radius inflation covers gaps between probes."""
    result = torch.empty(len(starts), dtype=torch.bool)  # (N,)
    for offset in range(0, len(starts), 64):
        a, b = starts[offset : offset + 64], ends[offset : offset + 64]  # (M, 3)
        samples = torch.ceil((b - a).norm(dim=-1) / step).clamp_min(1)  # (M,)
        t = (torch.arange(int(samples.max()) + 1)[None, :] / samples[:, None]).clamp_max(1)  # (M, K)
        points = a[:, None] + (b - a)[:, None] * t[:, :, None]  # (M, K, 3)
        occupied = torch.from_numpy(collision(points.reshape(-1, 3), radius + step / 2))  # (M*K,)
        result[offset : offset + len(a)] = ~occupied.reshape(len(a), -1).any(dim=1)
    return result  # (N,)


@std_dataclass
class Roadmap:
    points: torch.Tensor  # (N, 3), CPU; geometry API consumes Torch tensors
    edges: csr_matrix  # (N, N), symmetric edge lengths in metres
    wide_components: NDArray[np.int32]  # (N,), -1 for points without extra clearance
    extent: float


def sparse_graph(
    starts: torch.Tensor,  # (E,) node indices
    ends: torch.Tensor,  # (E,) node indices
    lengths: torch.Tensor,  # (E,) edge lengths in metres
    node_count: int,
) -> csr_matrix:  # (N, N)
    """Store each undirected edge in both directions for SciPy's graph routines."""
    edges = csr_matrix((lengths.numpy(), (starts.numpy(), ends.numpy())), shape=(node_count, node_count))  # (N, N)
    return edges + edges.T  # (N, N)


def build_roadmap(
    lower: torch.Tensor,  # (3,)
    upper: torch.Tensor,  # (3,)
    collision: CollisionQuery,
    clearance: float,
    config: RouteConfig,
) -> Roadmap:
    """Build a six-neighbor grid; reject excessive resolution rather than silently coarsening."""
    sizes = [max(2, ceil(float(size) / config.spacing) + 1) for size in upper - lower]
    if torch.any(upper <= lower):
        raise ValueError("Scene has no route sampling volume")
    if prod(sizes) > config.max_nodes:
        raise ValueError(f"Route grid needs {prod(sizes)} nodes; increase spacing or max_nodes")
    axes = [torch.linspace(float(lo), float(hi), n) for lo, hi, n in zip(lower, upper, sizes)]  # each (D,)
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)  # (G, 3)
    free = ~torch.from_numpy(collision(grid, clearance + config.probe_step / 2))  # (G,)
    points = grid[free]  # (N, 3)
    ids = torch.full((len(grid),), -1, dtype=torch.int64)  # (G,)
    ids[free] = torch.arange(len(points))
    ids = ids.reshape(sizes)  # (X, Y, Z)
    starts, ends, weights, wide_masks = [], [], [], []  # each contains three (C_axis,) CPU tensors
    for axis in range(3):
        a = ids.narrow(axis, 0, sizes[axis] - 1).flatten()  # (E,)
        b = ids.narrow(axis, 1, sizes[axis] - 1).flatten()  # (E,)
        valid = (a >= 0) & (b >= 0)  # (E,)
        a, b = a[valid], b[valid]  # (F,)
        clear = segments_clear(points[a], points[b], collision, clearance, config.probe_step)  # (F,)
        a, b = a[clear], b[clear]  # (C,)
        wide = segments_clear(
            points[a], points[b], collision, clearance + config.passage_margin, config.probe_step
        )  # (C,)
        starts.append(a)
        ends.append(b)
        weights.append((points[a] - points[b]).norm(dim=-1))  # (C,)
        wide_masks.append(wide)
    left, right, lengths, wide = map(torch.cat, (starts, ends, weights, wide_masks))  # each (E,)
    edges = sparse_graph(left, right, lengths, len(points))  # (N, N)
    wide_edges = sparse_graph(left[wide], right[wide], lengths[wide], len(points))  # (N, N)
    _, labels = connected_components(wide_edges, directed=False)  # (N,)
    # Isolated valid nodes have distinct labels; invalid nodes cannot mark a passage endpoint.
    invalid = collision(points, clearance + config.passage_margin + config.probe_step / 2)  # (N,)
    labels[invalid] = -1
    return Roadmap(
        points=points,
        edges=edges,
        wide_components=labels,  # (N,)
        extent=float((points.amax(dim=0) - points.amin(dim=0)).norm()) if len(points) else 0.0,
    )


def smooth_path(
    points: torch.Tensor,  # (K, 3)
    collision: CollisionQuery,
    clearance: float,
    step: float,
) -> tuple[Point, ...]:  # (P, 3)
    """Return (P, 3) polyline vertices after removing collision-free grid detours."""
    selected = [0]
    while selected[-1] < len(points) - 1:
        current = selected[-1]
        targets = points[current + 1 :]  # (K, 3)
        clear = segments_clear(points[current].expand_as(targets), targets, collision, clearance, step)  # (K,)
        selected.append(current + 1 + int(torch.nonzero(clear)[-1, 0]))
    return tuple(tuple(point) for point in points[selected].tolist())  # (P, 3)


def build_routes(graph: Roadmap, collision: CollisionQuery, clearance: float, config: RouteConfig) -> list[Route]:
    routes = []
    seen: set[tuple[int, int]] = set()
    for source in torch.randperm(len(graph.points))[: config.sources].tolist():
        distances, parents = dijkstra(
            graph.edges, directed=False, indices=source, return_predecessors=True
        )  # both (N,); one source at a time bounds memory
        separation = (graph.points - graph.points[source]).norm(dim=-1)  # (N,)
        separations = separation.tolist()
        candidates = [
            i for i, length in enumerate(distances) if length < float("inf") and separations[i] >= config.min_distance
        ]
        if not candidates:
            continue
        candidates.sort(key=separations.__getitem__)
        # Include a local and cross-scene route, with the rest spread over the reachable component.
        indices = torch.randperm(len(candidates))[: config.goals_per_source - 2].tolist()
        goals = [candidates[0], candidates[-1], *(candidates[i] for i in indices)]
        for goal in goals:
            pair = (min(source, goal), max(source, goal))
            if pair in seen:
                continue
            seen.add(pair)
            path = [goal]
            while path[-1] != source:
                path.append(int(parents[path[-1]]))
            points = graph.points[list(reversed(path))]  # (K, 3)
            direct = bool(segments_clear(points[:1], points[-1:], collision, clearance, config.probe_step)[0])
            smoothed = (
                (tuple(points[0].tolist()), tuple(points[-1].tolist()))
                if direct
                else smooth_path(points, collision, clearance, config.probe_step)
            )
            wide_start, wide_goal = graph.wide_components[source], graph.wide_components[goal]
            routes.append(
                Route(
                    points=smoothed,
                    direct_blocked=not direct,
                    constrained=not direct and wide_start >= 0 and wide_goal >= 0 and wide_start != wide_goal,
                    long=float(separation[goal]) >= config.long_fraction * graph.extent,
                    detour_threshold=config.detour_ratio,
                )
            )
    return routes


def sample_route(
    routes: list[Route],
    collision: CollisionQuery,
    clearance: float,
    config: RouteConfig,
    lower: torch.Tensor,  # (3,)
    upper: torch.Tensor,  # (3,)
) -> Route:
    """Renormalize weights over available levels; jitter endpoints through validated connectors."""
    groups = [[route for route in routes if route.difficulty == level] for level in DIFFICULTIES]
    weights = torch.tensor([weight if group else 0 for weight, group in zip(config.weights, groups)])  # (3,)
    if not bool(weights.sum() > 0):
        raise ValueError("No routes for any positively weighted difficulty in this scene")
    group = groups[int(torch.multinomial(weights, 1))]
    route = group[int(torch.randint(len(group), ()))]
    if bool(torch.randint(2, ())):
        route = replace(route, points=tuple(reversed(route.points)))
    if config.jitter:
        endpoints = torch.tensor([route.points[0], route.points[-1]])  # (2, 3)
        proposed = endpoints + (2 * torch.rand((2, 3)) - 1) * config.jitter  # (2, 3)
        valid = bool(((proposed >= lower) & (proposed <= upper)).all())
        valid = valid and float((proposed[1] - proposed[0]).norm()) >= config.min_distance
        if valid and bool(segments_clear(endpoints, proposed, collision, clearance, config.probe_step).all()):
            blocked = not bool(segments_clear(proposed[:1], proposed[1:], collision, clearance, config.probe_step)[0])
            # Preserve the selected difficulty; never turn a hard sample into a clear corridor.
            if blocked == route.direct_blocked:
                points = (
                    (tuple(proposed[0].tolist()), *route.points, tuple(proposed[1].tolist()))
                    if blocked
                    else (tuple(proposed[0].tolist()), tuple(proposed[1].tolist()))
                )  # (P, 3)
                route = replace(route, points=points)
    return route
