"""OMPL route generation using VisFly's host-side spherical collision queries.

SciPy queries the fixed PRM without threaded solves. Finite sampling can miss
passages; reference routes establish geometric feasibility, not optimality.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import ceil, prod

import numpy as np
import torch
from numpy.typing import NDArray
from ompl import base as ob
from ompl import geometric as og
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from rl2.envs.visfly.routes import DIFFICULTIES, Point, Route, RouteConfig

type CollisionQuery = Callable[[torch.Tensor, float], NDArray[np.bool_]]  # (N, 3), radius -> (N,)


def space_information(
    lower: torch.Tensor,  # (3,)
    upper: torch.Tensor,  # (3,)
    collision: CollisionQuery,
    radius: float,
    step: float,
) -> ob.SpaceInformation:
    space = ob.RealVectorStateSpace(3)
    bounds = ob.RealVectorBounds(3)
    for axis in range(3):
        bounds.setLow(axis, float(lower[axis]))
        bounds.setHigh(axis, float(upper[axis]))
    space.setBounds(bounds)
    space.setLongestValidSegmentFraction(min(0.99, step / space.getMaximumExtent()))

    def valid(state: ob.RealVectorStateType) -> bool:
        point = torch.tensor([[state[axis] for axis in range(3)]])  # (1, 3)
        # Inflate spheres to cover gaps between the discrete validator's probes.
        return not bool(collision(point, radius + step / 2)[0])

    info = ob.SpaceInformation(space)
    info.setStateValidityChecker(valid)
    info.setup()
    return info


def make_state(info: ob.SpaceInformation, point: Point) -> ob.RealVectorStateType:
    # OMPL 2's Python allocations own their memory and keep SpaceInformation alive.
    state = info.allocState()
    state[0:3] = point
    return state


def segments_clear(
    starts: torch.Tensor,  # (N, 3)
    ends: torch.Tensor,  # (N, 3)
    collision: CollisionQuery,
    radius: float,
    step: float,
) -> torch.Tensor:  # (N,)
    if not len(starts):
        return torch.empty(0, dtype=torch.bool)  # (0,)
    lower = torch.minimum(starts.amin(dim=0), ends.amin(dim=0)) - step  # (3,)
    upper = torch.maximum(starts.amax(dim=0), ends.amax(dim=0)) + step  # (3,)
    info = space_information(lower, upper, collision, radius, step)
    result = torch.empty(len(starts), dtype=torch.bool)  # (N,)
    for index, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        a = make_state(info, tuple(start))
        b = make_state(info, tuple(end))
        # OMPL assumes the start is valid when checking a motion.
        result[index] = info.isValid(a) and info.checkMotion(a, b)
    return result  # (N,)


class CatalogueSampler(ob.ValidStateSampler):
    """Feed prevalidated, seeded samples to PRM without OMPL's process-global RNG."""

    def __init__(self, info: ob.SpaceInformation, points: list[Point]) -> None:  # (N, 3)
        super().__init__(info)
        self.points = points  # (N, 3)
        self.index = 0

    def sample(self, state: ob.RealVectorStateType) -> bool:
        if self.exhausted():
            return False
        for axis, value in enumerate(self.points[self.index]):
            state[axis] = value
        self.index += 1
        return True

    def sampleNear(self, state: ob.RealVectorStateType, near: ob.RealVectorStateType, distance: float) -> bool:
        raise NotImplementedError("Catalogue sampling supports PRM growth only")

    def exhausted(self) -> bool:
        return self.index == len(self.points)


def prm_edges(
    points: torch.Tensor,  # (N, 3), prevalidated samples in insertion order
    info: ob.SpaceInformation,
    neighbors: int,
) -> torch.Tensor:  # (E, 2), each undirected edge once
    sampler = CatalogueSampler(info, [tuple(point) for point in points.tolist()])

    def allocate(space_info: ob.SpaceInformation) -> CatalogueSampler:
        return sampler

    info.setValidStateSamplerAllocator(allocate)
    planner = og.PRM(info)
    try:
        planner.setProblemDefinition(ob.ProblemDefinition(info))
        planner.setMaxNearestNeighbors(neighbors)
        planner.setup()
        planner.growRoadmapPtc(ob.PlannerTerminationCondition(sampler.exhausted))
        data = ob.PlannerData(info)
        planner.getPlannerData(data)
        # PlannerData reorders vertices and omits isolated milestones.
        indices = {point: index for index, point in enumerate(sampler.points)}
        order = [
            indices[tuple(data.getVertex(i).getState()[axis] for axis in range(3))] for i in range(data.numVertices())
        ]
        edges = [(order[a], order[b]) for a in range(data.numVertices()) for b in data.getEdges(a) if a < b]
        return torch.tensor(edges, dtype=torch.int64).reshape(-1, 2)  # (E, 2)
    finally:
        planner.clear()
        info.clearValidStateSamplerAllocator()


def shortcut_path(points: torch.Tensor, info: ob.SpaceInformation) -> tuple[Point, ...]:  # (K, 3) -> (P, 3)
    path = og.PathGeometric(info)
    for point in points.tolist():
        path.append(make_state(info, tuple(point)))
    og.PathSimplifier(info).ropeShortcutPath(path)
    # Newly inserted vertices can fail the inflated clearance test between probes.
    if not path.check():
        return tuple(tuple(point) for point in points.tolist())  # (K, 3)
    return tuple(tuple(path.getState(i)[axis] for axis in range(3)) for i in range(path.getStateCount()))  # (P, 3)


@dataclass
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
    """Build an OMPL PRM from a fixed budget of seeded uniform samples."""
    if torch.any(upper <= lower):
        raise ValueError("Scene has no route sampling volume")
    count = prod(max(2, ceil(float(size) / config.spacing) + 1) for size in upper - lower)
    if count > config.max_nodes:
        raise ValueError(f"Route roadmap needs {count} samples; increase spacing or max_nodes")
    samples = lower + torch.rand((count, 3)) * (upper - lower)  # (G, 3)
    free = ~torch.from_numpy(collision(samples, clearance + config.probe_step / 2))  # (G,)
    points = samples[free]  # (N, 3)
    info = space_information(lower, upper, collision, clearance, config.probe_step)
    pairs = prm_edges(points, info, config.neighbors)  # (E, 2)
    left, right = pairs.unbind(dim=1)  # each (E,)
    lengths = (points[left] - points[right]).norm(dim=-1)  # (E,)
    wide = segments_clear(
        points[left], points[right], collision, clearance + config.passage_margin, config.probe_step
    )  # (E,)
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


def build_routes(graph: Roadmap, collision: CollisionQuery, clearance: float, config: RouteConfig) -> list[Route]:
    if len(graph.points) < 2:
        return []
    lower = graph.points.amin(dim=0) - config.probe_step  # (3,)
    upper = graph.points.amax(dim=0) + config.probe_step  # (3,)
    info = space_information(lower, upper, collision, clearance, config.probe_step)
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
                (tuple(points[0].tolist()), tuple(points[-1].tolist())) if direct else shortcut_path(points, info)
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
    *,
    long_threshold: float,  # Metres; use the catalogue's generation threshold.
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
        distance = float((proposed[1] - proposed[0]).norm())
        valid = bool(((proposed >= lower) & (proposed <= upper)).all())
        valid = valid and distance >= config.min_distance and (distance >= long_threshold) == route.long
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
