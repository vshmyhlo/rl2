"""Host collision queries without per-point Torch conversion."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def collision_points(
    scene: Any,
    positions: NDArray[np.float32],  # (N, 3), VisFly coordinates
    radius: float,
) -> NDArray[np.bool_]:  # (N,)
    """Return (N,) collisions, preserving VisFly's coordinate and distance precision."""
    habitat = positions[:, [1, 2, 0]].astype(np.float64)  # (N, 3)
    habitat[:, (0, 2)] *= -1
    result = np.empty(len(positions), dtype=bool)  # (N,)
    threshold = np.float32(radius)
    for index, point in enumerate(habitat):  # (3,)
        hit = scene.get_closest_collision_point(pt=point.reshape(3, 1), max_search_radius=radius)
        result[index] = hit.is_out_bound or np.float32(np.linalg.norm(hit.hit_pos - point)) < threshold
    return result  # (N,)
