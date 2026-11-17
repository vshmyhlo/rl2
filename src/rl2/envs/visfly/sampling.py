"""Bounded collision-free position sampling for host-stepped VisFly scenes."""

from collections.abc import Callable

import numpy as np
import torch
from numpy.typing import NDArray


def sample_positions(
    lower: torch.Tensor,  # (3,) lower bounds in VisFly coordinates
    upper: torch.Tensor,  # (3,) upper bounds in VisFly coordinates
    count: int,
    collision: Callable[[torch.Tensor], NDArray[np.bool_]],  # (N, 3) -> (N,)
    *,
    reference: torch.Tensor | None = None,  # (count, 3) starting positions for goal sampling
    distance: tuple[float, float] = (2.0, 8.0),
    max_attempts: int = 128,
) -> torch.Tensor:
    """Return (count, 3) collision-free positions; optionally constrain distance from reference.

    Endpoints are checked against geometry; path connectivity is not guaranteed.
    """
    if torch.any(upper <= lower):
        raise ValueError("Scene has no sampling volume within the requested height and clearance")
    low = lower.expand(count, 3)  # (N, 3)
    high = upper.expand(count, 3)  # (N, 3)
    if reference is not None:
        low = torch.maximum(low, reference - distance[1])  # (N, 3)
        high = torch.minimum(high, reference + distance[1])  # (N, 3)
    positions = torch.empty((count, 3), dtype=lower.dtype, device=lower.device)  # (N, 3)
    pending = torch.arange(count, device=lower.device)  # (P,)
    for _ in range(max_attempts):
        candidates = low[pending] + torch.rand((len(pending), 3), device=lower.device) * (
            high[pending] - low[pending]
        )  # (P, 3)
        valid = ~torch.as_tensor(collision(candidates), device=lower.device)  # (P,)
        if reference is not None:
            separation = (candidates - reference[pending]).norm(dim=-1)  # (P,)
            valid &= (separation >= distance[0]) & (separation <= distance[1])
        positions[pending[valid]] = candidates[valid]
        pending = pending[~valid]  # (P,)
        if len(pending) == 0:
            return positions  # (N, 3)
    raise RuntimeError(f"Could not sample {len(pending)} collision-free positions after {max_attempts} attempts")
