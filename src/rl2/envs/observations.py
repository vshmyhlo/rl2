"""Observation pytrees supported by single-agent PPO."""

from typing import NamedTuple

import jax


class CraftaxObservation(NamedTuple):
    """Symbolic features; optional leading batch/time axes S."""

    grid: jax.Array  # (*S, H, W, C) float32 tile/entity/light features
    features: jax.Array  # (*S, F) float32 inventory, status and direction


type Observation = jax.Array | CraftaxObservation
