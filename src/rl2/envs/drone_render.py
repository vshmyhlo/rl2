"""Host-side position rendering shared by drone adapters."""

import jax
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray


def render_drones(
    positions: jax.Array,  # (N, 3)
    targets: jax.Array,  # (K, 3)
    extent: float,
) -> NDArray[np.uint8]:  # (400, 400, 3)
    """Render (N, 3) positions and (K, 3) targets as an RGB frame."""
    positions, targets = jax.device_get((positions, targets))  # (N, 3); (K, 3)
    figure = Figure(figsize=(5, 5), dpi=80)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot(projection="3d")
    axes.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(0, 3), xlabel="x", ylabel="y", zlabel="z")
    axes.scatter(*positions.T, c="tab:blue", s=50)
    axes.scatter(*targets.T, c="tab:orange", marker="x", s=50)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()  # (400, 400, 3)
