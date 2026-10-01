"""Shared numerical utilities."""

import numpy as np
from numpy.typing import ArrayLike


class RunningMeanStd:
    """Host-side parallel moments, using float64 to accumulate long runs."""

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, samples: ArrayLike) -> None:
        samples = np.asarray(samples, dtype=np.float64)
        if not len(samples):
            return
        batch_mean, batch_var = samples.mean(axis=0), samples.var(axis=0)
        total = self.count + len(samples)
        delta = batch_mean - self.mean
        self.var = (
            self.var * self.count + batch_var * len(samples) + delta**2 * self.count * len(samples) / total
        ) / total
        self.mean += delta * len(samples) / total
        self.count = total
