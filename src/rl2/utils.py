"""Shared numerical and file-storage utilities."""

import os
import tempfile
from pathlib import Path

import gcsfs
import numpy as np
from numpy.typing import ArrayLike


def read_bytes(path: str) -> bytes:
    if path.startswith("gs://"):
        return gcsfs.GCSFileSystem().cat_file(path)
    return Path(path).read_bytes()


def write_bytes(path: str, data: bytes) -> None:
    if path.startswith("gs://"):
        gcsfs.GCSFileSystem().pipe_file(path, data)
    else:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Commit a whole checkpoint, never truncate the previous good file.
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as f:
                temporary = Path(f.name)
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def read_optional(path: str) -> bytes | None:
    """Only a missing object means a new run; storage/authentication errors propagate."""
    try:
        return read_bytes(path)
    except FileNotFoundError:
        return None


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
