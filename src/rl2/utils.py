"""Shared numerical and file-storage utilities."""

import os
import tempfile
from pathlib import Path

import gcsfs
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray


@jax.jit
def gae(
    rewards: jax.Array,  # (T, B)
    dones: jax.Array,  # (T, B)
    values: jax.Array,  # (T, B)
    next_value: jax.Array,  # (B,)
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:  # both (T, B)
    """Compute generalized advantage estimates over T steps and B environments.

    Truncation bootstrap is already included in rewards; dones stop traces.

    Args:
        rewards: (T, B) rewards including truncation bootstrap.
        dones: (T, B) episode completion flags.
        values: (T, B) value estimates.
        next_value: (B,) bootstrap values after the final step.
        gamma: Discount factor.
        gae_lambda: Advantage trace decay.

    Returns:
        advantages: (T, B) generalized advantage estimates.
        returns: (T, B) value targets.
    """

    def step(
        carry: tuple[jax.Array, jax.Array],  # both (B,)
        transition: tuple[jax.Array, jax.Array, jax.Array],  # each (B,)
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:  # ((B,), (B,)), (B,)
        advantage, next_value = carry  # both (B,)
        reward, done, value = transition  # each (B,)
        discount = gamma * (1.0 - done)  # (B,)
        delta = reward + discount * next_value - value  # (B,)
        advantage = delta + discount * gae_lambda * advantage  # (B,)
        return (advantage, value), advantage  # ((B,), (B,)), (B,)

    _, advantages = jax.lax.scan(  # (T, B)
        step,
        (jnp.zeros_like(next_value), next_value),  # both (B,)
        (rewards, dones, values),  # each (T, B)
        reverse=True,
    )
    returns = advantages + values  # (T, B)
    return advantages, returns  # both (T, B)


def explained_variance(
    values: jax.Array,  # (*S,)
    returns: jax.Array,  # (*S,)
) -> jax.Array:  # ()
    variance = jnp.var(returns)  # ()
    return jnp.where(variance > 0, 1 - jnp.var(returns - values) / variance, jnp.nan)  # ()


def read_bytes(path: str) -> bytes:
    if path.startswith("gs://"):
        return gcsfs.GCSFileSystem().cat_file(path)
    return Path(path).read_bytes()


def write_bytes(path: str, data: bytes) -> None:
    if path.startswith("gs://"):
        gcsfs.GCSFileSystem().pipe_file(path, data)
        return

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Commit a whole checkpoint, never truncate the previous good file.
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as file:
            temporary = Path(file.name)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
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
        self.mean: NDArray[np.float64] = np.zeros(shape, dtype=np.float64)  # (*S,)
        self.var: NDArray[np.float64] = np.ones(shape, dtype=np.float64)  # (*S,)
        self.count = 1e-4

    def update(self, samples: ArrayLike) -> None:  # samples: (N, *S)
        """Merge (N, *S) samples over the leading batch axis; S is the configured shape."""
        samples = np.asarray(samples, dtype=np.float64)  # (N, *S)
        batch_count = len(samples)
        if batch_count == 0:
            return
        batch_mean = samples.mean(axis=0)  # (*S,)
        batch_var = samples.var(axis=0)  # (*S,)
        total = self.count + batch_count
        delta = batch_mean - self.mean  # (*S,)
        self.var = (
            self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total
        ) / total  # (*S,)
        self.mean += delta * batch_count / total  # (*S,)
        self.count = total
