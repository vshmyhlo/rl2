"""A fixed-size opponent pool; index zero preserves the initial policy."""

from functools import partial
from typing import Annotated, NamedTuple

import jax
import jax.numpy as jnp
import optax
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class SelfPlayConfig:
    pool_size: Annotated[int, Field(ge=2, strict=True)] = 8
    snapshot_interval: Annotated[int, Field(gt=0, strict=True)] = 100
    current_policy_fraction: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.5


class OpponentPool(NamedTuple):
    params: optax.Params  # leaves: (K, ...), K pool entries
    iteration: jax.Array  # (K,), -1 for unused entries


class MatchAssignment(NamedTuple):
    opponent: jax.Array  # (E,), pool index or -1 for current-policy self-play
    learner_team: jax.Array  # (E,), physical team 0 or 1


def init_pool(params: optax.Params, size: int, iteration: int = 0) -> OpponentPool:
    """Copy parameter leaves (...) into a pool with only index zero eligible."""
    return OpponentPool(
        params=jax.tree.map(lambda leaf: jnp.broadcast_to(leaf, (size, *leaf.shape)).copy(), params),
        iteration=jnp.full(size, -1, dtype=jnp.int32).at[0].set(iteration),  # (K,)
    )


def resize_pool(pool: OpponentPool, size: int) -> OpponentPool:
    """On restore, keep the initial snapshot and newest history using today's capacity."""
    old_size = pool.iteration.shape[0]
    if size == old_size:
        return pool
    count = min(size, old_size)
    recent = jnp.argsort(pool.iteration[1:], descending=True)[: count - 1] + 1  # (count-1,)
    indices = jnp.concatenate((jnp.zeros(1, jnp.int32), recent))  # (count,)
    resized = init_pool(jax.tree.map(lambda leaf: leaf[0], pool.params), size)
    return OpponentPool(
        params=jax.tree.map(lambda new, old: new.at[:count].set(old[indices]), resized.params, pool.params),
        iteration=resized.iteration.at[:count].set(pool.iteration[indices]),  # (K,)
    )


@jax.jit
def insert_snapshot(
    pool: OpponentPool,
    params: optax.Params,  # leaves: (...)
    iteration: int,
    active_opponent_indices: jax.Array,  # (E,), pool indices used by ongoing games; -1 for the current policy
) -> OpponentPool:
    """Replace the oldest unreferenced entry; skip if every replaceable entry is active."""
    indices = jnp.arange(pool.iteration.size)  # (K,)
    available = (indices != 0) & ~jnp.any(indices[:, None] == active_opponent_indices[None, :], axis=1)  # (K,)
    index = jnp.argmin(jnp.where(available, pool.iteration, jnp.iinfo(jnp.int32).max))  # ()

    def insert() -> OpponentPool:
        return OpponentPool(
            params=jax.tree.map(lambda history, current: history.at[index].set(current), pool.params, params),
            iteration=pool.iteration.at[index].set(iteration),  # (K,)
        )

    return jax.lax.cond(available.any(), insert, lambda: pool)


@partial(jax.jit, static_argnames=("config", "num_envs"))
def sample_opponents(
    key: jax.Array,  # () typed PRNG key
    pool: OpponentPool,
    config: SelfPlayConfig,
    num_envs: int,
) -> MatchAssignment:
    index_key, mode_key, team_key = jax.random.split(key, 3)  # each ()
    indices = jax.random.categorical(
        index_key, jnp.where(pool.iteration >= 0, 0.0, -jnp.inf), shape=(num_envs,)
    )  # (E,)
    current = jax.random.bernoulli(mode_key, config.current_policy_fraction, (num_envs,))  # (E,)
    return MatchAssignment(
        opponent=jnp.where(current, -1, indices).astype(jnp.int32),  # (E,)
        learner_team=jax.random.randint(team_key, (num_envs,), 0, 2),  # (E,)
    )
