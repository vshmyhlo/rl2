"""Conditional batch resets, local to each PPO environment shard."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, get_abstract_mesh


def reset_batch[T](
    reset: Callable[[jax.Array], T],  # () key -> (...) observation/state leaves
    keys: jax.Array,  # (B,) typed PRNG keys
    done: jax.Array,  # (B,) bool
    current: T,  # leaves: (B, ...)
) -> T:
    """Return (B, ...) leaves, resetting completed rows only when a shard needs it.

    Reset generation remains vectorized within a shard. A shard with no completed
    episodes skips it entirely, without synchronizing predicates across devices.
    """

    def reset_shard(
        keys: jax.Array,  # (b,) keys; b is the local environment count
        done: jax.Array,  # (b,) bool
        current: T,  # leaves: (b, ...)
    ) -> T:
        def reset_done() -> T:
            reset_values = jax.vmap(reset)(keys)  # leaves: (b, ...)

            def select(new: jax.Array, old: jax.Array) -> jax.Array:  # each (b, ...)
                mask = done.reshape((-1,) + (1,) * (old.ndim - 1))  # (b, 1, ...)
                return jnp.where(mask, new, old)  # (b, ...)

            return jax.tree.map(select, reset_values, current)  # leaves: (b, ...)

        return jax.lax.cond(done.any(), reset_done, lambda: current)  # leaves: (b, ...)

    mesh = get_abstract_mesh()
    if "batch" in mesh.axis_names and "batch" not in mesh.manual_axes:
        batched = PartitionSpec("batch")
        # Upstream reset branches lack manual-axis annotations; all outputs vary.
        return jax.shard_map(
            reset_shard,
            in_specs=(batched, batched, batched),
            out_specs=batched,
            axis_names={"batch"},
            check_vma=False,
        )(keys, done, current)  # leaves: (B, ...)
    return reset_shard(keys, done, current)  # leaves: (B, ...)
