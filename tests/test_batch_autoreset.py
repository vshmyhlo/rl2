from functools import partial
from operator import itemgetter

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, PartitionSpec

from rl2.envs.autoreset import reset_batch

type ResetState = dict[str, jax.Array]


def reset(key: jax.Array, scale: jax.Array) -> ResetState:  # key/scale: ()
    return {
        "vector": jax.random.uniform(key, (2,)) * scale,  # (O=2,)
        "count": jnp.asarray(0, jnp.int32),  # ()
        "key": key,  # () typed key
    }


@pytest.mark.parametrize("done", [[False], [True], [False, True, False, True]], ids=["continue", "reset", "mixed"])
def test_reset_selection_and_shapes(done: list[bool]) -> None:
    batch_size = len(done)
    keys = jax.random.split(jax.random.key(0), batch_size)  # (B,) keys
    current = {
        "vector": jnp.full((batch_size, 2), -1.0),  # (B, O)
        "count": jnp.ones(batch_size, jnp.int32),  # (B,)
        "key": jax.random.split(jax.random.key(1), batch_size),  # (B,) keys
    }
    result = jax.jit(partial(reset_batch, partial(reset, scale=jnp.asarray(2.0))))(
        keys, jnp.asarray(done), current
    )  # leaves: (B, ...)
    expected = jax.vmap(partial(reset, scale=jnp.asarray(2.0)))(keys)  # leaves: (B, ...)
    chex.assert_shape(result["vector"], (batch_size, 2))
    chex.assert_shape((result["count"], result["key"]), (batch_size,))
    for index, completed in enumerate(done):
        actual_row, expected_row = jax.tree.map(
            itemgetter(index), (result, expected if completed else current)
        )  # leaves: (...)
        chex.assert_trees_all_equal(actual_row, expected_row)


def test_sharded_reset_skips_idle_shards_without_collectives() -> None:
    if jax.device_count() < 2:
        pytest.skip("Requires two devices; use JAX_NUM_CPU_DEVICES=2")
    mesh = Mesh(devices=jax.devices()[:2], axis_names=("batch",), axis_types=(AxisType.Auto,))
    batched = PartitionSpec("batch")
    calls: list[None] = []

    def record_reset() -> None:
        calls.append(None)

    def step(
        keys: jax.Array,  # (B,) keys
        done: jax.Array,  # (B,)
        current: ResetState,  # leaves: (B, ...)
        scale: jax.Array,  # () replicated, captured by reset_one
    ) -> ResetState:
        def reset_one(key: jax.Array) -> ResetState:  # key: ()
            jax.debug.callback(record_reset)
            return reset(key, scale)  # leaves: (...)

        return reset_batch(reset_one, keys, done, current)  # leaves: (B, ...)

    keys = jax.random.split(jax.random.key(0), 4)  # (B=4,) keys
    expected_reset = np.asarray(reset(keys[2], jnp.asarray(2.0))["vector"])  # (O=2,)
    current = {
        "vector": jnp.full((4, 2), -1.0),  # (B, O=2)
        "count": jnp.ones(4, jnp.int32),  # (B,)
        "key": keys,  # (B,)
    }
    with jax.set_mesh(mesh):
        keys, done, current = jax.device_put((keys, jnp.zeros(4, bool), current), batched)  # leaves: (B, ...)
        scale = jax.device_put(jnp.asarray(2.0), PartitionSpec())  # ()
        compiled = (
            jax.jit(step, in_shardings=(batched, batched, batched, PartitionSpec()), out_shardings=batched)
            .lower(keys, done, current, scale)
            .compile()
        )
        first = compiled(keys, done, current, scale)  # leaves: (B, ...)
        chex.assert_trees_all_equal(first, current)
        jax.effects_barrier()
        assert not calls
        done = jax.device_put(jnp.asarray([False, False, True, False]), batched)  # (B,)
        result = compiled(keys, done, current, scale)
        # Leaves: (B, ...); only GPU/CPU shard 1 needs reset generation.
        np.testing.assert_array_equal(result["count"], [1, 1, 0, 1])
        np.testing.assert_array_equal(np.asarray(result["vector"])[:2], np.asarray(current["vector"])[:2])
        np.testing.assert_array_equal(np.asarray(result["vector"])[2], expected_reset)
        jax.effects_barrier()
        assert len(calls) == 1
    hlo = compiled.as_text()
    assert "all-reduce(" not in hlo and "all-gather(" not in hlo
