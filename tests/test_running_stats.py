import chex
import jax
import jax.numpy as jnp
import pytest

from rl2.running_stats import RunningSum


def test_running_sum_initializes_and_updates_nested_scalar_tree() -> None:
    values = {"reward": jnp.float32(1.5), "episodes": (jnp.int32(2),)}  # scalar leaves: ()
    initial = jax.jit(RunningSum.init)(values)  # scalar leaves: ()
    assert jax.tree.structure(initial) == jax.tree.structure(values)
    for total, value in zip(jax.tree.leaves(initial), jax.tree.leaves(values), strict=True):
        assert total.shape == ()
        assert total.dtype == value.dtype
        assert total == 0

    update = jax.jit(RunningSum.update)
    first = update(initial, values)  # scalar leaves: ()
    second = update(first, {"reward": jnp.float32(-2.0), "episodes": (jnp.int32(0),)})  # scalar leaves: ()
    assert first["reward"] == 1.5
    assert first["episodes"][0] == 2
    assert second["reward"] == -0.5
    assert second["episodes"][0] == 2
    for total in jax.tree.leaves(initial):  # ()
        assert total == 0
    for total, value in zip(jax.tree.leaves(second), jax.tree.leaves(values), strict=True):
        assert total.shape == ()
        assert total.dtype == value.dtype


def test_running_sum_scan_with_python_scalar_initialization() -> None:
    initial = RunningSum.init({"reward": 0.0, "episodes": 0})  # scalar leaves: ()
    values = {"reward": jnp.array([1.5, -2.0]), "episodes": jnp.array([2, 1])}  # leaves: (T,)

    def step(
        state: chex.ArrayTree,  # scalar leaves: ()
        value: chex.ArrayTree,  # scalar leaves: ()
    ) -> tuple[chex.ArrayTree, None]:  # state leaves: ()
        return RunningSum.update(state, value), None  # state leaves: ()

    final, _ = jax.lax.scan(step, initial, values)  # state leaves: ()
    assert final["reward"].shape == final["episodes"].shape == ()
    assert final["reward"] == -0.5
    assert final["episodes"] == 3


def test_running_sum_empty_tree() -> None:
    state = jax.jit(RunningSum.init)({})
    assert jax.jit(RunningSum.update)(state, {}) == {}


def test_running_sum_rejects_mismatched_tree() -> None:
    state = RunningSum.init({"reward": 0.0})  # scalar leaves: ()
    with pytest.raises(ValueError):
        RunningSum.update(state, {"episodes": 1})
