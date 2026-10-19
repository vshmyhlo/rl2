from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2 import utils


def test_gae_single_step_keeps_environments_independent() -> None:
    advantages, returns = utils.gae(
        jnp.array([[1.0, 2.0]]),  # (T, B)
        jnp.array([[True, False]]),  # (T, B)
        jnp.array([[0.5, 1.0]]),  # (T, B)
        jnp.array([100.0, 4.0]),  # (B,)
        0.5,
        0.8,
    )  # both (T, B)
    assert advantages.shape == returns.shape == (1, 2)
    np.testing.assert_allclose(advantages, [[0.5, 3.0]])
    np.testing.assert_allclose(returns, [[1.0, 4.0]])


@pytest.mark.parametrize("shape", [(3,), (1, 3), (1,)], ids=["steps", "rollout", "single-return"])
def test_explained_variance(shape: tuple[int, ...]) -> None:
    returns = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)  # (*S,)
    result = jax.jit(utils.explained_variance)(returns, returns)  # ()
    assert result.shape == ()
    assert result.dtype == jnp.float32
    if returns.size == 1:
        assert np.isnan(result)
        return
    assert result == 1.0
    assert utils.explained_variance(jnp.zeros_like(returns), returns) == 0.0
    assert utils.explained_variance(-returns, returns) == -3.0
    assert np.isnan(utils.explained_variance(returns, jnp.ones_like(returns)))


@pytest.mark.parametrize(
    ("rewards", "expected"),
    [
        ([0.0, 1.0], [-1.0, 1.0]),
        ([1.0, 1.0], [0.0, 0.0]),
        ([0.0, 0.0, 0.0, 1.0], [-1 / 3**0.5] * 3 + [3**0.5]),
    ],
    ids=["mixed", "equal", "imbalanced"],
)
def test_group_advantages(rewards: list[float], expected: list[float]) -> None:
    values = jnp.asarray(rewards, dtype=jnp.float32)  # (G,)
    advantages = jax.jit(utils.group_advantages)(values)  # (G,)
    assert advantages.shape == values.shape
    assert advantages.dtype == jnp.float32
    assert advantages.tolist() == pytest.approx(expected, abs=5e-4)
    assert float(advantages.mean()) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("batched", [False, True], ids=["single-group", "mixed-batch"])
def test_group_advantages_fractional_ties_are_exactly_zero(batched: bool) -> None:
    rewards = jnp.array([[0.7] * 8, [0.0, 1.0] * 4], dtype=jnp.float32)  # (B=2, G=8)
    if not batched:
        rewards = rewards[0]  # (G,)
    for normalize in (utils.group_advantages, jax.jit(utils.group_advantages)):
        advantages = normalize(rewards)  # (B, G) or (G,)
        assert advantages.shape == rewards.shape
        assert advantages.dtype == jnp.float32
        np.testing.assert_array_equal(advantages[0] if batched else advantages, 0.0)
        if batched:
            np.testing.assert_allclose(advantages[1], [-1.0, 1.0] * 4, atol=3e-4)


@pytest.mark.parametrize("shape", [(0,), (2, 1)], ids=["empty", "singleton-groups"])
def test_group_advantages_requires_multiple_rewards(shape: tuple[int, ...]) -> None:
    rewards = jnp.zeros(shape, dtype=jnp.float32)  # (*B, G)
    with pytest.raises(ValueError, match="two rewards"):
        utils.group_advantages(rewards)


def test_group_advantages_keeps_batch_rows_independent() -> None:
    rewards = jnp.array([[0, 1], [10, 10], [4, 2]], dtype=jnp.float32)  # (B=3, G=2)
    advantages = jax.jit(utils.group_advantages)(rewards)  # (B=3, G=2)
    assert advantages.shape == rewards.shape
    np.testing.assert_allclose(advantages, [[-1, 1], [0, 0], [1, -1]], atol=3e-4)
    single_group = jax.jit(utils.group_advantages)(rewards[:1])  # (B=1, G=2)
    assert single_group.shape == (1, 2)
    np.testing.assert_allclose(single_group, advantages[:1])


def test_checkpoint_io_preserves_previous_file_and_propagates_storage_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "checkpoint.msgpack"
    destination.write_bytes(b"old")

    def failed_replace(source: Path, target: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(utils.os, "replace", failed_replace)
    with pytest.raises(OSError, match="replace failure"):
        utils.write_bytes(str(destination), b"new")
    assert destination.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [destination]
    assert utils.read_optional(str(tmp_path / "missing")) is None

    def denied(path: str) -> bytes:
        raise PermissionError("storage access denied")

    monkeypatch.setattr(utils, "read_bytes", denied)
    with pytest.raises(PermissionError, match="denied"):
        utils.read_optional("gs://bucket/run/checkpoint.msgpack")


def test_local_storage_creates_parents_and_replaces_file(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "checkpoint.msgpack"
    utils.write_bytes(str(destination), b"old checkpoint")
    assert utils.read_bytes(str(destination)) == b"old checkpoint"
    utils.write_bytes(str(destination), b"new")
    assert utils.read_optional(str(destination)) == b"new"
    assert list(destination.parent.iterdir()) == [destination]


@pytest.mark.parametrize("shape", [(), (2, 4)], ids=["scalar", "tensor"])
def test_running_moments_shapes_and_empty_batches(shape: tuple[int, ...]) -> None:
    moments = utils.RunningMeanStd(shape)
    assert moments.mean.shape == moments.var.shape == shape
    np.testing.assert_array_equal(moments.mean, np.zeros(shape))
    np.testing.assert_array_equal(moments.var, np.ones(shape))
    prior_count = moments.count

    samples = np.stack([np.full(shape, 2), np.full(shape, 4)])  # (N, *S)
    moments.update(samples)
    total = prior_count + len(samples)
    assert moments.count == total
    assert moments.mean.shape == moments.var.shape == shape
    assert moments.mean.dtype == moments.var.dtype == np.float64
    np.testing.assert_allclose(moments.mean, 6 / total)
    np.testing.assert_allclose(moments.var, (prior_count + 20) / total - (6 / total) ** 2)

    mean, variance = jax.tree.map(np.copy, (moments.mean, moments.var))  # both (*S,)
    moments.update(np.empty((0, *shape)))
    np.testing.assert_array_equal(moments.mean, mean)
    np.testing.assert_array_equal(moments.var, variance)
    assert moments.count == total


def test_gae_stops_at_game_over() -> None:
    advantages, returns = utils.gae(
        np.array([[1.0], [2.0]], dtype=np.float32),  # (T, B)
        np.array([[True], [False]]),  # (T, B)
        np.array([[0.5], [1.0]], dtype=np.float32),  # (T, B)
        np.array([3.0], dtype=np.float32),  # (B,)
        0.9,
        0.8,
    )  # both (T, B)
    assert advantages.shape == returns.shape == (2, 1)
    np.testing.assert_allclose(advantages, [[0.5], [3.7]], rtol=1e-6)
    np.testing.assert_allclose(returns, [[1.0], [4.7]], rtol=1e-6)


def test_gae_timeout_bootstrap_and_trace() -> None:
    # The final reward contains gamma * V(final observation).
    advantages, returns = utils.gae(
        np.array([[1.0], [2.0 + 0.9 * 3.0]], dtype=np.float32),  # (T, B)
        np.array([[False], [True]]),  # (T, B)
        np.array([[0.5], [1.0]], dtype=np.float32),  # (T, B)
        np.array([100.0], dtype=np.float32),  # (B,)
        0.9,
        0.8,
    )  # both (T, B)
    assert advantages.shape == returns.shape == (2, 1)
    np.testing.assert_allclose(advantages, [[4.064], [3.7]], rtol=1e-6)
    np.testing.assert_allclose(returns, [[4.564], [4.7]], rtol=1e-6)
