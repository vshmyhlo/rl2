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
