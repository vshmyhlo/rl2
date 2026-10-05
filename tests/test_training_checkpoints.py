"""Checkpoint storage and resume state without running a training loop."""

from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import serialization
from flax.training.train_state import TrainState

from rl2 import train_karel_ast_grpo, train_karel_grpo, train_wm
from rl2.utils import read_bytes


@pytest.fixture(params=[train_karel_ast_grpo, train_karel_grpo], ids=["ast", "tokens"])
def grpo(request: pytest.FixtureRequest) -> ModuleType:
    return request.param


@pytest.fixture
def state() -> TrainState:
    # Nonzero Adam moments make missing optimizer state observable on restore.
    def apply(params: dict[str, jax.Array]) -> jax.Array:
        chex.assert_shape(params["weight"], (2,))
        chex.assert_type(params["weight"], jnp.float32)
        return params["weight"]

    initial = TrainState.create(apply_fn=apply, params={"weight": jnp.array([1.0, -1.0])}, tx=optax.adam(1e-3))
    return initial.apply_gradients(grads={"weight": jnp.array([0.5, -0.25])})


@pytest.mark.parametrize("remote", [False, True], ids=["local", "gcs"])
def test_grpo_checkpoint_restores_optimizer_rng_and_progress(
    grpo: ModuleType, state: TrainState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote: bool
) -> None:
    files: dict[str, bytes] = {}

    class FakeGCS:
        def cat_file(self, path: str) -> bytes:
            return files[path]

        def pipe_file(self, path: str, data: bytes) -> None:
            files[path] = data

    monkeypatch.setattr("rl2.utils.gcsfs.GCSFileSystem", FakeGCS)
    directory = "gs://test-bucket/checkpoints" if remote else str(tmp_path)
    rng = np.random.default_rng(12)
    rng.random(3)
    progress = grpo.TrainingProgress(state, jax.random.key(3), 2, 8, 1, 8)
    grpo._save_checkpoint(directory, progress, rng)
    data = read_bytes(f"{directory}/checkpoint.msgpack")
    fresh = state.replace(
        step=0, params=jax.tree.map(jnp.zeros_like, state.params), opt_state=state.tx.init(state.params)
    )
    restored_rng = np.random.default_rng(99)
    restored = grpo._restore_checkpoint(data, fresh, restored_rng)
    chex.assert_trees_all_equal(restored, progress)
    np.testing.assert_array_equal(restored_rng.integers(100, size=4), rng.integers(100, size=4))
    gradients = {"weight": jnp.array([-0.1, 0.2])}
    chex.assert_trees_all_equal(restored.state.apply_gradients(grads=gradients), state.apply_gradients(grads=gradients))
    payload = serialization.msgpack_restore(data)
    assert "config" not in payload and "reference_params" not in payload
    # Legacy decision counts must migrate to the recorded episode count.
    payload["steps"] = 100
    migrated = grpo._restore_checkpoint(serialization.msgpack_serialize(payload), fresh, restored_rng)
    assert migrated.steps == migrated.episodes == 8
    if remote:
        assert set(files) == {f"{directory}/checkpoint.msgpack"}


@pytest.mark.parametrize("damage", ["version", "shape", "counter"])
def test_grpo_rejects_invalid_checkpoint(grpo: ModuleType, state: TrainState, tmp_path: Path, damage: str) -> None:
    rng = np.random.default_rng(0)
    progress = grpo.TrainingProgress(state, jax.random.key(0), 1, 4, 0, 4)
    grpo._save_checkpoint(str(tmp_path), progress, rng)
    checkpoint = tmp_path / "checkpoint.msgpack"
    payload = serialization.msgpack_restore(checkpoint.read_bytes())
    if damage == "version":
        payload["version"] = 999
        message = "version"
    elif damage == "shape":
        payload["state"]["params"]["weight"] = np.zeros(3, np.float32)
        message = "incompatible"
    else:
        payload["iteration"] = -1
        message = "counters"
    data = serialization.msgpack_serialize(payload)
    checkpoint.write_bytes(data)
    with pytest.raises(ValueError, match=message):
        grpo._restore_checkpoint(checkpoint.read_bytes(), state, rng)
    assert checkpoint.read_bytes() == data


@pytest.mark.parametrize("remote", [False, True], ids=["local", "gcs"])
def test_world_model_checkpoint_roundtrip(
    state: TrainState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote: bool
) -> None:
    uploads: dict[str, bytes] = {}

    class FakeBlob:
        def upload_from_string(self, data: bytes) -> None:
            uploads["checkpoint"] = data

    blob = Mock(return_value=FakeBlob())
    monkeypatch.setattr(train_wm.storage, "Client", Mock())
    monkeypatch.setattr(train_wm.storage.Blob, "from_uri", blob)
    directory = "gs://test-bucket/wm" if remote else str(tmp_path / "checkpoints")
    train_wm.save_checkpoint(state, directory + "/")
    if remote:
        assert blob.call_args.args == (f"{directory}/checkpoint.msgpack",)
        data = uploads["checkpoint"]
    else:
        data = (Path(directory) / "checkpoint.msgpack").read_bytes()
        assert not list(Path(directory).glob("*.tmp"))
    restored = serialization.from_bytes(state, data)
    chex.assert_trees_all_equal(restored, state)
