from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import chex
import jax
import jax.numpy as jnp
import optax
import pytest
from flax.training.train_state import TrainState

from rl2.alphazero import checkpoints as ckpt
from rl2.alphazero.config import Config


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    def apply(params: dict[str, jax.Array]) -> jax.Array:  # weight: (1,)
        chex.assert_shape(params["weight"], (1,))
        chex.assert_type(params["weight"], jnp.float32)
        return params["weight"]  # (1,)

    train_state = TrainState.create(apply_fn=apply, params={"weight": jnp.ones(1)}, tx=optax.adam(1e-3))
    grads = {"weight": jnp.full(1, 0.5)}  # weight: (1,)
    trained = train_state.apply_gradients(grads=grads)
    config = Config()
    checkpoint = ckpt.Checkpoint(
        train_state=trained, key=jax.random.PRNGKey(3), iteration=1, steps=5, completed_games=1
    )
    empty_checkpoint = ckpt.Checkpoint(
        train_state=train_state, key=jax.random.PRNGKey(7), iteration=0, steps=0, completed_games=0
    )
    with ckpt.checkpoint_manager(str(tmp_path)) as manager:
        assert ckpt.restore_checkpoint(manager, empty_checkpoint) is None
        ckpt.save_checkpoint(manager, checkpoint, config)
    with ckpt.checkpoint_manager(str(tmp_path)) as manager:
        saved_config = manager.restore(1, args=ckpt.ocp.args.Composite(config=ckpt.ocp.args.JsonRestore())).config
        assert saved_config == asdict(config)
        assert not (tmp_path / "checkpoints" / "1" / "metadata").exists()
        restored_checkpoint = ckpt.restore_checkpoint(manager, empty_checkpoint)
        assert restored_checkpoint is not None
        chex.assert_trees_all_equal(restored_checkpoint, checkpoint)
        chex.assert_trees_all_equal(
            restored_checkpoint.train_state.apply_gradients(grads=grads), trained.apply_gradients(grads=grads)
        )
        for iteration in (2, 3):
            ckpt.save_checkpoint(manager, checkpoint._replace(iteration=iteration), config)
        assert manager.all_steps() == [3]
    assert {path.name for path in (tmp_path / "checkpoints").iterdir() if path.is_dir()} == {"3"}


def test_checkpoint_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with patch.object(ckpt.ocp, "CheckpointManager") as manager:
        ckpt.checkpoint_manager("gs://bucket/run/")
        assert manager.call_args.args[0] == "gs://bucket/run/checkpoints"
        ckpt.checkpoint_manager("logs/run")
        assert manager.call_args.args[0] == str(tmp_path / "logs/run/checkpoints")
