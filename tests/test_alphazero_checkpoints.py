from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import chex
import jax
import jax.numpy as jnp
import optax
import pytest
from flax.training.train_state import TrainState
from pydantic import ValidationError

from rl2.alphazero import checkpoints as ckpt
from rl2.alphazero.config import Config


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    def apply(params: dict[str, jax.Array]) -> jax.Array:  # weight: (1,)
        chex.assert_shape(params["weight"], (1,))
        chex.assert_type(params["weight"], jnp.float32)
        return params["weight"]  # (1,)

    state = TrainState.create(apply_fn=apply, params={"weight": jnp.ones(1)}, tx=optax.adam(1e-3))
    grads = {"weight": jnp.full(1, 0.5)}  # weight: (1,)
    trained = state.apply_gradients(grads=grads)
    config = Config()
    progress = ckpt.TrainingProgress(state=trained, key=jax.random.PRNGKey(3), iteration=1, steps=5, completed_games=1)
    with ckpt.checkpoint_manager(str(tmp_path)) as manager:
        assert ckpt.restore_checkpoint(manager, state, config) is None
        ckpt.save_checkpoint(manager, progress, config)
    with ckpt.checkpoint_manager(str(tmp_path)) as manager:
        metadata = manager.restore(1, args=ckpt.ocp.args.Composite(metadata=ckpt.ocp.args.JsonRestore())).metadata
        assert metadata["config"] == asdict(config)
        restored = ckpt.restore_checkpoint(manager, state, replace(config, iterations=200, learning_rate=0.01, seed=7))
        assert restored is not None
        chex.assert_trees_all_equal(restored, progress)
        chex.assert_trees_all_equal(restored.state.apply_gradients(grads=grads), trained.apply_gradients(grads=grads))
        for iteration in (2, 3):
            ckpt.save_checkpoint(manager, progress._replace(iteration=iteration), config)
        assert manager.all_steps() == [3]
    assert {path.name for path in (tmp_path / "checkpoints").iterdir() if path.is_dir()} == {"3"}


def test_checkpoint_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with patch.object(ckpt.ocp, "CheckpointManager") as manager:
        ckpt.checkpoint_manager("gs://bucket/run/")
        assert manager.call_args.args[0] == "gs://bucket/run/checkpoints"
        ckpt.checkpoint_manager("logs/run")
        assert manager.call_args.args[0] == str(tmp_path / "logs/run/checkpoints")


@pytest.mark.parametrize("metadata", [{"version": 2}, {"steps": -1}], ids=["version", "negative-progress"])
def test_checkpoint_metadata_validation(metadata: dict[str, int]) -> None:
    values = {"config": {}, "iteration": 1, "steps": 1, "completed_games": 0} | metadata
    with pytest.raises(ValidationError):
        ckpt.CheckpointMetadata.model_validate(values)


def test_restore_ignores_legacy_config() -> None:
    state = TrainState.create(apply_fn=jnp.add, params={"weight": jnp.ones(1)}, tx=optax.adam(1e-3))
    config = Config(ckpt_dir="new-location")
    metadata = ckpt.CheckpointMetadata(
        config={"ckpt_dir": "old-location", "log_dir": "legacy-logs", "learning_rate": 0.02},
        iteration=1,
        steps=2,
        completed_games=1,
    )
    manager = Mock()
    manager.latest_step.return_value = 1
    manager.restore.side_effect = [
        SimpleNamespace(metadata=metadata.model_dump()),
        SimpleNamespace(state={"train_state": state, "key": jax.random.PRNGKey(0)}),
    ]
    restored = ckpt.restore_checkpoint(manager, state, config)
    assert restored is not None
    assert (restored.iteration, restored.steps, restored.completed_games) == (1, 2, 1)
