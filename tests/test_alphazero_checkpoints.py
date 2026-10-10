from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import chex
import jax
import jax.numpy as jnp
import optax
import pytest
from flax.training.train_state import TrainState
from pydantic import ValidationError

from rl2.alphazero import checkpoints as ckpt
from rl2.alphazero.config import Config
from rl2.shape_checker import ShapeChecker


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    def apply(params: dict[str, jax.Array]) -> jax.Array:
        sc = ShapeChecker(W=1)
        sc.check(params["weight"], "W", jnp.float32)
        return params["weight"]

    state = TrainState.create(apply_fn=apply, params={"weight": jnp.ones(1)}, tx=optax.adam(1e-3))
    grads = {"weight": jnp.full(1, 0.5)}
    trained = state.apply_gradients(grads=grads)
    config = Config()
    progress = ckpt.TrainingProgress(trained, jax.random.PRNGKey(3), 1, 5, 1)
    with ckpt.checkpoint_manager(str(tmp_path)) as manager:
        assert ckpt.restore_checkpoint(manager, state, config) is None
        ckpt.save_checkpoint(manager, progress, config)
    with ckpt.checkpoint_manager(str(tmp_path)) as manager:
        restored = ckpt.restore_checkpoint(manager, state, replace(config, iterations=200))
        assert restored is not None
        chex.assert_trees_all_equal(restored, progress)
        chex.assert_trees_all_equal(restored.state.apply_gradients(grads=grads), trained.apply_gradients(grads=grads))
        with pytest.raises(ValueError, match="incompatible.*new run_id"):
            ckpt.restore_checkpoint(manager, state, replace(config, learning_rate=0.01))
        for iteration in (2, 3):
            ckpt.save_checkpoint(manager, progress._replace(iteration=iteration), config)
        assert manager.all_steps() == [2, 3]


def test_checkpoint_settings_allow_log_interval_changes() -> None:
    config = Config()
    settings = ckpt.checkpoint_settings(config)
    assert "log_interval_seconds" not in settings  # Also compatible with checkpoints predating this option.
    assert ckpt.checkpoint_settings(replace(config, log_interval_seconds=15.0)) == settings


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
