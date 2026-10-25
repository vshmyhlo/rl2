"""Orbax restart state without model loading, Ray, or vLLM."""

import asyncio
import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import chex
import jax
import jax.numpy as jnp
import pytest

from rl2 import train_grpo_v2 as training


@pytest.fixture
def progress() -> training.TrainingProgress:
    state = training.create_state(
        Mock(),
        {"weight": jnp.array([1.0, -1.0], dtype=jnp.float32)},  # (D=2,)
        training.Config(),
    )
    state = state.apply_gradients(grads={"weight": jnp.array([0.5, -0.25])})  # (D=2,)
    actor_class = training.Actor.__ray_metadata__.modified_class
    actor = actor_class.__new__(actor_class)
    actor.rng = random.Random(3)
    actor.rng.choice(range(10))
    return training.TrainingProgress(
        train_state=state,
        shuffle_key=jax.random.key(3),  # ()
        iteration=2,
        total_rollouts=2**31 + 4,
        sampling_state=actor.get_sampling_state(),
        stop_conditions=(
            "agent_completed",
            "environment_done",
            "max_turns",
            "max_tokens",
            "context_length",
            "unspecified",
        ),
    )


def test_checkpoint_roundtrip_and_retention(tmp_path: Path, progress: training.TrainingProgress) -> None:
    config = training.Config(ckpt_dir=str(tmp_path), run_id="resume-test")
    state = progress.train_state
    fresh = training.create_state(
        Mock(), jax.tree.map(jnp.zeros_like, state.params), replace(config, learning_rate=1e-4)
    )
    with training.checkpoint_manager(config.ckpt_dir) as manager:
        assert training.restore_checkpoint(manager, fresh, config) is None
        training.save_checkpoint(manager, replace(progress, iteration=1), config)
        training.save_checkpoint(manager, progress, config)
    with training.checkpoint_manager(config.ckpt_dir) as manager:
        metadata = manager.restore(
            2, args=training.ocp.args.Composite(metadata=training.ocp.args.JsonRestore())
        ).metadata
        assert metadata["config"] == json.loads(json.dumps(asdict(config)))
        restored = training.restore_checkpoint(
            manager, fresh, replace(config, seed=99, total_iterations=5, checkpoint_interval=2)
        )
        assert restored is not None
        assert restored.iteration == 2
        assert restored.total_rollouts == 2**31 + 4
        assert restored.stop_conditions == progress.stop_conditions
        assert restored.sampling_state == progress.sampling_state
        assert restored.shuffle_key.shape == ()
        assert restored.train_state.params["weight"].shape == (2,)
        chex.assert_trees_all_equal(
            (restored.train_state.params, restored.train_state.opt_state, restored.train_state.step),
            (state.params, state.opt_state, state.step),
        )
        assert restored.train_state.apply_fn == fresh.apply_fn
        assert restored.train_state.tx is fresh.tx
        chex.assert_trees_all_equal(
            jax.random.permutation(jax.random.fold_in(restored.shuffle_key, restored.train_state.step), 8),
            jax.random.permutation(jax.random.fold_in(progress.shuffle_key, state.step), 8),
        )
        gradients = {"weight": jnp.array([-0.1, 0.2])}  # (D=2,)
        expected = state.replace(tx=fresh.tx).apply_gradients(grads=gradients)
        actual = restored.train_state.apply_gradients(grads=gradients)
        chex.assert_trees_all_equal((actual.params, actual.opt_state), (expected.params, expected.opt_state))
        training.save_checkpoint(manager, replace(restored, iteration=3), config)
        assert manager.all_steps() == [2, 3]
    with training.checkpoint_manager(str(tmp_path / "new-run")) as manager:
        assert training.restore_checkpoint(manager, fresh, config) is None


def test_actor_sampling_state_resumes_task_sequence(progress: training.TrainingProgress) -> None:
    actor_class = training.Actor.__ray_metadata__.modified_class
    actor = actor_class.__new__(actor_class)
    actor.rng = random.Random(99)
    actor.tasks = [training.ALFWorldTask(id=str(i), config_file="config.yaml") for i in range(5)]
    actor.collect_group = AsyncMock(return_value=Mock(spec=training.RolloutGroup))
    actor.set_sampling_state(json.loads(json.dumps(progress.sampling_state)))
    for _ in range(8):
        asyncio.run(actor.fetch_next_group(group_size=2))
    expected = actor.collect_group.call_args_list
    actor.set_sampling_state(progress.sampling_state)
    actor.collect_group.reset_mock()
    for _ in range(8):
        asyncio.run(actor.fetch_next_group(group_size=2))
    assert actor.collect_group.call_args_list == expected


@pytest.mark.parametrize("condition", ["alfworld_done", "unknown", None], ids=["legacy", "unknown", "null"])
def test_checkpoint_stop_conditions(progress: training.TrainingProgress, condition: str | None) -> None:
    manager = Mock()
    config = training.Config()
    training.save_checkpoint(manager, progress, config)
    metadata = manager.save.call_args.kwargs["args"].metadata.item | {
        "stop_conditions": ["max_tokens", condition, "unspecified"]
    }
    manager.latest_step.return_value = progress.iteration
    manager.restore.side_effect = [
        SimpleNamespace(metadata=metadata),
        SimpleNamespace(
            state={
                "train_state": progress.train_state,
                "shuffle_key": jax.random.key_data(progress.shuffle_key),  # (K,)
            }
        ),
    ]
    if condition == "alfworld_done":
        restored = training.restore_checkpoint(manager, progress.train_state, config)
        assert restored is not None
        assert restored.stop_conditions == ("max_tokens", "environment_done", "unspecified")
    else:
        with pytest.raises(ValueError):
            training.restore_checkpoint(manager, progress.train_state, config)
        manager.restore.assert_called_once()


@pytest.mark.parametrize(
    "damage",
    [
        {"version": 2},
        {"algorithm": "ppo"},
        {"iteration": 3},
        {"total_rollouts": -1},
    ],
)
def test_invalid_checkpoint_metadata(progress: training.TrainingProgress, damage: dict[str, Any]) -> None:
    manager = Mock()
    config = training.Config()
    training.save_checkpoint(manager, progress, config)
    metadata = manager.save.call_args.kwargs["args"].metadata.item | damage
    manager.latest_step.return_value = progress.iteration
    manager.restore.return_value = SimpleNamespace(metadata=metadata)
    with pytest.raises(ValueError, match="checkpoint"):
        training.restore_checkpoint(manager, progress.train_state, config)
    manager.restore.assert_called_once()


def test_checkpoint_waits_for_writes_and_propagates_failure(
    progress: training.TrainingProgress, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Mock(monotonic=Mock(side_effect=[1.0, 6.0, 7.0]))
    monkeypatch.setattr(training, "time", clock)
    manager = Mock()

    def wait() -> None:
        assert clock.monotonic.call_count == 1

    manager.wait_until_finished.side_effect = wait
    assert training.save_checkpoint(manager, progress, training.Config()) == 5.0
    assert manager.save.call_args.kwargs["force"]
    manager.wait_until_finished.side_effect = OSError("write failed")
    with pytest.raises(OSError, match="write failed"):
        training.save_checkpoint(manager, progress, training.Config())


def test_checkpoint_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = Mock()
    monkeypatch.setattr(training.ocp, "CheckpointManager", manager)
    monkeypatch.chdir(tmp_path)
    training.checkpoint_manager("gs://bucket/run/")
    assert manager.call_args.args[0] == "gs://bucket/run/checkpoints"
    training.checkpoint_manager("runs/run/")
    assert manager.call_args.args[0] == str(tmp_path / "runs/run/checkpoints")
