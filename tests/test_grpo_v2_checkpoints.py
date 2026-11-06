"""Orbax restart state without model loading, Ray, or vLLM."""

import asyncio
import json
import random
from contextlib import ExitStack
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import chex
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from rl2 import train_grpo_v2 as training


@pytest.fixture
def progress() -> training.TrainingProgress:
    state = training.create_state(
        Mock(),
        {"model.layers.0.self_attn.q_proj.weight": jnp.array([[1.0, -1.0]], dtype=jnp.bfloat16)},  # (O=1, I=2)
        training.Config(),
    )
    state = state.apply_gradients(grads=jax.tree.map(jnp.ones_like, state.params))
    actor_class = training.Actor.__ray_metadata__.modified_class
    actor = actor_class.__new__(actor_class)
    actor.rng = random.Random(3)
    actor.taskset_scores = {"legs": -0.25, "arithmetic": 0.25}
    actor.rng.choice(range(10))
    return training.TrainingProgress(
        train_state=state,
        shuffle_key=jax.random.key(3),  # ()
        iteration=2,
        total_rollouts=2**31 + 4,
        sampling_state=actor.get_sampling_state(),
        rollout_pool=[{"task_id": "pending", "rollouts": [{"advantage": 0.25}]}],
        weight_version=7,
    )


def test_checkpoint_roundtrip_and_retention(tmp_path: Path, progress: training.TrainingProgress) -> None:
    config = training.Config(ckpt_dir=str(tmp_path), run_id="resume-test")
    state = progress.train_state
    fresh = training.create_state(
        Mock(), jax.tree.map(jnp.zeros_like, state.base_params), replace(config, learning_rate=1e-4)
    )
    mesh = Mesh(devices=jax.devices()[:2], axis_names=("batch",))
    fresh = jax.device_put(fresh, NamedSharding(mesh, PartitionSpec()))
    with training.checkpoint_manager(config.ckpt_dir) as manager:
        assert training.restore_checkpoint(manager, fresh, config) is None
        training.save_checkpoint(manager, replace(progress, iteration=1), config)
        training.save_checkpoint(manager, progress, config)
    with training.checkpoint_manager(config.ckpt_dir) as manager:
        assert manager.all_steps() == [2]
        assert not (tmp_path / "checkpoints" / "1").exists()
        metadata = manager.restore(
            2, args=training.ocp.args.Composite(metadata=training.ocp.args.JsonRestore())
        ).metadata
        assert metadata["config"] == json.loads(json.dumps(asdict(config)))
        assert "stop_conditions" not in metadata
        restored = training.restore_checkpoint(
            manager,
            fresh,
            replace(
                config, seed=99, total_iterations=5, checkpoint_interval=2, grad_accum_steps=4, parameter_sync_steps=3
            ),
        )
        assert restored is not None
        assert restored.iteration == 2
        assert restored.total_rollouts == 2**31 + 4
        assert restored.sampling_state == progress.sampling_state
        assert restored.rollout_pool == progress.rollout_pool
        assert restored.weight_version == 7
        assert restored.shuffle_key.shape == ()
        assert restored.train_state.params["model.layers.0.self_attn.q_proj.weight"].a.shape == (16, 2)
        chex.assert_trees_all_equal(restored.train_state.base_params, state.base_params)
        assert restored.train_state.lora_scale == state.lora_scale
        chex.assert_trees_all_equal(
            (restored.train_state.params, restored.train_state.opt_state, restored.train_state.step),
            (state.params, state.opt_state, state.step),
        )
        assert restored.train_state.apply_fn == fresh.apply_fn
        assert restored.train_state.tx is fresh.tx
        for value in jax.tree.leaves(restored.train_state):  # HF/LoRA parameter shapes and scalar optimizer counters.
            assert value.sharding.is_fully_replicated
            assert value.devices() == set(mesh.devices.flat)
        chex.assert_trees_all_equal(
            jax.random.permutation(jax.random.fold_in(restored.shuffle_key, restored.train_state.step), 8),
            jax.random.permutation(jax.random.fold_in(progress.shuffle_key, state.step), 8),
        )
        gradients = jax.tree.map(lambda value: jnp.full_like(value, -0.1), state.params)  # LoRA factor shapes.
        expected = state.replace(tx=fresh.tx).apply_gradients(grads=gradients)
        actual = restored.train_state.apply_gradients(grads=gradients)
        chex.assert_trees_all_equal((actual.params, actual.opt_state), (expected.params, expected.opt_state))
        training.save_checkpoint(manager, replace(restored, iteration=3), config)
        manager.wait_until_finished()
        assert manager.all_steps() == [3]
        assert not (tmp_path / "checkpoints" / "2").exists()
    with training.checkpoint_manager(str(tmp_path / "new-run")) as manager:
        assert training.restore_checkpoint(manager, fresh, config) is None


def test_actor_sampling_state_resumes_task_sequence() -> None:
    actor_class = training.Actor.__ray_metadata__.modified_class
    actor = actor_class.__new__(actor_class)
    actor.rng = random.Random(99)
    actor.taskset_scores = {}
    tasksets = {
        "legs": training.WeightedTasksetConfig(taskset=training.ReasoningGymConfig(), weight=3),
        "arithmetic": training.WeightedTasksetConfig(
            taskset=training.ReasoningGymConfig(task="basic_arithmetic"), weight=1
        ),
    }
    actor.tasks = training.stream_tasks(tasksets, actor.rng, timeout=120, taskset_scores=actor.taskset_scores)
    actor.collect_group = AsyncMock(return_value=Mock(spec=training.RolloutGroup))
    asyncio.run(actor.fetch_next_group(group_size=2))
    saved = json.loads(json.dumps(actor.get_sampling_state()))
    snapshot = actor.get_sampling_state()
    actor.collect_group.reset_mock()
    for _ in range(7):
        asyncio.run(actor.fetch_next_group(group_size=2))
    expected = actor.collect_group.call_args_list
    assert snapshot == saved
    actor.rng.seed(123)
    actor.tasks = training.stream_tasks(tasksets, actor.rng, timeout=120, taskset_scores=actor.taskset_scores)
    actor.set_sampling_state(saved)
    actor.collect_group.reset_mock()
    for _ in range(7):
        asyncio.run(actor.fetch_next_group(group_size=2))
    assert actor.collect_group.call_args_list == expected

    del saved["taskset_scores"]  # Legacy checkpoints restart the deterministic schedule.
    actor.set_sampling_state(saved)
    assert next(actor.tasks).taskset_id == "legs"


@pytest.mark.parametrize("mode", [None, "lora", "full"])
def test_checkpoint_ignores_legacy_stop_conditions(progress: training.TrainingProgress, mode: str | None) -> None:
    manager = Mock()
    config = training.Config(finetuning=training.FullFinetuningConfig() if mode == "full" else training.LoRAConfig())
    training.save_checkpoint(manager, progress, config)
    metadata = manager.save.call_args.kwargs["args"].metadata.item | {
        "stop_conditions": ["max_tokens", "alfworld_done", "unspecified"]
    }
    del metadata["rollout_pool"]
    del metadata["weight_version"]
    del metadata["config"]["finetuning"]
    metadata["config"] |= {"lora_rank": 16, "lora_alpha": 32.0}
    if mode is not None:
        metadata["config"]["finetuning_mode"] = mode
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
    restored = training.restore_checkpoint(manager, progress.train_state, config)
    assert restored is not None
    assert restored.iteration == progress.iteration
    assert restored.train_state is progress.train_state
    assert restored.rollout_pool == []
    assert restored.weight_version == -1


@pytest.mark.parametrize(
    "damage",
    [
        {"version": 1},
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


@pytest.mark.parametrize(
    "change",
    [
        {"model": "other"},
        {"finetuning": training.LoRAConfig(rank=8)},
        {"finetuning": training.LoRAConfig(alpha=16)},
        {"finetuning": training.FullFinetuningConfig()},
    ],
)
def test_checkpoint_rejects_incompatible_lora_config(
    progress: training.TrainingProgress, change: dict[str, Any]
) -> None:
    manager = Mock()
    config = training.Config()
    training.save_checkpoint(manager, progress, config)
    metadata = manager.save.call_args.kwargs["args"].metadata.item
    manager.latest_step.return_value = progress.iteration
    manager.restore.return_value = SimpleNamespace(metadata=metadata)
    with pytest.raises(ValueError, match="checkpoint .* does not match"):
        training.restore_checkpoint(manager, progress.train_state, replace(config, **change))
    manager.restore.assert_called_once()


def test_full_weight_checkpoint(tmp_path: Path, progress: training.TrainingProgress) -> None:
    config = training.Config(finetuning=training.FullFinetuningConfig(), ckpt_dir=str(tmp_path))
    state = training.create_state(Mock(), progress.train_state.base_params, config)
    updated = state.apply_gradients(grads=jax.tree.map(jnp.ones_like, state.params))
    expected = jax.tree.map(jnp.copy, updated)  # Parameter/moment shapes; () optimizer step.

    @partial(jax.jit, donate_argnums=(0,))
    def update(state: training.TrainState) -> training.TrainState:
        return state.apply_gradients(grads=jax.tree.map(jnp.ones_like, state.params))

    with training.checkpoint_manager(config.ckpt_dir) as manager:
        training.save_checkpoint(manager, replace(progress, train_state=updated), config)
        advanced = update(updated)
        assert int(advanced.step) == int(expected.step) + 1
        manager.wait_until_finished()
        restored = training.restore_checkpoint(manager, state, config)
        assert restored is not None
        assert type(restored.train_state) is training.TrainState
        chex.assert_trees_all_equal(
            (restored.train_state.params, restored.train_state.opt_state, restored.train_state.step),
            (expected.params, expected.opt_state, expected.step),
        )
        with pytest.raises(ValueError, match="checkpoint finetuning does not match"):
            training.restore_checkpoint(
                manager, progress.train_state, replace(config, finetuning=training.LoRAConfig())
            )


def test_checkpoint_dispatches_without_waiting_and_propagates_failure(
    progress: training.TrainingProgress, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Mock(monotonic=Mock(side_effect=[1.0, 6.0, 7.0]))
    monkeypatch.setattr(training, "time", clock)
    manager = Mock()

    assert training.save_checkpoint(manager, progress, training.Config()) == 5.0
    assert manager.save.call_args.kwargs["force"]
    manager.wait_until_finished.assert_not_called()
    manager.save.side_effect = OSError("write failed")
    with pytest.raises(OSError, match="write failed"):
        training.save_checkpoint(manager, progress, training.Config())


def test_async_checkpoint_failure_propagates_on_close(tmp_path: Path, progress: training.TrainingProgress) -> None:
    release = Event()

    def fail_finalize() -> None:
        assert release.wait(timeout=10), "save did not return while finalization was pending"
        raise OSError("background save failed")

    with pytest.raises(OSError, match="background save failed"), ExitStack() as resources:
        manager = resources.enter_context(
            training.ocp.CheckpointManager(
                str(tmp_path),
                options=training.ocp.CheckpointManagerOptions(
                    enable_async_checkpointing=True,
                    async_options=training.ocp.AsyncOptions(post_finalization_callback=fail_finalize),
                ),
            )
        )
        resources.callback(release.set)
        training.save_checkpoint(manager, progress, training.Config())
        assert manager.is_saving_in_progress()


def test_checkpoint_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = Mock()
    monkeypatch.setattr(training.ocp, "CheckpointManager", manager)
    monkeypatch.chdir(tmp_path)
    training.checkpoint_manager("gs://bucket/run/")
    assert manager.call_args.args[0] == "gs://bucket/run/checkpoints"
    assert manager.call_args.kwargs["options"].enable_async_checkpointing
    training.checkpoint_manager("runs/run/")
    assert manager.call_args.args[0] == str(tmp_path / "runs/run/checkpoints")
