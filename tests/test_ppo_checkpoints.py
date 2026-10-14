"""PPO checkpoint serialization and restart wiring without running an Atari model."""

import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState
from ppo_helpers import env_step, metric_history

from rl2 import ppo, train_mappo
from rl2.envs.base import BaseEnv
from rl2.lstm import initial_carry


@pytest.mark.parametrize("use_tf", [None, "true"], ids=["default-gcsfs", "explicit-tensorflow"])
def test_cloud_filesystem_backend_on_startup(use_tf: str | None) -> None:
    # Use a fresh process because epath caches backend selection. Fake cloud I/O
    # and TensorFlow availability so this regression needs neither credentials nor TF.
    # Check epath's backend calls; RLAX may independently import TensorFlow when installed.
    env = os.environ.copy()
    env.pop("EPATH_USE_TF", None)
    if use_tf is not None:
        env["EPATH_USE_TF"] = use_tf
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from unittest.mock import patch
from rl2 import ppo
from etils import epath
from etils.epath import backend

with (
    patch("importlib.util.find_spec", return_value=object()),
    patch.object(backend.fsspec_backend, "exists", return_value=True) as gcsfs_exists,
    patch.object(backend.tf_backend, "exists", return_value=True) as tf_exists,
):
    assert epath.Path("gs://bucket/run/checkpoints").exists()
    expected, unused = (tf_exists, gcsfs_exists) if sys.argv[1] == "true" else (gcsfs_exists, tf_exists)
    expected.assert_called_once_with("gs://bucket/run/checkpoints")
    unused.assert_not_called()
""",
            use_tf or "default",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def config(tmp_path: Path) -> ppo.Config:
    return replace(
        ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        ckpt_dir=str(tmp_path),
        run_id="resume-test",
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        total_steps=4,
        video_interval=0,
        eval_interval=0,
    )


@pytest.fixture
def state() -> TrainState:
    def apply(params: dict[str, jax.Array]) -> jax.Array:
        chex.assert_shape(params["weight"], (2,))
        chex.assert_type(params["weight"], jnp.float32)
        return params["weight"]

    state = TrainState.create(apply_fn=apply, params={"weight": jnp.array([1.0, -1.0])}, tx=optax.adam(1e-3))
    return state.apply_gradients(grads={"weight": jnp.array([0.5, -0.25])})


def test_checkpoint_restores_optimizer_rng_and_latest_progress(config: ppo.Config, state: TrainState) -> None:
    directory = f"{config.ckpt_dir}/{config.run_id}"
    progress = ppo.TrainingProgress(state, jax.random.key(3), 2, 8, 12.5, 40)
    with ppo.checkpoint_manager(directory) as manager:
        assert ppo.restore_checkpoint(manager, state, config) is None
        ppo.save_checkpoint(manager, progress._replace(rollout_count=1), config)
        manager.wait_until_finished()
        ppo.save_checkpoint(manager, progress, config)
    fresh = state.replace(
        step=0, params=jax.tree.map(jnp.zeros_like, state.params), opt_state=state.tx.init(state.params)
    )
    with ppo.checkpoint_manager(directory) as manager:
        metadata = manager.restore(2, args=ppo.ocp.args.Composite(metadata=ppo.ocp.args.JsonRestore())).metadata
        assert metadata["config"] == json.loads(json.dumps(asdict(config)))
        restored = ppo.restore_checkpoint(
            manager,
            fresh,
            replace(
                config,
                total_steps=8,
                log_interval=15,
                video_interval=600,
                eval_interval=1800,
            ),
        )
        assert restored is not None
        chex.assert_trees_all_equal(restored, progress)
        assert metadata["version"] == 3
        assert "recent_returns" not in metadata
        assert "recent_lengths" not in metadata
        assert "numpy_rng" not in metadata
        chex.assert_shape(restored.key, ())
        chex.assert_trees_all_equal(jax.random.permutation(restored.key, 4), jax.random.permutation(progress.key, 4))
        gradients = {"weight": jnp.array([-0.1, 0.2])}
        chex.assert_trees_all_equal(
            restored.train_state.apply_gradients(grads=gradients), state.apply_gradients(grads=gradients)
        )
        # Retention keeps two complete checkpoints, even after a manager restart.
        ppo.save_checkpoint(manager, progress._replace(rollout_count=3), config)
        manager.wait_until_finished()
        assert manager.all_steps() == [2, 3]
    with ppo.checkpoint_manager(f"{config.ckpt_dir}/different-run") as manager:
        assert ppo.restore_checkpoint(manager, fresh, config) is None


@pytest.mark.parametrize(
    "trainer,config_path",
    [(ppo, "configs/ppo.yaml"), (train_mappo, "configs/mappo/smax.yaml")],
    ids=["ppo", "mappo"],
)
def test_checkpoint_timing_includes_completed_writes(
    trainer: ModuleType, config_path: str, capsys: pytest.CaptureFixture[str]
) -> None:
    config = trainer.load_config(Path(__file__).resolve().parents[1] / config_path)
    progress = trainer.TrainingProgress(
        train_state=Mock(spec=TrainState),
        key=jax.random.key(0),  # () typed PRNG key
        rollout_count=2,
        episode_count=0,
        episode_return_sum=0.0,
        episode_length_sum=0,
    )
    clock = 10.0

    def advance(seconds: float, *args: object, **kwargs: object) -> None:
        nonlocal clock
        clock += seconds

    manager = Mock()
    manager.save.side_effect = partial(advance, 2.0)
    manager.wait_until_finished.side_effect = partial(advance, 3.0)
    with patch.object(trainer, "monotonic", side_effect=lambda: clock):
        assert trainer.save_checkpoint(manager, progress, config) == 5.0
        assert [call[0] for call in manager.mock_calls] == ["save", "wait_until_finished"]
        assert capsys.readouterr().out == "Saved checkpoint at iteration 2 in 5.0s\n"
        manager.wait_until_finished.side_effect = RuntimeError("write failed")
        with pytest.raises(RuntimeError, match="write failed"):
            trainer.save_checkpoint(manager, progress, config)
        assert capsys.readouterr().out == ""


@pytest.mark.parametrize("version", [2, 3], ids=["legacy-recent-history", "current-totals-only"])
def test_checkpoint_preserves_episode_totals(config: ppo.Config, state: TrainState, version: int) -> None:
    progress = ppo.TrainingProgress(state, jax.random.key(3), 2, 101, 300.0, 305)
    manager = Mock()
    ppo.save_checkpoint(manager, progress, config)
    saved = manager.save.call_args.kwargs["args"]
    metadata = json.loads(json.dumps(saved.metadata.item))
    assert metadata["version"] == 3
    assert "recent_returns" not in metadata
    assert "recent_lengths" not in metadata
    if version == 2:
        metadata.update(version=2, recent_returns=[2.0] * 100, recent_lengths=[3] * 100)
    manager.latest_step.return_value = progress.rollout_count
    manager.restore.side_effect = [SimpleNamespace(metadata=metadata), SimpleNamespace(state=saved.state.item)]
    restored = ppo.restore_checkpoint(manager, state, config)
    assert restored is not None
    assert restored.episode_length_sum == 305
    chex.assert_trees_all_equal(restored, progress)


@pytest.mark.parametrize(
    "trainer,config_path",
    [(ppo, "configs/ppo.yaml"), (train_mappo, "configs/mappo/smax.yaml")],
    ids=["ppo", "mappo"],
)
def test_checkpoint_ignores_saved_config(trainer: ModuleType, config_path: str, state: TrainState) -> None:
    config = trainer.load_config(Path(__file__).resolve().parents[1] / config_path)
    progress = trainer.TrainingProgress(
        train_state=state,
        key=jax.random.key(3),  # () typed PRNG key
        rollout_count=2,
        episode_count=8,
        episode_return_sum=12.5,
        episode_length_sum=40,
    )
    manager = Mock()
    trainer.save_checkpoint(manager, progress, config)
    saved = manager.save.call_args.kwargs["args"]
    metadata = json.loads(json.dumps(saved.metadata.item))
    metadata["config"].update(
        env={"id": "previous-environment", "retired_setting": True},
        model={"retired_model_setting": True},
        num_envs=config.num_envs + 1,
        num_steps=config.num_steps + 1,
    )
    if trainer is train_mappo:
        metadata["algorithm"] = "previous-environment_ippo"
    manager.latest_step.return_value = progress.rollout_count
    manager.restore.side_effect = [SimpleNamespace(metadata=metadata), SimpleNamespace(state=saved.state.item)]
    restored = trainer.restore_checkpoint(manager, state, config)
    chex.assert_trees_all_equal(restored, progress)


@pytest.mark.parametrize(
    "field,missing",
    [
        ("training_return_sum", True),
        ("training_return_sum", False),
        ("training_length_sum", True),
        ("training_length_sum", False),
    ],
    ids=["missing-return", "null-return", "missing-length", "null-length"],
)
def test_checkpoint_requires_training_totals(config: ppo.Config, state: TrainState, field: str, missing: bool) -> None:
    manager = Mock()
    progress = ppo.TrainingProgress(state, jax.random.key(3), 2, 101, 300.0, 305)
    ppo.save_checkpoint(manager, progress, config)
    metadata = manager.save.call_args.kwargs["args"].metadata.item
    if missing:
        del metadata[field]
        metadata["version"] = 1
    else:
        metadata[field] = None
    manager.latest_step.return_value = 2
    manager.restore.return_value = SimpleNamespace(metadata=metadata)
    with pytest.raises(ValueError, match=f"requires {field}"):
        ppo.restore_checkpoint(manager, state, config)
    manager.restore.assert_called_once()


@pytest.mark.parametrize("run_id", ["", " ", "..", "a/b", "a\\b"])
def test_invalid_run_id(config: ppo.Config, run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        replace(config, run_id=run_id)


@pytest.mark.parametrize("interval", [0, -1, 1.5, True])
def test_invalid_checkpoint_interval(config: ppo.Config, interval: float) -> None:
    with pytest.raises(ValueError, match="checkpoint_interval"):
        replace(config, checkpoint_interval=interval)


def test_checkpoint_paths_preserve_gcs_and_resolve_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with patch.object(ppo.ocp, "CheckpointManager") as manager:
        ppo.checkpoint_manager("gs://bucket/run")
        assert manager.call_args.args[0] == "gs://bucket/run/checkpoints"
        ppo.checkpoint_manager("logs/run")
        assert manager.call_args.args[0] == str(tmp_path / "logs/run/checkpoints")


@pytest.mark.parametrize("failure", ["manager", "reset", "writer", "logging", "writer_close"])
def test_train_closes_resources_on_failure(config: ppo.Config, failure: str) -> None:
    envs = Mock(action_space=ppo.DiscreteSpace(2), params={})
    envs.step_batch = partial(BaseEnv.step_batch, envs)
    envs.reset.return_value = (np.zeros(1, dtype=np.uint8), {})  # observation: (O=1,)
    manager = Mock()
    writer = MagicMock()
    model = Mock()
    model.init.return_value = {"params": {}}
    # Stop before rollout collection, with no real model or optimizer initialization.
    writer.log_text.side_effect = RuntimeError("logging")
    if failure == "reset":
        envs.reset.side_effect = RuntimeError(failure)
    elif failure == "writer_close":
        writer.__exit__.side_effect = RuntimeError(failure)

    with (
        patch.object(ppo, "configure_compilation_cache"),
        patch.object(ppo, "GymnaxEnv", return_value=envs),
        patch.object(ppo, "checkpoint_manager", return_value=manager) as make_manager,
        patch.object(ppo, "ActorCritic", return_value=model),
        patch.object(ppo.TrainState, "create"),
        patch.object(ppo, "restore_checkpoint", return_value=None),
        patch.object(ppo, "MLflowLogger", return_value=writer) as make_writer,
    ):
        if failure == "manager":
            make_manager.side_effect = RuntimeError(failure)
        elif failure == "writer":
            make_writer.side_effect = RuntimeError(failure)
        with pytest.raises(RuntimeError, match=failure):
            ppo.train(config)

    if failure == "manager":
        manager.close.assert_not_called()
    else:
        manager.close.assert_called_once_with()
    if failure in ("logging", "writer_close"):
        make_writer.assert_called_once_with(experiment="ppo", run_name=config.run_id, resume=True)
        writer.__exit__.assert_called_once()
    else:
        writer.__exit__.assert_not_called()


def test_train_checkpoints_at_iteration_intervals_and_resumes(config: ppo.Config) -> None:
    path = Path(config.ckpt_dir) / "config.yaml"
    settings = asdict(config)
    settings.update(env={"id": "Breakout-MinAtar"}, ckpt_dir=config.ckpt_dir + "/${run_id}/${path_name:${env.id}}")
    path.write_text(ppo.yaml.safe_dump(settings))
    config = ppo.load_config(path)
    assert config.checkpoint_interval == 600
    config = replace(config, checkpoint_interval=2, log_interval=2)
    obs = np.zeros((1, 1), dtype=np.uint8)
    zeros = np.zeros(1, dtype=np.float32)
    dones = np.ones(1, dtype=bool)
    carry = initial_carry(1, 1)
    envs = Mock(action_space=ppo.DiscreteSpace(2), params={})
    envs.step_batch = partial(BaseEnv.step_batch, envs)
    envs.reset.return_value = (obs[0], {})
    envs.step.return_value = env_step(obs[0], np.array(1.0), dones[0], ~dones[0], {})
    model = Mock()
    model.initial_carry.return_value = carry
    model.init.return_value = {"params": {"weight": jnp.zeros(1)}}
    clock = 0.0
    interrupted = False
    iterations: list[int] = []
    learning_rates: list[float] = []
    saved_iterations: list[int] = []
    saved_return_sums: list[float] = []
    saved_length_sums: list[int] = []

    def update(
        state: TrainState,
        batch: ppo.PPOBatch,
        config: ppo.Config,
        iteration: int,
    ) -> tuple[TrainState, ppo.MinibatchMetrics]:
        nonlocal clock, interrupted
        if iteration == 2 and not interrupted:
            interrupted = True
            raise RuntimeError("simulated interruption")
        # Wall time is independent of the checkpoint schedule.
        clock += 599.0 if iteration == 0 else 1.0
        iterations.append(iteration)
        chex.assert_shape(state.opt_state.hyperparams["learning_rate"], ())
        learning_rates.append(float(state.opt_state.hyperparams["learning_rate"]))
        return state.apply_gradients(grads={"weight": jnp.ones(1)}), ppo.MinibatchMetrics(
            policy_loss=jnp.asarray(0.0),  # ()
            value_loss=jnp.asarray(0.0),  # ()
            entropy_loss=jnp.asarray(0.0),  # ()
            entropy=jnp.asarray(0.0),  # ()
            approx_kl=jnp.asarray(0.0),  # ()
            clip_fraction=jnp.asarray(0.0),  # ()
        )

    checkpoint: ppo.TrainingProgress | None = None

    def record_save(
        manager: ppo.ocp.CheckpointManager,
        progress: ppo.TrainingProgress,
        config: ppo.Config,
    ) -> float:
        nonlocal checkpoint
        checkpoint = progress
        assert isinstance(progress.episode_count, int)
        assert isinstance(progress.episode_return_sum, float)
        assert isinstance(progress.episode_length_sum, int)
        saved_iterations.append(progress.rollout_count)
        saved_return_sums.append(progress.episode_return_sum)
        saved_length_sums.append(progress.episode_length_sum)
        return 2.5

    with (
        jax.disable_jit(),
        patch.object(ppo, "GymnaxEnv", return_value=envs),
        patch.object(ppo, "ActorCritic", return_value=model),
        patch.object(ppo, "MLflowLogger") as writer,
        patch.object(
            model,
            "apply",
            return_value=ppo.ActorCriticOutput(carry, ppo.CategoricalPolicy(jnp.array([[0.0, -jnp.inf]])), zeros),
        ),  # carry leaves: (B,H); (B,A); (B,)
        patch.object(ppo, "update", side_effect=update),
        patch.object(ppo, "monotonic", side_effect=lambda: clock + 0.01),
        patch.object(ppo, "checkpoint_manager"),
        patch.object(ppo, "restore_checkpoint", side_effect=lambda *args: checkpoint),
        patch.object(ppo, "save_checkpoint", side_effect=record_save),
        patch.object(ppo, "log_video") as video,
        patch.object(ppo, "log_evaluation") as evaluate,
    ):
        with pytest.raises(RuntimeError, match="simulated interruption"):
            ppo.train(config)
        assert saved_iterations == [2]
        writer.reset_mock()
        config = replace(config, checkpoint_interval=3, log_interval=3, video_interval=3, eval_interval=3)
        resumed = ppo.train(config)
        assert saved_iterations == [2, 3, 4]
        assert saved_return_sums == [2.0, 3.0, 4.0]
        assert saved_length_sums == [2, 3, 4]
        assert [call.args[4] for call in video.call_args_list] == [3, 4]
        assert [call.args[4] for call in evaluate.call_args_list] == [3, 4]
        assert all(isinstance(call.args[3], int) for call in video.call_args_list + evaluate.call_args_list)
        assert iterations == [0, 1, 2, 3]
        assert int(resumed.step) == 4
        np.testing.assert_allclose(learning_rates, [ppo.learning_rate_schedule(config)(i) for i in range(4)])
        writer.assert_called_once_with(experiment="ppo", run_name=config.run_id, resume=True)
        logged = metric_history(writer.return_value)
        assert logged["charts/total_episodes"] == [(3.0, 3), (4.0, 4)]
        assert logged["charts/total_observations"][-1] == (4.0, 4)
        assert logged["charts/training_progress"] == [(0.75, 3), (1.0, 4)]
        assert logged["charts/total_return_mean"] == [(1.0, 3), (1.0, 4)]
        assert logged["charts/total_episode_length_mean"] == [(1.0, 3), (1.0, 4)]
        assert [value for value, _ in logged["charts/observations_per_second"]] == [1.0, 1.0]
        assert logged["time/elapsed_seconds"] == [(1.0, 3), (2.0, 4)]
        # An already completed run does not perform more updates or duplicate a save.
        ppo.train(config)
        assert iterations == [0, 1, 2, 3]
        assert saved_iterations == [2, 3, 4]
        # Long runs must retain small increments beyond float32 precision and int32 range.
        assert checkpoint is not None
        checkpoint = checkpoint._replace(
            rollout_count=3,
            episode_count=2**31 - 1,
            episode_return_sum=float(2**24),
            episode_length_sum=2**31 - 1,
        )
        ppo.train(config)
        assert checkpoint.episode_count == 2**31
        assert checkpoint.episode_return_sum == float(2**24 + 1)
        assert checkpoint.episode_length_sum == 2**31
        # Restoring these totals must also avoid narrowing them to device scalars.
        ppo.train(config)
