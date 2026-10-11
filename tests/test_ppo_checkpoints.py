"""PPO checkpoint serialization and restart wiring without running an Atari model."""

import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from rl2 import ppo
from rl2.atari_eval import TrainingScores
from rl2.lstm import initial_carry


@pytest.mark.parametrize("use_tf", [None, "true"], ids=["default-gcsfs", "explicit-tensorflow"])
def test_cloud_filesystem_backend_on_startup(use_tf: str | None) -> None:
    # Use a fresh process because epath caches backend selection. Fake cloud I/O
    # and TensorFlow availability so this regression needs neither credentials nor TF.
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
assert "tensorflow" not in sys.modules
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
        env_id="ALE/Pong-v5",
        log_dir=str(tmp_path),
        run_id="resume-test",
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        total_steps=4,
        vector_env="sync",
        video_interval_seconds=0,
        eval_interval_seconds=0,
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
    directory = f"{config.log_dir}/{config.run_id}"
    progress = ppo.TrainingProgress(state, jax.random.key(3), 2, 8, (1.5, -2.0), (3, 7), 12.5, 40)
    with ppo.checkpoint_manager(directory) as manager:
        assert ppo.restore_checkpoint(manager, state, config) is None
        ppo.save_checkpoint(manager, progress._replace(iteration=1), config)
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
                num_steps=2,
                log_interval_seconds=15.0,
                video_interval_seconds=600.0,
                eval_interval_seconds=1800.0,
            ),
        )
        assert restored is not None
        chex.assert_trees_all_equal(restored, progress)
        assert metadata["version"] == 2
        assert "numpy_rng" not in metadata
        chex.assert_shape(restored.key, ())
        chex.assert_trees_all_equal(jax.random.permutation(restored.key, 4), jax.random.permutation(progress.key, 4))
        gradients = {"weight": jnp.array([-0.1, 0.2])}
        chex.assert_trees_all_equal(
            restored.state.apply_gradients(grads=gradients), state.apply_gradients(grads=gradients)
        )
        # Retention keeps two complete checkpoints, even after a manager restart.
        ppo.save_checkpoint(manager, progress._replace(iteration=3), config)
        manager.wait_until_finished()
        assert manager.all_steps() == [2, 3]
    with ppo.checkpoint_manager(f"{config.log_dir}/different-run") as manager:
        assert ppo.restore_checkpoint(manager, fresh, config) is None


def test_checkpoint_preserves_episode_length_history(config: ppo.Config, state: TrainState) -> None:
    progress = ppo.TrainingProgress(state, jax.random.key(3), 2, 101, (2.0,) * 100, (3,) * 100, 300.0, 305)
    manager = Mock()
    ppo.save_checkpoint(manager, progress, config)
    saved = manager.save.call_args.kwargs["args"]
    metadata = json.loads(json.dumps(saved.metadata.item))
    manager.latest_step.return_value = progress.iteration
    manager.restore.side_effect = [SimpleNamespace(metadata=metadata), SimpleNamespace(state=saved.state.item)]
    restored = ppo.restore_checkpoint(manager, state, config)
    assert restored is not None
    assert restored.training_length_sum == 305
    assert restored.recent_lengths == (3,) * 100


def test_legacy_checkpoint_leaves_training_total_unavailable(config: ppo.Config, state: TrainState) -> None:
    rng = np.random.default_rng(12)
    manager = Mock()
    manager.latest_step.return_value = 2
    manager.restore.side_effect = [
        SimpleNamespace(
            metadata={
                "version": 1,
                "config": {"env_id": config.env_id},  # Older checkpoints stored only a subset of settings.
                "iteration": 2,
                "completed_episodes": 101,
                "recent_returns": [2.0] * 100,
                "recent_lengths": [3] * 100,
                "numpy_rng": rng.bit_generator.state,
            }
        ),
        SimpleNamespace(state={"train_state": state, "key": jax.random.key_data(jax.random.key(3))}),
    ]
    restored = ppo.restore_checkpoint(manager, state, config)
    assert restored is not None
    assert restored.training_return_sum is None
    assert restored.training_length_sum is None
    assert restored.completed_episodes == 101
    assert restored.recent_returns == (2.0,) * 100
    report = TrainingScores(
        restored.completed_episodes,
        restored.training_return_sum,
        restored.recent_returns,
        length_sum=restored.training_length_sum,
        recent_lengths=restored.recent_lengths,
    ).as_report()
    assert report["total_return_mean"] is None
    assert report["return_mean_last_100"] == 2.0
    assert report["total_episode_length_mean"] is None
    assert report["episode_length_mean_last_100"] == 3.0
    assert report["episodes_last_100"] == 100


@pytest.mark.parametrize("run_id", ["", " ", "..", "a/b", "a\\b"])
def test_invalid_run_id(config: ppo.Config, run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        replace(config, run_id=run_id)


@pytest.mark.parametrize("interval", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_checkpoint_interval(config: ppo.Config, interval: float) -> None:
    with pytest.raises(ValueError, match="checkpoint_interval_seconds"):
        replace(config, checkpoint_interval_seconds=interval)


def test_checkpoint_paths_preserve_gcs_and_resolve_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with patch.object(ppo.ocp, "CheckpointManager") as manager:
        ppo.checkpoint_manager("gs://bucket/run")
        assert manager.call_args.args[0] == "gs://bucket/run/checkpoints"
        ppo.checkpoint_manager("logs/run")
        assert manager.call_args.args[0] == str(tmp_path / "logs/run/checkpoints")


@pytest.mark.parametrize("failure", ["manager", "reset", "writer", "logging", "writer_close"])
def test_train_closes_resources_on_failure(config: ppo.Config, failure: str) -> None:
    envs = Mock(single_action_space=ppo.gym.spaces.Discrete(2))
    envs.reset.return_value = (np.zeros((1, 1), dtype=np.uint8), {})  # observations: (B=1, O=1)
    manager = Mock()
    writer = Mock()
    model = Mock()
    model.init.return_value = {"params": {}}
    # Stop before rollout collection, with no real model or optimizer initialization.
    writer.add_text.side_effect = RuntimeError("logging")
    if failure == "reset":
        envs.reset.side_effect = RuntimeError(failure)
    elif failure == "writer_close":
        writer.close.side_effect = RuntimeError(failure)

    with (
        patch.object(ppo, "configure_compilation_cache"),
        patch.object(ppo.gym.vector, "SyncVectorEnv"),
        patch.object(ppo, "JaxVectorEnv", return_value=envs),
        patch.object(ppo, "checkpoint_manager", return_value=manager) as make_manager,
        patch.object(ppo, "ActorCritic", return_value=model),
        patch.object(ppo.TrainState, "create"),
        patch.object(ppo, "restore_checkpoint", return_value=None),
        patch.object(ppo, "SummaryWriter", return_value=writer) as make_writer,
    ):
        if failure == "manager":
            make_manager.side_effect = RuntimeError(failure)
        elif failure == "writer":
            make_writer.side_effect = RuntimeError(failure)
        with pytest.raises(RuntimeError, match=failure):
            ppo.train(config)

    envs.close.assert_called_once_with()
    if failure == "manager":
        manager.close.assert_not_called()
    else:
        manager.close.assert_called_once_with()
    if failure in ("logging", "writer_close"):
        writer.close.assert_called_once_with()
    else:
        writer.close.assert_not_called()


def test_train_checkpoints_after_ten_minutes_and_resumes(config: ppo.Config) -> None:
    run_dir = f"{config.log_dir}/{config.run_id}/ALE_Boxing-v5"
    path = Path(config.log_dir) / "config.yaml"
    settings = asdict(config)
    settings.update(env_id="ALE/Boxing-v5", log_dir=config.log_dir + "/${run_id}/${path_name:${env_id}}")
    path.write_text(ppo.yaml.safe_dump(settings))
    config = ppo.load_config(path)
    assert config.checkpoint_interval_seconds == 600.0
    obs = np.zeros((1, 1), dtype=np.uint8)
    zeros = np.zeros(1, dtype=np.float32)
    dones = np.ones(1, dtype=bool)
    carry = initial_carry(1, 1)
    envs = Mock(spec=ppo.gym.vector.VectorEnv, single_action_space=ppo.gym.spaces.Discrete(2))
    envs.reset.return_value = (obs, {})
    envs.step.return_value = (obs, np.ones(1), dones, ~dones, {})
    model = Mock()
    model.initial_carry.return_value = carry
    model.init.return_value = {"params": {"weight": jnp.zeros(1)}}
    clock = 0.0
    interrupted = False
    iterations: list[int] = []
    learning_rates: list[float] = []
    saved_iterations: list[int] = []
    saved_return_sums: list[float | None] = []
    saved_recent_returns: list[tuple[float, ...]] = []

    def update(
        state: TrainState, batch: ppo.PPOBatch, config: ppo.Config, iteration: int
    ) -> tuple[TrainState, ppo.PPOMetrics]:
        nonlocal clock, interrupted
        if iteration == 2 and not interrupted:
            interrupted = True
            raise RuntimeError("simulated interruption")
        # The first rollout is just before the interval; the second reaches it.
        clock += 599.0 if iteration == 0 else 1.0
        iterations.append(iteration)
        learning_rates.append(float(state.opt_state.hyperparams["learning_rate"]))
        return state.apply_gradients(grads={"weight": jnp.ones(1)}), ppo.PPOMetrics(
            policy_loss=jnp.asarray(0.0),  # ()
            value_loss=jnp.asarray(0.0),  # ()
            entropy_loss=jnp.asarray(0.0),  # ()
            entropy=jnp.asarray(0.0),  # ()
            approx_kl=jnp.asarray(0.0),  # ()
            clip_fraction=jnp.asarray(0.0),  # ()
        )

    original_save = ppo.save_checkpoint

    def record_save(
        manager: ppo.ocp.CheckpointManager,
        progress: ppo.TrainingProgress,
        config: ppo.Config,
    ) -> None:
        saved_iterations.append(progress.iteration)
        saved_return_sums.append(progress.training_return_sum)
        saved_recent_returns.append(progress.recent_returns)
        original_save(manager, progress, config)

    with (
        patch.object(ppo.gym.vector, "SyncVectorEnv", return_value=envs),
        patch.object(ppo, "ActorCritic", return_value=model),
        patch.object(ppo, "SummaryWriter") as writer,
        patch.object(ppo, "act", return_value=(jnp.zeros(1, dtype=jnp.int32), zeros, zeros, carry)),
        patch.object(ppo, "value", return_value=zeros),
        patch.object(ppo, "update", side_effect=update),
        patch.object(ppo, "monotonic", side_effect=lambda: clock + 0.01),
        patch.object(ppo, "save_checkpoint", side_effect=record_save),
    ):
        with pytest.raises(RuntimeError, match="simulated interruption"):
            ppo.train(config)
        assert saved_iterations == [2]
        assert envs.close.call_count == 1
        writer.reset_mock()
        resumed = ppo.train(config)
        assert saved_iterations == [2, 4]
        assert saved_return_sums == [2.0, 4.0]
        assert saved_recent_returns == [(1.0, 1.0), (1.0,) * 4]
        assert iterations == [0, 1, 2, 3]
        assert int(resumed.step) == 4
        np.testing.assert_allclose(learning_rates, [ppo.learning_rate_schedule(config)(i) for i in range(4)])
        writer.assert_called_once_with(logdir=run_dir, purge_step=3)
        episode_logs = [
            c.args[1:] for c in writer.return_value.add_scalar.call_args_list if c.args[0] == "charts/total_episodes"
        ]
        assert episode_logs == [(4.0, 4)]
        writer.return_value.add_scalar.assert_any_call("charts/total_observations", 4.0, 4)
        return_logs = [
            c.args[1:] for c in writer.return_value.add_scalar.call_args_list if c.args[0] == "charts/total_return_mean"
        ]
        assert return_logs == [(1.0, 4)]
        sps = [
            c.args[1]
            for c in writer.return_value.add_scalar.call_args_list
            if c.args[0] == "charts/observations_per_second"
        ]
        assert sps == [1.0]
        elapsed_logs = [
            c.args[1:] for c in writer.return_value.add_scalar.call_args_list if c.args[0] == "time/elapsed_seconds"
        ]
        assert elapsed_logs == [(2.0, 4)]
        # An already completed run does not perform more updates or duplicate a save.
        ppo.train(config)
        assert iterations == [0, 1, 2, 3]
        assert saved_iterations == [2, 4]
        assert envs.close.call_count == 3
    assert (Path(run_dir) / "checkpoints" / "4").is_dir()
