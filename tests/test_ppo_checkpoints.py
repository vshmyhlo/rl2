"""PPO checkpoint serialization and restart wiring without running an Atari model."""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from rl2 import ppo
from rl2.shape_checker import ShapeChecker


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
        log_dir=str(tmp_path),
        run_id="resume-test",
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        total_steps=4,
        vector_env="sync",
        video_every_episodes=0,
        eval_every_minutes=0,
    )


@pytest.fixture
def state() -> TrainState:
    def apply(params: dict[str, jax.Array]) -> jax.Array:
        sc = ShapeChecker(W=2)
        sc.check(params["weight"], "W", jnp.float32)
        return params["weight"]

    state = TrainState.create(apply_fn=apply, params={"weight": jnp.array([1.0, -1.0])}, tx=optax.adam(1e-3))
    return state.apply_gradients(grads={"weight": jnp.array([0.5, -0.25])})


def test_checkpoint_restores_optimizer_rng_and_latest_progress(config: ppo.Config, state: TrainState) -> None:
    directory = f"{config.log_dir}/{config.run_id}"
    rng = np.random.default_rng(12)
    rng.random(3)
    progress = ppo.TrainingProgress(state, jax.random.key(3), 2, 8, (1.5, -2.0), (3, 7))
    with ppo.checkpoint_manager(directory) as manager:
        assert ppo.restore_checkpoint(manager, state, rng, config) is None
        ppo.save_checkpoint(manager, progress._replace(iteration=1), rng, config)
        manager.wait_until_finished()
        ppo.save_checkpoint(manager, progress, rng, config)
    fresh = state.replace(
        step=0, params=jax.tree.map(jnp.zeros_like, state.params), opt_state=state.tx.init(state.params)
    )
    restored_rng = np.random.default_rng(99)
    with ppo.checkpoint_manager(directory) as manager:
        restored = ppo.restore_checkpoint(manager, fresh, restored_rng, replace(config, total_steps=8))
        assert restored is not None
        chex.assert_trees_all_equal(restored, progress)
        np.testing.assert_array_equal(restored_rng.integers(100, size=4), rng.integers(100, size=4))
        gradients = {"weight": jnp.array([-0.1, 0.2])}
        chex.assert_trees_all_equal(
            restored.state.apply_gradients(grads=gradients), state.apply_gradients(grads=gradients)
        )
        with pytest.raises(ValueError, match="incompatible.*new run_id"):
            ppo.restore_checkpoint(manager, fresh, restored_rng, replace(config, num_steps=2))
        # Retention keeps two complete checkpoints, even after a manager restart.
        ppo.save_checkpoint(manager, progress._replace(iteration=3), rng, config)
        manager.wait_until_finished()
        assert manager.all_steps() == [2, 3]
    with ppo.checkpoint_manager(f"{config.log_dir}/different-run") as manager:
        assert ppo.restore_checkpoint(manager, fresh, restored_rng, config) is None


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


@pytest.mark.parametrize(
    ("log_dir", "expected"),
    [
        ("gs://bucket/${run_id}/${env_id}/", "gs://bucket/run/ALE_Boxing-v5"),
        ("logs/${run_id}", "logs/run"),
        ("logs/${env_id}/${run_id}", "logs/ALE_Boxing-v5/run"),
        ("logs/", "logs/run"),
    ],
    ids=["cloud-with-environment", "run-only", "custom-order", "legacy-base-directory"],
)
def test_run_directory_interpolation(log_dir: str, expected: str) -> None:
    assert ppo.run_directory(log_dir, "run", "ALE/Boxing-v5") == expected


@pytest.mark.parametrize("log_dir", ["logs/${unknown}", "logs/${run_id"], ids=["unknown", "malformed"])
def test_run_directory_rejects_invalid_placeholders(log_dir: str) -> None:
    with pytest.raises(ValueError, match="log_dir supports only"):
        ppo.run_directory(log_dir, "run", "ALE/Boxing-v5")


def test_train_checkpoints_after_ten_minutes_and_resumes(config: ppo.Config) -> None:
    run_dir = f"{config.log_dir}/{config.run_id}/ALE_Boxing-v5"
    config = replace(config, env_id="ALE/Boxing-v5", log_dir=config.log_dir + "/${run_id}/${env_id}")
    assert config.checkpoint_interval_seconds == 600.0
    obs = np.zeros((1, 1), dtype=np.uint8)
    zeros = np.zeros(1, dtype=np.float32)
    dones = np.ones(1, dtype=bool)
    carry = ppo.initial_carry(1, 1)
    envs = Mock()
    envs.single_action_space.n = 2
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
        return state.apply_gradients(grads={"weight": jnp.ones(1)}), (jnp.asarray(0.0),) * 5

    original_save = ppo.save_checkpoint

    def record_save(
        manager: ppo.ocp.CheckpointManager,
        progress: ppo.TrainingProgress,
        rng: np.random.Generator,
        config: ppo.Config,
    ) -> None:
        saved_iterations.append(progress.iteration)
        original_save(manager, progress, rng, config)

    with (
        patch.object(ppo.gym.vector, "SyncVectorEnv", return_value=envs),
        patch.object(ppo, "make_model", return_value=model),
        patch.object(ppo, "SummaryWriter") as writer,
        patch.object(ppo, "act", return_value=(np.zeros(1, dtype=np.int32), zeros, zeros, carry)),
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
        assert iterations == [0, 1, 2, 3]
        assert int(resumed.step) == 4
        np.testing.assert_allclose(learning_rates, [ppo.learning_rate_schedule(config)(i) for i in range(4)])
        writer.assert_called_once_with(logdir=run_dir, purge_step=3)
        episode_logs = [
            c.args[1:] for c in writer.return_value.add_scalar.call_args_list if c.args[0] == "charts/total_episodes"
        ]
        assert episode_logs == [(3.0, 3), (4.0, 4)]
        sps = [
            c.args[1] for c in writer.return_value.add_scalar.call_args_list if c.args[0] == "charts/steps_per_second"
        ]
        assert sps == [1.0, 1.0]
        for tag, expected in (
            ("time/elapsed_seconds", [(1.0, 3), (2.0, 4)]),
            ("time/eta_seconds", [(1.0, 3), (0.0, 4)]),
        ):
            logged = [c.args[1:] for c in writer.return_value.add_scalar.call_args_list if c.args[0] == tag]
            assert logged == expected
        # An already completed run does not perform more updates or duplicate a save.
        ppo.train(config)
        assert iterations == [0, 1, 2, 3]
        assert saved_iterations == [2, 4]
        assert envs.close.call_count == 3
    assert (Path(run_dir) / "checkpoints" / "4").is_dir()
