import logging
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock, Mock

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml
from flax.training.train_state import TrainState
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

pytest.importorskip("pgx")
pytest.importorskip("mctx")
import pgx

from rl2 import train_alphazero as az
from rl2.alphazero.config import EvaluationConfig, ModelConfig


@pytest.fixture(scope="module")
def tiny_model() -> tuple[pgx.Env, az.PolicyValueNet, az.Parameters]:
    env = pgx.make("tic_tac_toe")
    model = az.PolicyValueNet(env.num_actions, channels=4, num_blocks=1)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 3, 3, 2), jnp.float32))["params"]
    return env, model, params


def test_sample_batch_excludes_padding() -> None:
    batch = az.Batch(
        observations=jnp.zeros((2, 3, 3, 2), jnp.float32),
        policy_targets=jnp.full((2, 9), 1 / 9, jnp.float32),
        value_targets=jnp.array([1.0, -1.0], jnp.float32),
        policy_mask=jnp.array([True, False]),
        value_mask=jnp.array([True, False]),
    )
    sampled = az.sample_batch(batch, jax.random.PRNGKey(0), batch_size=4)
    assert sampled.observations.shape == (4, 3, 3, 2)
    assert sampled.policy_mask.all()
    assert sampled.value_mask.all()
    np.testing.assert_array_equal(sampled.value_targets, np.ones(4))


def test_sample_batch_sparse_positions_preserves_fields() -> None:
    # Leading, interior, and trailing padding; valid targets have distinct labels.
    labels = jnp.arange(5, dtype=jnp.float32)
    batch = az.Batch(
        observations=labels[:, None, None, None],
        policy_targets=labels[:, None],
        value_targets=labels,
        policy_mask=jnp.array([False, True, False, True, False]),
        value_mask=jnp.array([False, True, False, False, False]),
    )
    sampled = az.sample_batch(batch, jax.random.PRNGKey(0), batch_size=128)
    assert sampled.observations.shape == (128, 1, 1, 1)
    assert sampled.policy_targets.shape == (128, 1)
    chex.assert_shape((sampled.value_targets, sampled.policy_mask, sampled.value_mask), (128,))
    assert sampled.policy_mask.all()
    np.testing.assert_array_equal(sampled.observations[:, 0, 0, 0], sampled.value_targets)
    np.testing.assert_array_equal(sampled.policy_targets[:, 0], sampled.value_targets)
    np.testing.assert_array_equal(sampled.value_mask, sampled.value_targets == 1)
    labels, counts = np.unique(sampled.value_targets, return_counts=True)
    np.testing.assert_array_equal(labels, [1, 3])
    # Generous deterministic bounds catch severe sampling bias without seed sweeps.
    assert np.all((counts > 32) & (counts < 96))


@pytest.mark.parametrize("fraction", [0.0, 0.25, 1.0], ids=["disabled", "mixture", "noise-only"])
def test_selfplay_prior_noise_uses_only_legal_actions(fraction: float, monkeypatch: pytest.MonkeyPatch) -> None:
    logits = jnp.log(jnp.array([[3, 1, 500, 500], [1, 1, 1, 1], [1, 1, 1, 1]], jnp.float32))
    legal = jnp.array([[True, True, False, False], [False, False, True, False], [False] * 4])
    # Most Gamma mass is on illegal actions; it must not dilute the noise weight.
    loggamma = Mock(return_value=jnp.log(jnp.array([[1, 3, 1000, 1000], [1, 2, 3, 4], [1, 1, 1, 1]], jnp.float32)))
    monkeypatch.setattr(az.jax.random, "loggamma", loggamma)
    noisy = az.selfplay_prior_logits(
        logits, legal, jax.random.PRNGKey(0), dirichlet_alpha=0.3, dirichlet_fraction=fraction
    )
    assert noisy.shape == logits.shape
    assert noisy.dtype == jnp.float32
    assert np.isfinite(noisy).all()
    np.testing.assert_allclose(
        jax.nn.softmax(noisy),
        [[0.75 - 0.5 * fraction, 0.25 + 0.5 * fraction, 0, 0], [0, 0, 1, 0], [0.25] * 4],
        rtol=1e-6,
    )
    if fraction == 0.0:
        loggamma.assert_not_called()


def test_masked_logits() -> None:
    logits = jnp.array([[1.0, 100.0, 2.0], [0.0, 0.0, 0.0]], jnp.float32)
    legal = jnp.array([[True, False, True], [False, False, False]])
    probs = jax.nn.softmax(az.masked_logits(logits, legal))
    assert probs[0, 1] == 0
    np.testing.assert_allclose(probs.sum(-1), 1.0)
    assert np.isfinite(probs).all()


def test_outcomes_use_player_ids_and_adjudicate_move_limit_draws(
    tiny_model: tuple[pgx.Env, az.PolicyValueNet, az.Parameters], monkeypatch: pytest.MonkeyPatch
) -> None:
    env, model, params = tiny_model
    config = az.Config(num_envs=4, max_moves=3)
    # Columns: win on the last allowed move, natural draw, move limit, env truncation.
    # Only the natural draw and env truncation have padding in the last row.
    # T: recorded time steps; B: games; P: players; A: actions.
    players = jnp.array([[1, 0, 0, 0], [0, 1, 1, 1], [1, 0, 0, 0]], jnp.int32)  # (T, B)
    valid = jnp.array([[True] * 4, [True] * 4, [True, False, True, False]])  # (T, B)
    # Nonterminal rewards must not leak into draw targets or masked targets.
    rewards = jnp.array([[-1, 1], [0, 0], [1, -1], [1, -1]], jnp.float32)  # (B, P)
    states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 4))  # Leaves: (B, ...)
    states = states.replace(
        rewards=rewards,  # (B, P)
        terminated=jnp.array([True, True, False, False]),  # (B,)
        truncated=jnp.array([False, False, False, True]),  # (B,)
    )
    observations = jnp.zeros((3, 4, *states.observation.shape[1:]))  # (T, B, 3, 3, 2)
    policies = jnp.full((3, 4, env.num_actions), 1 / env.num_actions)  # (T, B, A)
    monkeypatch.setattr(jax.lax, "scan", Mock(return_value=(states, (observations, policies, players, valid))))
    batch, metrics = az.collect_selfplay.__wrapped__(params, jax.random.PRNGKey(0), env=env, model=model, config=config)
    chex.assert_shape((batch.value_targets, batch.policy_mask, batch.value_mask), (12,))
    np.testing.assert_array_equal(batch.value_targets, [1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(
        batch.value_mask, [True, True, True, False, True, True, True, False, True, False, True, False]
    )
    np.testing.assert_array_equal(batch.policy_mask, valid.reshape(-1))
    assert metrics["value_positions"] == 8
    assert metrics["positions"] == 10
    assert metrics["completed_games"] == 3


def test_recurrent_step_terminal_reward_and_opponent_discount(
    tiny_model: tuple[pgx.Env, az.PolicyValueNet, az.Parameters],
) -> None:
    env, model, params = tiny_model
    state = env.init(jax.random.PRNGKey(0))
    # Player to move can win on square 2 after these four legal moves.
    for action in (0, 3, 1, 4):
        state = env.step(state, jnp.int32(action))

    def duplicate(x: jax.Array) -> jax.Array:
        return jnp.stack((x, x))

    output, next_states = az.recurrent_step(
        partial(model.apply, {"params": params}),
        jnp.array([2, 8], jnp.int32),
        jax.tree.map(duplicate, state),
        env=env,
    )
    np.testing.assert_array_equal(output.reward, [1, 0])
    np.testing.assert_array_equal(output.discount, [0, -1])
    assert output.value[0] == 0
    _, expected_values = model.apply({"params": params}, next_states.observation)
    np.testing.assert_allclose(output.value[1], expected_values[1])
    probs = jax.nn.softmax(output.prior_logits)
    assert np.all(np.asarray(probs[1])[~np.asarray(next_states.legal_action_mask[1])] == 0)


def test_selfplay_and_update(tiny_model: tuple[pgx.Env, az.PolicyValueNet, az.Parameters]) -> None:
    env, model, params = tiny_model
    config = az.Config(
        env_id="tic_tac_toe",
        num_envs=2,
        max_moves=10,  # One padding move after the longest possible game.
        num_simulations=2,
        model=ModelConfig(channels=4, num_blocks=1),
        batch_size=10,
    )
    batch, metrics = az.collect_selfplay(params, jax.random.PRNGKey(2), env=env, model=model, config=config)
    size = config.max_moves * config.num_envs
    chex.assert_shape(batch.observations, (size, 3, 3, 2))
    chex.assert_shape(batch.policy_targets, (size, 9))
    chex.assert_shape((batch.value_targets, batch.policy_mask, batch.value_mask), (size,))
    chex.assert_type((batch.observations, batch.policy_targets, batch.value_targets), jnp.float32)
    chex.assert_type((batch.policy_mask, batch.value_mask), jnp.bool_)
    assert metrics["completed_games"] == 2
    assert 10 <= metrics["positions"] <= 18
    np.testing.assert_array_equal(batch.value_mask, batch.policy_mask)
    assert not batch.policy_mask[-2:].any()
    np.testing.assert_allclose(batch.policy_targets.sum(-1), 1.0)
    occupied = np.asarray(batch.observations).any(axis=-1).reshape(size, 9)
    real_positions = np.asarray(batch.policy_mask)
    assert np.all(np.asarray(batch.policy_targets)[occupied & real_positions[:, None]] == 0)
    # Every valid move has the game's result, alternating with the player to move.
    for game in range(config.num_envs):
        game_mask = real_positions.reshape(config.max_moves, config.num_envs)[:, game]
        targets = np.asarray(batch.value_targets).reshape(config.max_moves, config.num_envs)[game_mask, game]
        np.testing.assert_array_equal(targets[:-1], -targets[1:])
        assert targets[-1] in (0, 1)

    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    updated, losses = az.train_step(state, batch, model=model)
    assert updated.step == 1
    assert all(np.isfinite(value) for value in losses.values())
    assert any(
        np.any(np.asarray(before) != np.asarray(after))
        for before, after in zip(jax.tree.leaves(params), jax.tree.leaves(updated.params), strict=True)
    )
    # Padding must not train either head; masked losses have zero gradient.
    empty = batch.replace(policy_mask=jnp.zeros(size, bool), value_mask=jnp.zeros(size, bool))
    (loss, _), grads = jax.value_and_grad(az.loss_fn, has_aux=True)(params, model, empty)
    assert loss == 0
    assert all(np.all(np.asarray(leaf) == 0) for leaf in jax.tree.leaves(grads))


def test_chess_search_smoke() -> None:
    env = pgx.make("chess")
    config = az.Config(num_envs=1, max_moves=1, num_simulations=1, model=ModelConfig(channels=4, num_blocks=1))
    model = az.PolicyValueNet(env.num_actions, channels=4, num_blocks=1)
    observation = env.init(jax.random.PRNGKey(0)).observation[None]  # (1, 8, 8, 119)
    params = model.init(jax.random.PRNGKey(0), observation)["params"]
    batch, metrics = az.collect_selfplay(params, jax.random.PRNGKey(0), env=env, model=model, config=config)
    assert batch.observations.shape == (1, *observation.shape[1:])
    assert batch.policy_targets.shape == (1, 4672)
    assert metrics["positions"] == 1
    chex.assert_shape((batch.value_targets, batch.policy_mask, batch.value_mask), (1,))
    assert metrics["completed_games"] == 1
    assert metrics["value_positions"] == 1
    assert batch.value_mask.all()
    assert batch.policy_mask.all()
    np.testing.assert_array_equal(batch.value_targets, [0.0])
    legal = env.init(jax.random.PRNGKey(0)).legal_action_mask  # (4672,)
    assert np.all(np.asarray(batch.policy_targets[0])[~np.asarray(legal)] == 0)
    np.testing.assert_allclose(batch.policy_targets.sum(), 1.0)


def test_main_wiring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "env_id: tic_tac_toe\niterations: 1\nrun_id: test\nlog_dir: runs/alphazero/${run_id}\nckpt_dir: ${log_dir}\n"
    )
    train = Mock()
    monkeypatch.setattr(az, "train", train)
    monkeypatch.setattr(sys, "argv", ["train_alphazero", "--config", str(path)])
    az.main()
    train.assert_called_once_with(
        replace(
            az.Config(),
            env_id="tic_tac_toe",
            iterations=1,
            run_id="test",
            log_dir="runs/alphazero/test",
            ckpt_dir="runs/alphazero/test",
        )
    )


def test_main_rejects_cli_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    train = Mock()
    monkeypatch.setattr(az, "train", train)
    monkeypatch.setattr(sys, "argv", ["train_alphazero", "--iterations", "1"])
    with pytest.raises(SystemExit) as error:
        az.main()
    assert error.value.code == 2
    train.assert_not_called()


def test_main_reports_config_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("num_envs: 0\n")
    train = Mock()
    monkeypatch.setattr(az, "train", train)
    monkeypatch.setattr(sys, "argv", ["train_alphazero", "--config", str(path)])
    with pytest.raises(SystemExit) as error:
        az.main()
    assert error.value.code == 2
    assert "num_envs" in capsys.readouterr().err
    train.assert_not_called()


@pytest.fixture
def logging_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> az.Config:
    """Exercise the training loop's logging without running a model or search."""
    config = az.Config(
        iterations=2,
        updates_per_iteration=2,
        run_id="test",
        log_dir=str(tmp_path / "test"),
        ckpt_dir=str(tmp_path / "checkpoints" / "test"),
    )
    manager = MagicMock()
    manager.__enter__.return_value = manager
    manager.latest_step.return_value = None
    monkeypatch.setattr(az, "checkpoint_manager", Mock(return_value=manager))
    env = Mock(num_actions=9)
    env.init.return_value.observation = jnp.zeros((3, 3, 2), jnp.float32)
    monkeypatch.setattr(az.pgx, "make", Mock(return_value=env))
    model = Mock()
    model.init.return_value = {"params": {"weight": jnp.zeros(1, jnp.float32)}}
    monkeypatch.setattr(az, "PolicyValueNet", Mock(return_value=model))
    monkeypatch.setattr(az, "sample_batch", Mock(return_value=None))
    rollouts = [
        (None, {"positions": jnp.int32(3), "completed_games": jnp.int32(1), "value_positions": jnp.int32(2)}),
        (None, {"positions": jnp.int32(2), "completed_games": jnp.int32(0), "value_positions": jnp.int32(0)}),
    ]
    monkeypatch.setattr(az, "collect_selfplay", Mock(side_effect=rollouts))

    def fake_step(state: TrainState, batch: az.Batch, *, model: az.PolicyValueNet) -> tuple[TrainState, az.Metrics]:
        loss = jnp.float32(2 * state.step + 1)
        return state.replace(step=state.step + 1), {"loss": loss, "policy_loss": loss, "value_loss": jnp.float32(0)}

    monkeypatch.setattr(az, "train_step", fake_step)
    return config


def test_tensorboard_progress(
    logging_config: az.Config, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    caplog.set_level(logging.INFO, logger=az.__name__)
    logging_config = replace(
        logging_config,
        iterations=5,
        log_dir=str(tmp_path / "test-run" / "chess"),
        run_id="test-run",
        ckpt_dir=str(tmp_path / "checkpoints" / "test-run"),
        log_interval_seconds=30.0,
    )
    clock = 0.0
    durations = [29.0, 1.0, 15.0, 15.0, 5.0]
    original_step = az.train_step
    rollouts = list(az.collect_selfplay.side_effect)
    az.collect_selfplay.side_effect = rollouts * 2 + rollouts[:1]

    def timed_step(state: TrainState, batch: az.Batch, *, model: az.PolicyValueNet) -> tuple[TrainState, az.Metrics]:
        nonlocal clock
        clock += durations[state.step // logging_config.updates_per_iteration] / logging_config.updates_per_iteration
        return original_step(state, batch, model=model)

    def now() -> float:
        return clock

    monkeypatch.setattr(az, "train_step", timed_step)
    monkeypatch.setattr(az, "monotonic", now)
    state = az.train(logging_config)
    assert state.step == 10
    events = EventAccumulator(str(tmp_path / "test-run" / "chess")).Reload()
    losses = events.Scalars("losses/loss")
    assert [event.step for event in losses] == [5, 10, 13]
    assert [event.value for event in losses] == [4, 12, 18]  # Mean over every update in each window.
    assert [event.value for event in events.Scalars("charts/completed_games")] == [1, 2, 3]
    assert [event.value for event in events.Scalars("selfplay/completed_games")] == [1, 1, 1]
    assert [event.value for event in events.Scalars("selfplay/positions")] == [5, 5, 3]
    assert [event.value for event in events.Scalars("selfplay/value_positions")] == [2, 2, 2]
    assert [event.value for event in events.Scalars("charts/iteration")] == [2, 4, 5]
    assert [event.value for event in events.Scalars("time/window_seconds")] == [30, 30, 5]
    assert [event.value for event in events.Scalars("time/iteration_seconds")] == [30, 30, 5]
    assert "time/rollout_seconds" not in events.Tags()["scalars"]
    assert "time/optimization_seconds" not in events.Tags()["scalars"]
    assert [event.value for event in events.Scalars("charts/SPS")] == pytest.approx([5 / 30, 5 / 30, 3 / 5])
    assert events.Scalars("time/eta_seconds")[-1].value == 0
    assert events.Scalars("model/params_millions")[0].value == pytest.approx(1e-6)
    for tag in ("charts/SPS", "time/elapsed_seconds"):
        assert all(np.isfinite(event.value) and event.value >= 0 for event in events.Scalars(tag))
    text = events.Tensors("config/text_summary")[0].tensor_proto.string_val[0].decode()
    saved_config = yaml.safe_load(text.removeprefix("```yaml\n").removesuffix("```"))
    assert saved_config["log_dir"] == str(tmp_path / "test-run" / "chess")
    assert saved_config["run_id"] == "test-run"
    assert saved_config["ckpt_dir"] == str(tmp_path / "checkpoints" / "test-run")
    az.checkpoint_manager.assert_called_once_with(str(tmp_path / "checkpoints" / "test-run"))
    assert "devices/text_summary" in events.Tags()["tensors"]
    messages = [record.getMessage() for record in caplog.records if record.name == az.__name__]
    assert "Run ID: test-run" in messages
    reports = [message for message in messages if message.startswith("Iteration ")]
    assert len(reports) == 3
    for report, iteration, steps, loss, positions, seconds in zip(
        reports, [2, 4, 5], [5, 10, 13], [4, 12, 18], [5, 5, 3], [30, 30, 5], strict=True
    ):
        assert report.startswith(f"Iteration {iteration}/5 | steps={steps} | ")
        assert f" | loss={loss} | " in report
        assert f" | positions={positions} | " in report
        assert report.endswith(f"seconds={seconds:.2f}")


def test_tensorboard_closes_on_error(logging_config: az.Config, monkeypatch: pytest.MonkeyPatch) -> None:
    logging_config = replace(logging_config, log_dir="logs/${run_id}", ckpt_dir="checkpoints/${run_id}")
    writer = MagicMock()
    writer.__enter__.return_value = writer
    make_writer = Mock(return_value=writer)
    monkeypatch.setattr(az, "SummaryWriter", make_writer)
    monkeypatch.setattr(az, "collect_selfplay", Mock(side_effect=RuntimeError("search failed")))
    with pytest.raises(RuntimeError, match="search failed"):
        az.train(logging_config)
    writer.__exit__.assert_called_once()
    assert writer.__exit__.call_args.args[0] is RuntimeError
    az.checkpoint_manager.return_value.__exit__.assert_called_once()
    run_dir = Path(make_writer.call_args.kwargs["logdir"])
    assert run_dir == Path("logs/${run_id}")
    az.checkpoint_manager.assert_called_once_with("checkpoints/${run_id}")


def test_checkpoint_cadence_and_resume(logging_config: az.Config, monkeypatch: pytest.MonkeyPatch) -> None:
    config = replace(logging_config, iterations=4, updates_per_iteration=1, run_id="resume")
    assert config.checkpoint_interval_seconds == 600.0
    clock = 0.0
    interrupted = False
    original_step = az.train_step

    def timed_step(state: TrainState, batch: az.Batch, *, model: az.PolicyValueNet) -> tuple[TrainState, az.Metrics]:
        nonlocal clock, interrupted
        if state.step == 2 and not interrupted:
            interrupted = True
            raise RuntimeError("interrupted")
        clock += 599.0 if state.step == 0 else 1.0
        return original_step(state, batch, model=model)

    def now() -> float:
        return clock

    rollout = Mock(return_value=(None, {"positions": 2, "completed_games": 1, "value_positions": 2}))
    restore = Mock(return_value=None)
    save = Mock()
    writer = MagicMock()
    writer.__enter__.return_value = writer
    make_writer = Mock(return_value=writer)
    monkeypatch.setattr(az, "collect_selfplay", rollout)
    monkeypatch.setattr(az, "train_step", timed_step)
    monkeypatch.setattr(az, "monotonic", now)
    monkeypatch.setattr(az, "restore_checkpoint", restore)
    monkeypatch.setattr(az, "save_checkpoint", save)
    monkeypatch.setattr(az, "SummaryWriter", make_writer)
    with pytest.raises(RuntimeError, match="interrupted"):
        az.train(config)
    save.assert_called_once()
    saved = save.call_args.args[1]
    assert (saved.iteration, saved.steps, saved.completed_games) == (2, 4, 2)
    manager = az.checkpoint_manager.return_value
    manager.__exit__.assert_called_once()
    writer.__exit__.assert_called_once()

    restore.return_value = saved
    writer.reset_mock()
    state = az.train(config)
    assert state.step == 4
    assert [call.args[1].iteration for call in save.call_args_list] == [2, 4]
    final = save.call_args.args[1]
    assert (final.steps, final.completed_games) == (8, 4)
    assert make_writer.call_args.kwargs["purge_step"] == 5
    # Replay the interrupted iteration with the same self-play RNG.
    np.testing.assert_array_equal(rollout.call_args_list[2].args[1], rollout.call_args_list[3].args[1])
    eta = [call.args[1] for call in writer.add_scalar.call_args_list if call.args[0] == "time/eta_seconds"]
    assert eta == [0.0]  # Resumed iterations share a final partial logging window.

    restore.return_value = final
    rollout.reset_mock()
    az.train(config)
    rollout.assert_not_called()
    assert save.call_count == 2


def test_evaluation_cadence_previous_checkpoint_and_resume(
    logging_config: az.Config, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=az.__name__)
    config = replace(logging_config, iterations=6, updates_per_iteration=1)
    clock = 0.0
    original_step = az.train_step

    def now() -> float:
        return clock

    def timed_step(state: TrainState, batch: az.Batch, *, model: az.PolicyValueNet) -> tuple[TrainState, az.Metrics]:
        nonlocal clock
        clock += 600.0
        state, metrics = original_step(state, batch, model=model)
        return state.replace(params={"weight": jnp.full(1, state.step, jnp.float32)}), metrics

    def evaluate(
        candidate_params: az.Parameters,
        opponent_params: az.Parameters,
        *,
        env: pgx.Env,
        model: az.PolicyValueNet,
        config: az.Config,
    ) -> dict[str, float]:
        nonlocal clock
        clock += 120.0  # Evaluation time must not trigger another immediate evaluation.
        return {"score": 0.5, "games": 2.0}

    rollout = Mock(return_value=(None, {"positions": 2, "completed_games": 1, "value_positions": 2}))
    run_evaluation = Mock(side_effect=evaluate)
    save = Mock()
    writer = MagicMock()
    writer.__enter__.return_value = writer
    monkeypatch.setattr(az, "collect_selfplay", rollout)
    monkeypatch.setattr(az, "train_step", timed_step)
    monkeypatch.setattr(az, "monotonic", now)
    monkeypatch.setattr(az, "save_checkpoint", save)
    monkeypatch.setattr(az, "evaluate", run_evaluation)
    monkeypatch.setattr(az, "SummaryWriter", Mock(return_value=writer))
    az.train(config)
    assert run_evaluation.call_count == 2
    for call, candidate, opponent in zip(run_evaluation.call_args_list, [3, 6], [2, 5], strict=True):
        assert call.kwargs["candidate_params"]["weight"].item() == candidate
        assert call.kwargs["opponent_params"]["weight"].item() == opponent
    logged = [call.args for call in writer.add_scalar.call_args_list if call.args[0].startswith("eval/")]
    assert ("eval/opponent_iteration", 2, 6) in logged
    assert ("eval/opponent_iteration", 5, 12) in logged
    assert ("eval/seconds", 120.0, 12) in logged
    assert sum(record.getMessage().startswith("Evaluation | ") for record in caplog.records) == 2
    saved_key = save.call_args.args[1].key
    saved_second_iteration = save.call_args_list[1].args[1]
    training_keys = [call.args[1] for call in rollout.call_args_list]

    # Turning off evaluations by extending the interval must not change training RNG.
    clock = 0.0
    rollout.reset_mock()
    az.train(replace(config, evaluation=EvaluationConfig(interval_seconds=10000.0)))
    for key, call in zip(training_keys, rollout.call_args_list, strict=True):
        np.testing.assert_array_equal(key, call.args[1])
    np.testing.assert_array_equal(saved_key, save.call_args.args[1].key)
    assert run_evaluation.call_count == 2

    # On resume, the restored checkpoint is the first opponent, even when a save is due.
    clock = 0.0
    monkeypatch.setattr(az, "restore_checkpoint", Mock(return_value=saved_second_iteration))
    run_evaluation.reset_mock()
    az.train(replace(config, iterations=3, evaluation=EvaluationConfig(interval_seconds=600.0)))
    run_evaluation.assert_called_once()
    assert run_evaluation.call_args.kwargs["opponent_params"]["weight"].item() == 2
    assert run_evaluation.call_args.kwargs["candidate_params"]["weight"].item() == 3


def test_evaluation_skips_missing_checkpoint(logging_config: az.Config, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = 0.0
    original_step = az.train_step

    def now() -> float:
        return clock

    def timed_step(state: TrainState, batch: az.Batch, *, model: az.PolicyValueNet) -> tuple[TrainState, az.Metrics]:
        nonlocal clock
        clock += 1800.0
        return original_step(state, batch, model=model)

    evaluate = Mock()
    monkeypatch.setattr(az, "monotonic", now)
    monkeypatch.setattr(az, "train_step", timed_step)
    monkeypatch.setattr(az, "evaluate", evaluate)
    az.train(replace(logging_config, iterations=1, updates_per_iteration=1))
    evaluate.assert_not_called()
