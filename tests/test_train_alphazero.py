import logging
import sys
from collections.abc import Generator
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
from jax.sharding import AxisType, Mesh

pytest.importorskip("pgx")
import pgx

from rl2.alphazero import train as az
from rl2.alphazero.config import ModelConfig


@pytest.fixture(autouse=True)
def single_device_mesh() -> Generator[None]:
    with jax.set_mesh(Mesh(devices=jax.devices()[:1], axis_names=("batch",), axis_types=(AxisType.Auto,))):
        yield


@pytest.fixture(scope="module")
def tiny_model() -> tuple[pgx.Env, az.PolicyValueNet, az.Parameters]:
    env = pgx.make("tic_tac_toe")
    model = az.PolicyValueNet(env.num_actions, channels=4, num_blocks=1)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 3, 3, 2), jnp.float32))["params"]
    return env, model, params


def test_sample_batch_excludes_padding() -> None:
    batch = az.Batch(
        observation=jnp.zeros((1, 2, 3, 3, 2), jnp.float32),  # (B, T, *O)
        action_weights=jnp.full((1, 2, 9), 1 / 9, jnp.float32),  # (B, T, A)
        value_targets=jnp.array([[1.0, -1.0]], jnp.float32),  # (B, T)
        policy_mask=jnp.array([[True, False]]),  # (B, T)
        value_mask=jnp.array([[True, False]]),  # (B, T)
    )
    sampled = az.sample_batch(batch, jax.random.PRNGKey(0), batch_size=4)
    assert sampled.observation.shape == (4, 3, 3, 2)
    assert sampled.policy_mask.all()
    assert sampled.value_mask.all()
    np.testing.assert_array_equal(sampled.value_targets, np.ones(4))


def test_sample_batch_sparse_positions_preserves_fields() -> None:
    # Leading, interior, and trailing padding; valid targets have distinct labels.
    labels = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)  # (B, T)
    batch = az.Batch(
        observation=labels[..., None, None, None],  # (B, T, 1, 1, 1)
        action_weights=labels[..., None],  # (B, T, 1)
        value_targets=labels,  # (B, T)
        policy_mask=jnp.array([[False, True, False], [True, False, False]]),  # (B, T)
        value_mask=jnp.array([[False, True, False], [False, False, False]]),  # (B, T)
    )
    sampled = az.sample_batch(batch, jax.random.PRNGKey(0), batch_size=128)
    assert sampled.observation.shape == (128, 1, 1, 1)
    assert sampled.action_weights.shape == (128, 1)
    chex.assert_shape((sampled.value_targets, sampled.policy_mask, sampled.value_mask), (128,))
    assert sampled.policy_mask.all()
    np.testing.assert_array_equal(sampled.observation[:, 0, 0, 0], sampled.value_targets)
    np.testing.assert_array_equal(sampled.action_weights[:, 0], sampled.value_targets)
    np.testing.assert_array_equal(sampled.value_mask, sampled.value_targets == 1)
    labels, counts = np.unique(sampled.value_targets, return_counts=True)
    np.testing.assert_array_equal(labels, [1, 3])
    # Generous deterministic bounds catch severe sampling bias without seed sweeps.
    assert np.all((counts > 32) & (counts < 96))


@pytest.mark.parametrize("fraction", [0.0, 0.25, 1.0], ids=["disabled", "mixture", "noise-only"])
def test_prepare_logits_uses_only_legal_actions(fraction: float, monkeypatch: pytest.MonkeyPatch) -> None:
    logits = jnp.log(jnp.array([3, 1, 500, 500], jnp.float32))  # (A,)
    legal = jnp.array([True, True, False, False])  # (A,)
    # Most Gamma mass is on illegal actions; it must not dilute the noise weight.
    loggamma = Mock(return_value=jnp.log(jnp.array([1, 3, 1000, 1000], jnp.float32)))
    monkeypatch.setattr(az.jax.random, "loggamma", loggamma)
    prepare = partial(az.prepare_logits, dirichlet_alpha=0.3, dirichlet_fraction=fraction)
    noisy = prepare(logits, legal, jax.random.PRNGKey(0))  # (A,)
    assert noisy.shape == (4,)
    assert noisy.dtype == jnp.float32
    np.testing.assert_allclose(jax.nn.softmax(noisy), [0.75 - 0.5 * fraction, 0.25 + 0.5 * fraction, 0, 0], rtol=1e-6)
    single = prepare(logits, jnp.array([False, False, True, False]), jax.random.PRNGKey(0))  # (A,)
    empty = prepare(logits, jnp.zeros(4, bool), jax.random.PRNGKey(0))  # (A,)
    np.testing.assert_array_equal(jax.nn.softmax(single), [0, 0, 1, 0])
    np.testing.assert_array_equal(jax.nn.softmax(empty), [0.25] * 4)
    if fraction == 0.0:
        loggamma.assert_not_called()


def test_masked_logits() -> None:
    # A large illegal logit must not erase differences between legal logits.
    logits = jnp.array([[1.0, 1e8, 2.0], [0.0, 0.0, 0.0]], jnp.float32)  # (B, A)
    legal = jnp.array([[True, False, True], [False, False, False]])  # (B, A)
    masked = jax.vmap(az.masked_logits)(logits, legal)  # (B, A)
    assert masked.shape == logits.shape
    assert masked.dtype == jnp.float32
    probs = jax.nn.softmax(masked)  # (B, A)
    np.testing.assert_allclose(probs, [[1 / (1 + np.e), 0, 1 / (1 + np.exp(-1))], [1 / 3] * 3], rtol=1e-6)


@pytest.mark.parametrize(
    ("terminated", "truncated", "policy_mask", "rewards", "expected"),
    [
        pytest.param(True, False, [True, True, True], [-1.0, 1.0], [1.0, -1.0, 1.0], id="last-move-win"),
        pytest.param(True, False, [True, True, False], [0.0, 0.0], [0.0] * 3, id="natural-draw"),
        pytest.param(False, False, [True, True, True], [1.0, -1.0], [0.0] * 3, id="move-limit-draw"),
        pytest.param(False, True, [True, True, False], [1.0, -1.0], [0.0] * 3, id="truncated"),
    ],
)
def test_outcomes_use_player_ids_and_adjudicate_move_limit_draws(
    terminated: bool,
    truncated: bool,
    policy_mask: list[bool],
    rewards: list[float],
    expected: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = pgx.make("tic_tac_toe")
    config = az.Config(max_moves=3)
    state = env.init(jax.random.PRNGKey(0)).replace(
        rewards=jnp.array(rewards),  # (P,)
        terminated=jnp.bool_(terminated),  # ()
        truncated=jnp.bool_(truncated),  # ()
    )
    rollout = az.SelfplayStep(
        observation=jnp.zeros((3, 3, 3, 2)),  # (T, *O)
        action_weights=jax.nn.one_hot(jnp.arange(3), env.num_actions),  # (T, A)
        current_player=jnp.array([1, 0, 1]),  # (T,)
        policy_mask=jnp.array(policy_mask),  # (T,)
    )
    monkeypatch.setattr(jax.lax, "scan", Mock(return_value=(state, rollout)))
    batch, metrics = az.collect_one_game({}, jax.random.PRNGKey(0), env=env, apply_fn=Mock(), config=config)
    chex.assert_shape((batch.value_targets, batch.policy_mask, batch.value_mask), (3,))
    np.testing.assert_array_equal(batch.observation, rollout.observation)
    np.testing.assert_array_equal(batch.action_weights, rollout.action_weights)
    np.testing.assert_array_equal(batch.value_targets, expected)
    np.testing.assert_array_equal(batch.value_mask, jnp.array(policy_mask) & ~jnp.bool_(truncated))
    np.testing.assert_array_equal(batch.policy_mask, policy_mask)
    assert metrics.value_target_count == (0 if truncated else sum(policy_mask))
    assert metrics.environment_steps == sum(policy_mask)
    assert metrics.completed_games == int(not truncated)


def test_transition_step_terminal_reward_and_opponent_discount(
    tiny_model: tuple[pgx.Env, az.PolicyValueNet, az.Parameters],
) -> None:
    env, model, params = tiny_model
    state = env.init(jax.random.PRNGKey(0))
    # Player to move can win on square 2 after these four legal moves.
    for action in (0, 3, 1, 4):
        state = env.step(state, jnp.int32(action))

    def duplicate(x: jax.Array) -> jax.Array:
        return jnp.stack((x, x))

    def predict(observation: jax.Array) -> tuple[jax.Array, jax.Array]:  # (*O,) -> (A,), ()
        logits, value = model.apply({"params": params}, observation[None])  # (1, A), (1,)
        chex.assert_rank(logits, 2)
        chex.assert_shape(logits, (1, logits.shape[-1]))
        chex.assert_shape(value, (1,))
        return logits[0], value[0]  # (A,), ()

    output = jax.jit(jax.vmap(partial(az.transition_step, predict, env=env)))(
        jax.tree.map(duplicate, state), jnp.array([2, 8], jnp.int32)
    )
    child_states = output.child_env_state  # Leaves: (B, ...)
    chex.assert_shape((output.reward, output.discount, output.child_value), (2,))
    chex.assert_shape(output.child_logits, (2, env.num_actions))
    chex.assert_shape(child_states.observation, (2, *state.observation.shape))
    chex.assert_shape(child_states.legal_action_mask, (2, env.num_actions))
    np.testing.assert_array_equal(child_states.terminated, [True, False])
    np.testing.assert_array_equal(output.reward, [1, 0])
    np.testing.assert_array_equal(output.discount, [0, -1])
    assert output.child_value[0] == 0
    _, expected_values = model.apply({"params": params}, child_states.observation)
    np.testing.assert_allclose(output.child_value[1], expected_values[1], atol=1e-7)
    probs = jax.nn.softmax(output.child_logits)
    assert np.all(np.asarray(probs[1])[~np.asarray(child_states.legal_action_mask[1])] == 0)


def test_collect_selfplay_matches_independent_games(
    tiny_model: tuple[pgx.Env, az.PolicyValueNet, az.Parameters],
) -> None:
    env, model, params = tiny_model
    config = az.Config(num_envs=2, max_moves=2, num_simulations=2)
    key = jax.random.PRNGKey(5)  # (2,)
    keys = jax.random.split(key, config.num_envs)  # (B, 2)
    collect = jax.jit(partial(az.collect_one_game, params, env=env, apply_fn=model.apply, config=config))
    games = [collect(game_key) for game_key in keys]

    def stack(*leaves: jax.Array) -> jax.Array:  # (...) -> (B, ...)
        return jnp.stack(leaves)  # (B, ...)

    expected_batch, per_game_metrics = jax.tree.map(stack, *games)
    batch, metrics = az.collect_selfplay(params, key, env=env, apply_fn=model.apply, config=config)
    chex.assert_trees_all_close(batch, expected_batch, atol=1e-6)
    chex.assert_trees_all_equal(metrics, jax.tree.map(jnp.sum, per_game_metrics))
    assert games[0][0].observation.shape == (2, 3, 3, 2)
    assert games[0][0].action_weights.shape == (2, 9)
    chex.assert_shape((games[0][0].value_targets, games[0][0].policy_mask, games[0][0].value_mask), (2,))
    chex.assert_shape(games[0][1], ())
    assert batch.observation.shape == (2, 2, 3, 3, 2)
    chex.assert_shape(metrics, ())


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
    batch, metrics = az.collect_selfplay(params, jax.random.PRNGKey(2), env=env, apply_fn=model.apply, config=config)
    size = config.max_moves * config.num_envs
    chex.assert_shape(batch.observation, (config.num_envs, config.max_moves, 3, 3, 2))
    chex.assert_shape(batch.action_weights, (config.num_envs, config.max_moves, 9))
    chex.assert_shape((batch.value_targets, batch.policy_mask, batch.value_mask), (config.num_envs, config.max_moves))
    chex.assert_type((batch.observation, batch.action_weights, batch.value_targets), jnp.float32)
    chex.assert_type((batch.policy_mask, batch.value_mask), jnp.bool_)
    assert metrics.completed_games == 2
    assert 10 <= metrics.environment_steps <= 18
    np.testing.assert_array_equal(batch.value_mask, batch.policy_mask)
    assert not batch.policy_mask[:, -1].any()
    np.testing.assert_allclose(batch.action_weights.sum(-1), 1.0)
    occupied = np.asarray(batch.observation).any(axis=-1).reshape(config.num_envs, config.max_moves, 9)  # (B, T, A)
    real_positions = np.asarray(batch.policy_mask)  # (B, T)
    assert np.all(np.asarray(batch.action_weights)[occupied & real_positions[..., None]] == 0)
    # Every valid move has the game's result, alternating with the player to move.
    for game in range(config.num_envs):
        game_mask = real_positions[game]  # (T,)
        targets = np.asarray(batch.value_targets)[game, game_mask]  # (valid_moves,)
        np.testing.assert_array_equal(targets[:-1], -targets[1:])
        assert targets[-1] in (0, 1)

    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    updated, _, losses = az.train_step(state, jax.random.PRNGKey(0), batch, batch_size=4, updates_per_iteration=1)
    assert updated.step == 1
    assert isinstance(losses, az.UpdateMetrics)
    chex.assert_shape(losses, ())
    assert losses.value_update_count == 1
    assert all(np.isfinite(value) for value in losses)
    assert any(
        np.any(np.asarray(before) != np.asarray(after))
        for before, after in zip(jax.tree.leaves(params), jax.tree.leaves(updated.params), strict=True)
    )

    # Padding must not train either head; masked losses have zero gradient.
    def flatten(x: jax.Array) -> jax.Array:  # (B, T, ...) -> (B * T, ...)
        return x.reshape((size, *x.shape[2:]))  # (B * T, ...)

    empty = jax.tree.map(flatten, batch).replace(
        policy_mask=jnp.zeros(size, bool), value_mask=jnp.zeros(size, bool)
    )  # Leaves: (B * T, ...).
    (loss, empty_metrics), grads = jax.value_and_grad(az.loss_fn, has_aux=True)(params, state.apply_fn, empty)
    assert loss == 0
    assert empty_metrics.value_update_count == 0
    assert all(np.all(np.asarray(leaf) == 0) for leaf in jax.tree.leaves(grads))


def test_chess_search_smoke() -> None:
    env = pgx.make("chess")
    config = az.Config(num_envs=1, max_moves=1, num_simulations=1, model=ModelConfig(channels=4, num_blocks=1))
    model = az.PolicyValueNet(env.num_actions, channels=4, num_blocks=1)
    observation = env.init(jax.random.PRNGKey(0)).observation[None]  # (1, 8, 8, 119)
    params = model.init(jax.random.PRNGKey(0), observation)["params"]
    batch, metrics = az.collect_selfplay(params, jax.random.PRNGKey(0), env=env, apply_fn=model.apply, config=config)
    assert batch.observation.shape == (1, 1, *observation.shape[1:])
    assert batch.action_weights.shape == (1, 1, 4672)
    assert metrics.environment_steps == 1
    chex.assert_shape((batch.value_targets, batch.policy_mask, batch.value_mask), (1, 1))
    assert metrics.completed_games == 1
    assert metrics.value_target_count == 1
    assert batch.value_mask.all()
    assert batch.policy_mask.all()
    np.testing.assert_array_equal(batch.value_targets, [[0.0]])
    legal = env.init(jax.random.PRNGKey(0)).legal_action_mask  # (4672,)
    assert np.all(np.asarray(batch.action_weights[0, 0])[~np.asarray(legal)] == 0)
    np.testing.assert_allclose(batch.action_weights.sum(), 1.0)


def test_main_wiring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("env_id: tic_tac_toe\niterations: 1\nrun_id: test\nckpt_dir: runs/alphazero/${run_id}\n")
    train = Mock()
    monkeypatch.setattr(az, "train", train)
    monkeypatch.setattr(sys, "argv", ["rl2.alphazero.train", "--config", str(path)])
    az.main()
    train.assert_called_once_with(
        replace(
            az.Config(),
            env_id="tic_tac_toe",
            iterations=1,
            run_id="test",
            ckpt_dir="runs/alphazero/test",
        )
    )


def test_main_rejects_cli_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    train = Mock()
    monkeypatch.setattr(az, "train", train)
    monkeypatch.setattr(sys, "argv", ["rl2.alphazero.train", "--iterations", "1"])
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
    monkeypatch.setattr(sys, "argv", ["rl2.alphazero.train", "--config", str(path)])
    with pytest.raises(SystemExit) as error:
        az.main()
    assert error.value.code == 2
    assert "num_envs" in capsys.readouterr().err
    train.assert_not_called()


@pytest.fixture
def logging_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> az.Config:
    """Exercise the training loop's logging without running a model or search."""
    monkeypatch.setattr(az, "training_mesh", Mock(return_value=jax.get_mesh()))
    config = az.Config(
        iterations=2,
        log_interval=2,
        updates_per_iteration=2,
        run_id="test",
        ckpt_dir=str(tmp_path / "checkpoints" / "test"),
    )
    writer = MagicMock()
    writer.__enter__.return_value = writer
    monkeypatch.setattr(az, "MLflowLogger", Mock(return_value=writer))
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
    rollouts = [
        (
            None,
            az.RolloutMetrics(
                environment_steps=jnp.int32(3), completed_games=jnp.int32(1), value_target_count=jnp.int32(2)
            ),
        ),
        (
            None,
            az.RolloutMetrics(
                environment_steps=jnp.int32(2), completed_games=jnp.int32(0), value_target_count=jnp.int32(0)
            ),
        ),
    ]
    monkeypatch.setattr(az, "collect_selfplay", Mock(side_effect=rollouts))

    def fake_step(
        state: TrainState,
        key: jax.Array,  # (2,)
        batch: az.Batch,  # Leaves: (B, T, ...).
        *,
        batch_size: int,
        updates_per_iteration: int,
    ) -> tuple[TrainState, jax.Array, az.UpdateMetrics]:
        loss = jnp.float32(updates_per_iteration * (2 * state.step + updates_per_iteration))  # ()
        for _ in range(updates_per_iteration):
            key, _ = jax.random.split(key)  # Each: (2,)
        return (
            state.replace(step=state.step + updates_per_iteration),
            key,  # (2,)
            az.UpdateMetrics(
                loss=loss,  # ()
                policy_loss=loss,  # ()
                value_loss=jnp.float32(0),  # ()
                value_update_count=jnp.int32(updates_per_iteration),  # ()
            ),
        )

    monkeypatch.setattr(az, "train_step", fake_step)
    return config


def test_mlflow_progress(
    logging_config: az.Config, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    caplog.set_level(logging.INFO, logger=az.__name__)
    logging_config = replace(
        logging_config,
        iterations=5,
        run_id="test-run",
        ckpt_dir=str(tmp_path / "checkpoints" / "test-run"),
        log_interval=2,
    )
    clock = 0.0
    durations = [600.0, 1.0, 2.0, 3.0, 5.0]
    original_step = az.train_step
    rollouts = list(az.collect_selfplay.side_effect)
    az.collect_selfplay.side_effect = rollouts * 2 + rollouts[:1]

    def timed_step(
        state: TrainState,
        key: jax.Array,  # (2,)
        batch: az.Batch,  # Leaves: (B, T, ...).
        *,
        batch_size: int,
        updates_per_iteration: int,
    ) -> tuple[TrainState, jax.Array, az.UpdateMetrics]:
        nonlocal clock
        clock += durations[state.step // updates_per_iteration]
        return original_step(state, key, batch, batch_size=batch_size, updates_per_iteration=updates_per_iteration)

    def now() -> float:
        return clock

    monkeypatch.setattr(az, "train_step", timed_step)
    monkeypatch.setattr(az, "monotonic", now)
    transfers: list[int] = []
    device_get = jax.device_get

    def get_metrics(metrics: az.WindowMetrics) -> chex.ArrayTree:  # Scalar leaves: () -> ().
        assert isinstance(metrics, az.WindowMetrics)
        assert isinstance(metrics.update_metrics, az.UpdateMetrics)
        chex.assert_shape(jax.tree.leaves(metrics), ())
        transfers.append(az.collect_selfplay.call_count)
        return device_get(metrics)

    monkeypatch.setattr(az.jax, "device_get", get_metrics)
    state = az.train(logging_config)
    assert transfers == [2, 4, 5]
    assert state.step == 10
    for call in az.collect_selfplay.call_args_list:
        assert call.kwargs["apply_fn"] is state.apply_fn
    writer = az.MLflowLogger.return_value
    az.MLflowLogger.assert_called_once_with(experiment="alphazero", run_name="test-run")
    metrics: dict[str, list[tuple[float, int]]] = {}
    for call in writer.log_metrics.call_args_list:
        for tag, value in call.args[0].items():
            metrics.setdefault(tag, []).append((value, call.args[1]))
    assert metrics["train/loss"] == [(4, 5), (12, 10), (18, 13)]
    assert metrics["train/progress_fraction"] == [(0.4, 5), (0.8, 10), (1.0, 13)]
    assert metrics["train/value_loss"] == [(0, 5), (0, 10), (0, 13)]
    for tag, values in {
        "rollout/completed_games_total": [1, 2, 3],
        "rollout/completed_games": [1, 1, 1],
        "rollout/environment_steps": [5, 5, 3],
        "rollout/value_target_count": [2, 2, 2],
        "train/iteration": [2, 4, 5],
        "train/window_seconds": [601, 5, 5],
        "train/iteration_seconds": [300.5, 2.5, 5],
    }.items():
        assert [value for value, _ in metrics[tag]] == values
    assert "time/rollout_seconds" not in metrics
    assert "time/optimization_seconds" not in metrics
    assert [value for value, _ in metrics["train/environment_steps_per_second"]] == pytest.approx(
        [5 / 601, 5 / 5, 3 / 5]
    )
    assert "train/elapsed_seconds" not in metrics
    assert "train/eta_seconds" not in metrics
    assert metrics["model/params_millions"][0][0] == pytest.approx(1e-6)
    artifacts = {call.args[1]: call.args[0] for call in writer.log_text.call_args_list}
    saved_config = yaml.safe_load(artifacts["config.yaml"])
    assert saved_config["run_id"] == "test-run"
    assert saved_config["ckpt_dir"] == str(tmp_path / "checkpoints" / "test-run")
    writer.log_params.assert_called_once_with(saved_config)
    az.checkpoint_manager.assert_called_once_with(str(tmp_path / "checkpoints" / "test-run"))
    assert "devices.txt" not in artifacts
    messages = [record.getMessage() for record in caplog.records if record.name == az.__name__]
    assert "Run ID: test-run" in messages
    reports = [message for message in messages if message.startswith("Iteration ")]
    assert len(reports) == 3
    for report, iteration, steps, loss, environment_steps, seconds in zip(
        reports, [2, 4, 5], [5, 10, 13], [4, 12, 18], [5, 5, 3], [601, 5, 5], strict=True
    ):
        assert report.startswith(f"Iteration {iteration}/5 | steps={steps} | ")
        assert f" | loss={loss} | " in report
        assert f" | environment_steps={environment_steps} | " in report
        assert report.endswith(f"seconds={seconds:.2f}")


def test_mlflow_closes_on_error(logging_config: az.Config, monkeypatch: pytest.MonkeyPatch) -> None:
    logging_config = replace(logging_config, ckpt_dir="checkpoints/${run_id}")
    writer = MagicMock()
    writer.__enter__.return_value = writer
    make_writer = Mock(return_value=writer)
    monkeypatch.setattr(az, "MLflowLogger", make_writer)
    monkeypatch.setattr(az, "collect_selfplay", Mock(side_effect=RuntimeError("search failed")))
    with pytest.raises(RuntimeError, match="search failed"):
        az.train(logging_config)
    writer.__exit__.assert_called_once()
    assert writer.__exit__.call_args.args[0] is RuntimeError
    az.checkpoint_manager.return_value.__exit__.assert_called_once()
    make_writer.assert_called_once_with(experiment="alphazero", run_name="test")
    az.checkpoint_manager.assert_called_once_with("checkpoints/${run_id}")


@pytest.mark.parametrize("has_targets", [True, False], ids=["mixed-with-measured-zero", "no-value-targets"])
def test_value_loss_logging_omits_updates_without_targets(
    logging_config: az.Config, monkeypatch: pytest.MonkeyPatch, has_targets: bool
) -> None:
    logging_config = replace(logging_config, iterations=4)
    az.collect_selfplay.side_effect = list(az.collect_selfplay.side_effect) * 2
    # The second window has no targets, even if the first window had eligible updates.
    losses = iter(([3.0, 0.0, 0.0, 0.0] if has_targets else [0.0] * 4) + [0.0] * 4)

    def step(
        state: TrainState,
        key: jax.Array,  # (2,)
        batch: az.Batch,  # Leaves: (B, T, ...).
        *,
        batch_size: int,
        updates_per_iteration: int,
    ) -> tuple[TrainState, jax.Array, az.UpdateMetrics]:
        value_loss = jnp.float32(sum(next(losses) for _ in range(updates_per_iteration)))  # ()
        return (
            state.replace(step=state.step + updates_per_iteration),
            key,  # (2,)
            az.UpdateMetrics(
                loss=2.0 * updates_per_iteration + value_loss,  # ()
                policy_loss=jnp.float32(2.0 * updates_per_iteration),  # ()
                value_loss=value_loss,  # ()
                value_update_count=jnp.int32(has_targets and state.step < 4),  # ()
            ),
        )

    def now() -> float:
        return 0.0

    monkeypatch.setattr(az, "train_step", step)
    monkeypatch.setattr(az, "monotonic", now)
    az.train(logging_config)
    first_window, second_window = az.MLflowLogger.return_value.log_metrics.call_args_list[-2:]
    metrics = first_window.args[0]
    assert metrics["train/policy_loss"] == 2.0
    assert metrics["train/loss"] == (2.75 if has_targets else 2.0)
    if has_targets:
        assert metrics["train/value_loss"] == 1.5
    else:
        assert "train/value_loss" not in metrics
    assert "train/value_loss" not in second_window.args[0]
    assert second_window.args[0]["train/loss"] == 2.0


def test_checkpoint_cadence_and_resume(logging_config: az.Config, monkeypatch: pytest.MonkeyPatch) -> None:
    config = replace(
        logging_config, iterations=5, updates_per_iteration=1, run_id="resume", log_interval=3, checkpoint_interval=2
    )
    clock = 0.0
    interrupted = False
    original_step = az.train_step

    def timed_step(
        state: TrainState,
        key: jax.Array,  # (2,)
        batch: az.Batch,  # Leaves: (B, T, ...).
        *,
        batch_size: int,
        updates_per_iteration: int,
    ) -> tuple[TrainState, jax.Array, az.UpdateMetrics]:
        nonlocal clock, interrupted
        if state.step == 2 and not interrupted:
            interrupted = True
            raise RuntimeError("interrupted")
        clock += 1000.0 if state.step == 0 else 1.0
        return original_step(state, key, batch, batch_size=batch_size, updates_per_iteration=updates_per_iteration)

    def now() -> float:
        return clock

    rollout = Mock(
        return_value=(
            None,
            az.RolloutMetrics(
                environment_steps=jnp.int32(2), completed_games=jnp.int32(1), value_target_count=jnp.int32(2)
            ),
        )
    )
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
    monkeypatch.setattr(az, "MLflowLogger", make_writer)
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
    assert state.step == 5
    assert [call.args[1].iteration for call in save.call_args_list] == [2, 4, 5]
    final = save.call_args.args[1]
    assert (final.steps, final.completed_games) == (10, 5)
    assert make_writer.call_args.kwargs["run_name"] == "resume"
    # Replay the interrupted iteration with the same self-play RNG.
    np.testing.assert_array_equal(rollout.call_args_list[2].args[1], rollout.call_args_list[3].args[1])
    iterations = [
        call.args[0]["train/iteration"]
        for call in writer.log_metrics.call_args_list
        if "train/iteration" in call.args[0]
    ]
    assert iterations == [3, 5]  # Regular boundary followed by a final partial logging window.
    assert [
        (call.args[0]["train/progress_fraction"], call.args[1])
        for call in writer.log_metrics.call_args_list
        if "train/progress_fraction" in call.args[0]
    ] == [(0.6, 6), (1.0, 10)]
    for call in writer.log_metrics.call_args_list:
        assert "train/elapsed_seconds" not in call.args[0]
        assert "train/eta_seconds" not in call.args[0]

    restore.return_value = final
    rollout.reset_mock()
    az.train(config)
    rollout.assert_not_called()
    assert save.call_count == 3


def test_evaluation_cadence_previous_checkpoint_and_resume(
    logging_config: az.Config, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=az.__name__)
    config = replace(logging_config, iterations=6, updates_per_iteration=1, checkpoint_interval=2, eval_interval=3)
    clock = 0.0
    original_step = az.train_step

    def now() -> float:
        return clock

    def timed_step(
        state: TrainState,
        key: jax.Array,  # (2,)
        batch: az.Batch,  # Leaves: (B, T, ...).
        *,
        batch_size: int,
        updates_per_iteration: int,
    ) -> tuple[TrainState, jax.Array, az.UpdateMetrics]:
        nonlocal clock
        clock += 1.0
        state, key, metrics = original_step(
            state, key, batch, batch_size=batch_size, updates_per_iteration=updates_per_iteration
        )
        return (
            state.replace(params={"weight": jnp.full(1, state.step, jnp.float32)}),  # weight: (1,)
            key,  # (2,)
            metrics,  # Scalar leaves: ().
        )

    def evaluate(
        candidate_params: az.Parameters,
        opponent_params: az.Parameters,
        *,
        env: pgx.Env,
        apply_fn: az.PolicyValueApplyFn,
        config: az.Config,
    ) -> dict[str, float]:
        nonlocal clock
        assert apply_fn is az.PolicyValueNet.return_value.apply
        clock += 120.0  # Evaluation time does not affect the iteration cadence.
        return {"score": 0.5, "games": 2.0}

    rollout = Mock(
        return_value=(
            None,
            az.RolloutMetrics(
                environment_steps=jnp.int32(2), completed_games=jnp.int32(1), value_target_count=jnp.int32(2)
            ),
        )
    )
    run_evaluation = Mock(side_effect=evaluate)
    save = Mock()
    writer = MagicMock()
    writer.__enter__.return_value = writer
    monkeypatch.setattr(az, "collect_selfplay", rollout)
    monkeypatch.setattr(az, "train_step", timed_step)
    monkeypatch.setattr(az, "monotonic", now)
    monkeypatch.setattr(az, "save_checkpoint", save)
    monkeypatch.setattr(az.evaluation, "evaluate", run_evaluation)
    monkeypatch.setattr(az, "MLflowLogger", Mock(return_value=writer))
    az.train(config)
    assert run_evaluation.call_count == 2
    # The middle logging window includes the evaluation after iteration 3.
    windows = [call.args[0] for call in writer.log_metrics.call_args_list if "train/window_seconds" in call.args[0]]
    assert [window["train/window_seconds"] for window in windows] == [2.0, 122.0, 2.0]
    assert [window["train/iteration_seconds"] for window in windows] == [1.0, 61.0, 1.0]
    for call, candidate, opponent in zip(run_evaluation.call_args_list, [3, 6], [2, 4], strict=True):
        assert call.kwargs["candidate_params"]["weight"].item() == candidate
        assert call.kwargs["opponent_params"]["weight"].item() == opponent
    logged = [
        (tag, value, call.args[1])
        for call in writer.log_metrics.call_args_list
        for tag, value in call.args[0].items()
        if tag.startswith("eval/")
    ]
    assert ("eval/opponent_iteration", 2, 6) in logged
    assert ("eval/opponent_iteration", 4, 12) in logged
    assert ("eval/seconds", 120.0, 12) in logged
    assert sum(record.getMessage().startswith("Evaluation | ") for record in caplog.records) == 2
    saved_key = save.call_args.args[1].key
    saved_second_iteration = save.call_args_list[0].args[1]
    training_keys = [call.args[1] for call in rollout.call_args_list]

    # Turning off evaluations by extending the interval must not change training RNG.
    clock = 0.0
    rollout.reset_mock()
    az.train(replace(config, eval_interval=10))
    for key, call in zip(training_keys, rollout.call_args_list, strict=True):
        np.testing.assert_array_equal(key, call.args[1])
    np.testing.assert_array_equal(saved_key, save.call_args.args[1].key)
    assert run_evaluation.call_count == 2

    # On resume, the restored checkpoint is the first opponent, even when a save is due.
    clock = 0.0
    monkeypatch.setattr(az, "restore_checkpoint", Mock(return_value=saved_second_iteration))
    run_evaluation.reset_mock()
    az.train(replace(config, iterations=3))
    run_evaluation.assert_called_once()
    assert run_evaluation.call_args.kwargs["opponent_params"]["weight"].item() == 2
    assert run_evaluation.call_args.kwargs["candidate_params"]["weight"].item() == 3


def test_evaluation_skips_missing_checkpoint(logging_config: az.Config, monkeypatch: pytest.MonkeyPatch) -> None:
    evaluate = Mock()
    monkeypatch.setattr(az.evaluation, "evaluate", evaluate)
    az.train(replace(logging_config, iterations=1, updates_per_iteration=1, eval_interval=1))
    evaluate.assert_not_called()
