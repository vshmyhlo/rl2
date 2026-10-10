from dataclasses import replace
from functools import partial
from math import log, sqrt
from typing import Any
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pgx = pytest.importorskip("pgx")
mctx = pytest.importorskip("mctx")

from rl2.alphazero import evaluation as ev
from rl2.alphazero.config import Config, EvaluationConfig
from rl2.alphazero.model import PolicyValueNet


def test_match_summary_separates_unfinished_games_and_accounts_for_pairs() -> None:
    rewards = jnp.tile(jnp.array([[1, -1], [0, 0]], jnp.float32), (25, 1))
    terminated = jnp.tile(jnp.array([[True, True], [True, False]]), (25, 1))
    metrics = ev.summarize_matches(rewards, terminated)
    assert metrics["games"] == 100
    assert metrics["completed_games"] == 75
    assert [metrics[name] for name in ("wins", "draws", "losses", "truncated_games")] == [25] * 4
    assert metrics["score"] == 0.5
    assert metrics["truncation_rate"] == 0.25
    assert [metrics[name] for name in ("win_rate", "draw_rate", "loss_rate")] == [1 / 3] * 3
    margin = sqrt(log(40) / (2 * 50))  # 50 independent pairs, not 100 independent games.
    assert metrics["score_lower_bound"] == pytest.approx(0.375 - margin)
    assert metrics["score_upper_bound"] == pytest.approx(0.625 + margin)


def test_no_finished_games_has_no_score() -> None:
    metrics = ev.summarize_matches(jnp.zeros((1, 2), jnp.float32), jnp.zeros((1, 2), jnp.bool_))
    assert metrics["draws"] == metrics["completed_games"] == 0
    assert metrics["truncated_games"] == 2
    assert metrics["truncation_rate"] == 1
    assert "score" not in metrics
    assert "draw_rate" not in metrics
    assert (metrics["score_lower_bound"], metrics["score_upper_bound"]) == (0, 1)


def test_openings_are_reproducible_varied_and_live() -> None:
    env = pgx.make("tic_tac_toe")
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    generate = jax.jit(jax.vmap(partial(ev.make_opening, env=env, moves=9)))
    first, second = generate(keys), generate(keys)
    for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        np.testing.assert_array_equal(a, b)
    assert not first.terminated.any()
    assert first.legal_action_mask.any(axis=-1).all()
    assert np.unique(np.asarray(first.observation).reshape(4, -1), axis=0).shape[0] > 1


def test_paired_search_is_reproducible_and_swaps_agents() -> None:
    env = pgx.make("tic_tac_toe")
    model = PolicyValueNet(env.num_actions, channels=2, num_blocks=1)
    observation = env.init(jax.random.PRNGKey(0)).observation[None]
    candidate = model.init(jax.random.PRNGKey(0), observation)["params"]
    opponent = model.init(jax.random.PRNGKey(1), observation)["params"]
    config = Config(
        env_id="tic_tac_toe",
        num_simulations=2,
        max_moves=9,
        evaluation=EvaluationConfig(num_openings=2, openings_per_batch=2, opening_moves=2),
    )
    ids = jnp.arange(2, dtype=jnp.int32)
    match = partial(ev.evaluate_batch, opening_ids=ids, env=env, model=model, config=config)
    rewards, terminated = match(candidate, opponent)
    repeated_rewards, repeated_terminated = match(candidate, opponent)
    swapped_rewards, swapped_terminated = match(opponent, candidate)
    assert terminated.all()
    np.testing.assert_array_equal(rewards, repeated_rewards)
    np.testing.assert_array_equal(terminated, repeated_terminated)
    np.testing.assert_array_equal(rewards, -swapped_rewards[:, ::-1])
    np.testing.assert_array_equal(terminated, swapped_terminated[:, ::-1])
    # Same search shape, but a deliberately insufficient move budget.
    _, unfinished = ev.evaluate_batch(
        candidate, opponent, ids, env=env, model=model, config=replace(config, max_moves=1)
    )
    assert not unfinished.any()


def test_search_uses_acting_network_and_greedy_legal_action(monkeypatch: pytest.MonkeyPatch) -> None:
    env = pgx.make("tic_tac_toe")
    state = env.init(jax.random.PRNGKey(0))
    for action in (0, 3, 1, 4):
        state = env.step(state, jnp.int32(action))
    candidate = {"tag": jnp.float32(1)}
    opponent = {"tag": jnp.float32(2)}
    model = Mock()
    model.apply.return_value = (jnp.zeros((1, 9), jnp.float32), jnp.zeros(1, jnp.float32))
    seen: list[float] = []

    def search(
        params: ev.Parameters,
        key: jax.Array,
        root: mctx.RootFnOutput,
        recurrent_fn: mctx.RecurrentFn,
        num_simulations: int,
        **kwargs: Any,
    ) -> mctx.PolicyOutput:
        seen.append(float(params["tag"]))
        assert num_simulations == 3
        assert kwargs["temperature"] == kwargs["dirichlet_fraction"] == 0
        np.testing.assert_array_equal(kwargs["invalid_actions"], ~state.legal_action_mask[None])
        # Illegal square 0 has the highest weight; legal square 2 wins the game.
        weights = jnp.array([[100, 0, 10, 0, 0, 0, 0, 0, 0]], jnp.float32)
        return mctx.PolicyOutput(action=jnp.array([8], jnp.int32), action_weights=weights, search_tree=None)

    monkeypatch.setattr(ev.mctx, "muzero_policy", search)
    with jax.disable_jit():
        for player, expected_reward in ((state.current_player, 1), (1 - state.current_player, -1)):
            reward, terminated = ev.play_game(
                candidate,
                opponent,
                state,
                player,
                jax.random.PRNGKey(0),
                env=env,
                model=model,
                num_simulations=3,
                max_moves=1,
            )
            assert terminated
            assert reward == expected_reward
    assert seen == [1, 2]


def test_chess_evaluation_search_smoke() -> None:
    env = pgx.make("chess")
    model = PolicyValueNet(env.num_actions, channels=2, num_blocks=1)
    observation = env.init(jax.random.PRNGKey(0)).observation[None]
    params = model.init(jax.random.PRNGKey(0), observation)["params"]
    config = Config(
        num_simulations=1,
        max_moves=1,
        evaluation=EvaluationConfig(num_openings=1, opening_moves=1, openings_per_batch=1),
    )
    metrics = ev.evaluate(params, params, env=env, model=model, config=config)
    assert metrics["games"] == metrics["truncated_games"] == 2
    assert metrics["completed_games"] == 0  # No illegal-action termination during either search.


def test_evaluation_batches_cover_openings_once(monkeypatch: pytest.MonkeyPatch) -> None:
    config = Config(evaluation=EvaluationConfig(num_openings=3, openings_per_batch=2))
    batches: list[list[int]] = []

    def batch(
        candidate: ev.Parameters, opponent: ev.Parameters, ids: jax.Array, **kwargs: Any
    ) -> tuple[jax.Array, jax.Array]:
        batches.append(ids.tolist())
        return jnp.zeros((ids.size, 2), jnp.float32), jnp.ones((ids.size, 2), jnp.bool_)

    monkeypatch.setattr(ev, "evaluate_batch", batch)
    result = ev.evaluate({}, {}, env=Mock(), model=Mock(), config=config)
    assert batches == [[0, 1], [2]]
    assert result["games"] == result["draws"] == 6
    assert result["score"] == 0.5
