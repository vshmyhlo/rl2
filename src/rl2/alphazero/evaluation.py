"""Reproducible, color-paired matches against a frozen checkpoint.

Openings are random legal prefixes generated from a fixed seed, independent of
training RNG and parameters. Both agents search with the same simulation budget,
without root noise, and choose the most-visited legal action (first-index ties).
Each search uses the acting agent's network throughout its tree.
"""

from functools import partial
from math import log, sqrt

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pgx

from rl2.alphazero import train
from rl2.alphazero.config import Config
from rl2.alphazero.model import Parameters, PolicyValueApplyFn


def make_opening(key: jax.Array, *, env: pgx.Env, moves: int) -> pgx.State:
    """Generate a live opening; reject moves that would end the game."""
    init_key, move_key = jax.random.split(key)
    env_state = env.init(init_key)

    def advance(move: int, env_state: pgx.State) -> pgx.State:
        logits = jnp.where(env_state.legal_action_mask, 0.0, -jnp.inf)
        action = jax.random.categorical(jax.random.fold_in(move_key, move), logits).astype(jnp.int32)
        next_env_state = env.step(env_state, action)

        def keep_live(old: jax.Array, new: jax.Array) -> jax.Array:
            return jnp.where(next_env_state.terminated | next_env_state.truncated, old, new)

        return jax.tree.map(keep_live, env_state, next_env_state)

    return jax.lax.fori_loop(0, moves, advance, env_state)


def play_game(
    candidate_params: Parameters,
    opponent_params: Parameters,
    env_state: pgx.State,  # Unbatched environment state leaves.
    candidate_player: jax.Array,  # ()
    key: jax.Array,  # (2,)
    *,
    env: pgx.Env,
    apply_fn: PolicyValueApplyFn,
    num_simulations: int,
    max_moves: int,
) -> tuple[jax.Array, jax.Array]:  # (), ()
    """Return scalar candidate reward and termination flag; false means unfinished."""

    def ongoing(carry: tuple[pgx.State, jax.Array]) -> jax.Array:
        env_state, move = carry
        return ~(env_state.terminated | env_state.truncated) & (move < max_moves)

    def advance(carry: tuple[pgx.State, jax.Array]) -> tuple[pgx.State, jax.Array]:
        env_state, move = carry

        def choose(candidate: jax.Array, opponent: jax.Array) -> jax.Array:
            return jnp.where(env_state.current_player == candidate_player, candidate, opponent)

        params = jax.tree.map(choose, candidate_params, opponent_params)

        def predict(observation: jax.Array) -> tuple[jax.Array, jax.Array]:  # (*O,) -> (A,), ()
            logits, value = apply_fn({"params": params}, observation[None])  # (1, A), (1,)
            chex.assert_rank(logits, 2)
            chex.assert_shape(logits, (1, logits.shape[-1]))
            chex.assert_shape(value, (1,))
            return logits[0], value[0]  # (A,), ()

        policy = train.puct_policy(
            predict,
            jax.random.fold_in(key, move),
            env_state,
            partial(train.transition_step, env=env),
            num_simulations,
            temperature=0.0,
        )
        # Explicit argmax avoids random tie breaking in the policy's sampled action.
        weights = jnp.where(env_state.legal_action_mask, policy.action_weights, -jnp.inf)  # (A,)
        action = jnp.argmax(weights).astype(jnp.int32)
        return env.step(env_state, action), move + 1

    final, _ = jax.lax.while_loop(ongoing, advance, (env_state, jnp.int32(0)))
    reward = final.rewards[candidate_player]
    return reward, final.terminated


@partial(jax.jit, static_argnames=("env", "apply_fn", "config"))
def evaluate_batch(
    candidate_params: Parameters,
    opponent_params: Parameters,
    opening_ids: jax.Array,  # (K,), opening count K.
    *,
    env: pgx.Env,
    apply_fn: PolicyValueApplyFn,
    config: Config,
) -> tuple[jax.Array, jax.Array]:  # (K, P), (K, P); P = 2 players.
    """Return (K, P) rewards and termination flags for K openings and P = 2 players."""
    seed = jax.random.PRNGKey(config.evaluation.seed)

    def play_pair(opening_id: jax.Array) -> tuple[jax.Array, jax.Array]:
        opening_key, search_key = jax.random.split(jax.random.fold_in(seed, opening_id))
        env_state = make_opening(opening_key, env=env, moves=config.evaluation.opening_moves)
        game = partial(
            play_game,
            candidate_params,
            opponent_params,
            env_state,
            key=search_key,
            env=env,
            apply_fn=apply_fn,
            num_simulations=config.num_simulations,
            max_moves=config.max_moves,
        )
        # PGX player IDs may be randomly assigned to colors. Using both IDs
        # on the identical state guarantees that the agents swap colors.
        rewards, terminated = jax.vmap(game)(jnp.arange(2, dtype=jnp.int32))
        return rewards, terminated

    rewards, terminated = jax.vmap(play_pair)(opening_ids)
    return rewards, terminated


def summarize_matches(rewards: jax.Array, terminated: jax.Array) -> dict[str, float]:
    """Score finished games; bound all-game score using independent opening pairs.

    The conservative 95% Hoeffding bounds treat each opening pair as one sample,
    and allow unfinished games any score in [0, 1]. They quantify variation across
    sampled openings, not strength against other opponents or opening distributions.
    No score/rates are emitted when no game finished; counts and bounds still are.
    """
    chex.assert_scalar_positive(rewards.shape[0])
    values, finished = np.asarray(rewards), np.asarray(terminated)
    wins = int(((values > 0) & finished).sum())
    draws = int(((values == 0) & finished).sum())
    losses = int(((values < 0) & finished).sum())
    games = values.size
    completed = wins + draws + losses
    unfinished = games - completed
    points = wins + 0.5 * draws
    margin = sqrt(log(2 / 0.05) / (2 * values.shape[0]))
    metrics = {
        "games": float(games),
        "completed_games": float(completed),
        "wins": float(wins),
        "draws": float(draws),
        "losses": float(losses),
        "truncated_games": float(unfinished),
        "truncation_rate": unfinished / games,
        "score_lower_bound": max(0.0, points / games - margin),
        "score_upper_bound": min(1.0, (points + unfinished) / games + margin),
    }
    if completed:
        metrics.update(
            score=points / completed,
            win_rate=wins / completed,
            draw_rate=draws / completed,
            loss_rate=losses / completed,
        )
    return metrics


def evaluate(
    candidate_params: Parameters,
    opponent_params: Parameters,
    *,
    env: pgx.Env,
    apply_fn: PolicyValueApplyFn,
    config: Config,
) -> dict[str, float]:
    """Run a fixed suite in bounded batches without consuming training RNG."""
    rewards, terminated = [], []
    evaluation = config.evaluation
    for start in range(0, evaluation.num_openings, evaluation.openings_per_batch):
        ids = jnp.arange(start, min(start + evaluation.openings_per_batch, evaluation.num_openings), dtype=jnp.int32)
        batch_rewards, batch_terminated = evaluate_batch(
            candidate_params, opponent_params, ids, env=env, apply_fn=apply_fn, config=config
        )
        rewards.append(batch_rewards)
        terminated.append(batch_terminated)
    return summarize_matches(jnp.concatenate(rewards), jnp.concatenate(terminated))
