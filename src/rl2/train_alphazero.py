"""Single-device AlphaZero sketch using PGX rules and Mctx PUCT search.

Run: uv run --extra alphazero python -m rl2.train_alphazero --config configs/alphazero_chess.yaml
For a cheap start, set env_id: tic_tac_toe and max_moves: 9 in the YAML config.

Each iteration starts fresh games and samples updates from that rollout only.
Games still active at max_moves are adjudicated as draws and train both heads;
terminal padding trains neither. Orbax checkpoints save every 10 minutes at
iteration boundaries and on normal completion; reusing a run directory resumes
the latest saved iteration.
Progress metrics aggregate over log_interval_seconds (default: 60), emitted
at iteration boundaries and on normal completion.

TensorBoard metrics use cumulative played positions as the step axis, excluding
terminal padding. Each logging window contains complete training iterations:

- losses/policy_loss: Mean policy cross-entropy against MCTS action weights,
  averaged over optimizer updates in the window. Padding is masked out.
- losses/value_loss: Mean squared error against final game outcomes, averaged
  over updates, including zero targets for move-limit draws. Environment-truncated
  games are excluded; an update with no eligible positions contributes zero.
- losses/loss: Sum of policy and value losses, averaged over updates. AdamW
  weight decay is applied by the optimizer and is not included in this loss.
- selfplay/positions: Total played positions collected in the window.
- selfplay/value_positions: Total positions with outcome targets, including move-limit draws,
  collected in the window, before sampling training minibatches.
- selfplay/completed_games: Total completed games in the window, including
  move-limit draws. Environment-truncated games are excluded.
- charts/iteration: Latest completed iteration number, starting at one.
- charts/completed_games: Cumulative completed games, including move-limit draws and restored
  checkpoint progress.
- charts/learning_rate: Configured constant optimizer learning rate.
- charts/SPS: Window positions divided by window wall time, in positions/sec.
- time/iteration_seconds: Sum of iteration wall times in the window, covering
  self-play and optimization; excludes logging, evaluation, and checkpoint saving.
- time/window_seconds: Wall time since the previous window boundary, including
  intervening logging, evaluation, and checkpoint overhead.
- time/elapsed_seconds: Wall time since this training invocation's loop began;
  resets on resume and excludes initialization and checkpoint restoration.
- time/eta_seconds: Estimated remaining training time from the mean iteration
  duration since this invocation began, including intervening overhead.

At startup, model/params_millions records the parameter count in millions;
config and devices record the resolved configuration and JAX devices as text.
Stdout reports the latest iteration and cumulative steps, window self-play
totals, mean losses, and window wall time (seconds).

Every evaluation.interval_seconds (default: 1800), at an iteration boundary,
the current model plays the most recently saved checkpoint, before a new save.
Evaluation pauses training and uses a separate fixed seed: 50 random legal
opening prefixes, each played twice with the agents swapping colors, equal
num_simulations, no root noise, and greedy visit-count actions. max_moves caps
each game after the opening. The timer restarts after evaluation and on resume.
If no checkpoint exists yet, evaluation is skipped until the next interval.

Evaluation metrics are logged immediately under eval/, without window averaging:
- wins, draws, losses, completed_games: Counts from terminated evaluation games.
- games, truncated_games, truncation_rate: Total attempted games and those
  unfinished at the move limit (or truncated by the environment).
- score: (wins + 0.5 * draws) / completed_games; win_rate, draw_rate, loss_rate
  also use completed_games. Omitted when no game finishes. Truncation can bias
  this conditional score, so always inspect truncation_rate alongside it.
- score_lower_bound, score_upper_bound: Conservative 95% bounds for all-game
  score over sampled openings, using opening pairs as independent samples and
  allowing unfinished games any outcome. These can be wide for small suites.
- candidate_iteration, opponent_iteration: Iterations identifying both models.
- seconds: Wall time for the evaluation, including any initial JIT compilation.
Evaluation does not alter training RNG, progress counters, or checkpoint policy.
This scaffold has no persistent replay.
PGX owns observation/action encodings and in-game draw rules. Values always use the
player-to-move perspective. This is not an exact reproduction of the paper.
"""

import argparse
import logging
import sys
from dataclasses import asdict
from functools import partial
from time import monotonic

import jax
import jax.numpy as jnp
import mctx
import optax
import pgx
import yaml
from flax import struct
from flax.training.train_state import TrainState
from omegaconf.errors import OmegaConfBaseException
from tensorboardX import SummaryWriter

from rl2.alphazero.checkpoints import TrainingProgress, checkpoint_manager, restore_checkpoint, save_checkpoint
from rl2.alphazero.config import Config
from rl2.alphazero.evaluation import evaluate
from rl2.alphazero.model import Parameters, PolicyValueNet
from rl2.alphazero.puct import puct_policy
from rl2.alphazero.search import PolicyValueFn, recurrent_step
from rl2.alphazero.utils import masked_logits
from rl2.configuration import load_config

logger = logging.getLogger(__name__)

type Metrics = dict[str, jax.Array]


@struct.dataclass
class Batch:
    # B: batch size; A: number of actions; observation_shape: environment observation dimensions.
    observations: jax.Array  # (B, *observation_shape)
    policy_targets: jax.Array  # (B, A)
    value_targets: jax.Array  # (B,)
    policy_mask: jax.Array  # (B,)
    value_mask: jax.Array  # (B,)


def selfplay_prior_logits(
    logits: jax.Array,
    legal: jax.Array,
    key: jax.Array,
    *,
    dirichlet_alpha: float,
    dirichlet_fraction: float,
) -> jax.Array:
    """Mix the policy with Dirichlet noise normalized over legal actions only.

    Args:
        logits: (B, A) float32 network policy logits.
        legal: (B, A) boolean legal-action mask.
        key: (2,) uint32 PRNG key.
        dirichlet_alpha: Positive concentration for each legal action.
        dirichlet_fraction: Noise mixture weight in [0, 1].

    Returns:
        (B, A) float32 noisy, masked logits; empty legal rows remain finite.
    """
    prior_logits = masked_logits(logits, legal)
    if dirichlet_fraction == 0.0:
        return prior_logits
    # Independent Gamma samples normalized over legal moves form a Dirichlet
    # sample on that subset. Log space avoids underflow for small concentrations.
    log_noise = jax.random.loggamma(key, dirichlet_alpha, shape=logits.shape)
    noise = jax.nn.softmax(masked_logits(log_noise, legal))
    probs = (1.0 - dirichlet_fraction) * jax.nn.softmax(prior_logits) + dirichlet_fraction * noise
    return masked_logits(jnp.log(jnp.maximum(probs, jnp.finfo(probs.dtype).tiny)), legal)


@partial(jax.jit, static_argnames=("env", "model", "config"))
def collect_selfplay(
    params: Parameters,
    key: jax.Array,  # (2,)
    *,
    env: pgx.Env,
    model: PolicyValueNet,
    config: Config,
) -> tuple[Batch, Metrics]:
    """Collect padded games, adjudicating games still active at max_moves as draws.

    B: games; T: max_moves; A: actions; O: environment observation dimensions.
    Environment truncations remain excluded from value supervision.

    Args:
        params: Model parameter pytree.
        key: (2,) uint32 PRNG key.
        env: Game rules and encodings.
        model: Policy and value network.
        config: Self-play settings.

    Returns:
        Batch and metrics, with fields:
            observations: (T * B, *O) float32 recorded observations.
            policy_targets: (T * B, A) float32 MCTS action weights.
            value_targets: (T * B,) float32 player-relative outcomes, zero for cutoff draws.
            policy_mask: (T * B,) bool non-padding positions.
            value_mask: (T * B,) bool positions with terminal or adjudicated outcomes.
            completed_games: () integer count including move-limit draws.
            positions: () integer count of non-padding positions.
            value_positions: () integer count of positions with outcome targets.
    """
    init_key, play_key = jax.random.split(key)  # Each: (2,)
    states = jax.vmap(env.init)(jax.random.split(init_key, config.num_envs))  # Leaves: (B, ...)
    predict = partial(model.apply, {"params": params})

    def recurrent_fn(
        model: PolicyValueFn,
        _key: jax.Array,  # (2,)
        actions: jax.Array,  # (B,)
        states: pgx.State,  # Leaves: (B, ...)
    ) -> tuple[mctx.RecurrentFnOutput, pgx.State]:
        # Mctx requires a PRNG key even for deterministic transitions.
        return recurrent_step(model, actions, states, env=env)

    def step(
        states: pgx.State,  # Leaves: (B, ...)
        inputs: tuple[jax.Array, jax.Array],  # (2,) key and () move index.
    ) -> tuple[pgx.State, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
        key, move_i = inputs  # key: (2,); move_i: () scalar move index.
        noise_key, search_key = jax.random.split(key)  # Each: (2,)
        logits, value = predict(states.observation)  # (B, A), (B,)
        done = states.terminated | states.truncated  # (B,)
        root = mctx.RootFnOutput(
            prior_logits=selfplay_prior_logits(
                logits,
                states.legal_action_mask,
                noise_key,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_fraction=config.dirichlet_fraction,
            ),  # (B, A)
            value=jnp.where(done, 0.0, value),  # (B,)
            embedding=states,  # Leaves: (B, ...)
        )
        # PGX supplies exact transitions for Mctx's PUCT, with no learned dynamics.
        policy = puct_policy(
            predict,
            search_key,
            root,
            recurrent_fn,
            config.num_simulations,
            invalid_actions=~states.legal_action_mask,
            temperature=jnp.where(move_i < config.exploration_moves, 1.0, 0.0),
        )
        next_states = jax.vmap(env.step)(states, policy.action)  # Leaves: (B, ...)

        def preserve_finished(
            old: jax.Array,  # (B, ...)
            new: jax.Array,  # (B, ...)
        ) -> jax.Array:  # (B, ...)
            """Preserve (B, ...) state leaves for games that have already stopped."""
            # Preserve final rewards: PGX otherwise clears them on later steps.
            return jnp.where(done.reshape((config.num_envs,) + (1,) * (old.ndim - 1)), old, new)  # (B, ...)

        next_states = jax.tree.map(preserve_finished, states, next_states)  # Leaves: (B, ...)

        return next_states, (
            states.observation.astype(jnp.float32),  # (B, *O)
            policy.action_weights,  # (B, A)
            states.current_player,  # (B,)
            ~done,  # (B,)
        )

    # Recorded arrays: (T, B, *O), (T, B, A), (T, B), (T, B).
    states, (observations, policies, players, valid) = jax.lax.scan(
        step,
        states,
        (jax.random.split(play_key, config.max_moves), jnp.arange(config.max_moves, dtype=jnp.int32)),
    )
    # T: recorded time steps; B: games. Map final rewards to each recorded actor.
    values = states.rewards[jnp.arange(config.num_envs)[None, :], players]  # (T, B)
    move_limit_draws = ~(states.terminated | states.truncated)  # (B,)
    completed = states.terminated | move_limit_draws  # (B,)
    value_mask = valid & completed[None, :]  # (T, B)
    # A win on the final allowed move keeps its outcome; only active games draw.
    values = jnp.where(valid & states.terminated[None, :], values, 0.0)  # (T, B)
    size = config.max_moves * config.num_envs
    batch = Batch(
        observations.reshape((size, *observations.shape[2:])),  # (T * B, *O)
        policies.reshape((size, env.num_actions)),  # (T * B, A)
        values.reshape(size),  # (T * B,)
        valid.reshape(size),  # (T * B,)
        value_mask.reshape(size),  # (T * B,)
    )
    return batch, {
        "completed_games": completed.sum(),  # ()
        "positions": valid.sum(),  # ()
        "value_positions": value_mask.sum(),  # ()
    }


def loss_fn(params: Parameters, model: PolicyValueNet, batch: Batch) -> tuple[jax.Array, Metrics]:
    logits, values = model.apply({"params": params}, batch.observations)
    policy_errors = optax.softmax_cross_entropy(logits, batch.policy_targets)
    value_errors = jnp.square(values - batch.value_targets)
    policy_loss = jnp.where(batch.policy_mask, policy_errors, 0.0).sum() / jnp.maximum(batch.policy_mask.sum(), 1)
    value_loss = jnp.where(batch.value_mask, value_errors, 0.0).sum() / jnp.maximum(batch.value_mask.sum(), 1)
    loss = policy_loss + value_loss
    return loss, {"loss": loss, "policy_loss": policy_loss, "value_loss": value_loss}


@partial(jax.jit, static_argnames=("model",))
def train_step(state: TrainState, batch: Batch, *, model: PolicyValueNet) -> tuple[TrainState, Metrics]:
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, model, batch)
    return state.apply_gradients(grads=grads), metrics


@partial(jax.jit, static_argnames=("batch_size",))
def sample_batch(batch: Batch, key: jax.Array, batch_size: int) -> Batch:
    """Sample real positions uniformly with replacement; at least one must exist."""
    # Invert cumulative counts instead of generating a (batch_size, rollout_size)
    # Gumbel array. Right insertion skips padding, including leading zero counts.
    counts = jnp.cumsum(batch.policy_mask, dtype=jnp.int32)
    ranks = jax.random.randint(key, (batch_size,), 0, counts[-1])
    indices = jnp.searchsorted(counts, ranks, side="right")

    def select(x: jax.Array) -> jax.Array:
        return x[indices]

    return jax.tree.map(select, batch)


def train(config: Config) -> TrainState:
    config = config.resolve_run()
    env = pgx.make(config.env_id)
    model = PolicyValueNet(env.num_actions, config.model.channels, config.model.num_blocks)
    key, init_key, env_key = jax.random.split(jax.random.PRNGKey(config.seed), 3)
    observation = env.init(env_key).observation[None].astype(jnp.float32)
    params = model.init(init_key, observation)["params"]
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
        ),
    )
    with (
        checkpoint_manager(config.ckpt_dir) as manager,
        SummaryWriter(
            logdir=config.log_dir,
            purge_step=(
                progress.steps + 1 if (progress := restore_checkpoint(manager, state, config)) is not None else None
            ),
        ) as writer,
    ):
        start_iteration = steps = completed_games = 0
        if progress is not None:
            state, key, start_iteration, steps, completed_games = progress
        opponent_params = state.params if progress is not None else None
        opponent_iteration = start_iteration
        writer.add_text("config", f"```yaml\n{yaml.safe_dump(asdict(config))}```", steps)
        writer.add_text("devices", str(jax.devices()), steps)
        parameter_count = sum(parameter.size for parameter in jax.tree.leaves(state.params))
        writer.add_scalar("model/params_millions", parameter_count / 1_000_000, steps)
        logger.info("Run ID: %s", config.run_id)
        logger.info("TensorBoard run: %s", config.log_dir)
        if progress is not None:
            logger.info("Resumed iteration %d, step %d", start_iteration, steps)
        start = monotonic()
        last_checkpoint_time = start
        last_evaluation_time = start
        window_start = start
        window_iterations = 0
        window_losses: dict[str, float] = {}
        window_rollout: dict[str, float] = {}
        window_iteration_seconds = 0.0
        for iteration in range(start_iteration, config.iterations):
            iteration_start = monotonic()
            key, play_key = jax.random.split(key)
            batch, rollout_metrics = collect_selfplay(state.params, play_key, env=env, model=model, config=config)
            for _ in range(config.updates_per_iteration):
                key, sample_key = jax.random.split(key)
                minibatch = sample_batch(batch, sample_key, config.batch_size)
                state, metrics = train_step(state, minibatch, model=model)
                for name, value in metrics.items():
                    window_losses[name] = window_losses.get(name, 0.0) + float(value) / config.updates_per_iteration
            rollout = {name: float(value) for name, value in rollout_metrics.items()}
            # Use actual played positions as the x-axis, excluding terminal padding.
            steps += int(rollout["positions"])
            completed_games += int(rollout["completed_games"])
            window_iteration_seconds += monotonic() - iteration_start
            window_iterations += 1
            for name, value in rollout.items():
                window_rollout[name] = window_rollout.get(name, 0.0) + value
            now = monotonic()
            window_seconds = now - window_start
            if window_seconds >= config.log_interval_seconds or iteration + 1 == config.iterations:
                # Each iteration has the same number of updates, so this is
                # also the mean over all optimizer updates in the window.
                mean_losses = {name: value / window_iterations for name, value in window_losses.items()}
                elapsed_seconds = now - start
                scalars = {
                    **{f"losses/{name}": value for name, value in mean_losses.items()},
                    **{f"selfplay/{name}": value for name, value in window_rollout.items()},
                    "charts/iteration": iteration + 1,
                    "charts/completed_games": completed_games,
                    "charts/learning_rate": config.learning_rate,
                    "charts/SPS": window_rollout["positions"] / max(window_seconds, 1e-9),
                    "time/iteration_seconds": window_iteration_seconds,
                    "time/window_seconds": window_seconds,
                    "time/elapsed_seconds": elapsed_seconds,
                    "time/eta_seconds": elapsed_seconds
                    * (config.iterations - iteration - 1)
                    / (iteration + 1 - start_iteration),
                }
                for tag, value in scalars.items():
                    writer.add_scalar(tag, value, steps)
                writer.flush()
                logger.info(
                    "Iteration %d/%d | steps=%d | %s | seconds=%.2f",
                    iteration + 1,
                    config.iterations,
                    steps,
                    " | ".join(f"{name}={value:.6g}" for name, value in (window_rollout | mean_losses).items()),
                    window_seconds,
                )
                window_start = now
                window_iterations = 0
                window_losses.clear()
                window_rollout.clear()
                window_iteration_seconds = 0.0
            if monotonic() - last_evaluation_time >= config.evaluation.interval_seconds:
                if opponent_params is None:
                    logger.info("Skipping evaluation: no previous checkpoint yet")
                else:
                    evaluation_start = monotonic()
                    evaluation_metrics = evaluate(
                        opponent_params=opponent_params,
                        candidate_params=state.params,
                        env=env,
                        model=model,
                        config=config,
                    )
                    evaluation_metrics.update(
                        candidate_iteration=iteration + 1,
                        opponent_iteration=opponent_iteration,
                        seconds=monotonic() - evaluation_start,
                    )
                    for name, value in evaluation_metrics.items():
                        writer.add_scalar(f"eval/{name}", value, steps)
                    writer.flush()
                    logger.info(
                        "Evaluation | %s",
                        " | ".join(f"{name}={value:.6g}" for name, value in evaluation_metrics.items()),
                    )
                last_evaluation_time = monotonic()
            if (
                monotonic() - last_checkpoint_time >= config.checkpoint_interval_seconds
                or iteration + 1 == config.iterations
            ):
                save_checkpoint(manager, TrainingProgress(state, key, iteration + 1, steps, completed_games), config)
                opponent_params = state.params
                opponent_iteration = iteration + 1
                last_checkpoint_time = monotonic()
        return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default="configs/alphazero_chess.yaml", help="Path to a YAML config")
    args = parser.parse_args()
    try:
        config = load_config(args.config, Config)
    except (OSError, TypeError, ValueError, yaml.YAMLError, OmegaConfBaseException) as error:
        parser.error(str(error))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout, force=True
    )
    train(config)


if __name__ == "__main__":
    main()
