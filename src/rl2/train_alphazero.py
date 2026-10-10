"""Single-device AlphaZero sketch using PGX rules and Mctx PUCT search.

Run: uv run --extra alphazero python -m rl2.train_alphazero --config configs/alphazero_chess.yaml
For a cheap start, set env_id: tic_tac_toe and max_moves: 9 in the YAML config.

Each iteration starts fresh games and samples updates from that rollout only.
Unfinished games train the policy but not the value; terminal padding trains
neither. This scaffold has no persistent replay, evaluation, or checkpoints.
PGX owns observation/action encodings and draw rules. Values always use the
player-to-move perspective. This is not an exact reproduction of the paper.
"""

import argparse
import json
from dataclasses import asdict, replace
from functools import partial
from time import monotonic
from typing import Any

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

from rl2.alphazero.config import Config, load_config
from rl2.alphazero.model import PolicyValueNet
from rl2.configuration import resolve_settings
from rl2.shape_checker import ShapeChecker

type Parameters = dict[str, Any]
type Metrics = dict[str, jax.Array]


def masked_logits(logits: jax.Array, legal: jax.Array) -> jax.Array:
    sc = ShapeChecker()
    sc.check(logits, "BA", jnp.float32)
    sc.check(legal, "BA", jnp.bool_)
    # Finite masking also keeps terminal nodes with no legal moves well-defined.
    return jnp.where(legal, logits - logits.max(axis=-1, keepdims=True), jnp.finfo(jnp.float32).min)


def recurrent_step(
    params: Parameters,
    key: jax.Array,
    actions: jax.Array,
    states: pgx.State,
    *,
    env: pgx.Env,
    model: PolicyValueNet,
) -> tuple[mctx.RecurrentFnOutput, pgx.State]:
    """Expand real game states; back up the opponent's value with a minus sign."""
    sc = ShapeChecker(K=2, P=2)
    sc.check(key, "K", jnp.uint32)
    sc.check(actions, "B", jnp.int32)
    sc.check(states.current_player, "B", jnp.int32)
    next_states = jax.vmap(env.step)(states, actions)
    logits, value = model.apply({"params": params}, next_states.observation)
    sc.check(next_states.rewards, "BP", jnp.float32)
    sc.check([next_states.terminated, next_states.truncated], "B", jnp.bool_)
    reward = jnp.take_along_axis(next_states.rewards, states.current_player[:, None], axis=1)[:, 0]
    done = next_states.terminated | next_states.truncated
    output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=jnp.where(done, 0.0, -1.0),
        prior_logits=masked_logits(logits, next_states.legal_action_mask),
        value=jnp.where(done, 0.0, value),
    )
    sc.check([output.reward, output.discount, output.value], "B", jnp.float32)
    return output, next_states


@struct.dataclass
class Batch:
    observations: jax.Array
    policy_targets: jax.Array
    value_targets: jax.Array
    policy_mask: jax.Array
    value_mask: jax.Array

    def validate(self) -> None:
        sc = ShapeChecker()
        sc.check(self.observations, "BHWC", jnp.float32)
        sc.check(self.policy_targets, "BA", jnp.float32)
        sc.check(self.value_targets, "B", jnp.float32)
        sc.check([self.policy_mask, self.value_mask], "B", jnp.bool_)


def outcome_targets(
    players: jax.Array, valid: jax.Array, rewards: jax.Array, terminated: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """One game per column: map its final rewards to each recorded actor."""
    sc = ShapeChecker(P=2)
    sc.check(players, "TB", jnp.int32)
    sc.check(valid, "TB", jnp.bool_)
    sc.check(rewards, "BP", jnp.float32)
    sc.check(terminated, "B", jnp.bool_)
    targets = rewards[jnp.arange(players.shape[1])[None, :], players]
    mask = valid & terminated[None, :]
    targets = jnp.where(mask, targets, 0.0)
    sc.check(targets, "TB", jnp.float32)
    sc.check(mask, "TB", jnp.bool_)
    return targets, mask


@partial(jax.jit, static_argnames=("env", "model", "config"))
def collect_selfplay(
    params: Parameters, key: jax.Array, *, env: pgx.Env, model: PolicyValueNet, config: Config
) -> tuple[Batch, Metrics]:
    """Collect one fixed-length, terminal-padded game per environment."""
    sc = ShapeChecker(K=2)
    sc.check(key, "K", jnp.uint32)
    init_key, play_key = jax.random.split(key)
    states = jax.vmap(env.init)(jax.random.split(init_key, config.num_envs))
    recurrent_fn = partial(recurrent_step, env=env, model=model)

    def step(
        states: pgx.State, inputs: tuple[jax.Array, jax.Array]
    ) -> tuple[pgx.State, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
        key, move = inputs
        step_sc = ShapeChecker(K=2, B=config.num_envs, A=env.num_actions)
        step_sc.check(key, "K", jnp.uint32)
        step_sc.check(move, "", jnp.int32)
        logits, value = model.apply({"params": params}, states.observation)
        done = states.terminated | states.truncated
        root = mctx.RootFnOutput(
            prior_logits=masked_logits(logits, states.legal_action_mask),
            value=jnp.where(done, 0.0, value),
            embedding=states,
        )
        # Mctx calls this muzero_policy; using PGX as the exact transition
        # function makes it AlphaZero-style PUCT, with no learned dynamics.
        policy = mctx.muzero_policy(
            params,
            key,
            root,
            recurrent_fn,
            config.num_simulations,
            invalid_actions=~states.legal_action_mask,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_fraction=config.dirichlet_fraction,
            temperature=jnp.where(move < config.exploration_moves, 1.0, 0.0),
        )
        step_sc.check(policy.action, "B", jnp.int32)
        step_sc.check(policy.action_weights, "BA", jnp.float32)
        next_states = jax.vmap(env.step)(states, policy.action)

        def preserve_finished(old: jax.Array, new: jax.Array) -> jax.Array:
            # Preserve final rewards: PGX otherwise clears them on later steps.
            return jnp.where(done.reshape((config.num_envs,) + (1,) * (old.ndim - 1)), old, new)

        next_states = jax.tree.map(preserve_finished, states, next_states)
        return next_states, (
            states.observation.astype(jnp.float32),
            policy.action_weights,
            states.current_player,
            ~done,
        )

    states, (observations, policies, players, valid) = jax.lax.scan(
        step, states, (jax.random.split(play_key, config.max_moves), jnp.arange(config.max_moves, dtype=jnp.int32))
    )
    values, value_mask = outcome_targets(players, valid, states.rewards, states.terminated)
    size = config.max_moves * config.num_envs
    batch = Batch(
        observations.reshape((size, *observations.shape[2:])),
        policies.reshape((size, env.num_actions)),
        values.reshape(size),
        valid.reshape(size),
        value_mask.reshape(size),
    )
    batch.validate()
    return batch, {
        "completed_games": states.terminated.sum(),
        "positions": valid.sum(),
        "value_positions": value_mask.sum(),
    }


def loss_fn(params: Parameters, model: PolicyValueNet, batch: Batch) -> tuple[jax.Array, Metrics]:
    batch.validate()
    logits, values = model.apply({"params": params}, batch.observations)
    sc = ShapeChecker()
    sc.check([logits, batch.policy_targets], "BA", jnp.float32)
    sc.check([values, batch.value_targets], "B", jnp.float32)
    policy_errors = optax.softmax_cross_entropy(logits, batch.policy_targets)
    value_errors = jnp.square(values - batch.value_targets)
    policy_loss = jnp.where(batch.policy_mask, policy_errors, 0.0).sum() / jnp.maximum(batch.policy_mask.sum(), 1)
    value_loss = jnp.where(batch.value_mask, value_errors, 0.0).sum() / jnp.maximum(batch.value_mask.sum(), 1)
    loss = policy_loss + value_loss
    sc.check([loss, policy_loss, value_loss], "", jnp.float32)
    return loss, {"loss": loss, "policy_loss": policy_loss, "value_loss": value_loss}


@partial(jax.jit, static_argnames=("model",))
def train_step(state: TrainState, batch: Batch, *, model: PolicyValueNet) -> tuple[TrainState, Metrics]:
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, model, batch)
    return state.apply_gradients(grads=grads), metrics


@partial(jax.jit, static_argnames=("batch_size",))
def sample_batch(batch: Batch, key: jax.Array, batch_size: int) -> Batch:
    """Sample real positions uniformly with replacement; at least one must exist."""
    batch.validate()
    sc = ShapeChecker(K=2, N=batch_size)
    sc.check(key, "K", jnp.uint32)
    sampling_logits = jnp.where(batch.policy_mask, 0.0, -jnp.inf)
    indices = jax.random.categorical(key, sampling_logits, shape=(batch_size,))
    sc.check(indices, "N", jnp.int32)

    def select(x: jax.Array) -> jax.Array:
        return x[indices]

    sampled = jax.tree.map(select, batch)
    sampled.validate()
    return sampled


def train(config: Config) -> TrainState:
    settings = resolve_settings(asdict(config))
    config = replace(config, run_id=settings["run_id"], log_dir=settings["log_dir"])
    run_name = config.run_id
    run_dir = config.log_dir
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
    writer = SummaryWriter(logdir=run_dir)
    try:
        writer.add_text("config", f"```yaml\n{yaml.safe_dump(asdict(config))}```", 0)
        writer.add_text("devices", str(jax.devices()), 0)
        parameter_count = sum(parameter.size for parameter in jax.tree.leaves(state.params))
        writer.add_scalar("model/params_millions", parameter_count / 1_000_000, 0)
        print(f"Run ID: {run_name}\nTensorBoard run: {run_dir}", flush=True)
        steps = 0
        completed_games = 0
        start = monotonic()
        for iteration in range(config.iterations):
            iteration_start = monotonic()
            key, play_key = jax.random.split(key)
            batch, rollout_metrics = jax.block_until_ready(
                collect_selfplay(state.params, play_key, env=env, model=model, config=config)
            )
            rollout_seconds = monotonic() - iteration_start
            optimization_start = monotonic()
            losses: dict[str, float] = {}
            for _ in range(config.updates_per_iteration):
                key, sample_key = jax.random.split(key)
                minibatch = sample_batch(batch, sample_key, config.batch_size)
                state, metrics = train_step(state, minibatch, model=model)
                for name, value in metrics.items():
                    losses[name] = losses.get(name, 0.0) + float(value) / config.updates_per_iteration
            jax.block_until_ready(state)
            optimization_seconds = monotonic() - optimization_start
            rollout = {name: float(value) for name, value in rollout_metrics.items()}
            # Use actual played positions as the x-axis, excluding terminal padding.
            steps += int(rollout["positions"])
            completed_games += int(rollout["completed_games"])
            seconds = monotonic() - iteration_start
            elapsed_seconds = monotonic() - start
            scalars = {
                **{f"losses/{name}": value for name, value in losses.items()},
                **{f"selfplay/{name}": value for name, value in rollout.items()},
                "charts/iteration": iteration + 1,
                "charts/completed_games": completed_games,
                "charts/learning_rate": config.learning_rate,
                "charts/SPS": rollout["positions"] / max(seconds, 1e-9),
                "time/rollout_seconds": rollout_seconds,
                "time/optimization_seconds": optimization_seconds,
                "time/iteration_seconds": seconds,
                "time/elapsed_seconds": elapsed_seconds,
                "time/eta_seconds": elapsed_seconds * (config.iterations - iteration - 1) / (iteration + 1),
            }
            for tag, value in scalars.items():
                writer.add_scalar(tag, value, steps)
            writer.flush()
            report = dict(rollout | losses, iteration=iteration + 1, steps=steps, seconds=seconds)
            print(json.dumps(report), flush=True)
        return state
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default="configs/alphazero_chess.yaml", help="Path to a YAML config")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except (OSError, TypeError, ValueError, yaml.YAMLError, OmegaConfBaseException) as error:
        parser.error(str(error))
    train(config)


if __name__ == "__main__":
    main()
