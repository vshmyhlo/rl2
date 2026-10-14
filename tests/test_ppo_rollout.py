from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from rl2 import ppo
from rl2.envs.gymnax_env import GymnaxEnv, encode_observations
from rl2.model.encoders.rgb import ConvStage


def policy(
    variables: dict[str, jax.Array],
    observations: jax.Array,  # (B, C, H, W)
    memory: ppo.RecurrentCarry,  # leaves: (B, H)
    starts: jax.Array,  # (B,)
    *,
    method: str,
) -> ppo.ActorCriticOutput:
    def advance(leaf: jax.Array) -> jax.Array:  # (B, H) -> (B, H)
        return jnp.where(starts[:, None], 0, leaf) + 1  # (B, H)

    memory = jax.tree.map(advance, memory)  # leaves: (B, H)
    return ppo.ActorCriticOutput(
        memory, ppo.CategoricalPolicy(jnp.zeros((observations.shape[0], 3))), jnp.full(observations.shape[0], 2.0)
    )


@pytest.mark.parametrize("num_envs,num_steps", [(1, 1), (2, 3)], ids=["single-transition", "batched-sequence"])
def test_compiled_rollout_continues_rng_memory_and_bootstraps_timeouts(num_envs: int, num_steps: int) -> None:
    config = replace(
        ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        num_envs=num_envs,
        num_steps=num_steps,
        num_minibatches=1,
        gamma=0.5,
        gae_lambda=1.0,
    )
    env = GymnaxEnv(config.env_id, max_steps_in_episode=2)
    key = jax.random.key(0)  # ()
    observations, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(key, num_envs), env.params
    )  # (B,4,10,10); leaves: (B,...)
    memory = ppo.ActorCritic(
        ppo.DiscreteSpace(3), ppo.ModelConfig(encoder=ppo.RGBEncoderConfig(), recurrent=ppo.LSTMConfig(hidden_size=1))
    ).initial_carry(num_envs)  # leaves: (B,1)
    runner = ppo.RolloutCarry(
        env_state,
        observations,
        memory,
        jnp.ones(num_envs, bool),
        jnp.arange(1, num_envs + 1, dtype=jnp.float32),  # (B,) returns carried into this rollout
        jnp.zeros(num_envs, jnp.int32),
    )
    state = TrainState.create(apply_fn=policy, params={}, tx=optax.sgd(0.1))
    first, rollout = ppo.collect_rollout(runner, state, env, config, key)
    second, continued = ppo.collect_rollout(first, state, env, config, rollout.key)
    duplicate, repeated = ppo.collect_rollout(runner, state, env, config, key)
    chex.assert_trees_all_equal((duplicate, repeated), (first, rollout))
    for result in (rollout, continued):
        chex.assert_shape(result.batch.observations, (num_steps, num_envs, 4, 10, 10))
        chex.assert_shape((result.batch.actions, result.batch.returns), (num_steps, num_envs))
        chex.assert_shape((result.episode_count, result.episode_return_sum, result.episode_length_sum), ())
        chex.assert_type((result.episode_count, result.episode_length_sum), jnp.int32)
        chex.assert_type(result.episode_return_sum, jnp.float32)
        chex.assert_type(result.batch.observations, jnp.uint8)
        chex.assert_type(result.batch.actions, jnp.int32)
        # No brick hits within two steps. All boundaries here are timeouts.
        expected_count = num_envs * ((num_steps + (0 if result is rollout else num_steps % 2)) // 2)
        assert int(result.episode_count) == expected_count
        completes_initial_episodes = (result is rollout) == (num_steps > 1)
        expected_return = num_envs * (num_envs + 1) / 2 if completes_initial_episodes else 0
        assert float(result.episode_return_sum) == expected_return
        assert int(result.episode_length_sum) == 2 * expected_count
        expected_key = key if result is rollout else rollout.key  # ()
        for _ in range(num_steps):
            expected_key = jax.random.split(expected_key, 3)[0]  # ()
        chex.assert_trees_all_equal(result.key, expected_key)
    np.testing.assert_array_equal(second.episode_step_counts, 2 * num_steps % 2)
    np.testing.assert_array_equal(second.episode_starts, True)
    np.testing.assert_array_equal(continued.batch.returns[-1], 1.0)  # gamma * value(final), not zero
    for leaf in jax.tree.leaves(second.model_carry):  # (B,1)
        np.testing.assert_array_equal(leaf, 2)
    chex.assert_trees_all_equal(continued.batch.model_carry, first.model_carry)
    np.testing.assert_array_equal(runner.episode_step_counts, 0)


def test_train_step_matches_separate_rollout_and_updates_with_real_model() -> None:
    config = replace(
        ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        num_envs=2,
        num_steps=2,
        num_minibatches=2,
        update_epochs=2,
        total_steps=16,
        anneal_lr=True,
        entropy_decay="cosine",
        bf16=False,
        target_kl=None,
        model=ppo.ModelConfig(
            # Two-channel LayerNorm magnifies FP32 fusion differences in this comparison.
            encoder=ppo.RGBEncoderConfig(stages=(ConvStage(4, blocks=0),), embedding_size=4),
            recurrent=ppo.LSTMConfig(hidden_size=4),
        ),
    )
    env = GymnaxEnv(config.env_id)
    observations, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(jax.random.key(0), 2), env.params
    )  # (B=2,4,10,10); leaves: (B,...)
    model = ppo.ActorCritic(ppo.DiscreteSpace(3), config.model)
    carry = model.initial_carry(2)  # leaves: (B,4)
    starts = jnp.ones(2, bool)  # (B,)
    params = model.init(jax.random.key(1), observations[None], carry, starts[None])["params"]
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.inject_hyperparams(optax.adam)(learning_rate=config.learning_rate, eps=1e-5),
    )
    runner = ppo.RolloutCarry(env_state, observations, carry, starts, jnp.zeros(2), jnp.zeros(2, jnp.int32))
    key = jax.random.key(2)  # ()
    iteration = jnp.asarray(1)  # () nonzero iteration exercises schedule forwarding
    result = ppo.train_step(runner, state, env, config, key, ppo.init_metrics(), iteration)

    def copy_buffer(array: jax.Array | int) -> jax.Array:  # arbitrary parameter/optimizer shape
        return jnp.array(array, copy=True)  # same shape

    owned_state = jax.tree.map(copy_buffer, state)  # parameter/optimizer leaves
    donated = ppo.train_step_donated(runner, owned_state, env, config, key, ppo.init_metrics(), iteration)
    chex.assert_trees_all_close((donated.train_state, donated.metrics), (result.train_state, result.metrics), atol=1e-6)
    chex.assert_trees_all_equal((donated.rollout_carry, donated.key), (result.rollout_carry, result.key))
    assert all(leaf.is_deleted() for leaf in jax.tree.leaves(owned_state.params))
    advanced, rollout = ppo.collect_rollout(runner, state, env, config, key)
    expected = state.replace(
        opt_state=state.opt_state._replace(
            hyperparams={
                **state.opt_state.hyperparams,
                "learning_rate": ppo.learning_rate_schedule(config)(iteration),  # ()
            }
        )
    )
    expected_key = rollout.key  # ()
    metrics: list[ppo.MinibatchMetrics] = []  # scalar leaves
    for _ in range(config.update_epochs):
        expected_key, shuffle_key = jax.random.split(expected_key)  # each ()
        permutation = jax.random.permutation(shuffle_key, config.num_envs)  # (B,)
        for indices in jnp.split(permutation, config.num_minibatches):  # (M=1,)
            expected, update_metrics = ppo.update(expected, rollout.batch.select_sequences(indices), config, iteration)
            metrics.append(update_metrics)

    def sum_metrics(*leaves: jax.Array) -> jax.Array:  # each ()
        return jnp.stack(leaves).sum(axis=0)  # ()

    minibatch_metrics = jax.tree.map(sum_metrics, *metrics)  # scalar leaves
    chex.assert_trees_all_close(result.train_state, expected, atol=1e-6)
    chex.assert_trees_all_close(result.metrics.window.minibatch_metrics, minibatch_metrics, atol=1e-6)
    chex.assert_trees_all_equal(result.rollout_carry, advanced)
    chex.assert_trees_all_equal(result.key, expected_key)
    chex.assert_shape(jax.tree.leaves(result.metrics.window.minibatch_metrics), ())
    chex.assert_shape(result.metrics.window.explained_variance_sum, ())
    assert (
        int(result.train_state.step)
        == int(result.metrics.window.update_count)
        == int(result.metrics.window.minibatch_count)
        == 4
    )
    assert not bool(result.metrics.window.early_stop_count)
    np.testing.assert_allclose(
        result.metrics.window.explained_variance_sum,
        ppo.explained_variance(rollout.batch.values, rollout.batch.returns),
    )
    assert all(
        np.isfinite(leaf).all()
        for leaf in jax.tree.leaves((result.train_state.params, result.metrics.window.minibatch_metrics))
    )
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(state.params), jax.tree.leaves(result.train_state.params))
    )


@pytest.mark.parametrize("reject_at", [0, 1, 2], ids=["first-minibatch", "mid-epoch", "next-epoch"])
def test_train_step_stops_updates_and_rng_at_rejected_minibatch(reject_at: int) -> None:
    config = replace(
        ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        num_envs=2,
        num_steps=1,
        num_minibatches=2,
        update_epochs=2,
        target_kl=0.01,
    )
    env = GymnaxEnv(config.env_id, max_steps_in_episode=1)
    key = jax.random.key(0)  # ()
    observations, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(key, 2), env.params
    )  # (B=2,4,10,10); leaves: (B,...)
    memory = ppo.ActorCritic(
        ppo.DiscreteSpace(3), ppo.ModelConfig(encoder=ppo.RGBEncoderConfig(), recurrent=ppo.LSTMConfig(hidden_size=1))
    ).initial_carry(2)  # leaves: (B,1)
    runner = ppo.RolloutCarry(
        env_state, observations, memory, jnp.ones(2, bool), jnp.array([1.0, 2.0]), jnp.zeros(2, jnp.int32)
    )
    state = TrainState.create(
        apply_fn=policy, params={"weight": jnp.zeros(1)}, tx=optax.inject_hyperparams(optax.adam)(learning_rate=0.1)
    )

    def update(
        state: TrainState,
        batch: ppo.PPOBatch,  # observations: (T=1,M=1,4,10,10); targets: (T,M); memory: (M,1)
        config: ppo.Config,
        iteration: int | jax.Array,  # ()
    ) -> tuple[TrainState, ppo.MinibatchMetrics]:
        chex.assert_shape(batch.observations, (1, 1, 4, 10, 10))
        rejected = state.step == reject_at  # ()
        loss = jnp.asarray(state.step + 1, jnp.float32)  # ()
        metrics = ppo.MinibatchMetrics(loss, loss, loss, loss, rejected.astype(jnp.float32), loss)
        state = jax.lax.cond(rejected, lambda: state, lambda: state.apply_gradients(grads={"weight": jnp.ones(1)}))
        return state, metrics

    with patch.object(ppo, "update", side_effect=update):
        result = ppo.train_step(runner, state, env, config, key, ppo.init_metrics())
    assert int(result.train_state.step) == int(result.metrics.window.update_count) == reject_at
    assert int(result.metrics.window.minibatch_count) == reject_at + 1
    assert bool(result.metrics.window.early_stop_count)
    assert float(result.metrics.window.minibatch_metrics.policy_loss) == (reject_at + 1) * (reject_at + 2) / 2
    assert float(result.metrics.window.minibatch_metrics.approx_kl) == 1.0
    expected = state.replace(
        opt_state=state.opt_state._replace(
            hyperparams={
                **state.opt_state.hyperparams,
                "learning_rate": jnp.asarray(config.learning_rate, jnp.float32),  # ()
            }
        )
    )
    for _ in range(reject_at):
        expected = expected.apply_gradients(grads={"weight": jnp.ones(1)})
    chex.assert_trees_all_close(result.train_state, expected)
    expected_key = jax.random.split(key, 3)[0]  # () after the single rollout step
    for _ in range(reject_at // config.num_minibatches + 1):
        expected_key = jax.random.split(expected_key)[0]  # () one shuffle per started epoch
    chex.assert_trees_all_equal(result.key, expected_key)
    chex.assert_shape(result.rollout_carry.observations, (2, 4, 10, 10))
    chex.assert_shape(jax.tree.leaves(result.rollout_carry.model_carry), (2, 1))
    chex.assert_shape(jax.tree.leaves(result.metrics.window.minibatch_metrics), ())
    chex.assert_shape(jax.tree.leaves(result.metrics), ())
    assert int(result.metrics.total.episode_count) == int(result.metrics.total.episode_length_sum) == 2
    assert float(result.metrics.total.episode_return_sum) == 3.0

    if reject_at == 1:
        # Exercise continued windows and resets once, including totals beyond 32-bit precision.
        x64_enabled = jax.config.x64_enabled
        metrics = ppo.init_metrics(ppo.TrainingProgress(state, key, 0, 2**31 - 1, float(2**24), 2**31 - 1))._replace(
            window=result.metrics.window
        )  # scalar leaves
        with patch.object(ppo, "update", side_effect=update):
            continued = ppo.train_step(runner, state, env, config, key, metrics)
            window = continued.metrics.window  # scalar leaves
            assert int(window.rollout_count) == int(window.early_stop_count) == int(window.update_count) == 2
            assert (
                int(window.minibatch_count)
                == int(window.episodes.episode_count)
                == int(window.episodes.episode_length_sum)
                == 4
            )
            assert float(window.minibatch_metrics.policy_loss) == float(window.episodes.episode_return_sum) == 6.0
            assert (
                int(continued.metrics.total.episode_count)
                == int(continued.metrics.total.episode_length_sum)
                == 2**31 + 1
            )
            assert float(continued.metrics.total.episode_return_sum) == float(2**24 + 3)
            chex.assert_type(
                (continued.metrics.total.episode_count, continued.metrics.total.episode_length_sum), jnp.int64
            )
            chex.assert_type(continued.metrics.total.episode_return_sum, jnp.float64)
            with jax.enable_x64():
                reset = continued.metrics._replace(window=ppo.RunningSum.init(window))  # scalar leaves
            assert all(float(leaf) == 0 for leaf in jax.tree.leaves(reset.window))
            after_reset = ppo.train_step(runner, state, env, config, key, reset)
        chex.assert_trees_all_close(after_reset.metrics.window, result.metrics.window)
        assert (
            int(after_reset.metrics.total.episode_count)
            == int(after_reset.metrics.total.episode_length_sum)
            == 2**31 + 3
        )
        assert float(after_reset.metrics.total.episode_return_sum) == float(2**24 + 6)
        assert jax.config.x64_enabled == x64_enabled


def test_terminal_and_timeout_targets_use_final_observations() -> None:
    config = replace(
        ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        num_envs=4,
        num_steps=1,
        num_minibatches=1,
        gamma=0.5,
    )
    env = GymnaxEnv(config.env_id, max_steps_in_episode=2)
    _, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(jax.random.key(0), 4), env.params
    )  # leaves: (B=4,...)
    env_state = env_state.replace(
        ball_y=jnp.array([8, 3, 8, 3]),
        ball_x=jnp.zeros(4, jnp.int32),
        ball_dir=jnp.full(4, 2),
        time=jnp.array([0, 1, 1, 0]),
    )
    observations = encode_observations(jax.vmap(env.env.get_obs)(env_state))  # (B,4,10,10)
    memory = ppo.ActorCritic(
        ppo.DiscreteSpace(3), ppo.ModelConfig(encoder=ppo.RGBEncoderConfig(), recurrent=ppo.LSTMConfig(hidden_size=1))
    ).initial_carry(4)  # leaves: (B,1)

    def position_policy(
        variables: dict[str, jax.Array],
        observations: jax.Array,  # (B,C,H,W)
        memory: ppo.RecurrentCarry,  # leaves: (B,1)
        starts: jax.Array,  # (B,)
        *,
        method: str,
    ) -> ppo.ActorCriticOutput:
        memory, logits, _ = policy(variables, observations, memory, starts, method=method)
        positions = jnp.argmax(observations[:, 1].reshape((4, -1)), axis=-1).astype(jnp.float32)  # (B,)
        return ppo.ActorCriticOutput(memory, logits, positions)  # leaves: (B,1); (B,3); (B,)

    state = TrainState.create(apply_fn=position_policy, params={}, tx=optax.sgd(0.1))
    runner = ppo.RolloutCarry(
        env_state,
        observations,
        memory,
        jnp.ones(4, bool),  # (B,)
        jnp.array([2.0, -3.0, 4.0, 100.0]),  # (B,)
        jnp.array([0, 1, 1, 0]),  # (B,)
    )
    advanced, rollout = ppo.collect_rollout(runner, state, env, config, jax.random.key(1))
    np.testing.assert_array_equal(rollout.batch.returns, [[0, 20.5, 0, 20.5]])
    np.testing.assert_array_equal(advanced.episode_starts, [True, True, True, False])
    assert int(rollout.episode_count) == 3
    assert float(rollout.episode_return_sum) == 3.0  # Excludes bootstrap values and unfinished returns.
    assert int(rollout.episode_length_sum) == 5
    np.testing.assert_array_equal(advanced.episode_returns, [0, 0, 0, 100])
    advanced, rollout = ppo.collect_rollout(advanced, state, env, config, rollout.key)
    assert int(rollout.episode_count) == 1
    assert float(rollout.episode_return_sum) == 100.0
    assert int(rollout.episode_length_sum) == 2
    np.testing.assert_array_equal(rollout.batch.episode_starts, [[True, True, True, False]])
    for leaf in jax.tree.leaves(advanced.model_carry):  # (B,1)
        np.testing.assert_array_equal(leaf[:, 0], [1, 1, 1, 2])
