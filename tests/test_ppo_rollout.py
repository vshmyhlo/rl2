from dataclasses import replace
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from rl2 import ppo
from rl2.envs.gymnax_env import GymnaxEnv
from rl2.model.encoders.rgb import ConvStage


def policy(
    variables: dict[str, jax.Array],
    observations: jax.Array,  # (B, C, H, W)
    memory: ppo.RecurrentCarry,  # leaves: (B, H)
    starts: jax.Array,  # (B,)
    *,
    method: str,
) -> tuple[ppo.RecurrentCarry, jax.Array, jax.Array]:
    def advance(leaf: jax.Array) -> jax.Array:  # (B, H) -> (B, H)
        return jnp.where(starts[:, None], 0, leaf) + 1  # (B, H)

    memory = jax.tree.map(advance, memory)  # leaves: (B, H)
    return memory, jnp.zeros((observations.shape[0], 3)), jnp.full(observations.shape[0], 2.0)


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
    memory = ppo.ActorCritic(3, ppo.ModelConfig(recurrent=ppo.LSTMConfig(hidden_size=1))).initial_carry(
        num_envs
    )  # leaves: (B,1)
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
        chex.assert_shape((result.completed_count, result.completed_return_sum, result.completed_length_sum), ())
        chex.assert_type((result.completed_count, result.completed_length_sum), jnp.int32)
        chex.assert_type(result.completed_return_sum, jnp.float32)
        chex.assert_type(result.batch.observations, jnp.uint8)
        chex.assert_type(result.batch.actions, jnp.int32)
        # No brick hits within two steps. All boundaries here are timeouts.
        expected_count = num_envs * ((num_steps + (0 if result is rollout else num_steps % 2)) // 2)
        assert int(result.completed_count) == expected_count
        completes_initial_episodes = (result is rollout) == (num_steps > 1)
        expected_return = num_envs * (num_envs + 1) / 2 if completes_initial_episodes else 0
        assert float(result.completed_return_sum) == expected_return
        assert int(result.completed_length_sum) == 2 * expected_count
        expected_key = key if result is rollout else rollout.key  # ()
        for _ in range(num_steps):
            expected_key = jax.random.split(expected_key, 3)[0]  # ()
        chex.assert_trees_all_equal(result.key, expected_key)
    np.testing.assert_array_equal(second.episode_lengths, 2 * num_steps % 2)
    np.testing.assert_array_equal(second.episode_starts, True)
    np.testing.assert_array_equal(continued.batch.returns[-1], 1.0)  # gamma * value(final), not zero
    for leaf in jax.tree.leaves(second.carry):  # (B,1)
        np.testing.assert_array_equal(leaf, 2)
    chex.assert_trees_all_equal(continued.batch.carry, first.carry)
    np.testing.assert_array_equal(runner.episode_lengths, 0)


def test_rollout_and_update_with_real_breakout_model() -> None:
    config = replace(
        ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        num_envs=1,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        bf16=False,
        target_kl=None,
        model=ppo.ModelConfig(
            encoder=ppo.RGBEncoderConfig(stages=(ConvStage(2, blocks=0),), embedding_size=2),
            recurrent=ppo.LSTMConfig(hidden_size=2),
        ),
    )
    env = GymnaxEnv(config.env_id)
    observations, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(jax.random.key(0), 1), env.params
    )  # (1,4,10,10); leaves: (1,...)
    model = ppo.ActorCritic(3, config.model)
    carry = model.initial_carry(1)  # leaves: (1,2)
    starts = jnp.ones(1, bool)  # (1,)
    params = model.init(jax.random.key(1), observations[None], carry, starts[None])["params"]
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    runner = ppo.RolloutCarry(env_state, observations, carry, starts, jnp.zeros(1), jnp.zeros(1, jnp.int32))
    _, rollout = ppo.collect_rollout(runner, state, env, config, jax.random.key(2))
    updated, metrics = ppo.update(state, rollout.batch, config)
    assert int(updated.step) == 1
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves((updated.params, metrics)))
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(state.params), jax.tree.leaves(updated.params)))


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
    from rl2.envs.gymnax_env import encode_observations

    observations = encode_observations(jax.vmap(env.env.get_obs)(env_state))  # (B,4,10,10)
    memory = ppo.ActorCritic(3, ppo.ModelConfig(recurrent=ppo.LSTMConfig(hidden_size=1))).initial_carry(
        4
    )  # leaves: (B,1)

    def position_policy(
        variables: dict[str, jax.Array],
        observations: jax.Array,  # (B,C,H,W)
        memory: ppo.RecurrentCarry,  # leaves: (B,1)
        starts: jax.Array,  # (B,)
        *,
        method: str,
    ) -> tuple[ppo.RecurrentCarry, jax.Array, jax.Array]:
        memory, logits, _ = policy(variables, observations, memory, starts, method=method)
        positions = jnp.argmax(observations[:, 1].reshape((4, -1)), axis=-1).astype(jnp.float32)  # (B,)
        return memory, logits, positions  # leaves: (B,1); (B,3); (B,)

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
    assert int(rollout.completed_count) == 3
    assert float(rollout.completed_return_sum) == 3.0  # Excludes bootstrap values and unfinished returns.
    assert int(rollout.completed_length_sum) == 5
    np.testing.assert_array_equal(advanced.episode_returns, [0, 0, 0, 100])
    advanced, rollout = ppo.collect_rollout(advanced, state, env, config, rollout.key)
    assert int(rollout.completed_count) == 1
    assert float(rollout.completed_return_sum) == 100.0
    assert int(rollout.completed_length_sum) == 2
    np.testing.assert_array_equal(rollout.batch.episode_starts, [[True, True, True, False]])
    for leaf in jax.tree.leaves(advanced.carry):  # (B,1)
        np.testing.assert_array_equal(leaf[:, 0], [1, 1, 1, 2])
