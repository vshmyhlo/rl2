from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.training.train_state import TrainState

from rl2 import ppo
from rl2.jax_env import JaxVectorEnv
from rl2.lstm import initial_carry


@pytest.mark.parametrize(
    ("num_steps", "num_envs", "obs_shape"),
    [(1, 1, (1,)), (2, 2, (2, 1))],
    ids=["single-transition", "sequence-with-partial-reset"],
)
def test_collect_rollout_continues_memory_rng_and_episodes(
    num_steps: int, num_envs: int, obs_shape: tuple[int, ...]
) -> None:
    config = replace(
        ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        num_steps=num_steps,
        num_envs=num_envs,
        num_minibatches=1,
        gamma=1.0,
        gae_lambda=1.0,
    )
    obs = jnp.zeros((num_envs, *obs_shape), dtype=jnp.uint8)  # (B, *O)
    carry = initial_carry(num_envs, 1)  # leaves: (B, H=1)
    starts = jnp.ones(num_envs, dtype=bool)  # (B,)
    zeros = jnp.zeros(num_envs, dtype=jnp.float32)  # (B,)
    rollout_state = ppo.RolloutCarry(
        observations=obs,  # (B, *O)
        carry=carry,  # leaves: (B, H)
        episode_starts=starts,  # (B,)
        episode_returns=zeros,  # (B,)
        episode_lengths=jnp.zeros(num_envs, dtype=jnp.int32),  # (B,)
    )
    key = jax.random.key(0)  # () typed PRNG key
    state = Mock(spec=TrainState)
    host_env = Mock(spec=gym.vector.VectorEnv)
    # The first environment finishes at the end of the second rollout.
    done = np.arange(num_envs) == 0  # (B,)
    host_env.step.side_effect = [
        (
            np.full(obs.shape, step, dtype=np.uint8),  # (B, *O)
            np.ones(num_envs, dtype=np.float32),  # (B,)
            done if step == 2 * num_steps else np.zeros(num_envs, dtype=bool),  # (B,)
            np.zeros(num_envs, dtype=bool),  # (B,)
            {},
        )
        for step in range(1, 2 * num_steps + 1)
    ]
    reset_obs = np.full(obs.shape, 2 * num_steps, dtype=np.uint8)  # (B, *O)
    reset_obs[0] = 0
    host_env.reset.return_value = (reset_obs, {})
    envs = JaxVectorEnv(host_env)

    def act(
        state: TrainState,
        observations: jax.Array,  # (B, *O)
        memory: ppo.RecurrentCarry,  # leaves: (B, H)
        episode_starts: jax.Array,  # (B,)
        action_key: jax.Array,  # () typed PRNG key
    ) -> tuple[jax.Array, jax.Array, jax.Array, ppo.RecurrentCarry]:
        chex.assert_shape(observations, (num_envs, *obs_shape))
        chex.assert_shape(episode_starts, (num_envs,))

        def advance(leaf: jax.Array) -> jax.Array:  # (B, H) -> (B, H)
            return jnp.where(episode_starts[:, None], 0, leaf) + 1  # (B, H)

        advanced = jax.tree.map(advance, memory)  # leaves: (B, H)
        return (
            jax.random.randint(action_key, (num_envs,), 0, 2),  # (B,)
            jnp.full(num_envs, -0.5),  # (B,)
            zeros,  # (B,)
            advanced,  # leaves: (B, H)
        )

    with (
        patch.object(ppo, "act", side_effect=act),
        patch.object(ppo, "value", return_value=jnp.full(num_envs, 3.0)) as bootstrap,
    ):
        first_carry, first = ppo.collect_rollout(
            rollout_state, state, envs, config, key
        )  # carry: (B, ...); rollout arrays: (T, B, ...)
        host_env.reset.assert_not_called()
        second_carry, second = ppo.collect_rollout(
            first_carry, state, envs, config, first.key
        )  # carry: (B, ...); rollout arrays: (T, B, ...)

    expected_key = key  # () typed PRNG key
    for index, (result_carry, result) in enumerate(((first_carry, first), (second_carry, second))):
        batch = result.batch  # observations: (T, B, *O); targets/starts: (T, B); memory leaves: (B, H)
        chex.assert_shape(batch.observations, (num_steps, num_envs, *obs_shape))
        chex.assert_shape(
            (batch.actions, batch.old_log_probs, batch.advantages, batch.returns, batch.episode_starts, batch.values),
            (num_steps, num_envs),
        )
        chex.assert_type(batch.observations, jnp.uint8)
        chex.assert_type(batch.actions, jnp.int32)
        chex.assert_type(batch.episode_starts, jnp.bool_)
        chex.assert_type((batch.old_log_probs, batch.advantages, batch.returns, batch.values), jnp.float32)
        for step in range(num_steps):
            expected_key, action_key = jax.random.split(expected_key)  # each () typed PRNG key
            np.testing.assert_array_equal(batch.actions[step], jax.random.randint(action_key, (num_envs,), 0, 2))
            np.testing.assert_array_equal(batch.observations[step], index * num_steps + step)
        chex.assert_trees_all_equal(result.key, expected_key)
        np.testing.assert_array_equal(batch.values, 0)
        np.testing.assert_array_equal(batch.old_log_probs, -0.5)
        expected_returns = np.broadcast_to(
            np.arange(num_steps, 0, -1, dtype=np.float32)[:, None] + 3, (num_steps, num_envs)
        ).copy()  # (T, B)
        if index == 1:
            expected_returns[:, 0] -= 3  # Terminal episodes do not bootstrap.
        np.testing.assert_array_equal(batch.returns, expected_returns)
        np.testing.assert_array_equal(batch.advantages, expected_returns)
        for leaf in jax.tree.leaves(batch.carry):  # (B, H)
            np.testing.assert_array_equal(leaf, index * num_steps)
        for leaf in jax.tree.leaves(result_carry.carry):  # (B, H)
            np.testing.assert_array_equal(leaf, (index + 1) * num_steps)
        chex.assert_shape(result_carry.observations, (num_envs, *obs_shape))
        chex.assert_shape(
            (result_carry.episode_starts, result_carry.episode_returns, result_carry.episode_lengths), (num_envs,)
        )
        np.testing.assert_array_equal(bootstrap.call_args_list[index].args[1], result_carry.observations)
        chex.assert_trees_all_equal(bootstrap.call_args_list[index].args[2], result_carry.carry)
        np.testing.assert_array_equal(bootstrap.call_args_list[index].args[3], result_carry.episode_starts)

    assert first.completed_returns == first.completed_lengths == ()
    assert second.completed_returns == (float(2 * num_steps),)
    assert second.completed_lengths == (2 * num_steps,)
    np.testing.assert_array_equal(first_carry.episode_returns, num_steps)
    np.testing.assert_array_equal(first_carry.episode_lengths, num_steps)
    np.testing.assert_array_equal(first.batch.episode_starts[0], True)
    np.testing.assert_array_equal(first.batch.episode_starts[1:], False)
    np.testing.assert_array_equal(second.batch.episode_starts, False)
    np.testing.assert_array_equal(second_carry.observations, reset_obs)
    np.testing.assert_array_equal(second_carry.episode_starts, done)
    np.testing.assert_array_equal(second_carry.episode_returns, np.where(done, 0, 2 * num_steps))
    np.testing.assert_array_equal(second_carry.episode_lengths, np.where(done, 0, 2 * num_steps))
    host_env.reset.assert_called_once()
    np.testing.assert_array_equal(host_env.reset.call_args.kwargs["options"]["reset_mask"], done)
    # Collection leaves the supplied state untouched and returns new arrays.
    np.testing.assert_array_equal(rollout_state.observations, 0)
    np.testing.assert_array_equal(rollout_state.episode_returns, 0)
    for leaf in jax.tree.leaves(rollout_state.carry):  # (B, H)
        np.testing.assert_array_equal(leaf, 0)
