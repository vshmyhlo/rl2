from dataclasses import replace
from operator import itemgetter
from pathlib import Path
from typing import Literal, NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import struct
from flax.training.train_state import TrainState

from rl2 import ppo
from rl2.actions import ContinuousSpace, DiscreteSpace
from rl2.envs.kinetix_env import KinetixConfig, KinetixEnv


@struct.dataclass
class Params:
    max_timesteps: int = 2


class State(NamedTuple):
    timestep: jax.Array  # ()
    terminal: jax.Array  # ()
    color: jax.Array  # ()


class Physics:
    """Cheap transitions with controllable boundaries and shaped rewards."""

    def get_obs(self, state: State) -> jax.Array:  # scalar state leaves -> (F=2,)
        return jnp.array([state.timestep, state.color], dtype=jnp.float32)  # (F=2,)

    def reset(self, key: jax.Array, params: Params) -> tuple[jax.Array, State]:  # key: ()
        state = State(jnp.array(0), jnp.array(False), jax.random.uniform(key))  # scalar leaves
        return self.get_obs(state), state  # observation: (F=2,); scalar state leaves

    def step_env(
        self,
        key: jax.Array,  # ()
        state: State,  # scalar leaves
        action: jax.Array,  # ()
        params: Params,
    ) -> tuple[jax.Array, State, jax.Array, jax.Array, dict[str, jax.Array]]:
        state = state._replace(timestep=state.timestep + 1, color=state.color + 0.1)  # scalar leaves
        done = state.terminal | (state.timestep >= params.max_timesteps)  # ()
        return self.get_obs(state), state, jnp.array(0.25), done, {}  # observation: (F=2,); scalar state/reward/done


def test_video_render_preserves_orientation_and_clips_pixels() -> None:
    env = object.__new__(KinetixEnv)
    pixels = jnp.arange(18, dtype=jnp.float32).reshape(3, 2, 3) * 16 - 0.5  # (W=3,H=2,C=3)

    def render_pixels(state: State) -> jax.Array:  # scalar state leaves -> (W,H,C)
        return pixels  # (W,H,C)

    env.render_pixels = render_pixels
    state = State(jnp.array(0), jnp.array(False), jnp.array(0.0))  # scalar leaves
    frame = env.render(state)  # (H,W,C)
    expected = np.rint(np.clip(np.asarray(pixels), 0, 255)).astype(np.uint8).transpose(1, 0, 2)[::-1]  # (H,W,C)
    assert frame.shape == (2, 3, 3) and frame.dtype == jnp.uint8
    np.testing.assert_array_equal(frame, expected)


def test_termination_timeout_and_partial_autoreset() -> None:
    env = object.__new__(KinetixEnv)
    env.env = Physics()
    env.params = Params(max_timesteps=2)
    key = jax.random.key(0)  # ()
    observation, single_state = env.reset(key, env.params)  # (F=2,); scalar state leaves
    observation, single_state, reward, single_done, single_info = env.step(
        key, single_state, jnp.array(0), env.params.replace(max_timesteps=1)
    )  # (F=2,); scalar state/reward/done/flags; final_observation: (F=2,)
    assert observation.shape == single_info["final_observation"].shape == (2,)
    chex.assert_shape((reward, single_done, single_info["terminated"], single_info["truncated"]), ())
    assert bool(single_done) and bool(single_info["truncated"]) and int(single_state.timestep) == 0
    observations, initial = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(key, 4), env.params
    )  # (B=4,F=2); leaves: (B,)
    repeated = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(key, 4), env.params)  # (B,F=2); leaves: (B,)
    chex.assert_trees_all_equal((observations, initial), repeated)
    assert np.unique(initial.color).size == 4
    # Terminal, timeout, both, continuing. Nonzero shaped rewards do not imply termination.
    state = initial._replace(timestep=jnp.array([0, 1, 1, 0]), terminal=jnp.array([True, False, True, False]))
    key = jax.random.key(1)  # ()
    next_observations, next_state, rewards, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        jax.random.split(key, 4), state, jnp.zeros(4, jnp.int32), env.params
    )  # (B,F=2); leaves: (B,...); rewards/done/flags: (B,); final_observation: (B,F=2)
    np.testing.assert_array_equal(info["terminated"], [True, False, True, False])
    np.testing.assert_array_equal(info["truncated"], [False, True, True, False])
    np.testing.assert_array_equal(done, [True, True, True, False])
    np.testing.assert_array_equal(next_state.timestep, [0, 0, 0, 1])
    np.testing.assert_array_equal(info["final_state"].timestep, [1, 2, 2, 1])
    final = jax.tree.map(itemgetter(0), info["final_state"])  # scalar state leaves
    np.testing.assert_array_equal(env.env.get_obs(final), info["final_observation"][0])
    np.testing.assert_array_equal(rewards, np.full(4, 0.25))
    assert next_observations.shape == info["final_observation"].shape == (4, 2)
    reset_keys = jax.vmap(jax.random.split)(jax.random.split(key, 4))[:, 1]  # (B,) typed keys
    reset_observations, reset_state = jax.vmap(env.reset, in_axes=(0, None))(
        reset_keys, env.params
    )  # (B,F=2); leaves: (B,)
    np.testing.assert_array_equal(next_observations[:3], reset_observations[:3])
    np.testing.assert_array_equal(next_state.color[:3], reset_state.color[:3])
    np.testing.assert_array_equal(next_observations[3], info["final_observation"][3])
    np.testing.assert_allclose(next_state.color[3], initial.color[3] + 0.1)
    expected_final = jax.vmap(env.env.get_obs)(state._replace(timestep=state.timestep + 1, color=state.color + 0.1))
    # (B,F=2), including terminal observations before reset
    np.testing.assert_array_equal(info["final_observation"], expected_final)
    # No boundaries: observations and states continue without autoreset.
    next_observations, next_state, rewards, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        jax.random.split(key, 4), initial, jnp.zeros(4, jnp.int32), env.params
    )  # (B,F=2); leaves: (B,...); rewards/done/flags: (B,); final_observation: (B,F=2)
    np.testing.assert_array_equal(next_state.timestep, 1)
    np.testing.assert_array_equal(next_observations, info["final_observation"])


def test_real_kinetix_rollout_update_and_video() -> None:
    pytest.importorskip("kinetix")
    config = replace(
        ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo/kinetix.yaml"),
        num_envs=1,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        target_kl=None,
        bf16=False,
        env=KinetixConfig(action_type="continuous", levels=("s/h4_thrust_aim.json",), max_steps=2),
        model=ppo.ModelConfig(
            encoder=ppo.KinetixEncoderConfig(hidden_size=4, embedding_size=2),
            recurrent=ppo.LSTMConfig(hidden_size=2),
        ),
    )
    env = ppo.make_env(config)
    assert isinstance(env, KinetixEnv) and isinstance(env.action_space, ContinuousSpace)
    observations, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(jax.random.key(0), 1), env.params
    )  # (B=1,F); leaves: (1,...)
    feature_shape = env.env.observation_space(env.params).shape
    assert observations.shape == (1, *feature_shape) and observations.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(observations)))
    model = ppo.ActorCritic(env.action_space, config.model)
    carry = model.initial_carry(1)  # leaves: (1,2)
    starts = jnp.ones(1, bool)  # (1,)
    params = model.init(jax.random.key(1), observations[None], carry, starts[None])["params"]  # parameter pytree
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    runner = ppo.RolloutCarry(env_state, observations, carry, starts, jnp.zeros(1), jnp.zeros(1, jnp.int32))
    advanced, rollout = ppo.collect_rollout(runner, state, env, config, jax.random.key(2))
    assert rollout.batch.observations.shape == (2, 1, *feature_shape)
    assert rollout.batch.actions.shape == (2, 1, env.action_space.size)
    assert rollout.batch.actions.dtype == jnp.float32
    assert rollout.batch.returns.shape == (2, 1)
    output = model.apply({"params": params}, rollout.batch.observations, carry, rollout.batch.episode_starts)
    np.testing.assert_allclose(output.policy.log_prob(rollout.batch.actions), rollout.batch.old_log_probs, atol=1e-5)
    chex.assert_shape((rollout.episode_count, rollout.episode_return_sum, rollout.episode_length_sum), ())
    assert int(rollout.episode_count) == 1
    assert int(rollout.episode_length_sum) == 2
    np.testing.assert_array_equal(advanced.env_state.timestep, 0)
    updated, metrics = ppo.update(state, rollout.batch, config)
    assert int(updated.step) == 1
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves((updated.params, metrics)))
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(state.params), jax.tree.leaves(updated.params)))
    score, length, frames = ppo.play_episode(state, config, jax.random.key(3), carry, record=True, env=env)
    assert np.isfinite(score) and length == 2 and len(frames) == 3
    assert all(frame.shape == (125, 125, 3) and frame.dtype == np.uint8 for frame in frames)


def test_procedural_reset_is_seeded_and_batched() -> None:
    pytest.importorskip("kinetix")
    env = KinetixEnv(KinetixConfig())
    key = jax.random.key(7)  # ()
    observations, state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(key, 2), env.params
    )  # (B=2,F); leaves: (B,...)
    assert observations.shape == (2, *env.env.observation_space(env.params).shape)
    assert observations.dtype == jnp.float32
    assert all(leaf.shape[0] == 2 for leaf in jax.tree.leaves(state))
    chex.assert_trees_all_equal(
        (observations, state), jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(key, 2), env.params)
    )
    assert not np.array_equal(observations[0], observations[1])


@pytest.mark.parametrize("action_type", ["discrete", "continuous"])
def test_kinetix_action_space(action_type: Literal["discrete", "continuous"]) -> None:
    config = KinetixConfig(action_type=action_type, levels=("s/h4_thrust_aim.json",))
    env = KinetixEnv(config)
    if action_type == "discrete":
        assert env.action_space == DiscreteSpace(11)
    else:
        assert isinstance(env.action_space, ContinuousSpace)
        assert env.action_space.shape == (env.action_space.size,)
        assert env.action_space.low == (-1.0,) * env.action_space.size
        assert env.action_space.high == (1.0,) * env.action_space.size
