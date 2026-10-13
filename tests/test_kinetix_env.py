from dataclasses import replace
from operator import itemgetter
from pathlib import Path
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import struct
from flax.training.train_state import TrainState

from rl2 import ppo
from rl2.envs.kinetix_env import KinetixConfig, KinetixEnv, encode_pixels
from rl2.model.encoders.rgb import ConvStage


class Pixels(NamedTuple):
    image: jax.Array  # (W=3, H=2, C=3)


@struct.dataclass
class Params:
    max_timesteps: int = 2


class State(NamedTuple):
    timestep: jax.Array  # ()
    terminal: jax.Array  # ()
    color: jax.Array  # ()


class Physics:
    """Cheap transitions with controllable boundaries and shaped rewards."""

    def get_obs(self, state: State) -> Pixels:
        return Pixels(jnp.full((3, 2, 3), state.color))  # image: (W, H, C)

    def reset(self, key: jax.Array, params: Params) -> tuple[Pixels, State]:  # key: ()
        state = State(jnp.array(0), jnp.array(False), jax.random.uniform(key))  # scalar leaves
        return self.get_obs(state), state  # image: (W, H, C); scalar state leaves

    def step_env(
        self,
        key: jax.Array,  # ()
        state: State,  # scalar leaves
        action: jax.Array,  # ()
        params: Params,
    ) -> tuple[Pixels, State, jax.Array, jax.Array, dict[str, jax.Array]]:
        state = state._replace(timestep=state.timestep + 1, color=state.color + 0.1)  # scalar leaves
        done = state.terminal | (state.timestep >= params.max_timesteps)  # ()
        return self.get_obs(state), state, jnp.array(0.25), done, {}  # image: (W,H,C); scalar state/reward/done


@pytest.mark.parametrize("batched", [False, True], ids=["frame", "batch"])
def test_pixel_encoding_preserves_scale_orientation_and_shapes(batched: bool) -> None:
    pixels = jnp.arange(18, dtype=jnp.float32).reshape(3, 2, 3) * 15 / 255  # (W=3,H=2,C=3)
    images = pixels[None] if batched else pixels  # (1,W,H,C) or (W,H,C)
    observations = encode_pixels(images)  # (1,C,H,W) or (C,H,W)
    expected = np.moveaxis(np.rint(np.asarray(pixels) * 255), (2, 1, 0), (0, 1, 2))[:, ::-1, :]  # (C,H,W)
    np.testing.assert_array_equal(observations, expected[None] if batched else expected)
    assert observations.dtype == jnp.uint8
    frame = KinetixEnv.render_observation(observations[0] if batched else observations)  # (H,W,C)
    np.testing.assert_array_equal(frame, expected.transpose(1, 2, 0))


def test_termination_timeout_and_partial_autoreset() -> None:
    env = object.__new__(KinetixEnv)
    env.env = Physics()
    env.params = Params(max_timesteps=2)
    key = jax.random.key(0)  # ()
    observation, single_state = env.reset(key, env.params)  # (3,2,3); scalar state leaves
    observation, single_state, reward, single_done, single_info = env.step(
        key, single_state, jnp.array(0), env.params.replace(max_timesteps=1)
    )  # (3,2,3); scalar state/reward/done/flags; final_observation: (3,2,3)
    assert observation.shape == single_info["final_observation"].shape == (3, 2, 3)
    chex.assert_shape((reward, single_done, single_info["terminated"], single_info["truncated"]), ())
    assert bool(single_done) and bool(single_info["truncated"]) and int(single_state.timestep) == 0
    observations, initial = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(key, 4), env.params
    )  # (B=4,3,2,3); leaves: (B,)
    repeated = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(key, 4), env.params)  # (B,3,2,3); leaves: (B,)
    chex.assert_trees_all_equal((observations, initial), repeated)
    assert np.unique(initial.color).size == 4
    # Terminal, timeout, both, continuing. Nonzero shaped rewards do not imply termination.
    state = initial._replace(timestep=jnp.array([0, 1, 1, 0]), terminal=jnp.array([True, False, True, False]))
    key = jax.random.key(1)  # ()
    next_observations, next_state, rewards, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        jax.random.split(key, 4), state, jnp.zeros(4, jnp.int32), env.params
    )  # (B,C,H,W); leaves: (B,...); rewards/done/flags: (B,); final_observation: (B,C,H,W)
    np.testing.assert_array_equal(info["terminated"], [True, False, True, False])
    np.testing.assert_array_equal(info["truncated"], [False, True, True, False])
    np.testing.assert_array_equal(done, [True, True, True, False])
    np.testing.assert_array_equal(next_state.timestep, [0, 0, 0, 1])
    np.testing.assert_array_equal(info["final_state"].timestep, [1, 2, 2, 1])
    final = jax.tree.map(itemgetter(0), info["final_state"])  # scalar state leaves
    np.testing.assert_array_equal(env.render(final), env.render_observation(info["final_observation"][0]))
    np.testing.assert_array_equal(rewards, np.full(4, 0.25))
    assert next_observations.shape == info["final_observation"].shape == (4, 3, 2, 3)
    reset_keys = jax.vmap(jax.random.split)(jax.random.split(key, 4))[:, 1]  # (B,) typed keys
    reset_observations, reset_state = jax.vmap(env.reset, in_axes=(0, None))(
        reset_keys, env.params
    )  # (B,3,2,3); leaves: (B,)
    np.testing.assert_array_equal(next_observations[:3], reset_observations[:3])
    np.testing.assert_array_equal(next_state.color[:3], reset_state.color[:3])
    np.testing.assert_array_equal(next_observations[3], info["final_observation"][3])
    np.testing.assert_allclose(next_state.color[3], initial.color[3] + 0.1)
    expected_final = encode_pixels(jax.vmap(env.env.get_obs)(state._replace(color=state.color + 0.1)).image)
    # (B,3,2,3), including terminal frames before reset
    np.testing.assert_array_equal(info["final_observation"], expected_final)
    # No boundaries: observations and states continue without autoreset.
    next_observations, next_state, rewards, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        jax.random.split(key, 4), initial, jnp.zeros(4, jnp.int32), env.params
    )  # (B,C,H,W); leaves: (B,...); rewards/done/flags: (B,); final_observation: (B,C,H,W)
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
        kinetix=KinetixConfig(levels=("s/h4_thrust_aim.json",), max_steps=2),
        model=ppo.ModelConfig(
            encoder=ppo.RGBEncoderConfig(stages=(ConvStage(2, blocks=0, resize_factor=0.0625),), embedding_size=2),
            recurrent=ppo.LSTMConfig(hidden_size=2),
        ),
    )
    env = ppo.make_env(config)
    assert isinstance(env, KinetixEnv) and env.num_actions == 11
    observations, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(jax.random.key(0), 1), env.params
    )  # (1,3,125,125); leaves: (1,...)
    assert observations.shape == (1, 3, 125, 125) and observations.dtype == jnp.uint8
    model = ppo.ActorCritic(env.num_actions, config.model)
    carry = model.initial_carry(1)  # leaves: (1,2)
    starts = jnp.ones(1, bool)  # (1,)
    params = model.init(jax.random.key(1), observations[None], carry, starts[None])["params"]  # parameter pytree
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    runner = ppo.RolloutCarry(env_state, observations, carry, starts, jnp.zeros(1), jnp.zeros(1, jnp.int32))
    advanced, rollout = ppo.collect_rollout(runner, state, env, config, jax.random.key(2))
    assert rollout.batch.observations.shape == (2, 1, 3, 125, 125)
    chex.assert_shape((rollout.batch.actions, rollout.batch.returns), (2, 1))
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
    )  # (B=2,3,125,125); leaves: (B,...)
    assert observations.shape == (2, 3, 125, 125) and observations.dtype == jnp.uint8
    assert all(leaf.shape[0] == 2 for leaf in jax.tree.leaves(state))
    chex.assert_trees_all_equal(
        (observations, state), jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(key, 2), env.params)
    )
    assert not np.array_equal(observations[0], observations[1])
