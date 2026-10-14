from dataclasses import replace
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState
from pydantic import ValidationError

from rl2 import ppo
from rl2.envs.foragax_env import ForagaxConfig, ForagaxEnv
from rl2.model.encoders.rgb import ConvStage

PRESET = Path(__file__).resolve().parents[1] / "configs/ppo/foragax/square_wave_two_biome.yaml"


@pytest.mark.parametrize(
    "observation_type,aperture,channels",
    [("color", 9, 3), ("object", 1, 6), ("rgb", 9, 3)],
    ids=["symbolic-colors", "single-cell-objects", "pixels"],
)
def test_observation_shapes(observation_type: str, aperture: int, channels: int) -> None:
    env = ForagaxEnv(ForagaxConfig(observation_type=observation_type, aperture_size=aperture))
    observation, state = env.reset(jax.random.key(0), env.params)  # (H, W, C); state leaves: (...)
    assert observation.shape == (aperture, aperture, channels)
    assert observation.dtype == jnp.float32
    assert bool(jnp.all((observation >= 0) & (observation <= 1)))
    np.testing.assert_array_equal(observation, env.env.get_obs(state, env.params))
    assert env.action_space.n == 4
    if observation_type == "color":
        for object_id in (1, 3):  # Oyster colors alias across biomes.
            changed = state.replace(
                object_state=state.object_state.replace(
                    object_id=state.object_state.object_id.at[state.pos[1], state.pos[0]].set(object_id)  # (15, 24)
                )
            )
            np.testing.assert_array_equal(env.env.get_obs(changed, env.params)[4, 4], [1, 0, 0])


def test_batched_timeout_preserves_final_observation_and_resets_clock() -> None:
    env = ForagaxEnv(ForagaxConfig(max_steps=2))
    observation, state = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(
        jax.random.split(jax.random.key(0), 2), env.params
    )  # (B=2, H=9, W=9, C=3); state leaves: (B, ...)
    chex.assert_shape(observation, (2, 9, 9, 3))
    state = state.replace(time=jnp.array([0, 1], jnp.int32))  # leaves: (B, ...)
    keys = jax.random.split(jax.random.key(1), 2)  # (B,)
    result = jax.jit(env.step_batch)(keys, state, jnp.zeros(2, jnp.int32), env.params)
    chex.assert_shape(result.observation, (2, 9, 9, 3))
    chex.assert_shape((result.reward, result.done, result.info["truncated"]), (2,))
    np.testing.assert_array_equal(result.done, [False, True])
    np.testing.assert_array_equal(result.info["truncated"], [False, True])
    np.testing.assert_array_equal(result.info["terminated"], False)
    np.testing.assert_array_equal(result.info["final_state"].time, [1, 2])
    np.testing.assert_array_equal(result.state.time, [1, 0])
    np.testing.assert_array_equal(result.state.offset, 0)
    final_obs = jax.vmap(env.env.get_obs, in_axes=(0, None))(result.info["final_state"], env.params)  # (B,H,W,C)
    np.testing.assert_array_equal(result.info["final_observation"], final_obs)
    np.testing.assert_array_equal(result.observation[0], final_obs[0])
    reset_observation, reset_state = env.reset(jax.random.split(keys[1])[1], env.params)  # (H,W,C); leaves: (...)
    chex.assert_trees_all_equal(jax.tree.map(lambda x: x[1], result.state), reset_state)
    np.testing.assert_array_equal(result.observation[1], reset_observation)
    frame = env.render(jax.tree.map(lambda x: x[0], result.info["final_state"]))  # (135,216,3)
    assert frame.shape == (135, 216, 3) and frame.dtype == jnp.uint8


@pytest.mark.parametrize("delay", [0, 2], ids=["immediate", "delayed"])
def test_collection_rewards_switch_phase_and_respect_delay(delay: int) -> None:
    env = ForagaxEnv(ForagaxConfig(period=8, reward_delay=delay))
    _, state = env.reset(jax.random.key(0), env.params)  # (H,W,C); state leaves: (...)
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (8, *jnp.shape(x))), state)  # leaves: (B=8,...)
    states = states.replace(
        pos=jnp.zeros((8, 2), jnp.int32),  # (B, 2)
        time=jnp.repeat(jnp.array([2, 6], jnp.int32), 4),  # (B,): opposite wave phases
        object_state=states.object_state.replace(
            object_id=jnp.zeros((8, 15, 24), jnp.int32).at[:, 0, 1].set(jnp.tile(jnp.arange(1, 5), 2))  # (B,H,W)
        ),
    )
    keys = jax.random.split(jax.random.key(1), 8)  # (B,)
    actions = jnp.ones(8, jnp.int32)  # (B,): right
    step = jax.jit(env.step_batch)
    result = step(keys, states, actions, env.params)
    expected = jnp.array([4, -2, -8, -14, -14, -8, -2, 4], jnp.float32)  # (B,)
    np.testing.assert_array_equal(result.reward, expected if delay == 0 else jnp.zeros(8))
    np.testing.assert_array_equal(result.state.pos, jnp.tile(jnp.array([1, 0]), (8, 1)))
    np.testing.assert_array_equal(result.state.object_state.object_id[:, 0, 1], 0)
    second = step(keys, result.state, actions, env.params)
    np.testing.assert_array_equal(second.reward, 0)
    third = step(keys, second.state, actions, env.params)
    np.testing.assert_array_equal(third.reward, expected if delay else jnp.zeros(8))
    np.testing.assert_array_equal(third.done, False)


def test_movement_wraps_and_walls_block() -> None:
    env = ForagaxEnv(ForagaxConfig())
    _, state = env.reset(jax.random.key(0), env.params)  # (H,W,C); state leaves: (...)
    state = state.replace(
        pos=jnp.zeros(2, jnp.int32),  # (2,)
        object_state=state.object_state.replace(object_id=jnp.zeros((15, 24), jnp.int32)),  # grid: (H,W)
    )
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (4, *jnp.shape(x))), state)  # leaves: (B=4,...)
    states = states.replace(
        object_state=states.object_state.replace(object_id=states.object_state.object_id.at[0, 1, 0].set(5))
    )
    result = jax.jit(env.step_batch)(
        jax.random.split(jax.random.key(1), 4), states, jnp.arange(4, dtype=jnp.int32), env.params
    )
    np.testing.assert_array_equal(result.state.pos, [[0, 0], [1, 0], [0, 14], [23, 0]])
    np.testing.assert_array_equal(result.reward, 0)


@pytest.mark.parametrize(
    "settings",
    [{"aperture_size": 0}, {"period": 1}, {"reward_delay": -1}, {"max_steps": 0}, {"env_id": "unknown"}],
)
def test_invalid_settings(settings: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ForagaxConfig(**settings)


def test_requires_cnn() -> None:
    config = ppo.load_config(PRESET, ppo.Config)
    with pytest.raises(ValidationError, match="Foragax requires model.encoder.type: rgb"):
        replace(config, model=ppo.ModelConfig(encoder=ppo.MLPEncoderConfig()))


def test_ppo_update_and_playback() -> None:
    config = replace(
        ppo.load_config(PRESET, ppo.Config),
        env=ForagaxConfig(max_steps=2),
        num_envs=1,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        target_kl=None,
        model=ppo.ModelConfig(
            encoder=ppo.RGBEncoderConfig(
                stages=(ConvStage(channels=2, blocks=0, resize_factor=1.0),), embedding_size=2
            ),
            recurrent=ppo.LSTMConfig(hidden_size=2),
        ),
    )
    env = ppo.make_env(config)
    assert isinstance(env, ForagaxEnv)
    observations, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(jax.random.key(0), 1), env.params
    )  # (B=1,H=9,W=9,C=3); state leaves: (B,...)
    model = ppo.ActorCritic(action_space=env.action_space, model=config.model)
    carry = model.initial_carry(1)  # leaves: (B,H=2)
    starts = jnp.ones(1, bool)  # (B,)
    params = model.init(jax.random.key(1), observations[None], carry, starts[None])["params"]  # leaves: (...)
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    runner = ppo.RolloutCarry(
        env_state=env_state,  # leaves: (B,...)
        observations=observations,  # (B,H,W,C)
        model_carry=carry,  # leaves: (B,H)
        episode_starts=starts,  # (B,)
        episode_returns=jnp.zeros(1),  # (B,)
        episode_step_counts=jnp.zeros(1, jnp.int32),  # (B,)
    )
    advanced, rollout = ppo.collect_rollout(runner, train_state, env, config, jax.random.key(2))
    chex.assert_shape(rollout.batch.observations, (2, 1, 9, 9, 3))
    chex.assert_shape((rollout.batch.actions, rollout.batch.returns), (2, 1))
    assert int(rollout.episode_count) == 1 and int(rollout.episode_length_sum) == 2
    np.testing.assert_array_equal(advanced.env_state.time, 0)
    updated, _ = ppo.update(train_state, rollout.batch, config)
    assert int(updated.step) == 1
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(updated.params)))
    episode = ppo.play_episode(updated, config, jax.random.key(3), carry, record=False, env=env)
    assert episode.length == 2 and not episode.frames
