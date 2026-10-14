from dataclasses import replace
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState
from jaxtari.paths import get_base_sprite_dir
from pydantic import ValidationError

from rl2 import ppo
from rl2.actions import DiscreteSpace
from rl2.envs.jaxtari_env import JAXtariConfig, JAXtariEnv
from rl2.model.encoders.rgb import ConvStage

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/ppo/jaxtari/breakout.yaml"


@pytest.fixture(scope="module")
def env() -> JAXtariEnv:
    if not (Path(get_base_sprite_dir()) / "breakout/background.npy").is_file():
        pytest.skip("Install sprites with: uv run python -m jaxtari.install_sprites")
    return JAXtariEnv(JAXtariConfig(max_steps=2))


@pytest.mark.parametrize("settings", [{"game": "unknown"}, {"max_steps": 0}, {"max_steps": True}])
def test_invalid_settings(settings: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        JAXtariConfig(**settings)


def test_preset_and_encoder_validation() -> None:
    config = ppo.load_config(CONFIG_PATH)
    assert config.env == JAXtariConfig()
    assert config.model.encoder.type == "rgb"
    assert config.run_id == "ppo_JAXtari_breakout_seed1"
    with pytest.raises(ValidationError, match="JAXtari requires model.encoder.type: rgb"):
        replace(config, model=replace(config.model, encoder=ppo.MLPEncoderConfig()))


def test_native_frames_actions_and_unclipped_reward(env: JAXtariEnv) -> None:
    key = jax.random.key(0)  # ()
    observation, state = env.reset(key, env.params)  # (210,160,3); leaves: (...)
    assert observation.shape == (210, 160, 3) and observation.dtype == jnp.uint8
    assert env.action_space == DiscreteSpace(n=4)
    np.testing.assert_array_equal(observation, env.env.render(state))
    idle = env.step(key, state, jnp.array(0, jnp.int32), env.params)
    fired = env.step(key, state, jnp.array(1, jnp.int32), env.params)
    right = env.step(key, state, jnp.array(2, jnp.int32), env.params)
    assert not bool(idle.state.game_started) and bool(fired.state.game_started)
    assert int(right.state.player_x) > int(state.player_x)
    chex.assert_shape((fired.reward, fired.done, fired.info["truncated"]), ())
    # A ball rising into the top row earns seven points, without reward clipping.
    near_brick = state.replace(
        ball_x=jnp.array(20),  # ()
        ball_y=jnp.array(64),  # ()
        ball_vel_x=jnp.array(0),  # ()
        ball_vel_y=jnp.array(-1),  # ()
        ball_direction_idx=jnp.array(2),  # ()
        game_started=jnp.array(1),  # ()
    )
    hit = env.step(key, near_brick, jnp.array(0, jnp.int32), env.params)
    assert float(hit.reward) == 7.0 and hit.reward.dtype == jnp.float32
    assert int(hit.state.score) == 7
    assert int(hit.state.blocks.sum()) == int(state.blocks.sum()) - 1


def test_batched_episode_boundaries_and_final_frames(env: JAXtariEnv) -> None:
    keys = jax.random.split(jax.random.key(0), 4)  # (B=4,)
    observations, initial = jax.vmap(env.reset, in_axes=(0, None))(keys, env.params)  # (B,210,160,3); leaves: (B,...)
    assert observations.shape == (4, 210, 160, 3)
    # Terminal, timeout, both, and continuing environments.
    state = initial.replace(lives=jnp.array([0, 5, 0, 5]), step_counter=jnp.array([0, 1, 1, 0]))  # leaves: (B,...)
    actions = jnp.full(4, 2, jnp.int32)  # (B,) RIGHT
    result = jax.jit(env.step_batch)(keys, state, actions, env.params)
    np.testing.assert_array_equal(result.info["terminated"], [True, False, True, False])
    np.testing.assert_array_equal(result.info["truncated"], [False, True, True, False])
    np.testing.assert_array_equal(result.done, [True, True, True, False])
    np.testing.assert_array_equal(result.state.step_counter, [0, 0, 0, 1])
    np.testing.assert_array_equal(result.info["final_state"].step_counter, [1, 2, 2, 1])
    np.testing.assert_array_equal(result.observation[:3], observations[:3])
    np.testing.assert_array_equal(result.observation[3], result.info["final_observation"][3])
    np.testing.assert_array_equal(result.info["final_observation"], jax.vmap(env.render)(result.info["final_state"]))
    assert result.info["final_observation"].shape == result.observation.shape == (4, 210, 160, 3)
    assert not np.array_equal(result.info["final_observation"][0], observations[0])
    chex.assert_shape((result.done, result.reward), (4,))


def test_losing_one_life_continues_the_game(env: JAXtariEnv) -> None:
    key = jax.random.key(0)  # ()
    _, state = env.reset(key, env.params)  # (210,160,3); leaves: (...)
    state = state.replace(ball_y=jnp.array(220), game_started=jnp.array(1))  # scalar fields
    result = env.step(key, state, jnp.array(0, jnp.int32), env.params)
    assert int(result.state.lives) == 4 and not bool(result.done)
    assert not bool(result.state.game_started)  # FIRE launches the next ball.


def test_real_breakout_ppo_rollout_update_and_video(env: JAXtariEnv) -> None:
    config = replace(
        ppo.load_config(CONFIG_PATH),
        env=JAXtariConfig(max_steps=2),
        num_envs=1,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        target_kl=None,
        model=ppo.ModelConfig(
            encoder=ppo.RGBEncoderConfig(
                stages=(ConvStage(channels=2, blocks=0, resize_factor=0.125),), embedding_size=2
            ),
            recurrent=ppo.LSTMConfig(hidden_size=2),
        ),
    )
    observations, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(jax.random.key(0), 1), env.params
    )  # (B=1,210,160,3); leaves: (B,...)
    model = ppo.ActorCritic(action_space=env.action_space, model=config.model, dtype=jnp.bfloat16)
    carry = model.initial_carry(1)  # leaves: (B,H=2)
    starts = jnp.ones(1, bool)  # (B,)
    params = model.init(jax.random.key(1), observations[None], carry, starts[None])["params"]
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    runner = ppo.RolloutCarry(
        env_state=env_state,  # leaves: (B,...)
        observations=observations,  # (B,210,160,3)
        model_carry=carry,  # leaves: (B,H)
        episode_starts=starts,  # (B,)
        episode_returns=jnp.zeros(1),  # (B,)
        episode_step_counts=jnp.zeros(1, jnp.int32),  # (B,)
    )
    advanced, rollout = ppo.collect_rollout(runner, train_state, env, config, jax.random.key(2))
    batch = rollout.batch
    assert batch.observations.shape == (2, 1, 210, 160, 3) and batch.observations.dtype == jnp.uint8
    assert batch.actions.shape == batch.returns.shape == (2, 1)
    assert batch.actions.dtype == jnp.int32
    assert bool(jnp.all((batch.actions >= 0) & (batch.actions < 4)))
    output = model.apply({"params": params}, batch.observations, carry, batch.episode_starts)
    np.testing.assert_allclose(output.policy.log_prob(batch.actions), batch.old_log_probs, atol=1e-5)
    assert int(rollout.episode_count) == 1 and int(rollout.episode_length_sum) == 2
    np.testing.assert_array_equal(advanced.env_state.step_counter, 0)
    updated, _ = ppo.update(train_state, batch, config)
    assert int(updated.step) == 1
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(updated.params)))
    episode = ppo.play_episode(train_state, config, jax.random.key(3), carry, record=True, env=env)
    assert episode.length == 2 and len(episode.frames) == 3
    assert all(frame.shape == (210, 160, 3) and frame.dtype == np.uint8 for frame in episode.frames)
