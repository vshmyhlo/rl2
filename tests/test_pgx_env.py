from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml
from flax.training.train_state import TrainState
from pydantic import ValidationError

from rl2 import ppo
from rl2.actions import DiscreteSpace
from rl2.envs.pgx_env import PgxConfig, PgxEnv
from rl2.model.encoders.conv2d import ConvStage


@pytest.fixture
def config() -> ppo.Config:
    return ppo.Config(
        env=PgxConfig(max_steps=2),
        model=ppo.ModelConfig(
            encoder=ppo.Conv2DEncoderConfig(
                stages=(ConvStage(channels=2, blocks=0, resize_factor=1.0),), embedding_size=2
            ),
            recurrent=ppo.LSTMConfig(hidden_size=2),
        ),
        seed=0,
        total_steps=2,
        num_envs=1,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        learning_rate=0.001,
        anneal_lr=False,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        target_kl=None,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        video_fps=10.0,
        ckpt_dir="/tmp/ppo-pgx-test",
        run_id="test",
    )


def test_config_loading_and_encoder_validation(config: ppo.Config, tmp_path: Path) -> None:
    path = tmp_path / "2048.yaml"
    settings = asdict(config)
    path.write_text(yaml.safe_dump(settings))
    loaded = ppo.load_config(path, ppo.Config)
    assert loaded == config
    assert isinstance(ppo.make_env(loaded), PgxEnv)
    with pytest.raises(ValidationError, match="Pgx requires model.encoder.type: conv2d"):
        replace(config, model=replace(config.model, encoder=ppo.MLPEncoderConfig()))


@pytest.mark.parametrize("settings", [{"game": "chess"}, {"max_steps": 0}, {"max_steps": True}])
def test_invalid_settings(settings: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        PgxConfig(**settings)


def test_native_observation_actions_and_reward() -> None:
    env = PgxEnv(PgxConfig())
    key = jax.random.key(0)  # ()
    observation, state = jax.jit(env.reset)(key, env.params)  # (4,4,31); leaves: (...)
    assert observation.shape == (4, 4, 31) and observation.dtype == jnp.bool_
    np.testing.assert_array_equal(observation.sum(axis=-1), 1)
    assert env.action_space == DiscreteSpace(n=4)
    state = state.replace(_board=jnp.array([1, 1, 0, 0] + [0] * 12, jnp.int32))  # board: (16,)
    result = jax.jit(env.step)(key, state, jnp.array(0, jnp.int32), env.params)
    assert result.reward.shape == result.done.shape == ()
    assert float(result.reward) == 4.0 and result.reward.dtype == jnp.float32
    assert int(result.state._board[0]) == 2
    assert not bool(result.done)
    np.testing.assert_array_equal(result.observation, env.env.observe(result.state))
    frame = env.render(result.state)  # (256,256,3)
    assert frame.shape == (256, 256, 3) and frame.dtype == np.uint8


def test_batched_termination_timeout_and_autoreset() -> None:
    env = PgxEnv(PgxConfig(max_steps=2))
    keys = jax.random.split(jax.random.key(0), 4)  # (B=4,)
    observations, state = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(keys, env.params)
    assert observations.shape == (4, 4, 4, 31)
    # A single top-left tile cannot move left; right is legal.
    board = jnp.array([1] + [0] * 15, jnp.int32)  # (16,)
    state = state.replace(
        _board=jnp.broadcast_to(board, (4, 16)),  # (B,16)
        legal_action_mask=jnp.broadcast_to(jnp.array([False, False, True, True]), (4, 4)),  # (B,4)
        _step_count=jnp.array([0, 1, 1, 0], jnp.int32),  # (B,)
    )
    result = jax.jit(env.step_batch)(keys, state, jnp.array([0, 2, 0, 2], jnp.int32), env.params)
    np.testing.assert_array_equal(result.info["terminated"], [True, False, True, False])
    np.testing.assert_array_equal(result.info["truncated"], [False, True, True, False])
    np.testing.assert_array_equal(result.done, [True, True, True, False])
    np.testing.assert_array_equal(result.reward, [-1, 0, -1, 0])
    np.testing.assert_array_equal(result.state._step_count, [0, 0, 0, 1])
    np.testing.assert_array_equal(result.info["final_state"]._step_count, [1, 2, 2, 1])
    reset_keys = jax.vmap(jax.random.split)(keys)[:, 1]  # (B,)
    reset_observations, _ = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env.params)
    np.testing.assert_array_equal(result.observation[:3], reset_observations[:3])
    np.testing.assert_array_equal(result.observation[3], result.info["final_observation"][3])
    assert result.observation.shape == result.info["final_observation"].shape == (4, 4, 4, 31)
    np.testing.assert_array_equal(
        result.info["final_observation"], jax.vmap(env.env.observe)(result.info["final_state"])
    )


def test_game_over_after_last_empty_cell_is_filled() -> None:
    env = PgxEnv(PgxConfig())
    key = jax.random.key(0)  # ()
    _, state = env.reset(key, env.params)  # (4,4,31); leaves: (...)
    state = state.replace(
        _board=jnp.array(list(range(1, 16)) + [0], jnp.int32),  # (16,)
        legal_action_mask=jnp.array([False, False, True, True]),  # (4,)
    )
    result = jax.jit(env.step)(key, state, jnp.array(2, jnp.int32), env.params)
    assert bool(result.info["terminated"]) and not bool(result.info["truncated"])
    assert float(result.reward) == 0.0
    assert bool(jnp.all(result.info["final_state"]._board > 0))
    assert int(result.state._step_count) == 0


@pytest.mark.usefixtures("ppo_mesh")
def test_ppo_rollout_update_and_video(config: ppo.Config) -> None:
    env = ppo.make_env(config)
    observations, state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(jax.random.key(0), 1), env.params
    )  # (B=1,4,4,31); leaves: (B,...)
    model = ppo.ActorCritic(action_space=env.action_space, model=config.model, dtype=jnp.float32)
    carry = model.initial_carry(1)  # leaves: (B,H=2)
    starts = jnp.ones(1, bool)  # (B,)
    params = model.init(jax.random.key(1), observations[:, None], carry, starts[:, None])["params"]
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(0.001))
    runner = ppo.RolloutCarry(
        env_state=state,  # leaves: (B,...)
        observations=observations,  # (B,4,4,31)
        model_carry=carry,  # leaves: (B,H)
        episode_starts=starts,  # (B,)
        episode_returns=jnp.zeros(1),  # (B,)
        episode_step_counts=jnp.zeros(1, jnp.int32),  # (B,)
    )
    _, rollout = ppo.collect_rollout(runner, train_state, env, config, jax.random.key(2))
    batch = rollout.batch
    assert batch.observations.shape == (1, 2, 4, 4, 31) and batch.observations.dtype == jnp.bool_
    assert batch.actions.shape == batch.returns.shape == (1, 2)
    output = model.apply({"params": params}, batch.observations, carry, batch.episode_starts)
    np.testing.assert_allclose(output.policy.log_prob(batch.actions), batch.old_log_probs, atol=1e-5)
    assert int(rollout.episode_count) >= 1
    updated, _ = ppo.update(train_state, batch, config)
    assert int(updated.step) == 1
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(updated.params)))
    episode = ppo.play_episode(train_state, config, jax.random.key(3), carry, record=True, env=env)
    assert 1 <= episode.length <= 2 and len(episode.frames) == episode.length + 1
    assert all(frame.shape == (256, 256, 3) and frame.dtype == np.uint8 for frame in episode.frames)
