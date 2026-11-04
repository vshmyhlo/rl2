from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml
from flax.training.train_state import TrainState
from pydantic import ValidationError

from rl2 import ppo
from rl2.actions import ContinuousSpace
from rl2.envs.brax_env import BraxConfig, BraxEnv


@pytest.fixture
def config() -> ppo.Config:
    return ppo.Config(
        env=BraxConfig(max_steps=2),
        seed=0,
        total_steps=2,
        num_envs=1,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        model=ppo.ModelConfig(
            encoder=ppo.MLPEncoderConfig(hidden_size=4, embedding_size=2),
            recurrent=ppo.LSTMConfig(hidden_size=2),
        ),
        bf16=False,
        learning_rate=1e-3,
        anneal_lr=False,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        target_kl=None,
        entropy_coef=0.001,
        max_grad_norm=0.5,
        video_fps=20.0,
    )


@pytest.fixture(scope="module")
def env() -> BraxEnv:
    return BraxEnv(BraxConfig(max_steps=2))


@pytest.mark.parametrize(
    "settings", [{"task": "unknown"}, {"backend": "unknown"}, {"max_steps": 0}, {"max_steps": True}]
)
def test_invalid_settings(settings: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        BraxConfig(**settings)


def test_cli(config: ppo.Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "ppo.yaml"
    config_path.write_text(yaml.safe_dump(asdict(config)))
    config = ppo.load_config(config_path, ppo.Config)
    train = Mock()
    monkeypatch.setattr(ppo, "train", train)
    monkeypatch.setattr("sys.argv", ["ppo", "--config", str(config_path)])
    ppo.main()
    train.assert_called_once_with(config)


@pytest.mark.parametrize(
    "env_config,observation_size,action_size",
    [
        (BraxConfig(task="halfcheetah", max_steps=2), 17, 6),
        (BraxConfig(task="inverted_pendulum", max_steps=2), 4, 1),
        (BraxConfig(task="reacher", max_steps=2), 11, 2),
    ],
    ids=["locomotion", "balance", "manipulation"],
)
def test_starter_tasks(config: ppo.Config, env_config: BraxConfig, observation_size: int, action_size: int) -> None:
    env = ppo.make_env(replace(config, env=env_config))
    assert isinstance(env, BraxEnv)
    assert env.action_space == ContinuousSpace(
        shape=(action_size,), low=(-1.0,) * action_size, high=(1.0,) * action_size
    )
    observation, initial = env.reset(jax.random.key(0), env.params)  # (O,); leaves: (...)
    assert observation.shape == (observation_size,) and observation.dtype == jnp.float32
    states = jax.tree.map(lambda x: jnp.stack([x, x]), initial)  # leaves: (B=2, ...)
    states = states.replace(step_count=jnp.array([0, 1], jnp.int32))
    keys = jax.random.split(jax.random.key(1), 2)  # (B,)
    actions = jnp.zeros((2, action_size), jnp.float32)  # (B, A)
    result = jax.jit(env.step_batch)(keys, states, actions, env.params)
    chex.assert_shape((result.observation, result.info["final_observation"]), (2, observation_size))
    chex.assert_shape((result.reward, result.done), (2,))
    np.testing.assert_array_equal(result.done, [False, True])
    np.testing.assert_array_equal(result.info["terminated"], [False, False])
    np.testing.assert_array_equal(result.info["truncated"], [False, True])
    np.testing.assert_array_equal(result.state.step_count, [1, 0])
    np.testing.assert_array_equal(result.info["final_observation"], result.info["final_state"].physics.obs)
    np.testing.assert_array_equal(result.observation[0], result.info["final_observation"][0])
    reset_observation, _ = env.reset(keys[1], env.params)  # (O,)
    np.testing.assert_array_equal(result.observation[1], reset_observation)


def test_pendulum_fall_terminates() -> None:
    env = BraxEnv(BraxConfig(task="inverted_pendulum"))
    _, state = env.reset(jax.random.key(0), env.params)  # (O=4,); leaves: (...)
    q = env.env.sys.init_q.at[1].set(0.5)  # (2,) pole beyond the 0.2-radian limit
    pipeline_state = env.env.pipeline_init(q, jnp.zeros(2))  # leaves: (...)
    state = state.replace(physics=state.physics.replace(pipeline_state=pipeline_state))
    key = jax.random.key(1)  # ()
    result = env.step(key, state, jnp.zeros(1), env.params)
    assert bool(result.done) and bool(result.info["terminated"])
    assert not bool(result.info["truncated"])
    assert int(result.state.step_count) == 0
    assert abs(float(result.info["final_observation"][1])) > 0.2
    reset_observation, _ = env.reset(key, env.params)  # (O,)
    np.testing.assert_array_equal(result.observation, reset_observation)


def test_native_spaces_rewards_and_batched_timeout(env: BraxEnv) -> None:
    assert env.action_space == ContinuousSpace(shape=(7,), low=(-1.0,) * 7, high=(1.0,) * 7)
    assert float(env.env.dt) == pytest.approx(0.05)
    observation, initial = env.reset(jax.random.key(0), env.params)  # (23,); leaves: (...)
    assert observation.shape == (23,) and observation.dtype == jnp.float32
    np.testing.assert_array_equal(observation[:7], initial.physics.pipeline_state.q[:7])
    np.testing.assert_array_equal(observation[7:14], initial.physics.pipeline_state.qd[:7])
    states = jax.tree.map(lambda x: jnp.stack([x, x]), initial)  # leaves: (B=2, ...)
    states = states.replace(step_count=jnp.array([0, 1], jnp.int32))
    actions = jnp.array([[0.0] * 7, [1.0, -1.0, 0.5, -0.5, 0.0, 0.0, 0.0]], jnp.float32)  # (B, 7)
    keys = jax.random.split(jax.random.key(1), 2)  # (B,)
    result = jax.jit(env.step_batch)(keys, states, actions, env.params)
    chex.assert_shape(result.observation, (2, 23))
    chex.assert_shape((result.reward, result.done, result.info["terminated"], result.info["truncated"]), (2,))
    near = jnp.linalg.norm(observation[17:20] - observation[14:17])  # () pre-step tip/object distance
    distance = jnp.linalg.norm(observation[17:20] - observation[20:23])  # () pre-step object/goal distance
    expected = -distance - 0.5 * near - 0.1 * jnp.square(2 * actions).sum(axis=-1)  # (B,)
    np.testing.assert_allclose(result.reward, expected, rtol=1e-6)
    np.testing.assert_allclose(
        result.info["final_state"].physics.metrics["reward_ctrl"], -jnp.square(2 * actions).sum(-1)
    )
    np.testing.assert_array_equal(result.done, [False, True])
    np.testing.assert_array_equal(result.info["truncated"], [False, True])
    np.testing.assert_array_equal(result.info["terminated"], False)
    np.testing.assert_array_equal(result.state.step_count, [1, 0])
    np.testing.assert_array_equal(result.info["final_state"].step_count, [1, 2])
    np.testing.assert_array_equal(result.info["final_observation"], result.info["final_state"].physics.obs)
    np.testing.assert_array_equal(result.observation[0], result.info["final_observation"][0])
    reset_observation, _ = env.reset(keys[1], env.params)  # (23,)
    np.testing.assert_array_equal(result.observation[1], reset_observation)
    assert not np.array_equal(result.observation[1], result.info["final_observation"][1])


def test_render_passes_pipeline_state(env: BraxEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    _, state = env.reset(jax.random.key(0), env.params)  # state leaves: (...)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)  # (H, W, 3)
    render = Mock(return_value=[frame])
    monkeypatch.setattr(env.env, "render", render)
    assert env.render(state) is frame
    render.assert_called_once_with([state.physics.pipeline_state], height=240, width=320)


@pytest.mark.usefixtures("ppo_mesh")
def test_real_pusher_rollout_and_update(env: BraxEnv, config: ppo.Config) -> None:
    observations, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(jax.random.key(0), 1), env.params
    )  # (B=1,23); leaves: (B,...)
    model = ppo.ActorCritic(action_space=env.action_space, model=config.model)
    carry = model.initial_carry(1)  # leaves: (B,2)
    starts = jnp.ones(1, bool)  # (B,)
    params = model.init(jax.random.key(1), observations[:, None], carry, starts[:, None])["params"]
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    runner = ppo.RolloutCarry(
        env_state=env_state,  # leaves: (B,...)
        observations=observations,  # (B,23)
        model_carry=carry,  # leaves: (B,2)
        episode_starts=starts,  # (B,)
        episode_returns=jnp.zeros(1),  # (B,)
        episode_step_counts=jnp.zeros(1, jnp.int32),  # (B,)
    )
    advanced, rollout = ppo.collect_rollout(runner, train_state, env, config, jax.random.key(2))
    batch = rollout.batch
    assert batch.observations.shape == (1, 2, 23)
    assert batch.actions.shape == (1, 2, 7) and batch.actions.dtype == jnp.float32
    assert batch.returns.shape == (1, 2)
    assert bool(jnp.all((batch.actions >= -1) & (batch.actions <= 1)))
    output = model.apply({"params": params}, batch.observations, carry, batch.episode_starts)
    np.testing.assert_allclose(output.policy.log_prob(batch.actions), batch.old_log_probs, atol=1e-5)
    assert int(rollout.episode_count) == 1 and int(rollout.episode_length_sum) == 2
    np.testing.assert_array_equal(advanced.env_state.step_count, 0)
    updated, _ = ppo.update(train_state, batch, config)
    assert int(updated.step) == 1
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(updated.params)))
    episode = ppo.play_episode(train_state, config, jax.random.key(3), carry, record=False, env=env)
    assert episode.length == 2 and episode.frames == []
