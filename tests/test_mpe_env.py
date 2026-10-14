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

from rl2 import train_mappo as ppo
from rl2.envs.mpe_env import MPEConfig, MPEEnv, MPEEnvId

PRESET = Path(__file__).resolve().parents[1] / "configs/mappo/jaxmarl/mpe/simple_spread.yaml"


@pytest.mark.parametrize("num_agents,num_landmarks", [(3, 3), (1, 2)], ids=["cooperative", "single-agent"])
def test_shapes_rewards_and_timeout(num_agents: int, num_landmarks: int) -> None:
    env = MPEEnv(MPEConfig(num_agents=num_agents, num_landmarks=num_landmarks, max_steps=2))
    observation, state = env.reset(jax.random.key(0))  # (N, O)/(N, A); state leaves: (...)
    features = 4 * num_agents + 2 * num_landmarks
    chex.assert_shape(observation.observations, (num_agents, features))
    chex.assert_shape(observation.action_masks, (num_agents, 5))
    assert observation.observations.dtype == jnp.float32
    assert observation.action_masks.dtype == jnp.bool_
    np.testing.assert_array_equal(observation.action_masks, True)
    raw = env.env.get_obs(state)  # agent leaves: (O,)
    for i, agent in enumerate(env.agents):
        np.testing.assert_array_equal(observation.observations[i], raw[agent])
    actions = jnp.zeros(num_agents, jnp.int32)  # (N,)
    first = env.step(jax.random.key(1), state, actions)
    assert not bool(first.done) and not bool(first.terminated)
    np.testing.assert_array_equal(first.agent_dones, False)
    np.testing.assert_array_equal(env.memory_resets(first.state), False)
    chex.assert_trees_all_equal(first.state, first.final_state)
    rewards = env.env.rewards(first.final_state)  # agent leaves: ()
    np.testing.assert_allclose(first.rewards, jnp.stack([rewards[agent] for agent in env.agents]))
    states = jax.tree.map(lambda x: jnp.stack([x, x]), first.state)  # leaves: (E=2, ...)
    result = jax.jit(jax.vmap(env.step))(
        jax.random.split(jax.random.key(2), 2), states, jnp.zeros((2, num_agents), jnp.int32)
    )
    chex.assert_shape(result.observation.observations, (2, num_agents, features))
    chex.assert_shape(result.observation.action_masks, (2, num_agents, 5))
    chex.assert_shape(result.rewards, (2, num_agents))
    chex.assert_shape(result.done, (2,))
    chex.assert_shape(result.agent_dones, (2, num_agents))
    np.testing.assert_array_equal(result.done, True)
    np.testing.assert_array_equal(result.truncated, True)
    np.testing.assert_array_equal(result.terminated, False)
    np.testing.assert_array_equal(result.agent_dones, True)
    np.testing.assert_array_equal(result.final_state.step, 2)
    np.testing.assert_array_equal(result.state.step, 0)
    np.testing.assert_array_equal(jax.vmap(env.memory_resets)(result.final_state), False)
    assert not np.array_equal(result.observation.observations, result.final_observation.observations)
    raw_final = jax.vmap(env.env.get_obs)(result.final_state)  # agent leaves: (E, O)
    for i, agent in enumerate(env.agents):
        np.testing.assert_array_equal(result.final_observation.observations[:, i], raw_final[agent])


@pytest.mark.parametrize(
    "overrides",
    [{"num_agents": 0}, {"num_landmarks": 0}, {"max_steps": 0}, {"local_ratio": -0.1}, {"local_ratio": 1.1}],
)
def test_invalid_mpe_settings(overrides: dict[str, int | float]) -> None:
    with pytest.raises(ValidationError):
        MPEConfig(**overrides)


def test_config_and_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ppo.load_config(PRESET)
    assert config.env.id == "MPE_simple_spread_v3"
    assert isinstance(ppo.make_env(config), MPEEnv)
    with pytest.raises(ValueError, match="num_agents"):
        ppo.make_env(replace(config, num_minibatches=5))
    calls: list[ppo.Config] = []
    monkeypatch.setattr(ppo, "train", calls.append)
    monkeypatch.setattr("sys.argv", ["mappo", "--config", str(PRESET)])
    ppo.main()
    assert calls == [config]


def test_reference_communication_movement_and_timeout() -> None:
    config = ppo.load_config(PRESET.with_name("simple_reference.yaml"))
    env = ppo.make_env(replace(config, env=replace(config.env, max_steps=2)))
    observation, state = env.reset(jax.random.key(0))  # (N=2, O=21)/(N, A=50); state leaves: (...)
    chex.assert_shape(observation.observations, (2, 21))
    chex.assert_shape(observation.action_masks, (2, 50))
    np.testing.assert_array_equal(observation.action_masks, True)
    np.testing.assert_array_equal(observation.observations[:, -10:], 0)
    assert env.action_space.n == 50
    # Each agent observes the partner's target, which it must communicate back.
    goals = jnp.argmax(observation.observations[:, 8:11], axis=-1)  # (N,)
    np.testing.assert_array_equal(goals, state.goal[::-1])
    first = env.step(jax.random.key(1), state, jnp.array([12, 46], jnp.int32))
    messages = jax.nn.one_hot(jnp.array([2, 9]), 10)  # (N, C=10)
    np.testing.assert_array_equal(first.state.c, messages)
    np.testing.assert_array_equal(first.observation.observations[:, -10:], messages[::-1])
    assert float(first.state.p_vel[0, 0]) > 0
    assert float(first.state.p_vel[1, 0]) < 0
    assert not bool(first.done)
    np.testing.assert_allclose(first.rewards[0], first.rewards[1])
    states = jax.tree.map(lambda x: jnp.stack([x, x]), first.state)  # leaves: (E=2, ...)
    result = jax.jit(jax.vmap(env.step))(
        jax.random.split(jax.random.key(2), 2), states, jnp.tile(jnp.array([0, 49], jnp.int32), (2, 1))
    )
    chex.assert_shape(result.observation.observations, (2, 2, 21))
    chex.assert_shape(result.observation.action_masks, (2, 2, 50))
    chex.assert_shape(result.rewards, (2, 2))
    assert bool((result.final_state.p_pos[:, 0, 0] > state.p_pos[0, 0]).all())
    assert bool((result.final_state.p_pos[:, 1, 0] < state.p_pos[1, 0]).all())
    np.testing.assert_array_equal(result.truncated, True)
    np.testing.assert_array_equal(result.terminated, False)
    np.testing.assert_array_equal(result.agent_dones, True)
    np.testing.assert_array_equal(result.final_state.step, 2)
    np.testing.assert_array_equal(result.state.step, 0)
    np.testing.assert_array_equal(result.observation.observations[..., -10:], 0)
    final_messages = jax.nn.one_hot(jnp.array([9, 0]), 10)  # (N, C), received from partner
    np.testing.assert_array_equal(result.final_observation.observations[..., -10:], jnp.tile(final_messages, (2, 1, 1)))
    np.testing.assert_array_equal(jax.vmap(env.memory_resets)(result.final_state), False)


@pytest.mark.parametrize("num_agents,num_landmarks", [(3, 3), (2, 2)])
def test_reference_requires_fixed_agent_and_landmark_counts(num_agents: int, num_landmarks: int) -> None:
    with pytest.raises(ValueError, match="requires 2 agents and 3 landmarks"):
        MPEConfig(id="MPE_simple_reference_v3", num_agents=num_agents, num_landmarks=num_landmarks)


@pytest.mark.parametrize("env_id", ["MPE_simple_spread_v3", "MPE_simple_reference_v3"])
def test_update_playback_and_checkpoint(tmp_path: Path, env_id: MPEEnvId) -> None:
    config = replace(
        ppo.load_config(PRESET),
        env=MPEConfig(id=env_id, num_agents=2, num_landmarks=3, max_steps=1),
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        total_steps=1,
        model=ppo.ModelConfig(encoder=ppo.MLPEncoderConfig(embedding_size=4), recurrent=ppo.LSTMConfig(hidden_size=4)),
    )
    env = ppo.make_env(config)
    observation, state = jax.vmap(env.reset)(
        jax.random.split(jax.random.key(0), 1)
    )  # (E, N, O)/(E, N, A); state leaves: (E, ...)
    model = ppo.ActorCritic(action_space=env.action_space, model=config.model)
    memory = model.initial_carry(2)  # leaves: (B=2, H=4)
    starts = jnp.ones(2, bool)  # (B,)
    params = model.init(
        jax.random.key(1), observation.observations, memory, starts[None], observation.action_masks
    )  # leaves: (...)
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params["params"],
        tx=optax.inject_hyperparams(optax.adam)(learning_rate=config.learning_rate),
    )
    runner = ppo.RolloutCarry(
        env_state=state,  # leaves: (E, ...)
        observations=observation,  # (E, N, O)/(E, N, A)
        model_carry=memory,  # leaves: (B, H)
        episode_starts=starts,  # (B,)
        episode_returns=jnp.zeros(1),  # (E,)
        episode_step_counts=jnp.zeros(1, jnp.int32),  # (E,)
    )
    result = ppo.train_step(runner, train_state, env, config, jax.random.key(2), ppo.init_metrics())
    assert int(result.train_state.step) == 1
    assert int(result.metrics.total.episode_count) == 1
    assert int(result.metrics.total.episode_length_sum) == 1
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(jax.tree.leaves(train_state.params), jax.tree.leaves(result.train_state.params))
    )
    episode = ppo.play_episode(result.train_state, config, jax.random.key(3), memory, record=True, env=env)
    assert episode.length == 1 and len(episode.frames) == 2
    assert all(frame.shape == (400, 400, 3) and frame.dtype == np.uint8 for frame in episode.frames)
    progress = ppo.TrainingProgress(
        train_state=result.train_state,
        key=result.key,  # ()
        rollout_count=1,
        episode_count=1,
        episode_return_sum=episode.episode_return,
        episode_length_sum=1,
    )
    with ppo.checkpoint_manager(str(tmp_path)) as manager:
        ppo.save_checkpoint(manager, progress, config)
        manager.wait_until_finished()
        restored = ppo.restore_checkpoint(manager, train_state, config)
        chex.assert_trees_all_equal(restored, progress)
        with pytest.raises(ValueError, match="environment settings"):
            ppo.restore_checkpoint(manager, train_state, replace(config, env=replace(config.env, local_ratio=0.0)))
