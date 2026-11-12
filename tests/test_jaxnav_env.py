from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState
from jaxmarl.environments.jaxnav.jaxnav_env import State
from pydantic import ValidationError

from rl2 import train_mappo as ppo
from rl2.envs.jaxnav_env import JaxNavConfig, JaxNavEnv


@pytest.fixture(scope="module")
def env() -> JaxNavEnv:
    return JaxNavEnv(JaxNavConfig(num_agents=2, map_size=7, fill=0.1, lidar_num_beams=8, max_steps=2))


@pytest.fixture(scope="module")
def state(env: JaxNavEnv) -> State:
    state = env.reset(jax.random.key(0))[1]
    # Known empty interior, separated stationary robots and distant goals.
    walls = jnp.ones((7, 7), jnp.int32).at[1:-1, 1:-1].set(0)  # (Y, X)
    return state.replace(
        pos=jnp.array([[2.0, 2.0], [4.0, 4.0]]),
        theta=jnp.zeros(2),
        vel=jnp.zeros((2, 2)),
        goal=jnp.array([[4.0, 2.0], [2.0, 4.0]]),
        map_data=walls,  # (N, 2); (Y, X)
    )


def test_reset_packing_batch_shapes_and_render(env: JaxNavEnv, state: State) -> None:
    observation, reset_state = env.reset(jax.random.key(0))
    raw = env.env.get_obs(reset_state)  # agent leaves: (O,)
    assert env.action_space.n == 15
    chex.assert_shape(observation.observation, (2, 13))
    chex.assert_shape(observation.actions_mask, (2, 15))
    assert observation.actions_mask.dtype == jnp.bool_
    np.testing.assert_array_equal(observation.actions_mask, True)
    for i, agent in enumerate(env.agents):
        np.testing.assert_allclose(observation.observation[i], raw[agent])
    states = jax.tree.map(lambda x: jnp.stack([x, x]), state)  # leaves: (E=2, ...)
    result = jax.jit(jax.vmap(env.step))(
        jax.random.split(jax.random.key(1), 2),
        states,
        jnp.full((2, 2), 2),
    )
    chex.assert_shape(result.observation.observation, (2, 2, 13))
    chex.assert_shape(result.reward, (2, 2))
    chex.assert_shape(result.agent_inactive, (2, 2))
    assert result.agent_inactive.dtype == jnp.bool_
    np.testing.assert_array_equal(result.agent_inactive, False)
    chex.assert_shape(result.done, (2,))
    assert bool(jnp.isfinite(result.reward).all())
    np.testing.assert_array_equal(result.done, False)
    np.testing.assert_array_equal(result.final_state.step, 1)
    frame = env.render(state)  # (H, W, 3)
    assert frame.shape == (400, 400, 3) and frame.dtype == np.uint8


@pytest.mark.parametrize("boundary", ["timeout", "goals", "collision", "partial-goal"])
def test_boundaries_and_action_masks(env: JaxNavEnv, state: State, boundary: str) -> None:
    if boundary == "timeout":
        state = state.replace(step=jnp.asarray(1))
    elif boundary == "goals":
        state = state.replace(goal=state.pos, step=jnp.asarray(1))
    elif boundary == "collision":
        state = state.replace(pos=state.pos.at[0].set(jnp.array([0.5, 0.5])))
    else:
        state = state.replace(goal=state.goal.at[0].set(state.pos[0]))
    result = env.step(jax.random.key(2), state, jnp.full(2, 2))
    assert bool(result.done) == (boundary in ("timeout", "goals"))
    assert bool(result.terminated) == (boundary == "goals")
    assert bool(result.truncated) == (boundary in ("timeout", "goals"))
    chex.assert_shape(result.agent_inactive, (2,))
    np.testing.assert_array_equal(result.agent_inactive, [boundary == "collision", False])
    if bool(result.done):
        assert int(result.state.step) == 0 and int(result.final_state.step) == 2
        np.testing.assert_array_equal(result.agent_done, True)
        assert not np.array_equal(result.observation.observation, result.final_observation.observation)
    else:
        chex.assert_trees_all_equal(result.state, result.final_state)
        if boundary == "collision":
            np.testing.assert_array_equal(result.agent_done, [True, False])
            np.testing.assert_array_equal(result.observation.actions_mask, True)
            assert float(result.reward[0]) < float(result.reward[1])
        else:
            # Native non-evaporating agents can leave their goals to make room for teammates.
            np.testing.assert_array_equal(result.agent_done, False)
            np.testing.assert_array_equal(result.observation.actions_mask, True)
            np.testing.assert_array_equal(result.final_state.goal_reached, [True, False])


def test_jaxnav_config_matches_smax_and_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1] / "configs/mappo"
    config = ppo.load_config(root / "jaxnav.yaml", ppo.Config)
    reference = ppo.load_config(root / "smax.yaml", ppo.Config)
    for name in asdict(reference):
        if name not in {"env", "run_id", "ckpt_dir", "video_fps"}:
            assert getattr(config, name) == getattr(reference, name)
    assert config.video_fps == 24 and reference.video_fps == 5
    assert config.env.id == "JaxNav" and hash(config)
    assert config.run_id == "mappo_JaxNav_4agents_11x11_seed1"
    assert config.ckpt_dir == f"/runs/mappo/{config.run_id}"
    assert isinstance(ppo.make_env(config), JaxNavEnv)
    with pytest.raises(ValueError, match="num_agents"):
        ppo.make_env(replace(config, num_minibatches=3))
    with pytest.raises(ValidationError, match="fill"):
        JaxNavConfig(fill=1.0)
    with pytest.raises(ValidationError, match="num_agents"):
        JaxNavConfig(num_agents=0)
    with pytest.raises(ValidationError, match="map_size"):
        JaxNavConfig(map_size=2)
    train = Mock()
    monkeypatch.setattr(ppo, "train", train)
    monkeypatch.setattr("sys.argv", ["mappo", "--config", str(root / "jaxnav.yaml")])
    ppo.main()
    train.assert_called_once_with(config)


def constant_policy(
    variables: dict[str, Any],
    observation: jax.Array,  # (B, O)
    carry: ppo.RecurrentCarry,  # leaves: (B, H)
    starts: jax.Array,  # (B,)
    actions_mask: jax.Array,  # (B, A)
    *,
    method: str,
) -> ppo.ActorCriticOutput:
    logits = jnp.full(actions_mask.shape, -1e10).at[:, 2].set(0.0)  # (B, A), stop only
    return ppo.ActorCriticOutput(carry, ppo.CategoricalPolicy(logits), jnp.full(observation.shape[0], 2.0))


@pytest.mark.usefixtures("ppo_mesh")
def test_jaxnav_rollout_timeout_update_and_evaluation(env: JaxNavEnv, state: State) -> None:
    config = replace(
        ppo.load_config(Path(__file__).resolve().parents[1] / "configs/mappo/jaxnav.yaml", ppo.Config),
        env=env.config,
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        total_iterations=1,
        gamma=0.5,
        gae_lambda=1.0,
        target_kl=None,
        model=ppo.GenericModelConfig(
            encoder=ppo.MLPEncoderConfig(embedding_size=4), recurrent=ppo.LSTMConfig(hidden_size=4)
        ),
    )
    model = ppo.make_actor_critic(env.action_space, config.model)
    memory = model.initial_carry(2)  # leaves: (B=2, H=4)
    final_input = state.replace(step=jnp.asarray(1))
    observation = env._observation(env.env.get_obs(final_input))  # (N, O)/(N, A)
    batched_state, observation = jax.tree.map(lambda x: jnp.asarray(x)[None], (final_input, observation))
    runner = ppo.RolloutCarry(
        batched_state,
        observation,
        memory,
        jnp.ones(2, bool),
        jnp.zeros(1),
        jnp.zeros(1, jnp.int32),
    )
    fake = TrainState.create(apply_fn=constant_policy, params={}, tx=optax.sgd(0.1))
    advanced, rollout = ppo.collect_rollout(runner, fake, env, config, jax.random.key(3))
    expected = env.step(jax.random.key(4), final_input, jnp.full(2, 2))
    np.testing.assert_allclose(rollout.batch.return_target[:, 0], expected.reward + 1.0, atol=1e-6)
    chex.assert_shape(rollout.batch.observation, (2, 1, 13))
    chex.assert_shape(rollout.batch.actions_mask, (2, 1, 15))
    assert int(rollout.episode_count) == 1
    np.testing.assert_allclose(rollout.episode_return_sum, expected.reward.mean())
    np.testing.assert_array_equal(advanced.episode_start, True)
    params = model.init(
        jax.random.key(5),
        observation.observation[0, :, None],  # (B, T=1, O)
        memory,
        jnp.ones((2, 1), bool),  # (B, T=1)
        observation.actions_mask[0, :, None],  # (B, T=1, A)
    )
    real = TrainState.create(
        apply_fn=model.apply,
        params=params["params"],
        tx=optax.inject_hyperparams(optax.adam)(learning_rate=config.learning_rate),
    )
    result = ppo.train_step(runner, real, env, config, jax.random.key(6), ppo.init_metrics())
    assert int(result.train_state.step) == 1 and int(result.metrics.total.episode_count) == 1
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(result.train_state.params))
    episode = ppo.play_episode(result.train_state, config, jax.random.key(7), memory, record=True, env=env)
    assert 1 <= episode.length <= 2 and len(episode.frames) == episode.length + 1
    assert np.isfinite(episode.episode_return)
