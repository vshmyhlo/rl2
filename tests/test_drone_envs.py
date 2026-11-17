from dataclasses import asdict
from pathlib import Path
from typing import Literal
from unittest.mock import Mock

import chex
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import optax
import pytest
import yaml
from flax.training.train_state import TrainState
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec
from pydantic import ValidationError

from rl2 import train_mappo as mappo

# isort: split

from rl2 import ppo
from rl2.actions import ContinuousSpace, TruncatedGaussianPolicy
from rl2.envs.crazyflow_env import CrazyflowConfig, CrazyflowEnv
from rl2.envs.crazyrl_env import CrazyRLConfig, CrazyRLEnv

pytestmark = pytest.mark.usefixtures("ppo_mesh")


@pytest.fixture(scope="module")
def flow_env() -> CrazyflowEnv:
    return CrazyflowEnv(CrazyflowConfig(max_steps=2))


@pytest.fixture
def ppo_config() -> ppo.Config:
    return ppo.Config(
        run_id="test_drone_ppo",
        video_max_frames=3,
        env=CrazyflowConfig(max_steps=2),
        seed=0,
        total_steps=2,
        num_envs=1,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        learning_rate=1e-3,
        anneal_lr=False,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        target_kl=None,
        entropy_coef=0.0,
        max_grad_norm=0.5,
        video_fps=50.0,
        bf16=False,
        model=ppo.ModelConfig(
            encoder=ppo.MLPEncoderConfig(hidden_size=4, embedding_size=2),
            recurrent=ppo.LSTMConfig(hidden_size=2),
        ),
    )


@pytest.fixture
def mappo_config() -> mappo.Config:
    return mappo.Config(
        run_id="test_drone_mappo",
        video_max_frames=3,
        env=CrazyRLConfig(max_steps=2),
        seed=0,
        total_iterations=1,
        num_envs=1,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        learning_rate=1e-3,
        anneal_lr=False,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        target_kl=None,
        entropy_coef=0.0,
        value_coef=1.0,
        max_grad_norm=0.5,
        video_fps=10.0,
        dtype="float32",
        model=mappo.GenericModelConfig(
            encoder=mappo.MLPEncoderConfig(embedding_size=2), recurrent=mappo.LSTMConfig(hidden_size=2)
        ),
    )


@pytest.mark.parametrize("task", ["hover", "circle", "surround", "escort", "catch"])
def test_crazyrl_native_transition_and_timeout(task: Literal["hover", "circle", "surround", "escort", "catch"]) -> None:
    env = CrazyRLEnv(CrazyRLConfig(task=task, max_steps=2))
    key = jax.random.key(0)  # ()
    observation, initial = jax.jit(env.reset)(key)  # (N=2, O)/(N, 0); leaves: (...)
    size = 6 if task in ("hover", "circle") else 9
    assert observation.observation.shape == (2, size)
    assert observation.actions_mask.shape == (2, 0)
    state = jdc.replace(initial, timestep=jnp.asarray(1, jnp.int32))
    actions = jnp.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]], jnp.float32)  # (N, 3)
    native_obs, reward, _, _, _, native_state = env.env.step(
        state, actions, jax.random.split(key)[0]
    )  # (N, O); (N,); state leaves: (...)
    result = jax.jit(env.step)(key, state, actions)
    chex.assert_trees_all_close(result.final_state, native_state)
    np.testing.assert_allclose(result.reward, reward, atol=1e-6)
    np.testing.assert_allclose(result.final_observation.observation, native_obs, atol=1e-6)
    np.testing.assert_array_equal(result.observation.observation, observation.observation)
    assert bool(result.done) and bool(result.truncated) and not bool(result.terminated)
    np.testing.assert_array_equal(result.agent_done, True)
    np.testing.assert_array_equal(result.agent_inactive, False)
    assert int(result.state.timestep) == 0 and int(result.final_state.timestep) == 2


def test_crazyrl_collision_and_batched_resets() -> None:
    env = CrazyRLEnv(CrazyRLConfig(max_steps=2))
    keys = jax.random.split(jax.random.key(0), 2)  # (E=2,)
    _, state = jax.jit(jax.vmap(env.reset))(keys)  # leaves: (E=2, ...)
    # First world hits the floor; the second reaches its time limit.
    positions = state.agents_locations.at[0, 0, 2].set(0.1)  # (E, N=2, 3)
    state = jdc.replace(state, agents_locations=positions, timestep=jnp.array([0, 1], jnp.int32))
    result = jax.jit(jax.vmap(env.step))(keys, state, jnp.zeros((2, 2, 3)))
    assert result.observation.observation.shape == (2, 2, 9)
    assert result.reward.shape == (2, 2)
    np.testing.assert_array_equal(result.terminated, [True, False])
    np.testing.assert_array_equal(result.truncated, [False, True])
    np.testing.assert_array_equal(result.reward[0], -10.0)
    np.testing.assert_array_equal(result.state.timestep, 0)
    np.testing.assert_array_equal(result.final_state.timestep, [1, 2])
    assert env.render(jax.tree.map(lambda x: x[0], result.state)).shape == (400, 400, 3)


def test_crazyrl_one_drone_circle() -> None:
    env = CrazyRLEnv(CrazyRLConfig(task="circle", initial_positions=((0.0, 0.0, 1.0),)))
    obs, state = env.reset(jax.random.key(0))  # (N=1, O=6)/(N, 0); state leaves: (...)
    assert obs.observation.shape == (1, 6)
    result = jax.jit(env.step)(jax.random.key(1), state, jnp.zeros((1, 3)))
    assert result.reward.shape == (1,) and not bool(result.done)


@pytest.mark.parametrize(
    "settings",
    [{"initial_positions": ((0.0, 0.0, 1.0),)}, {"max_steps": 201}, {"target_position": (0, 0, 4)}],
    ids=["surround-needs-two-drones", "native-horizon", "target-outside-map"],
)
def test_crazyrl_invalid_config(settings: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        CrazyRLConfig(**settings)


@pytest.mark.parametrize("task", ["reach_pos", "reach_vel", "landing", "figure_eight"])
def test_crazyflow_tasks_and_batch_timeout(
    task: Literal["reach_pos", "reach_vel", "landing", "figure_eight"], flow_env: CrazyflowEnv
) -> None:
    env = flow_env if task == "reach_pos" else CrazyflowEnv(CrazyflowConfig(task=task, max_steps=2, n_samples=2))
    keys = jax.random.split(jax.random.key(0), 2)  # (B=2,)
    obs, states = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(keys, None)  # (B, O); leaves: (B, ...)
    size = 19 if task == "figure_eight" else 16
    assert obs.shape == (2, size) and obs.dtype == jnp.float32
    states = states.replace(step_count=jnp.array([0, 1], jnp.int32))
    low, high = jnp.asarray((env.action_space.low, env.action_space.high))  # each (4,)
    actions = jnp.broadcast_to((low + high) / 2, (2, 4))  # (B, 4)
    result = jax.jit(env.step_batch)(keys, states, actions, None)
    chex.assert_shape((result.observation, result.info["final_observation"]), (2, size))
    chex.assert_shape((result.reward, result.done), (2,))
    np.testing.assert_array_equal(result.info["terminated"], False)
    np.testing.assert_array_equal(result.info["truncated"], [False, True])
    np.testing.assert_array_equal(result.state.step_count, [1, 0])
    np.testing.assert_array_equal(result.info["final_state"].step_count, [1, 2])
    np.testing.assert_array_equal(result.info["final_state"].physics.core.steps, 10)
    np.testing.assert_allclose(result.observation[1], obs[1], atol=1e-6)
    np.testing.assert_array_equal(result.observation[0], result.info["final_observation"][0])
    final = result.info["final_state"]
    pos, vel = final.physics.states.pos[:, 0, 0], final.physics.states.vel[:, 0, 0]  # each (B, 3)
    if task == "figure_eight":
        goal = env.trajectory[final.step_count - 1]  # (B, 3)
    else:
        goal = final.goal[:, 0]  # (B, 3)
    distance = jnp.linalg.norm(goal - (vel if task == "reach_vel" else pos), axis=-1)  # (B,)
    expected = jnp.exp(-distance * (1 if task == "reach_vel" else 2))  # (B,)
    if task == "landing":
        expected *= jnp.exp(-2 * jnp.linalg.norm(vel, axis=-1))  # (B,)
    np.testing.assert_allclose(result.reward, expected, rtol=1e-5)


def test_crazyflow_ground_collision_and_seed(flow_env: CrazyflowEnv) -> None:
    key = jax.random.key(0)  # ()
    obs, state = flow_env.reset(key, None)  # (O=16,); state leaves: (...)
    chex.assert_trees_all_equal((obs, state), flow_env.reset(key, None))
    other, _ = flow_env.reset(jax.random.key(1), None)  # (O=16,)
    assert not np.array_equal(obs, other)
    physics = state.physics.replace(
        states=state.physics.states.replace(
            pos=state.physics.states.pos.at[..., 2].set(-0.01),  # (1, 1, 3)
            vel=jnp.zeros((1, 1, 3)),  # (1, 1, 3)
        )
    )
    state = state.replace(physics=physics, step_count=jnp.asarray(1, jnp.int32))
    result = jax.jit(flow_env.step)(key, state, jnp.zeros(4), None)
    assert bool(result.info["terminated"]) and bool(result.info["truncated"])
    assert float(result.reward) == -1.0
    assert int(result.state.step_count) == 0
    assert float(result.info["final_observation"][2]) < 0
    assert flow_env.render(result.state).dtype == np.uint8


@pytest.mark.parametrize("settings", [{"freq": 60}, {"max_steps": 0}, {"task": "figure_eight", "max_steps": 501}])
def test_crazyflow_invalid_config(settings: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        CrazyflowConfig(**settings)


@pytest.mark.parametrize("trainer", ["ppo", "mappo"])
def test_drone_cli(
    trainer: Literal["ppo", "mappo"],
    ppo_config: ppo.Config,
    mappo_config: mappo.Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, config = (ppo, ppo_config) if trainer == "ppo" else (mappo, mappo_config)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(asdict(config)))
    train = Mock()
    monkeypatch.setattr(module, "train", train)
    monkeypatch.setattr("sys.argv", [trainer, "--config", str(path)])
    module.main()
    train.assert_called_once_with(config)


def test_crazyrl_rollout_and_continuous_update(mappo_config: mappo.Config) -> None:
    env = mappo.make_env(mappo_config)
    assert isinstance(env, CrazyRLEnv)
    observation, state = jax.vmap(env.reset)(jax.random.split(jax.random.key(0), 1))  # (E=1, N=2, O)/(E, N, 0)
    model = mappo.make_actor_critic(env.action_space, mappo_config.model, dtype=jnp.bfloat16)
    memory = model.initial_carry(2)  # leaves: (B=2, H=2)
    local = jax.tree.map(lambda x: x.reshape((2, 1, *x.shape[2:])), observation)  # (B, T=1, O)/(B, T, 0)
    params = model.init(jax.random.key(1), local.observation, memory, jnp.ones((2, 1), bool), local.actions_mask)[
        "params"
    ]
    train_state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.inject_hyperparams(optax.adam)(learning_rate=1e-3)
    )
    runner = mappo.RolloutCarry(
        env_state=state,
        observation=observation,
        model_carry=memory,
        episode_start=jnp.ones(2, bool),  # (B=2,)
        episode_return=jnp.zeros(1),  # (E=1,)
        episode_step_count=jnp.zeros(1, jnp.int32),  # (E=1,)
    )
    _, rollout = mappo.collect_rollout(runner, train_state, env, mappo_config, jax.random.key(2))
    batch = rollout.batch
    assert batch.action.shape == (2, 2, 3) and batch.action.dtype == jnp.float32
    assert batch.actions_mask.shape == (2, 2, 0)
    assert bool(jnp.all((batch.action >= -1) & (batch.action <= 1)))
    output, _ = model.apply({"params": params}, batch.observation, memory, batch.episode_start, batch.actions_mask)
    assert isinstance(output.policy, TruncatedGaussianPolicy)
    assert output.policy.mean.dtype == jnp.float32
    np.testing.assert_allclose(output.policy.log_prob(batch.action), batch.old_log_prob, atol=1e-5)
    assert output.value.shape == (2, 2)
    result = mappo.train_step(runner, train_state, env, mappo_config, jax.random.key(2), mappo.init_metrics())
    assert int(result.train_state.step) == 1
    assert any(
        not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(result.train_state.params))
    )

    episode = mappo.play_episode(train_state, mappo_config, jax.random.key(3), memory, record=True, env=env)
    assert episode.length == 2 and len(episode.frames) == 3
    assert all(frame.shape == (400, 400, 3) and frame.dtype == np.uint8 for frame in episode.frames)


def test_crazyflow_rollout_and_update(ppo_config: ppo.Config, flow_env: CrazyflowEnv) -> None:
    env = flow_env
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(jax.random.key(0), 1), None
    )  # (B=1, O=16); state leaves: (B, ...)
    model = ppo.ActorCritic(action_space=env.action_space, model=ppo_config.model)
    memory = model.initial_carry(1)  # leaves: (B=1, H=2)
    starts = jnp.ones(1, bool)  # (B,)
    params = model.init(jax.random.key(1), obs[:, None], memory, starts[:, None])["params"]
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    runner = ppo.RolloutCarry(
        env_state=state,
        observations=obs,
        model_carry=memory,
        episode_starts=starts,
        episode_returns=jnp.zeros(1),  # (B=1,)
        episode_step_counts=jnp.zeros(1, jnp.int32),  # (B=1,)
    )
    _, rollout = ppo.collect_rollout(runner, train_state, env, ppo_config, jax.random.key(2))
    batch = rollout.batch
    assert batch.actions.shape == (1, 2, 4)
    output = model.apply({"params": params}, batch.observations, memory, batch.episode_starts)
    np.testing.assert_allclose(output.policy.log_prob(batch.actions), batch.old_log_probs, atol=1e-5)
    assert int(rollout.episode_count) == 1
    updated, _ = ppo.update(train_state, batch, ppo_config)
    assert int(updated.step) == 1
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(updated.params)))

    episode = ppo.play_episode(train_state, ppo_config, jax.random.key(3), memory, record=True, env=env)
    assert episode.length == 2 and len(episode.frames) == 3
    assert all(frame.shape == (400, 400, 3) and frame.dtype == np.uint8 for frame in episode.frames)


def test_mappo_continuous_event_shape(mappo_config: mappo.Config) -> None:
    space = ContinuousSpace(shape=(2, 1), low=(-2.0, 0.0), high=(4.0, 1.0))
    model = mappo.make_actor_critic(space, mappo_config.model)
    memory = model.initial_carry(1)  # leaves: (B=1, H=2)
    obs = jnp.zeros((1, 2, 3))  # (B=1, T=2, O=3)
    starts = jnp.ones((1, 2), bool)  # (B, T)
    masks = jnp.empty((1, 2, 0), bool)  # (B, T, 0)
    params = model.init(jax.random.key(0), obs, memory, starts, masks)["params"]
    output, _ = model.apply({"params": params}, obs, memory, starts, masks)
    actions, log_probs = output.policy.sample_and_log_prob(key=jax.random.key(1))  # (B, T=2, 2, 1); (B, T)
    assert actions.shape == (1, 2, 2, 1) and log_probs.shape == (1, 2)
    assert output.policy.entropy().shape == (1, 2)
    low, high = jnp.asarray((space.low, space.high)).reshape(2, 2, 1)  # each (2, 1)
    assert bool(jnp.all((actions >= low) & (actions <= high)))
    np.testing.assert_allclose(output.policy.log_prob(actions), log_probs, atol=1e-6)


@pytest.mark.parametrize("family", ["crazyflow", "crazyrl"])
def test_drone_two_device_reset_and_step(family: Literal["crazyflow", "crazyrl"]) -> None:
    if jax.device_count() < 2:
        pytest.skip("Requires two devices")
    mesh = Mesh(devices=jax.devices()[:2], axis_names=("batch",), axis_types=(AxisType.Auto,))
    sharding = NamedSharding(mesh, PartitionSpec("batch"))
    with jax.set_mesh(mesh):
        keys = jax.device_put(jax.random.split(jax.random.key(0), 2), sharding)  # (B=2,)
        if family == "crazyflow":
            env = CrazyflowEnv(CrazyflowConfig(task="figure_eight", max_steps=1, n_samples=2))
            _, state = jax.jit(jax.vmap(env.reset, in_axes=(0, None)), out_shardings=sharding)(keys, None)
            result = jax.jit(env.step_batch, out_shardings=sharding)(keys, state, jnp.zeros((2, 4)), None)
            assert result.observation.shape == (2, 19)
            assert result.observation.sharding == sharding
            np.testing.assert_array_equal(result.state.step_count, 0)
            np.testing.assert_array_equal(result.info["final_state"].step_count, 1)
        else:
            env = CrazyRLEnv(CrazyRLConfig(task="circle", max_steps=1))
            _, state = jax.jit(jax.vmap(env.reset), out_shardings=sharding)(keys)
            result = jax.jit(jax.vmap(env.step), out_shardings=sharding)(keys, state, jnp.zeros((2, 2, 3)))
            assert result.observation.observation.shape == (2, 2, 6)
            assert result.observation.observation.sharding == sharding
            np.testing.assert_array_equal(result.state.timestep, 0)
            np.testing.assert_array_equal(result.final_state.timestep, 1)
        np.testing.assert_array_equal(result.done, True)
