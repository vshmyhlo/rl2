from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import struct
from flax.training.train_state import TrainState
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec
from pydantic import ValidationError

from rl2 import ppo
from rl2.actions import ContinuousSpace
from rl2.envs import mujoco_playground_env, playground_assets
from rl2.envs.mujoco_playground_env import PlaygroundConfig, PlaygroundEnv, PlaygroundParams


@pytest.fixture
def config() -> ppo.Config:
    return ppo.Config(
        env=PlaygroundConfig(),
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
        video_fps=50.0,
    )


@struct.dataclass
class PhysicsState:
    obs: jax.Array | dict[str, jax.Array]  # (O,) or named (O,)/(P,) vectors
    reward: jax.Array  # ()
    done: jax.Array  # ()


@dataclass
class Physics:
    """Cheap deterministic transitions for adapter boundaries and PPO wiring."""

    observation_size: int = 66
    dictionary_observation: bool = False

    def reset(self, key: jax.Array) -> PhysicsState:  # key: ()
        observation = jax.random.uniform(key, (self.observation_size,))  # (O,)
        obs = (
            {"state": observation, "privileged_state": jnp.full(123, 42.0)}
            if self.dictionary_observation
            else observation
        )  # (O,) or state: (O,), privileged_state: (P=123,)
        return PhysicsState(obs=obs, reward=jnp.array(0.0), done=jnp.array(0.0))

    def step(self, state: PhysicsState, action: jax.Array) -> PhysicsState:  # leaves: (...); action: (A,)
        return state.replace(
            obs=jax.tree.map(lambda x: x + 1, state.obs), reward=action.sum()
        )  # obs: (...); reward: ()


@pytest.fixture
def env() -> PlaygroundEnv:
    adapter = object.__new__(PlaygroundEnv)
    adapter.task = "PandaPickCube"
    adapter.env = Physics()
    adapter.params = PlaygroundParams(max_steps=2)
    adapter.action_space = ContinuousSpace(shape=(8,), low=(-1.0,) * 8, high=(1.0,) * 8)
    return adapter


@pytest.mark.parametrize(
    "env_config",
    [PlaygroundConfig(), PlaygroundConfig(task="CartpoleSwingup"), PlaygroundConfig(task="Go1JoystickFlatTerrain")],
)
def test_config_round_trip_and_factory(
    config: ppo.Config, env_config: PlaygroundConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(config, env=env_config)
    restored = ppo.Config(**asdict(config))
    assert restored == config and hash(restored) == hash(config)
    constructor = Mock()
    monkeypatch.setattr(ppo, "PlaygroundEnv", constructor)
    assert ppo.make_env(config) is constructor.return_value
    constructor.assert_called_once_with(config.env)


@pytest.mark.parametrize(
    "settings,match",
    [
        ({"task": "unknown"}, "task"),
        ({"max_steps": 0}, "max_steps"),
        ({"max_steps": True}, "max_steps"),
    ],
)
def test_invalid_settings(settings: dict[str, object], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        PlaygroundConfig(**settings)


@pytest.mark.parametrize(
    "env_id,encoder,match",
    [
        ("MuJoCo-Playground", "conv2d", "requires model.encoder.type: mlp"),
        ("Brax", "conv2d", "requires model.encoder.type: mlp"),
        ("Breakout-MinAtar", "mlp", "mlp encoder requires"),
    ],
)
def test_config_rejects_mismatches(config: ppo.Config, env_id: str, encoder: str, match: str) -> None:
    settings = asdict(config)
    settings.update(env={"id": env_id}, model={"encoder": {"type": encoder}})
    with pytest.raises(ValidationError, match=match):
        ppo.Config(**settings)


@pytest.mark.parametrize("config", [PlaygroundConfig(), PlaygroundConfig(task="Go1JoystickFlatTerrain")])
def test_missing_assets_is_actionable(
    config: PlaygroundConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mujoco_playground_env.mjx_env, "MENAGERIE_PATH", tmp_path)
    with pytest.raises(FileNotFoundError, match="uv run python -m rl2.envs.playground_assets"):
        PlaygroundEnv(config)


def test_asset_setup_downloads_required_files_and_reuses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(playground_assets.mjx_env, "MENAGERIE_PATH", tmp_path)
    download = Mock(
        side_effect=[
            BytesIO(b'<mujoco><asset><mesh file="hand.obj"/></asset></mujoco>'),
            BytesIO(b"license"),
            BytesIO(b"mesh"),
        ]
    )
    monkeypatch.setattr(playground_assets, "urlopen", download)
    playground_assets.download_assets()
    playground_assets.download_assets()
    assert download.call_count == 3
    assert (tmp_path / "franka_emika_panda/assets/hand.obj").read_bytes() == b"mesh"
    assert playground_assets.mjx_env.MENAGERIE_COMMIT_SHA in download.call_args.args[0]


def test_go1_asset_setup_uses_bundled_mesh_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(playground_assets.mjx_env, "MENAGERIE_PATH", tmp_path / "menagerie")
    monkeypatch.setattr(playground_assets.mjx_env, "ROOT_PATH", tmp_path / "playground")
    manifest = tmp_path / "playground/locomotion/go1/xmls/go1_mjx_feetonly.xml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('<mujoco><asset><mesh file="../../unitree_go1/assets/trunk.stl"/></asset></mujoco>')
    download = Mock(side_effect=[BytesIO(b"license"), BytesIO(b"mesh")])
    monkeypatch.setattr(playground_assets, "urlopen", download)
    playground_assets.download_assets(robot="go1")
    playground_assets.download_assets(robot="go1")
    assert download.call_count == 2
    assert (tmp_path / "menagerie/unitree_go1/assets/trunk.stl").read_bytes() == b"mesh"
    assert download.call_args.args[0].endswith("/unitree_go1/assets/trunk.stl")
    assert playground_assets.mjx_env.MENAGERIE_COMMIT_SHA in download.call_args.args[0]


@pytest.mark.parametrize("dictionary_observation", [False, True], ids=["vector", "go1-policy-state"])
def test_batched_boundaries_preserve_final_observations(env: PlaygroundEnv, dictionary_observation: bool) -> None:
    observation_size = 48 if dictionary_observation else 66
    env.task = "Go1JoystickFlatTerrain" if dictionary_observation else "PandaPickCube"
    env.env = Physics(observation_size=observation_size, dictionary_observation=dictionary_observation)
    keys = jax.random.split(jax.random.key(0), 4)  # (B=4,)
    obs, initial = jax.vmap(env.reset, in_axes=(0, None))(keys, env.params)  # (B,O); leaves: (B,...)
    assert obs.shape == (4, observation_size) and obs.dtype == jnp.float32
    # Termination, timeout, both, and continuation in one batch.
    state = initial.replace(
        physics=initial.physics.replace(done=jnp.array([1.0, 0.0, 1.0, 0.0])),  # (B,)
        step_count=jnp.array([0, 1, 1, 0], jnp.int32),  # (B,)
    )
    actions = jnp.full((4, 8), 0.5)  # (B,8)
    result = jax.jit(env.step_batch)(keys, state, actions, env.params)
    np.testing.assert_array_equal(result.info["terminated"], [True, False, True, False])
    np.testing.assert_array_equal(result.info["truncated"], [False, True, True, False])
    np.testing.assert_array_equal(result.done, [True, True, True, False])
    np.testing.assert_array_equal(result.state.step_count, [0, 0, 0, 1])
    np.testing.assert_array_equal(result.info["final_state"].step_count, [1, 2, 2, 1])
    np.testing.assert_array_equal(result.info["final_observation"], obs + 1)
    np.testing.assert_array_equal(result.observation[:3], obs[:3])
    np.testing.assert_array_equal(result.observation[3], obs[3] + 1)
    np.testing.assert_array_equal(result.reward, jnp.full(4, 4.0))
    chex.assert_shape((result.done, result.reward), (4,))


@pytest.mark.usefixtures("ppo_mesh")
def test_vector_policy_rollout_update_and_playback(env: PlaygroundEnv, config: ppo.Config) -> None:
    observations, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(jax.random.key(0), 1), env.params
    )  # (B=1,66); leaves: (B,...)
    model = ppo.ActorCritic(action_space=env.action_space, model=config.model)
    carry = model.initial_carry(1)  # leaves: (B,2)
    starts = jnp.ones(1, bool)  # (B,)
    params = model.init(jax.random.key(1), observations[:, None], carry, starts[:, None])["params"]
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    runner = ppo.RolloutCarry(
        env_state=env_state,  # leaves: (B,...)
        observations=observations,  # (B,66)
        model_carry=carry,  # leaves: (B,2)
        episode_starts=starts,  # (B,)
        episode_returns=jnp.zeros(1),  # (B,)
        episode_step_counts=jnp.zeros(1, jnp.int32),  # (B,)
    )
    advanced, rollout = ppo.collect_rollout(runner, train_state, env, config, jax.random.key(2))
    batch = rollout.batch
    assert batch.observations.shape == (1, 2, 66)
    assert batch.actions.shape == (1, 2, 8) and batch.actions.dtype == jnp.float32
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


def test_real_panda_spaces_controls_reward_and_timeout(config: ppo.Config) -> None:
    if not (Path(mujoco_playground_env.mjx_env.MENAGERIE_PATH) / "franka_emika_panda/mjx_panda.xml").exists():
        pytest.skip("Install Panda assets with: uv run python -m rl2.envs.playground_assets")
    env = ppo.make_env(replace(config, env=PlaygroundConfig(max_steps=1)))
    assert isinstance(env, PlaygroundEnv)
    assert env.action_space == ContinuousSpace(shape=(8,), low=(-1.0,) * 8, high=(1.0,) * 8)
    obs, state = env.reset(jax.random.key(0), env.params)  # (66,); leaves: (...)
    assert obs.shape == (66,) and obs.dtype == jnp.float32
    actions = jnp.array([1.0, -1.0, 0.0, 0.5, 0.0, 0.0, 0.0, -1.0])  # (8,)
    result = env.step(jax.random.key(1), state, actions, env.params)
    final = result.info["final_state"].physics  # leaves: (...)
    np.testing.assert_allclose(final.data.ctrl, state.physics.data.ctrl + 0.04 * actions, atol=1e-7)
    expected_reward = sum(final.metrics[name] * scale for name, scale in env.env._config.reward_config.scales.items())
    np.testing.assert_allclose(result.reward, expected_reward, rtol=1e-6)
    chex.assert_shape((result.reward, result.done), ())
    assert bool(result.done) and bool(result.info["truncated"]) and not bool(result.info["terminated"])
    assert int(result.state.step_count) == 0 and int(result.info["final_state"].step_count) == 1
    np.testing.assert_array_equal(result.info["final_observation"], final.obs)
    reset_obs, _ = env.reset(jax.random.key(1), env.params)  # (66,); leaves: (...)
    np.testing.assert_array_equal(result.observation, reset_obs)
    batched_obs, _ = jax.eval_shape(
        jax.vmap(env.reset, in_axes=(0, None)), jax.random.split(jax.random.key(0), 2), env.params
    )  # (B=2,66); leaves: (B,...)
    assert batched_obs.shape == (2, 66)


@pytest.mark.parametrize(
    "env_config,observation_size,action_size,dt",
    [
        (PlaygroundConfig(task="CartpoleSwingup", max_steps=1), 5, 1, 0.01),
        (PlaygroundConfig(task="Go1JoystickFlatTerrain", max_steps=1), 48, 12, 0.02),
    ],
    ids=["dm-control", "locomotion"],
)
def test_real_starter_tasks(
    config: ppo.Config, env_config: PlaygroundConfig, observation_size: int, action_size: int, dt: float
) -> None:
    if (
        env_config.task == "Go1JoystickFlatTerrain"
        and not (Path(mujoco_playground_env.mjx_env.MENAGERIE_PATH) / "unitree_go1/assets/trunk.stl").exists()
    ):
        pytest.skip("Install Go1 assets with: uv run python -m rl2.envs.playground_assets --robot go1")
    env = ppo.make_env(replace(config, env=env_config))
    assert isinstance(env, PlaygroundEnv)
    assert env.action_space == ContinuousSpace(
        shape=(action_size,), low=(-1.0,) * action_size, high=(1.0,) * action_size
    )
    assert env.env.dt == pytest.approx(dt)
    assert env.env._config.impl == "jax" and env.env._config.episode_length == 1
    obs, state = env.reset(jax.random.key(0), env.params)  # (O,); leaves: (...)
    assert obs.shape == (observation_size,) and obs.dtype == jnp.float32
    if env_config.task == "CartpoleSwingup":
        assert abs(float(state.physics.data.qpos[1])) == pytest.approx(np.pi, abs=0.05)
    else:
        np.testing.assert_array_equal(obs, state.physics.obs["state"])
    actions = jnp.zeros(action_size, jnp.float32)  # (A,)
    key = jax.random.key(1)  # ()
    result = env.step(key, state, actions, env.params)
    chex.assert_shape((result.observation, result.info["final_observation"]), (observation_size,))
    chex.assert_shape((result.reward, result.done), ())
    assert bool(result.done) and bool(result.info["truncated"])
    assert not bool(result.info["terminated"])
    assert int(result.state.step_count) == 0 and int(result.info["final_state"].step_count) == 1
    final = result.info["final_state"].physics  # leaves: (...)
    expected = final.obs["state"] if env_config.task == "Go1JoystickFlatTerrain" else final.obs  # (O,)
    np.testing.assert_array_equal(result.info["final_observation"], expected)
    reset_obs, _ = env.reset(key, env.params)  # (O,); leaves: (...)
    np.testing.assert_array_equal(result.observation, reset_obs)
    batched_obs, _ = jax.eval_shape(
        jax.vmap(env.reset, in_axes=(0, None)), jax.random.split(jax.random.key(0), 2), env.params
    )  # (B=2,O); leaves: (B,...)
    assert batched_obs.shape == (2, observation_size)


def test_cartpole_initialization_and_step_on_two_devices() -> None:
    if jax.device_count() < 2:
        pytest.skip("Requires two devices; use JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=2")
    mesh = Mesh(devices=jax.devices()[:2], axis_names=("batch",), axis_types=(AxisType.Auto,))
    batched = NamedSharding(mesh, PartitionSpec("batch"))
    with jax.set_mesh(mesh):
        env = PlaygroundEnv(PlaygroundConfig(task="CartpoleSwingup", max_steps=1))
        keys = jax.device_put(jax.random.split(jax.random.key(0), 2), batched)  # (B=2,)
        reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)), out_shardings=batched)
        observations, states = reset(keys, env.params)  # (B, O=5); leaves: (B, ...)
        actions = jax.device_put(jnp.zeros((2, 1)), batched)  # (B, A=1)
        result = jax.jit(env.step_batch, out_shardings=batched)(keys, states, actions, env.params)
    chex.assert_shape((observations, result.observation, result.info["final_observation"]), (2, 5))
    chex.assert_shape((result.reward, result.done), (2,))
    assert observations.sharding == result.observation.sharding == batched
    np.testing.assert_array_equal(result.info["terminated"], False)
    np.testing.assert_array_equal(result.info["truncated"], True)
    np.testing.assert_array_equal(result.state.step_count, 0)
    np.testing.assert_array_equal(result.info["final_state"].step_count, 1)
    np.testing.assert_array_equal(result.observation, observations)


def test_render_delegates_physics_state(env: PlaygroundEnv) -> None:
    _, state = env.reset(jax.random.key(0), env.params)  # (66,); leaves: (...)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)  # (H,W,3)
    env.env.render = Mock(return_value=[frame])
    assert env.render(state) is frame
    env.env.render.assert_called_once_with([state.physics], height=240, width=320)
