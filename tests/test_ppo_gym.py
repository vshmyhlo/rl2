"""Host rollout contracts, VisFly timeout semantics, and compiled PPO integration."""

import json
from contextlib import closing
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import MagicMock, Mock, patch

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch
from flax.training.train_state import TrainState
from numpy.typing import NDArray
from pydantic import TypeAdapter, ValidationError

from rl2 import train_ppo_gym as ppo
from rl2.actions import BetaPolicy, CategoricalPolicy, ContinuousSpace, Policy, TruncatedGaussianPolicy
from rl2.envs import GymStyleEnv, GymStyleStep
from rl2.envs.gym_env import GymConfig, GymEnv
from rl2.envs.gym_style_env import GymStyleInfo, RouteSamplingInfo, StopReason
from rl2.envs.visfly import generate_manifest as manifest_generation
from rl2.envs.visfly.power import PowerConfig, PowerUsage
from rl2.envs.visfly.render import RenderContext, diagnostics_panel, top_down_panel
from rl2.envs.visfly.routes import CATEGORIES, Route, RouteConfig
from rl2.envs.visfly.scenarios import EvaluationConfig, Manifest, Scenario, SceneCoverage
from rl2.envs.visfly_env import RawVisFlyObservation, VisFlyConfig, VisFlyEnv
from rl2.model.encoders.conv2d import ConvStage

pytestmark = pytest.mark.usefixtures("ppo_mesh")


@pytest.fixture
def config() -> ppo.Config:
    return ppo.Config(
        env=GymConfig(env_id="Pendulum-v1", kwargs={"max_episode_steps": 2}),
        model=ppo.ModelConfig(
            encoder=ppo.MLPEncoderConfig(hidden_size=4, embedding_size=4),
            recurrent=ppo.LSTMConfig(hidden_size=4),
        ),
        seed=7,
        total_steps=8,
        num_envs=2,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        learning_rate=0.001,
        anneal_lr=False,
        gamma=0.5,
        gae_lambda=1.0,
        clip_coef=0.2,
        target_kl=None,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        video_fps=20.0,
        bf16=False,
    )


class VisFlyBatch:
    """VisFly's tensor-valued, four-result API with all end-flag combinations."""

    def __init__(self, num_envs: int = 4, num_scenes: int = 1) -> None:
        self.num_envs = num_envs
        self.num_scenes = num_scenes
        self.action_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.closed = False
        self.steps = 0
        self.state = torch.zeros((num_envs, 13))  # (B, S)
        self.state[:, 3] = 1  # identity wxyz quaternion
        self.target = torch.zeros((num_envs, 3))  # (B, 3)
        self.rgb = torch.zeros((num_envs, 3, 4, 4), dtype=torch.uint8)  # (B, C, H, W)

    def get_observation(self) -> RawVisFlyObservation[torch.Tensor]:
        return RawVisFlyObservation(state=self.state, target=self.target, rgb=self.rgb)

    def reset(self) -> RawVisFlyObservation[torch.Tensor]:
        return self.get_observation()

    def step(self, actions: NDArray[np.float32]) -> tuple[Any, ...]:  # (B, 4)
        assert isinstance(actions, np.ndarray)
        assert actions.shape == (self.num_envs, 4)
        assert np.all((actions >= -1) & (actions <= 1))
        self.steps += 1
        self.state[:] = -1
        self.rgb[:] = 255
        infos = []
        for index in range(self.num_envs):
            infos.append(
                {
                    "episode_done": index in (0, 2),
                    "is_success": index == 0,
                    "episode": {"extra": {"collision": index == 2}},
                    "TimeLimit.truncated": index in (1, 2),
                    "terminal_observation": {
                        "state": torch.full((13,), 10.0 * (index + 1)),  # (S,)
                        "target": self.target[index],  # (3,)
                        "rgb": torch.full((3, 4, 4), index + 1, dtype=torch.uint8),  # (C, H, W)
                    },
                }
            )
        return (
            self.reset(),
            torch.arange(1, self.num_envs + 1, dtype=torch.float32),
            torch.arange(self.num_envs) < 3,
            infos,
        )

    def render(self) -> list[NDArray[np.uint8]]:
        return [np.full((4, 4, 4), self.steps, dtype=np.uint8) for _ in range(self.num_scenes)]  # each (H, W, RGBA)

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("num_envs", [1, 2], ids=["single", "batched"])
def test_gymnasium_autoreset_keeps_timeout_observation(num_envs: int) -> None:
    with closing(GymEnv(config=GymConfig(kwargs={"max_episode_steps": 1}), num_envs=num_envs, seed=3)) as env:
        assert isinstance(env, GymStyleEnv)
        initial = env.reset()  # (B, 4)
        result = env.step(np.zeros(num_envs, dtype=np.int32))
        assert isinstance(result, GymStyleStep)
        assert isinstance(initial, np.ndarray)
        assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(result))
        assert result.info is None
        assert result.rewards.dtype == np.float32
        assert result.terminated.dtype == result.truncated.dtype == np.bool_
        assert initial.shape == result.observations.shape == result.final_observations.shape == (num_envs, 4)
        assert result.rewards.shape == result.truncated.shape == result.terminated.shape == (num_envs,)
        np.testing.assert_array_equal(result.truncated, True)
        np.testing.assert_array_equal(result.terminated, False)
        np.testing.assert_array_equal(result.rewards, 1)
        assert not np.array_equal(result.observations, result.final_observations)
        np.testing.assert_array_equal(env.reset(), initial)


def test_visfly_factory_tensors_terminal_observations_and_cleanup() -> None:
    backend = VisFlyBatch(num_scenes=2)
    settings = VisFlyConfig(num_scenes=2, kwargs={"device": "cpu", "visual": True})
    with (
        patch("rl2.envs.visfly_env.load_env_creator", return_value=Mock(return_value=backend)) as loader,
        closing(VisFlyEnv(config=settings, num_envs=4, seed=17)) as env,
    ):
        initial = env.reset()
        result = env.step(np.zeros((4, 4), dtype=np.float32))
        assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves((initial, result)))
        assert result.observations["rgb"].shape == (4, 4, 4, 3)
        assert result.final_observations["state"].shape == (4, 13)
        assert result.info["goal_distance"].shape == (4,)
        assert result.info["goal_distance"].dtype == np.float32
        assert "energy_hover_seconds" not in result.info
        np.testing.assert_allclose(result.info["goal_distance"], np.sqrt(3) * np.array([10, 20, 30, 1]))
        np.testing.assert_array_equal(result.terminated, [True, False, True, False])
        np.testing.assert_array_equal(result.truncated, [False, True, True, False])
        np.testing.assert_array_equal(result.final_observations["state"][:, 0], [10, 20, 30, -1])
        np.testing.assert_array_equal(result.observations["state"], -1)
        expected_state = np.zeros((4, 13), dtype=np.float32)  # (B, S)
        expected_state[:, 3] = 1
        np.testing.assert_array_equal(
            initial["state"], expected_state
        )  # simulator buffer reuse cannot mutate a rollout
        loader.return_value.assert_called_once_with(
            num_scene=2,
            num_agent_per_scene=2,
            seed=17,
            requires_grad=False,
            device="cpu",
            visual=True,
        )
        assert env.render().shape == (4, 8, 10, 3)
    assert backend.closed


def test_visfly_requires_timeout_final_observation() -> None:
    backend = VisFlyBatch()
    step = backend.step

    def missing_terminal(actions: NDArray[np.float32]) -> tuple[Any, ...]:  # (B, 4)
        observations, rewards, done, infos = step(actions)
        del infos[1]["terminal_observation"]
        return observations, rewards, done, infos

    backend.step = missing_terminal
    with (
        patch("rl2.envs.visfly_env.load_env_creator", return_value=Mock(return_value=backend)),
        closing(VisFlyEnv(config=VisFlyConfig(), num_envs=4, seed=1)) as env,
    ):
        env.reset()
        with pytest.raises(ValueError, match="terminal_observation"):
            env.step(np.zeros((4, 4), dtype=np.float32))


@pytest.mark.parametrize(
    ("env_config", "evaluation"),
    [
        pytest.param(GymConfig(), False, id="gym-training"),
        pytest.param(GymConfig(), True, id="gym-evaluation"),
        pytest.param(VisFlyConfig(num_scenes=2), False, id="visfly-training"),
        pytest.param(VisFlyConfig(num_scenes=2), True, id="visfly-evaluation"),
    ],
)
def test_factory_selects_backend(config: ppo.Config, env_config: ppo.EnvConfig, evaluation: bool) -> None:
    config = replace(config, env=env_config)
    with patch.object(ppo, "GymEnv") as gym_env, patch.object(ppo, "VisFlyEnv") as visfly_env:
        env = ppo.make_env(config, seed=17, evaluation=evaluation, record=True)
    if isinstance(env_config, VisFlyConfig):
        assert env is visfly_env.return_value
        visfly_env.assert_called_once_with(
            config=replace(env_config, kwargs={**env_config.kwargs, "scene_split": "test" if evaluation else "train"}),
            num_envs=1 if evaluation else config.num_envs,
            seed=17,
            evaluation=evaluation,
        )
        gym_env.assert_not_called()
        assert "scene_split" not in env_config.kwargs
    else:
        assert env is gym_env.return_value
        gym_env.assert_called_once_with(
            config=env_config, num_envs=1 if evaluation else config.num_envs, seed=17, record=True
        )
        visfly_env.assert_not_called()


@pytest.mark.parametrize("space", [gym.spaces.MultiDiscrete([2, 3]), gym.spaces.Discrete(2, start=1)])
def test_reject_unsupported_actions(space: gym.Space) -> None:
    with pytest.raises(ValueError):
        ppo.action_space(space)


def test_scalar_continuous_actions() -> None:
    space = ppo.action_space(gym.spaces.Box(-2.0, 3.0, shape=(), dtype=np.float32))
    assert space == ContinuousSpace(shape=(), low=(-2.0,), high=(3.0,))


def value_policy(
    variables: dict[str, Any],
    observations: dict[str, jax.Array],  # leaves: (B, *O)
    memory: ppo.RecurrentCarry,  # leaves: (B, 1)
    starts: jax.Array,  # (B,)
    *,
    method: str,
) -> ppo.ActorCriticOutput:
    memory = jax.tree.map(lambda leaf: jnp.where(starts[:, None], 0, leaf) + 1, memory)  # (B, 1) leaves
    count = starts.shape[0]
    return ppo.ActorCriticOutput(
        carry=memory,
        policy=TruncatedGaussianPolicy(
            mean=jnp.zeros((count, 4)),
            log_std=jnp.zeros((count, 4)),
            action_space=ContinuousSpace(shape=(4,), low=(-1.0,) * 4, high=(1.0,) * 4),
        ),
        values=observations["state"][:, 0] + memory[0][0][:, 0],  # (B,)
    )


@pytest.mark.parametrize("action_shape", [None, (), (4,)], ids=["discrete", "scalar", "vector"])
def test_sample_policy_step_preserves_rng_and_outputs(action_shape: tuple[int, ...] | None) -> None:
    policy: Policy
    if action_shape is None:
        policy = CategoricalPolicy(jnp.zeros((2, 3)))  # logits: (B, A)
    else:
        size = int(np.prod(action_shape))
        policy = TruncatedGaussianPolicy(
            mean=jnp.zeros((2, size)),  # (B, D)
            log_std=jnp.zeros((2, size)),  # (B, D)
            action_space=ContinuousSpace(shape=action_shape, low=(-1.0,) * size, high=(1.0,) * size),
        )

    def apply(
        variables: dict[str, Any],
        observations: jax.Array,  # (B, O)
        memory: ppo.RecurrentCarry,  # leaves: (B, H)
        starts: jax.Array,  # (B,)
        *,
        method: str,
    ) -> ppo.ActorCriticOutput:
        memory = jax.tree.map(lambda leaf: jnp.where(starts[:, None], 0, leaf) + 1, memory)  # leaves: (B, H)
        return ppo.ActorCriticOutput(carry=memory, policy=policy, values=observations[:, 0])

    observations = jnp.asarray([[2.0], [3.0]])  # (B, O)
    memory = ((jnp.ones((2, 1)), jnp.ones((2, 1))),)  # leaves: (B, H)
    starts = jnp.asarray([True, False])  # (B,)
    key = jax.random.key(9)  # ()
    next_key, action_key = jax.random.split(key)  # each ()
    expected_actions, expected_log_probs = policy.sample_and_log_prob(key=action_key)  # (B, *A); (B,)
    sampled = ppo.sample_policy_step(apply, {}, observations, memory, starts, key)
    assert sampled.actions.shape == (2, *(action_shape or ()))
    assert sampled.values.shape == sampled.log_probs.shape == (2,)
    assert all(leaf.shape == (2, 1) for leaf in jax.tree.leaves(sampled.memory))
    np.testing.assert_array_equal(jax.random.key_data(sampled.key), jax.random.key_data(next_key))
    np.testing.assert_allclose(sampled.actions, expected_actions, atol=1e-6)
    np.testing.assert_allclose(sampled.log_probs, expected_log_probs, atol=1e-6)
    np.testing.assert_array_equal(sampled.values, [2, 3])
    for leaf in jax.tree.leaves(sampled.memory):
        np.testing.assert_array_equal(leaf, [[1], [2]])


@pytest.mark.parametrize(
    ("kind", "action_shape"),
    [
        pytest.param("categorical", (), id="categorical_ties"),
        pytest.param("gaussian", (), id="gaussian_scalar"),
        pytest.param("gaussian", (2, 2), id="gaussian_matrix"),
        pytest.param("beta", (), id="beta_scalar"),
        pytest.param("beta", (4,), id="beta_vector"),
    ],
)
def test_deterministic_policy_step(kind: str, action_shape: tuple[int, ...]) -> None:
    policy: Policy
    if kind == "categorical":
        policy = CategoricalPolicy(jnp.asarray([[0.0, 2.0, 1.0], [3.0, 3.0, 0.0]]))  # (B, A)
        expected = jnp.asarray([1, 0])  # (B,)
    else:
        size = int(np.prod(action_shape))
        space = ContinuousSpace(shape=action_shape, low=(-2.0,) * size, high=(4.0,) * size)
        if kind == "gaussian":
            policy = TruncatedGaussianPolicy(
                mean=jnp.broadcast_to(jnp.asarray([[-3.0], [5.0]]), (2, size)),  # (B, D)
                log_std=jnp.zeros((2, size)),  # (B, D)
                action_space=space,
            )
            expected = jnp.broadcast_to(jnp.asarray([[-2.0], [4.0]]), (2, size))  # (B, D)
        else:
            policy = BetaPolicy(
                alpha=jnp.full((2, size), 0.5),  # (B, D)
                beta=jnp.full((2, size), 1.5),  # (B, D)
                action_space=space,
            )
            expected = jnp.full((2, size), -0.5)  # (B, D)
        expected = expected.reshape((2, *action_shape))  # (B, *action_shape)

    def apply(
        variables: dict[str, Any],
        observations: jax.Array,  # (B, O)
        memory: ppo.RecurrentCarry,  # leaves: (B, H)
        starts: jax.Array,  # (B,)
        *,
        method: str,
    ) -> ppo.ActorCriticOutput:
        memory = jax.tree.map(lambda leaf: jnp.where(starts[:, None], 0, leaf) + 1, memory)  # leaves: (B, H)
        return ppo.ActorCriticOutput(carry=memory, policy=policy, values=observations[:, 0])

    observations = jnp.zeros((2, 1))  # (B, O)
    memory = ((jnp.ones((2, 1)), jnp.ones((2, 1))),)  # leaves: (B, H)
    starts = jnp.asarray([True, False])  # (B,)
    next_memory, actions = ppo.deterministic_policy_step(apply, {}, observations, memory, starts)
    assert actions.shape == (2, *action_shape)
    np.testing.assert_allclose(jax.device_get(actions), jax.device_get(expected))
    for leaf in jax.tree.leaves(next_memory):
        assert leaf.shape == (2, 1)
        np.testing.assert_array_equal(jax.device_get(leaf), [[1], [2]])


def test_step_info_metrics_weight_samples_preserve_zeros_and_omit_missing_values() -> None:
    metrics = ppo.RolloutInfoMetrics()
    metrics.add(GymStyleInfo(stop_reason=np.zeros(2, dtype=np.int32)))  # (B,)
    assert metrics.metrics() == {}
    metrics.add(
        GymStyleInfo(
            energy_hover_seconds=np.array([0.04, 0.0], dtype=np.float32),  # (B,)
            power_ratio=np.array([2.0, 0.0], dtype=np.float32),  # (B,)
            power_penalty=np.zeros(2, dtype=np.float32),  # (B,)
            route_sampling=RouteSamplingInfo(
                initial_goal_distance_meters=np.array([3.0, 6.0, 9.0]),  # (N,)
                reference_route_length_meters=np.array([3.0, 6.0, 15.0]),  # (N,)
                direct_path_blocked=np.array([False, False, True]),  # (N,)
                difficulty=np.array(["easy", "easy", "hard"]),  # (N,)
            ),
        )
    )
    # A later step has fewer sampled starts; averages must weight starts, not steps.
    metrics.add(
        GymStyleInfo(
            energy_hover_seconds=np.array([0.02, 0.06], dtype=np.float32),  # (B,)
            power_ratio=np.array([1.0, 3.0], dtype=np.float32),  # (B,)
            power_penalty=np.zeros(2, dtype=np.float32),  # (B,)
            route_sampling=RouteSamplingInfo(
                initial_goal_distance_meters=np.array([2.0]),  # (N,)
                reference_route_length_meters=np.array([2.0]),  # (N,)
                direct_path_blocked=np.array([False]),  # (N,)
                difficulty=np.array(["easy"]),  # (N,)
            ),
        )
    )
    assert metrics.metrics() == pytest.approx(
        {
            "rollout/power_ratio": 1.5,
            "rollout/power_penalty": 0.0,
            "rollout/sampled_episode_count": 4,
            "rollout/initial_goal_distance_meters": 5.0,
            "rollout/reference_route_length_meters": 6.5,
            "rollout/direct_path_blocked_fraction": 0.25,
            "rollout/difficulty_fraction/easy": 0.75,
            "rollout/difficulty_fraction/medium": 0.0,
            "rollout/difficulty_fraction/hard": 0.25,
        }
    )


def test_rollout_bootstraps_timeout_with_final_observation_and_memory(config: ppo.Config) -> None:
    config = replace(config, env=VisFlyConfig(power=PowerConfig()), num_envs=4)
    backend = VisFlyBatch()
    backend.envs = SimpleNamespace(dynamics=SimpleNamespace(dt=0.01, _interval_steps=2))
    backend.power_usage = PowerUsage(
        energy=torch.full((4,), 0.04),  # (B,)
        excess=torch.zeros(4),  # (B,)
    )
    with (
        patch("rl2.envs.visfly_env.load_env_creator", return_value=Mock(return_value=backend)),
        closing(ppo.make_env(config)) as env,
    ):
        runner = ppo.RolloutCarry(
            observations=jax.tree.map(jnp.asarray, env.reset()),
            model_carry=((jnp.zeros((4, 1)), jnp.zeros((4, 1))),),
            episode_starts=jnp.ones(4, dtype=bool),
            episode_returns=jnp.zeros(4),
            episode_step_counts=jnp.zeros(4, dtype=jnp.int32),
        )
        state = TrainState.create(apply_fn=value_policy, params={}, tx=optax.sgd(0.1))
        key = jax.random.key(9)  # ()
        # Catch implicit scalar readbacks on CPU too, where transfer guards are ineffective.
        with (
            jax.transfer_guard_device_to_host("disallow"),
            patch.object(type(runner.episode_starts), "__bool__", side_effect=AssertionError("Device scalar readback")),
        ):
            info_metrics = ppo.RolloutInfoMetrics()
            advanced, rollout = ppo.collect_rollout(runner, state, env, config, key, info_metrics)
        assert info_metrics.metrics() == pytest.approx(
            {
                "rollout/power_ratio": 2.0,
                "rollout/power_penalty": 0.0,
            }
        )
        batch = rollout.batch
        assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves((advanced, batch)))
        assert batch.actions.shape == (4, 2, 4)
        assert batch.observations["rgb"].shape == (4, 2, 4, 4, 3)
        assert batch.returns.shape == (4, 2)
        # Bootstrap sees the terminal observation (20), and memory advanced twice (2).
        np.testing.assert_allclose(batch.returns[:3], [[1, 1], [13, 13], [3, 3]])
        np.testing.assert_array_equal(batch.episode_starts, [[True, True]] * 3 + [[True, False]])
        np.testing.assert_array_equal(advanced.model_carry[0][0][:, 0], [1, 1, 1, 2])
        assert int(rollout.episode_count) == 6
        assert float(rollout.episode_return_sum) == 12.0  # raw rewards, excluding timeout correction
        assert int(rollout.episode_length_sum) == 6
        np.testing.assert_array_equal(advanced.episode_returns, [0, 0, 0, 8])
        np.testing.assert_array_equal(batch.observations["state"][:, 0], jax.device_get(runner.observations["state"]))


def test_real_gym_rollout_compiled_update_and_checkpoint(config: ppo.Config, tmp_path: Path) -> None:
    with closing(ppo.make_env(config)) as env:
        observations = jax.tree.map(jnp.asarray, env.reset())  # (B, 3)
        model = ppo.ActorCritic(action_space=ppo.action_space(env.action_space), model=config.model)
        memory = model.initial_carry(2)  # leaves: (B, H)
        starts = jnp.ones(2, dtype=bool)  # (B,)
        state = TrainState.create(
            apply_fn=model.apply,
            params=model.init(jax.random.key(0), observations[:, None], memory, starts[:, None])["params"],
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=config.learning_rate),
        )
        runner = ppo.RolloutCarry(
            observations=observations,
            model_carry=memory,
            episode_starts=starts,
            episode_returns=jnp.zeros(2),
            episode_step_counts=jnp.zeros(2, dtype=jnp.int32),
        )
        original = np.asarray(state.params["value_output"]["kernel"]).copy()  # (H, 1)
        result = ppo.train_step(runner, state, env, config, jax.random.key(1), ppo.init_metrics())
        assert int(result.train_state.step) == 1
        assert int(result.metrics.total.episode_count) == 2
        assert not np.array_equal(result.train_state.params["value_output"]["kernel"], original)
        assert result.rollout_carry.observations.shape == (2, 3)
        with closing(ppo.checkpoint_manager(str(tmp_path))) as manager:
            progress = ppo.TrainingProgress(
                train_state=result.train_state,
                key=result.key,
                rollout_count=1,
                episode_count=2,
                episode_return_sum=float(result.metrics.total.episode_return_sum),
                episode_length_sum=4,
                env_steps=4,
            )
            ppo.save_checkpoint(manager, progress, config)
            manager.wait_until_finished()
            metadata = manager.restore(1, args=ppo.ocp.args.Composite(metadata=ppo.ocp.args.JsonRestore())).metadata
            assert metadata["config"] == TypeAdapter(ppo.Config).dump_python(config, mode="json")
            restored = ppo.restore_checkpoint(manager, result.train_state, replace(config, gamma=0.9))
            assert restored.env_steps == 4
            assert int(restored.train_state.step) == 1


@pytest.mark.parametrize("encoder_type", ["mlp", "conv2d", "visfly", "visfly_rgbd"])
def test_mixed_image_vector_encoder_and_sequence_shapes(config: ppo.Config, encoder_type: str) -> None:
    encoder = ppo.Conv2DEncoderConfig(stages=(ConvStage(channels=2, blocks=0),), embedding_size=4)
    settings = config.model
    if encoder_type == "conv2d":
        settings = replace(settings, encoder=encoder)
    elif encoder_type.startswith("visfly"):
        settings = replace(
            settings,
            encoder=ppo.VisFlyEncoderConfig(
                stages=encoder.stages,
                embedding_size=4,
                goal_scale=2,
                velocity_scale=4,
                angular_velocity_scale=8,
                depth_scale=4,
            ),
        )
    model = ppo.ActorCritic(action_space=ppo.DiscreteSpace(n=2), model=settings)
    obs = {"rgb": jnp.full((2, 2, 4, 4, 3), 255, dtype=jnp.uint8), "state": jnp.zeros((2, 2, 3))}  # (B, T, *O)
    if encoder_type.startswith("visfly"):
        obs["state"] = jnp.tile(
            jnp.array([3, 4, 5, 1, 0, 0, 0, 4, 8, 12, 8, 16, 24], jnp.float32), (2, 2, 1)
        )  # (B, T, 13)
        obs["target"] = jnp.tile(jnp.array([5, 8, 11], jnp.float32), (2, 2, 1))  # (B, T, 3)
    if encoder_type == "visfly_rgbd":
        obs["depth"] = jnp.tile(
            jnp.array([-1, 0, 2, 20], jnp.float32)[None, None, None, :, None], (2, 2, 4, 1, 1)
        )  # (B, T, H, W, 1)
    memory = model.initial_carry(2)  # leaves: (B, H)
    starts = jnp.zeros((2, 2), dtype=bool)  # (B, T)
    params = model.init(jax.random.key(1), obs, memory, starts)
    output = jax.jit(model.apply)(params, obs, memory, starts)
    assert output.values.shape == (2, 2)
    assert output.policy.logits.shape == (2, 2, 2)
    assert all(leaf.shape == (2, 4) for leaf in jax.tree.leaves(output.carry))
    if encoder_type.startswith("visfly"):
        generic = ppo.ActorCritic(action_space=ppo.DiscreteSpace(n=2), model=replace(settings, encoder=encoder))
        prepared = {
            "rgb": obs["rgb"],  # (B, T, H, W, 3)
            "state": jnp.tile(
                jnp.array([1, 0, 0, 0, 1, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3], jnp.float32), (2, 2, 1)
            ),  # (B, T, 15)
        }
        if encoder_type == "visfly_rgbd":
            expected_depth = jnp.tile(
                jnp.array([0, 0, 0.5, 1], jnp.float32)[None, None, None, :, None], (2, 2, 4, 1, 1)
            )  # (B, T, H, W, 1)
            prepared["rgb"] = jnp.concatenate(
                (jnp.ones_like(obs["rgb"], dtype=jnp.float32), expected_depth), axis=-1
            )  # (B, T, H, W, 4)
        expected = jax.jit(generic.apply)(params, prepared, memory, starts)
        np.testing.assert_allclose(jax.device_get(output.values), jax.device_get(expected.values), atol=1e-6)
        np.testing.assert_allclose(
            jax.device_get(output.policy.logits), jax.device_get(expected.policy.logits), atol=1e-6
        )
        step = jax.jit(partial(model.apply, method="step"))(
            params, jax.tree.map(lambda leaf: leaf[:, 0], obs), memory, starts[:, 0]
        )
        np.testing.assert_allclose(jax.device_get(step.values), jax.device_get(output.values[:, 0]), atol=1e-6)


@pytest.mark.parametrize("terminal_available", [True, False], ids=["terminal_observation", "missing_terminal"])
def test_evaluation_uses_fresh_environment_and_closes_on_failure(config: ppo.Config, terminal_available: bool) -> None:
    backend = VisFlyBatch(num_envs=1)
    step = backend.step

    def backend_step(actions: NDArray[np.float32]) -> tuple[Any, ...]:  # (B, 4)
        observations, rewards, done, infos = step(actions)
        if not terminal_available:
            infos[0].pop("terminal_observation")
        return observations, rewards, done, infos

    backend.step = backend_step
    config = replace(config, env=VisFlyConfig(), video_max_frames=1)
    state = TrainState.create(apply_fn=value_policy, params={}, tx=optax.sgd(0.1))
    memory = ((jnp.zeros((1, 1)), jnp.zeros((1, 1))),)  # leaves: (1, 1)
    with (
        patch("rl2.envs.visfly_env.load_env_creator", return_value=Mock(return_value=backend)),
        patch.object(VisFlyEnv, "render_outcome", autospec=True, side_effect=VisFlyEnv.render_outcome) as outcome,
    ):
        result = ppo.play_episode(state, config, jax.random.key(0), memory, record=True)
        assert len(result.frames) == 1
        assert result.frames[0].shape == (8, 10, 3)
        assert result.frames[0].dtype == np.uint8
        assert result.length == 1
        assert result.stop_reason == StopReason.SUCCESS
        assert (result.final_goal_distance is not None) == terminal_available
        assert outcome.call_count == int(terminal_available)
        assert result.terminated and not result.truncated
        assert backend.steps == 1
        assert backend.closed
        backend.closed = False
        with (
            patch.object(backend, "render", side_effect=RuntimeError("renderer failed")),
            pytest.raises(RuntimeError, match="renderer failed"),
        ):
            ppo.play_episode(state, config, jax.random.key(0), memory, record=True)
        assert backend.closed


def test_video_frame_limit_stops_after_last_recorded_action(config: ppo.Config) -> None:
    backend = VisFlyBatch(num_envs=1)
    step = backend.step

    def ongoing_step(actions: NDArray[np.float32]) -> tuple[Any, ...]:  # (B, 4)
        observations, rewards, done, infos = step(actions)
        done[:] = False
        return observations, rewards, done, infos

    backend.step = ongoing_step
    config = replace(config, env=VisFlyConfig(), video_max_frames=2)
    state = TrainState.create(apply_fn=value_policy, params={}, tx=optax.sgd(0.1))
    memory = ((jnp.zeros((1, 1)), jnp.zeros((1, 1))),)  # leaves: (1, 1)
    with patch("rl2.envs.visfly_env.load_env_creator", return_value=Mock(return_value=backend)):
        result = ppo.play_episode(state, config, jax.random.key(0), memory, record=True)
    assert len(result.frames) == result.length == backend.steps == 2
    assert result.stop_reason is None
    assert not result.terminated and not result.truncated
    assert backend.closed


@pytest.mark.parametrize("frame_cap", [None, 3, 2], ids=["full_hold", "partial_hold", "cap_at_end"])
def test_visfly_video_diagnostics_preserve_terminal_state_and_frame_cap(
    config: ppo.Config, frame_cap: int | None
) -> None:
    backend = VisFlyBatch(num_envs=1)
    step = backend.step

    def delayed_done(actions: NDArray[np.float32]) -> tuple[Any, ...]:  # (B, 4)
        observations, rewards, done, infos = step(actions)
        if backend.steps == 1:
            done[:] = False
        return observations, rewards, done, infos

    backend.step = delayed_done
    config = replace(config, env=VisFlyConfig(), video_max_frames=frame_cap, video_fps=2)
    state = TrainState.create(apply_fn=value_policy, params={}, tx=optax.sgd(0.1))
    memory = ((jnp.zeros((1, 1)), jnp.zeros((1, 1))),)  # leaves: (1, 1)
    with (
        patch("rl2.envs.visfly_env.load_env_creator", return_value=Mock(return_value=backend)),
        patch("rl2.envs.visfly_env.diagnostics_panel", wraps=diagnostics_panel) as panel,
        patch.object(backend, "render", wraps=backend.render) as render,
    ):
        result = ppo.play_episode(state, config, jax.random.key(0), memory, record=True)
    assert result.length == backend.steps == 2
    assert result.episode_return == 2
    assert result.stop_reason == StopReason.SUCCESS
    assert len(result.frames) == (4 if frame_cap is None else frame_cap)
    assert render.call_count == 2  # No render of the autoreset state.
    initial: RenderContext = panel.call_args_list[0].kwargs["context"]
    assert initial.step == initial.episode_return == 0
    assert initial.reward is initial.action is None
    ongoing: RenderContext = panel.call_args_list[1].kwargs["context"]
    assert ongoing.step == ongoing.episode_return == ongoing.reward == 1
    assert ongoing.action.shape == (4,)
    terminal: RenderContext = panel.call_args.kwargs["context"]
    assert terminal.step == terminal.episode_return == 2
    assert terminal.reward == 1
    assert terminal.action.shape == (4,)
    assert terminal.stop_reason == StopReason.SUCCESS
    np.testing.assert_array_equal(panel.call_args.args[0], 10)  # Terminal state, not autoreset's -1.
    assert all(frame.shape == (8, 10, 3) for frame in result.frames)
    assert backend.closed


@pytest.mark.parametrize("rotation_interval", [None, 8], ids=["autoreset", "scene_rotation"])
def test_visfly_render_tracks_each_episode_across_resets(rotation_interval: int | None) -> None:
    backend = VisFlyBatch()
    settings = VisFlyConfig(scene_rotation_env_steps=rotation_interval)
    actions = np.full((4, 4), 0.5, dtype=np.float32)  # (B, A)
    with (
        patch("rl2.envs.visfly_env.load_env_creator", return_value=Mock(return_value=backend)),
        patch("rl2.envs.visfly_env.diagnostics_panel", wraps=diagnostics_panel) as panel,
        patch("rl2.envs.visfly_env.top_down_panel", wraps=top_down_panel) as trail,
        closing(VisFlyEnv(config=settings, num_envs=4, seed=1)) as env,
    ):
        env.reset()
        env.render()
        panel.reset_mock()
        env.step(actions)
        transition = env.step(actions)
        actions[:] = 0  # Caller buffer reuse must not change the recorded action.
        frames = env.render()  # (B, H, W, 3)
        assert frames.shape == (4, 8, 10, 3)
        for call in trail.call_args_list[-4:-1]:
            np.testing.assert_array_equal(call.args[0], [[-1, -1, -1]])
        expected_ongoing = [[0, 0, 0], [-1, -1, -1], [-1, -1, -1]] if rotation_interval is None else [[-1, -1, -1]]
        np.testing.assert_array_equal(trail.call_args.args[0], expected_ongoing)
        current = [call.kwargs["context"] for call in panel.call_args_list]
        assert [context.step for context in current] == [0, 0, 0, 2 if rotation_interval is None else 0]
        assert [context.episode_return for context in current] == [0, 0, 0, 8 if rotation_interval is None else 0]
        for context in current[:3] if rotation_interval is None else current:
            assert context.reward is context.action is context.stop_reason is None
        panel.reset_mock()
        trail.reset_mock()
        for index in range(4):
            outcome = env.render_outcome(frames[index], transition.final_observations, index=index)  # (H, W, 3)
            assert outcome.shape == frames[index].shape
            np.testing.assert_array_equal(outcome[:4], frames[index, :4])
            expected_trail = [[-1, -1, -1], [10 * (index + 1)] * 3] if index < 3 else [[0] * 3, [-1] * 3, [-1] * 3]
            np.testing.assert_array_equal(trail.call_args.args[0], expected_trail)
        terminal = [call.kwargs["context"] for call in panel.call_args_list]
        assert [context.step for context in terminal] == [1, 1, 1, 2]
        assert [context.episode_return for context in terminal] == [1, 2, 3, 8]
        assert [context.reward for context in terminal] == [1, 2, 3, 4]
        assert [context.stop_reason for context in terminal] == [
            StopReason.SUCCESS,
            StopReason.TIME_LIMIT,
            StopReason.COLLISION,
            None if rotation_interval is None else StopReason.SCENE_ROTATION,
        ]
        for context in terminal:
            assert context.action.shape == (4,)
            np.testing.assert_array_equal(context.action, 0.5)
        panel.reset_mock()
        env.step(actions)
        env.render()
        ongoing: RenderContext = panel.call_args.kwargs["context"]
        assert ongoing.step == (3 if rotation_interval is None else 1)
        assert ongoing.episode_return == (12 if rotation_interval is None else 4)
        panel.reset_mock()
        env.reset()
        env.render()
        for call in trail.call_args_list[-4:]:
            np.testing.assert_array_equal(call.args[0], [[-1, -1, -1]])
        for call in panel.call_args_list:
            assert call.kwargs["context"] == RenderContext(step=0, episode_return=0.0)


def test_visfly_render_channels_last_rectangular_camera_and_final_views() -> None:
    backend = VisFlyBatch(num_envs=1)
    backend.rgb = torch.full((1, 2, 4, 3), 150, dtype=torch.uint8)  # (B, H, W, C)
    raw = backend.get_observation()  # leaves: (B, *O)
    raw["depth"] = torch.full((1, 2, 4, 1), 10.0)  # (B, H, W, 1)
    with (
        patch.object(backend, "get_observation", return_value=raw),
        patch("rl2.envs.visfly_env.load_env_creator", return_value=Mock(return_value=backend)),
        closing(VisFlyEnv(config=VisFlyConfig(channels_first=False), num_envs=1, seed=1)) as env,
    ):
        env.reset()
        frame = env.render()[0]  # (2 * H, H + W_right, 3)
        assert frame.shape == (8, 10, 3)
        np.testing.assert_array_equal(frame[:4, :4], 0)
        np.testing.assert_array_equal(frame[1, 4:7], 150)
        assert frame[1, 7:].any()
        before = frame.copy()  # (H, W, 3)
        raw = backend.get_observation()  # leaves: (B, *O)
        with patch.object(
            backend,
            "step",
            return_value=(
                raw,
                np.zeros(1),  # (B,)
                np.ones(1, dtype=bool),  # (B,)
                [{"TimeLimit.truncated": True, "terminal_observation": {key: value[0] for key, value in raw.items()}}],
            ),
        ):
            transition = env.step(np.zeros((1, 4), dtype=np.float32))
        final = env.render_outcome(frame, transition.final_observations)  # (H, W, 3)
        np.testing.assert_array_equal(frame, before)
        np.testing.assert_array_equal(final[:4], frame[:4])
        assert not np.array_equal(final[4:, 4:], frame[4:, 4:])


def test_configuration_round_trip(config: ppo.Config) -> None:
    values = asdict(config)
    values["env"] = {"id": "VisFly", "num_scenes": 2, "kwargs": {"visual": True}, "power": {"energy_weight": 0.1}}
    values["model"]["encoder"] = asdict(
        ppo.VisFlyEncoderConfig(
            stages=(ConvStage(channels=2, blocks=0),), embedding_size=4, goal_scale=2.0, depth_scale=4.0
        )
    )
    parsed = TypeAdapter(ppo.Config).validate_json(json.dumps(values))
    assert isinstance(parsed.env, VisFlyConfig)
    assert parsed.env.kwargs == {"visual": True}
    assert parsed.env.power == PowerConfig(energy_weight=0.1)
    assert isinstance(parsed.model.encoder, ppo.VisFlyEncoderConfig)
    assert parsed.model.encoder.goal_scale == 2.0
    assert parsed.model.encoder.depth_scale == 4.0
    hash(parsed)  # Configurations with arbitrary environment kwargs can specialize JIT updates.


@pytest.mark.parametrize(
    "scales",
    [
        {"goal_scale": 0.0},
        {"velocity_scale": -1.0},
        {"angular_velocity_scale": float("inf")},
        {"depth_scale": 0.0},
        {"depth_scale": float("nan")},
    ],
)
def test_visfly_encoder_requires_positive_scales(scales: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ppo.VisFlyEncoderConfig(**scales)


def test_visfly_encoder_requires_navigation_environment(config: ppo.Config) -> None:
    with pytest.raises(ValueError, match="visfly encoder requires a VisFly"):
        replace(config, model=replace(config.model, encoder=ppo.VisFlyEncoderConfig()))


@pytest.mark.parametrize(
    ("manifest", "episodes", "error"),
    [
        pytest.param(False, 2, None, id="seeded"),
        pytest.param(True, None, None, id="manifest"),
        pytest.param(True, 2, "mutually exclusive", id="both_modes"),
        pytest.param(False, None, "eval_episodes is required", id="neither_mode"),
        pytest.param(False, 0, "greater than 0", id="empty_seeded_evaluation"),
    ],
)
def test_evaluation_configuration_modes(
    config: ppo.Config, manifest: bool, episodes: int | None, error: str | None
) -> None:
    values = asdict(config)
    values["env"] = asdict(
        VisFlyConfig(evaluation=EvaluationConfig(manifest_path="/tmp/eval.json") if manifest else None)
    )
    values["eval_episodes"] = episodes
    adapter = TypeAdapter(ppo.Config)
    if error is not None:
        with pytest.raises(ValueError, match=error):
            adapter.validate_json(json.dumps(values))
    else:
        parsed = adapter.validate_json(json.dumps(values))
        assert parsed.eval_episodes == episodes
        assert adapter.validate_json(adapter.dump_json(parsed)) == parsed


def test_evaluation_scores_without_rendering(config: ppo.Config) -> None:
    config = replace(config, env=VisFlyConfig(power=PowerConfig()), eval_episodes=2, video_max_frames=1)
    backends = []

    def create(**kwargs: Any) -> VisFlyBatch:
        assert kwargs["num_scene"] == kwargs["num_agent_per_scene"] == 1
        assert kwargs["scene_split"] == "test"
        backend = VisFlyBatch(num_envs=1)
        backend.envs = SimpleNamespace(dynamics=SimpleNamespace(dt=0.01, _interval_steps=2))
        step = backend.step

        def delayed_done(actions: NDArray[np.float32]) -> tuple[Any, ...]:  # (1, 4)
            observations, rewards, done, infos = step(actions)
            backend.power_usage = PowerUsage(
                energy=torch.tensor([0.02 * backend.steps]),  # (B,)
                excess=torch.zeros(1),  # (B,)
            )
            if backend.steps == 1:
                done[:] = False
            return observations, rewards, done, infos

        backend.step = Mock(side_effect=delayed_done)
        backends.append(backend)
        return backend

    state = TrainState.create(apply_fn=value_policy, params={}, tx=optax.sgd(0.1))
    memory = ((jnp.zeros((1, 1)), jnp.zeros((1, 1))),)  # leaves: (1, 1)
    writer = Mock()
    with patch("rl2.envs.visfly_env.load_env_creator", return_value=create):
        ppo.log_evaluation(state, config, writer, 0, 4, initial_carry=memory)
    assert len(backends) == 2
    assert all(backend.closed and backend.steps == 2 for backend in backends)  # Scoring ignores the video cap.
    for backend in backends:
        actions = backend.step.call_args.args[0]  # (1, 4)
        assert actions.shape == (1, 4)
        np.testing.assert_array_equal(actions, np.zeros((1, 4), dtype=np.float32))
    scalars, step = writer.log_metrics.call_args.args
    assert step == 4
    assert scalars["eval/return"] == 2
    assert scalars["eval/return_std"] == scalars["eval/return_sem"] == 0
    assert scalars["eval/stop_fraction/success"] == 1
    assert scalars["eval/stop_fraction/terminated"] == scalars["eval/stop_fraction/collision"] == 0
    assert scalars["eval/final_goal_distance_meters"] == pytest.approx(np.sqrt(300))
    assert scalars["eval/termination_fraction"] == 1
    assert scalars["eval/truncation_fraction"] == 0
    assert "eval/energy_hover_seconds" not in scalars
    assert "eval/energy_hover_seconds/success" not in scalars
    assert scalars["eval/power_ratio"] == pytest.approx(1.5)
    assert "eval/power_ratio_max" not in scalars
    report = json.loads(writer.log_text.call_args.args[0])
    assert report["episodes"][0]["power_ratio"] == pytest.approx(1.5)
    assert all("energy_hover_seconds" not in episode for episode in report["episodes"])
    assert all("power_ratio_max" not in episode for episode in report["episodes"])
    writer.log_video.assert_not_called()


def test_evaluation_preserves_zero_power_for_failed_episode(config: ppo.Config) -> None:
    config = replace(config, eval_episodes=1, eval_video=False)
    episode = ppo.EpisodeResult(
        episode_return=0.0,
        length=1,
        frames=[],
        stop_reason=StopReason.COLLISION,
        terminated=True,
        power_ratio=0.0,
    )
    writer = Mock()
    with patch.object(ppo, "play_episode", return_value=episode):
        ppo.log_evaluation(Mock(), config, writer, 0, 4, initial_carry=())
    metrics = writer.log_metrics.call_args.args[0]
    assert metrics["eval/power_ratio"] == 0.0
    assert "eval/energy_hover_seconds/success" not in metrics


@pytest.mark.parametrize("availability", ["complete", "missing", "partial", "zero_distance"])
def test_evaluation_stop_fractions_exclude_video_replay(
    config: ppo.Config, availability: Literal["complete", "missing", "partial", "zero_distance"]
) -> None:
    config = replace(config, eval_episodes=2, eval_video=True)
    complete = availability in ("complete", "zero_distance")
    episodes = [
        ppo.EpisodeResult(
            episode_return=reward,
            length=1,
            frames=[],
            stop_reason=(StopReason.SUCCESS if success else StopReason.COLLISION)
            if complete or (availability == "partial" and success)
            else None,
            terminated=True,
            truncated=not success,  # Simultaneous termination and timeout retain both flags.
            final_goal_distance=(0.0 if success or availability == "zero_distance" else 2.0)
            if complete or (availability == "partial" and success)
            else None,
            power_ratio=0.0 if complete or (availability == "partial" and success) else None,
        )
        for reward, success in [(1.0, True), (0.0, False)]
    ]
    replay = ppo.EpisodeResult(
        episode_return=1.0,
        length=1,
        frames=[np.zeros((2, 2, 3), dtype=np.uint8)],  # each (H, W, 3)
        stop_reason=StopReason.SUCCESS,
        terminated=True,
        final_goal_distance=100.0,  # Replay must not affect scored distances.
        power_ratio=100.0,
    )
    writer = Mock()
    with patch.object(ppo, "play_episode", side_effect=[*episodes, replay]) as play:
        ppo.log_evaluation(Mock(), config, writer, 0, 4, initial_carry=())
    assert play.call_count == 3
    assert all(call.kwargs["deterministic"] for call in play.call_args_list)
    scalars, step = writer.log_metrics.call_args.args
    fractions = {key: value for key, value in scalars.items() if key.startswith("eval/stop_fraction/")}
    assert step == 4
    assert scalars["eval/termination_fraction"] == 1
    assert scalars["eval/truncation_fraction"] == 0.5
    if complete:
        assert fractions == {
            "eval/stop_fraction/success": 0.5,
            "eval/stop_fraction/collision": 0.5,
            "eval/stop_fraction/time_limit": 0.0,
            "eval/stop_fraction/terminated": 0.0,
            "eval/stop_fraction/scene_rotation": 0.0,
        }
        assert sum(fractions.values()) == 1
        assert scalars["eval/final_goal_distance_meters"] == (0 if availability == "zero_distance" else 1)
        assert scalars["eval/power_ratio"] == 0
    else:
        assert fractions == {}
        assert "eval/final_goal_distance_meters" not in scalars
        assert "eval/energy_hover_seconds" not in scalars
        assert "eval/energy_hover_seconds/success" not in scalars
        assert "eval/power_ratio" not in scalars
    report = json.loads(writer.log_text.call_args.args[0])
    assert len(report["episodes"]) == 2
    assert ("stop_reason" in report["episodes"][0]) == (availability != "missing")
    if complete:
        assert [episode["stop_reason"] for episode in report["episodes"]] == ["success", "collision"]
        assert report["episodes"][0]["final_goal_distance_meters"] == 0
    assert report["episodes"][1]["terminated"] and report["episodes"][1]["truncated"]
    writer.log_video.assert_called_once()


@pytest.mark.parametrize(
    ("accepted", "resume"),
    [(True, False), (False, False), (True, True)],
    ids=["accepted_updates", "all_updates_rejected", "resume_changed_batch_size"],
)
def test_train_orchestration_uses_host_env_and_closes_resources(
    config: ppo.Config, accepted: bool, resume: bool
) -> None:
    config = replace(config, total_steps=11 if resume else 4, anneal_lr=True, run_id="host-test")
    env = Mock(action_space=gym.spaces.Discrete(2))
    env.reset.return_value = np.zeros((2, 3), dtype=np.float32)  # (B, O)
    model = Mock()
    model.initial_carry.side_effect = lambda batch: ((jnp.zeros((batch, 1)), jnp.zeros((batch, 1))),)
    model.init.return_value = {"params": {"weight": jnp.ones(1)}}
    manager = MagicMock()
    writer = MagicMock()
    writer.__enter__.return_value = writer

    def restore(manager: Any, state: TrainState, settings: ppo.Config) -> ppo.TrainingProgress | None:
        if not resume:
            return None
        return ppo.TrainingProgress(
            train_state=state.replace(step=2),
            key=jax.random.key(3),  # ()
            rollout_count=3,
            episode_count=5,
            episode_return_sum=10.0,
            episode_length_sum=15,
            env_steps=6,  # Saved batch size was 2; current batch size is 4.
        )

    def iteration(
        runner: ppo.RolloutCarry,
        state: TrainState,
        envs: ppo.PPOEnv,
        settings: ppo.Config,
        key: jax.Array,  # ()
        metrics: ppo.TrainingMetrics,
        schedule_index: int,
        info_metrics: ppo.RolloutInfoMetrics,
    ) -> ppo.TrainStepResult:
        assert envs is env
        info_metrics.add(GymStyleInfo(power_ratio=np.array([1.25, 0.0], dtype=np.float32)))  # (B,)
        assert schedule_index == (1 if resume else 0)
        window = metrics.window._replace(
            minibatch_count=jnp.asarray(3),
            rollout_count=jnp.asarray(1),
            update_count=jnp.asarray(2 if accepted else 0),
            minibatch_metrics=metrics.window.minibatch_metrics.replace(
                policy_loss=jnp.asarray(0.6),  # () sum over three evaluated minibatches
                value_loss=jnp.asarray(1.2),  # ()
                entropy=jnp.asarray(6.0),  # ()
                entropy_loss=jnp.asarray(-0.06),  # () coefficient-weighted negative entropy sum
                gradient_norm=jnp.asarray(6.0 if accepted else 0.0),  # () sum over accepted updates
                gradient_clip_fraction=jnp.asarray(1.0 if accepted else 0.0),  # () accepted clipping count
            ),
            final_approx_kl_sum=jnp.asarray(0.03),  # ()
        )
        return ppo.TrainStepResult(
            train_state=state.replace(step=state.step + (2 if accepted else 0)),
            rollout_carry=runner,
            key=key,
            metrics=metrics._replace(window=window),
        )

    with (
        patch.object(ppo, "make_env", return_value=env),
        patch.object(ppo, "ActorCritic", return_value=model),
        patch.object(ppo, "checkpoint_manager", return_value=manager),
        patch.object(ppo, "restore_checkpoint", side_effect=restore),
        patch.object(ppo, "MLflowLogger", return_value=writer),
        patch.object(ppo, "train_step", side_effect=iteration),
    ):
        result = ppo.train(config)
    assert int(result.step) == (4 if resume else 2 if accepted else 0)
    env.reset.assert_called_once_with()
    env.close.assert_called_once_with()
    manager.close.assert_called_once_with()
    manager.save.assert_called_once()
    assert manager.save.call_args.args[0] == (4 if resume else 1)
    scalars = {key: value for call in writer.log_metrics.call_args_list for key, value in call.args[0].items()}
    assert scalars["train/env_steps_total"] == (10 if resume else 4)
    assert scalars["train/iterations_total"] == (4 if resume else 1)
    assert scalars["train/learning_rate"] == pytest.approx(config.learning_rate / (2 if resume else 1))
    assert scalars["train/progress_fraction"] == pytest.approx(10 / 11 if resume else 1)
    assert "rollout/return" not in scalars
    assert scalars["train/final_approx_kl"] == pytest.approx(0.03)
    assert scalars["train/entropy"] == 2
    assert scalars["train/entropy_loss"] == pytest.approx(-0.02)
    assert scalars["train/total_loss"] == pytest.approx(
        scalars["train/policy_loss"] + config.value_coef * scalars["train/value_loss"] + scalars["train/entropy_loss"]
    )
    assert scalars["rollout/episode_count"] == 0
    assert scalars["rollout/power_ratio"] == 0.625
    assert scalars["train/episodes_total"] == (5 if resume else 0)
    assert "rollout/episode_length_steps" not in scalars
    if resume:
        assert scalars["train/return"] == 2.0
        assert scalars["train/episode_length_steps"] == 3.0
    else:
        assert "train/return" not in scalars
        assert "train/episode_length_steps" not in scalars
    assert "train/explained_variance" not in scalars
    assert "train/kl_early_stop_fraction" not in scalars
    assert "actor/video_seconds" not in scalars
    assert not any(key.startswith("eval/") for key in scalars)
    if accepted:
        assert scalars["train/gradient_norm"] == 3
        assert scalars["train/gradient_clip_fraction"] == 0.5
    else:
        assert "train/gradient_norm" not in scalars
        assert "train/gradient_clip_fraction" not in scalars


def test_checkpoint_shutdown_failure_reaches_logger(config: ppo.Config) -> None:
    env = Mock(action_space=gym.spaces.Discrete(2))
    env.reset.return_value = np.zeros((2, 3), dtype=np.float32)  # (B, O)
    model = Mock()
    model.initial_carry.return_value = ()
    model.init.return_value = {"params": {"weight": jnp.ones(1)}}  # (1,)
    manager = Mock()
    failure = RuntimeError("async checkpoint write failed")
    manager.close.side_effect = failure
    writer = MagicMock()
    writer.__enter__.return_value = writer

    def completed_progress(
        manager: ppo.ocp.CheckpointManager, state: TrainState, settings: ppo.Config
    ) -> ppo.TrainingProgress:
        return ppo.TrainingProgress(
            train_state=state,
            key=jax.random.key(0),  # ()
            rollout_count=2,
            episode_count=0,
            episode_return_sum=0.0,
            episode_length_sum=0,
            env_steps=settings.total_steps,
        )

    with (
        patch.object(ppo, "make_env", return_value=env),
        patch.object(ppo, "ActorCritic", return_value=model),
        patch.object(ppo, "checkpoint_manager", return_value=manager),
        patch.object(ppo, "restore_checkpoint", side_effect=completed_progress),
        patch.object(ppo, "MLflowLogger", return_value=writer),
        pytest.raises(RuntimeError, match="async checkpoint write failed"),
    ):
        ppo.train(config)
    env.close.assert_called_once_with()
    manager.close.assert_called_once_with()
    writer.__exit__.assert_called_once()
    assert writer.__exit__.call_args.args[-2] is failure


def scalar_policy(
    variables: dict[str, dict[str, jax.Array]],  # parameter leaves: ()
    observations: jax.Array,  # (B, T, 1)
    memory: ppo.RecurrentCarry,  # leaves: (B, H)
    starts: jax.Array,  # (B, T)
) -> ppo.ActorCriticOutput:
    weight = variables["params"]["weight"]  # () shared actor/critic parameter
    return ppo.ActorCriticOutput(
        carry=memory,
        policy=TruncatedGaussianPolicy(
            mean=weight + observations,  # (B, T, 1)
            log_std=jnp.zeros_like(observations),  # (B, T, 1)
            action_space=ContinuousSpace(shape=(1,), low=(-1.0,), high=(1.0,)),
        ),
        values=jnp.broadcast_to(weight, starts.shape),  # (B, T)
    )


@pytest.fixture
def scalar_batch() -> ppo.PPOBatch:
    observations = jnp.array([[[0.0]], [[0.5]]])  # (B, T, O), distinct states test full-batch KL reduction
    starts = jnp.ones((2, 1), dtype=bool)  # (B, T)
    policy = scalar_policy({"params": {"weight": jnp.asarray(0.0)}}, observations, (), starts).policy
    return ppo.PPOBatch(
        observations=observations,
        actions=jnp.zeros((2, 1, 1)),  # (B, T, D)
        old_log_probs=policy.log_prob(jnp.zeros((2, 1, 1))),  # (B, T)
        values=jnp.zeros((2, 1)),  # (B, T)
        advantages=jnp.zeros((2, 1)),  # (B, T), isolate the known critic gradient
        returns=jnp.full((2, 1), 2.0),  # (B, T)
        model_carry=(),
        episode_starts=starts,
    )


@pytest.mark.parametrize("kind", ["normalized", "constant", "singleton"])
def test_update_advantage_normalization(config: ppo.Config, scalar_batch: ppo.PPOBatch, kind: str) -> None:
    config = replace(config, value_coef=0.0, entropy_coef=0.0)
    if kind == "singleton":
        scalar_batch = jax.tree.map(lambda leaf: leaf[1:], scalar_batch)  # leaves: (B=1, T=1, ...)
        advantages = jnp.full((1, 1), 0.7)  # (B, T)
    elif kind == "constant":
        advantages = jnp.full((2, 1), 0.7)  # (B, T)
    else:
        advantages = jnp.asarray([[3.0], [1.0]])  # (B, T), normalized to [1, -1]
    state = TrainState.create(
        apply_fn=scalar_policy,
        params={"weight": jnp.asarray(0.0)},  # ()
        tx=optax.sgd(0.1),
    )
    updated, metrics = jax.jit(partial(ppo.update, config=config))(state, scalar_batch.replace(advantages=advantages))
    assert metrics.policy_loss.shape == metrics.gradient_norm.shape == ()
    assert int(updated.step) == 1
    assert float(metrics.policy_loss) == pytest.approx(-0.7 if kind == "singleton" else 0.0, abs=1e-6)
    if kind == "constant":
        assert float(metrics.gradient_norm) == 0.0
        assert float(updated.params["weight"]) == 0.0
    else:
        assert float(metrics.gradient_norm) > 0.0
        assert (float(updated.params["weight"]) < 0.0) == (kind == "singleton")


@pytest.mark.parametrize(
    ("max_grad_norm", "rejected"),
    [(0.5, False), (3.0, False), (0.5, True)],
    ids=["clipped", "unclipped", "kl_rejected"],
)
def test_update_gradient_metrics(
    config: ppo.Config, scalar_batch: ppo.PPOBatch, max_grad_norm: float, rejected: bool
) -> None:
    config = replace(config, entropy_coef=0.0, max_grad_norm=max_grad_norm, target_kl=0.02)
    if rejected:
        scalar_batch = scalar_batch.replace(old_log_probs=scalar_batch.old_log_probs - 1.0)  # (B, T)
    state = TrainState.create(
        apply_fn=scalar_policy,
        params={"weight": jnp.asarray(0.0)},  # ()
        tx=optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.sgd(0.1)),
    )
    updated, metrics = jax.jit(partial(ppo.update, config=config))(state, scalar_batch)
    assert metrics.gradient_norm.shape == metrics.gradient_clip_fraction.shape == ()
    assert float(metrics.gradient_norm) == (0.0 if rejected else 2.0)
    assert float(metrics.gradient_clip_fraction) == float(not rejected and max_grad_norm < 2.0)
    assert int(updated.step) == int(not rejected)
    expected_weight = 0.0 if rejected else 0.1 * min(2.0, max_grad_norm)
    assert float(updated.params["weight"]) == pytest.approx(expected_weight)


@pytest.mark.parametrize("early_stop", [False, True], ids=["all_epochs", "kl_stop"])
def test_final_kl_uses_final_parameters_and_full_batch(
    config: ppo.Config, scalar_batch: ppo.PPOBatch, early_stop: bool
) -> None:
    config = replace(
        config,
        num_steps=1,
        num_minibatches=2,
        learning_rate=0.1,
        entropy_coef=0.0,
        target_kl=1e-12 if early_stop else None,
    )

    def optimizer(learning_rate: float) -> optax.GradientTransformation:
        return optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.sgd(learning_rate))

    state = TrainState.create(
        apply_fn=scalar_policy,
        params={"weight": jnp.asarray(0.0)},  # ()
        tx=optax.inject_hyperparams(optimizer)(learning_rate=config.learning_rate),
    )
    rollout = ppo.RolloutResult(
        batch=scalar_batch,
        key=jax.random.key(0),  # key: ()
        episode_return_sum=jnp.asarray(0.0),  # ()
        episode_length_sum=jnp.asarray(0),  # ()
        episode_count=jnp.asarray(0),  # ()
    )
    carry = ppo.RolloutCarry(
        observations=scalar_batch.observations[:, 0],  # (B, O)
        model_carry=(),
        episode_starts=jnp.zeros(2, dtype=bool),  # (B,)
        episode_returns=jnp.zeros(2),  # (B,)
        episode_step_counts=jnp.zeros(2, dtype=jnp.int32),  # (B,)
    )
    result = ppo.optimize_rollout(carry, rollout, state, config, ppo.init_metrics())
    output = scalar_policy(
        {"params": result.train_state.params}, scalar_batch.observations, (), scalar_batch.episode_starts
    )
    log_ratio = output.policy.log_prob(scalar_batch.actions) - scalar_batch.old_log_probs  # (B, T)
    expected = (jnp.expm1(log_ratio) - log_ratio).mean()  # ()
    window = result.metrics.window
    assert window.final_approx_kl_sum.shape == ()
    assert float(window.final_approx_kl_sum) == pytest.approx(float(expected), rel=1e-4, abs=1e-9)
    assert float(window.final_approx_kl_sum) > 0
    assert int(window.update_count) == (1 if early_stop else 2)
    assert int(window.early_stop_count) == int(early_stop)
    assert float(window.minibatch_metrics.gradient_norm) == pytest.approx(2.0 if early_stop else 3.95)


@pytest.fixture
def flight_manifest(tmp_path: Path) -> Manifest:
    route = Route(
        points=((1.0, 1.0, 1.0), (1.0, 5.0, 1.0), (8.0, 5.0, 1.0)),
        direct_blocked=True,
        constrained=False,
        long=True,
        detour_threshold=1.25,
    )
    scenarios = tuple(
        Scenario(
            scene=scene,
            category=category,
            route=route,
            orientation=(1.0, 0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            max_episode_steps=2000,
            seed=seed,
        )
        for scene, category, seed in [
            ("test/a.json", "long_traversal", 1),
            ("test/a.json", "obstacle_avoidance", 2),
            ("test/b.json", "long_traversal", 3),
        ]
    )
    return Manifest(
        scenarios=scenarios,
        coverage=tuple(
            SceneCoverage(
                scene=scene,
                aliases=(),
                unavailable=tuple(
                    category
                    for category in CATEGORIES
                    if category not in {s.category for s in scenarios if s.scene == scene}
                ),
            )
            for scene in ("test/a.json", "test/b.json", "test/unavailable.json")
        ),
        seed=8,
        generation=EvaluationConfig(manifest_path=str(tmp_path / "manifest.json")),
        control_dt=0.02,
    )


def test_manifest_evaluation_scores_all_routes_and_replays_group_failures_and_best(
    config: ppo.Config,
    flight_manifest: Manifest,
) -> None:
    config = replace(
        config, env=VisFlyConfig(evaluation=flight_manifest.generation), eval_episodes=None, eval_video=True
    )
    scored = [
        ppo.EpisodeResult(
            episode_return=float(i),
            length=1,
            frames=[],
            terminated=True,
            stop_reason=StopReason.SUCCESS if i < 2 else StopReason.COLLISION,
        )
        for i in range(3)
    ]
    replay = ppo.EpisodeResult(
        episode_return=1000.0, length=1, frames=[np.zeros((2, 2, 3), dtype=np.uint8)]
    )  # (H, W, 3)
    writer = Mock()
    with (
        patch.object(ppo, "evaluation_manifest", return_value=flight_manifest),
        patch.object(ppo, "play_episode", side_effect=[*scored, replay, replay]) as play,
        patch.object(ppo, "make_env", return_value=Mock()) as make,
    ):
        ppo.log_evaluation(Mock(), config, writer, 9, 40, initial_carry=())
    assert play.call_count == 5
    make.assert_called_once()
    assert all(call.kwargs["env"] is make.return_value for call in play.call_args_list)
    make.return_value.close.assert_called_once()
    assert [call.kwargs["scenario"] for call in play.call_args_list[:3]] == list(flight_manifest.scenarios)
    # In test/long_traversal, the failure is also best; test/obstacle_avoidance is separate.
    assert [call.kwargs["scenario"] for call in play.call_args_list[3:]] == [
        flight_manifest.scenarios[1],
        flight_manifest.scenarios[2],
    ]
    metrics, step = writer.log_metrics.call_args.args
    assert step == 40
    assert metrics["eval/return"] == 1
    assert metrics["eval/return_sem"] == pytest.approx(1 / np.sqrt(3))
    assert metrics["eval/stop_fraction/success"] == pytest.approx(2 / 3)
    assert metrics["eval/scene_success_fraction"] == 0.5
    assert metrics["eval/scene_collision_fraction"] == 0.5
    assert metrics["eval/success_fraction/scene_0000"] == 1
    assert metrics["eval/success_fraction/long_traversal"] == 0.5
    assert metrics["eval/collision_fraction/obstacle_avoidance"] == 0
    assert "eval/success_fraction/scene_0002" not in metrics
    assert metrics["eval/scene_coverage_fraction"] == pytest.approx(2 / 3)
    assert metrics["eval/category_coverage_fraction/constrained_passage"] == 0
    report = json.loads(writer.log_text.call_args.args[0])
    assert len(report["episodes"]) == 3
    assert report["episodes"][0]["scenario"]["scene"] == "test/a.json"
    assert len(report["manifest"]["coverage"]) == 3
    assert writer.log_video.call_count == 2


def test_scenario_replays_group_by_folder_and_category(flight_manifest: Manifest) -> None:
    cases = [
        # Multiple scenes in one group: first failure and highest return, with first-on-tie behavior.
        ("box30_test/a.json", "long_traversal", 0.0, StopReason.SUCCESS),
        ("box30_test/b.json", "long_traversal", -1.0, StopReason.COLLISION),
        ("box30_test/c.json", "long_traversal", 3.0, StopReason.SUCCESS),
        ("box30_test/d.json", "long_traversal", 3.0, StopReason.SUCCESS),
        # Same category in another group; a timed-out best episode also fills the failure role.
        ("garage_test/a.json", "long_traversal", 2.0, StopReason.TIME_LIMIT),
        ("garage_test/b.json", "long_traversal", 1.0, StopReason.COLLISION),
        # Same group, different category; unknown outcomes must not add a failure replay.
        ("box30_test/a.json", "obstacle_avoidance", -2.0, None),
        ("box30_test/b.json", "obstacle_avoidance", 2.0, StopReason.SUCCESS),
        # Root-level scenes share a group instead of being grouped by filename.
        ("a.json", "long_traversal", 1.0, StopReason.SUCCESS),
        ("b.json", "long_traversal", 2.0, StopReason.SUCCESS),
    ]
    scenarios = tuple(
        replace(flight_manifest.scenarios[0], scene=scene, category=category) for scene, category, _, _ in cases
    )
    episodes = [
        ppo.EpisodeResult(episode_return=reward, length=1, frames=[], stop_reason=reason)
        for _, _, reward, reason in cases
    ]
    assert ppo.scenario_replay_indices(episodes, scenarios) == [1, 2, 4, 7, 9]


def test_fixed_scenario_overrides_training_sampling_horizon_and_rotation(
    config: ppo.Config,
    flight_manifest: Manifest,
) -> None:
    scenario = flight_manifest.scenarios[0]
    config = replace(config, env=VisFlyConfig(scene_rotation_env_steps=1, route_sampling=RouteConfig()))
    backend = VisFlyBatch(num_envs=1)
    creator = Mock(return_value=backend)
    state = TrainState.create(apply_fn=value_policy, params={}, tx=optax.sgd(0.1))
    memory = ((jnp.zeros((1, 1)), jnp.zeros((1, 1))),)  # leaves: (1, 1)
    with patch("rl2.envs.visfly_env.load_env_creator", return_value=creator):
        result = ppo.play_episode(
            state, config, jax.random.key(0), memory, record=False, deterministic=True, scenario=scenario
        )
    kwargs = creator.call_args.kwargs
    assert kwargs["scenario"] == asdict(scenario)
    assert kwargs["seed"] == scenario.seed
    assert kwargs["max_episode_steps"] == scenario.max_episode_steps
    assert "route_sampling" not in kwargs
    assert kwargs["random_kwargs"] == {"scene_iter": False}
    assert result.stop_reason == StopReason.SUCCESS
    assert backend.closed


def test_scenario_metrics_omit_unavailable_outcomes(flight_manifest: Manifest) -> None:
    episodes = [ppo.EpisodeResult(episode_return=0, length=1, frames=[]) for _ in flight_manifest.scenarios]
    metrics = ppo.scenario_metrics(episodes, flight_manifest)
    assert "eval/scene_success_fraction" not in metrics
    assert not any(key.startswith("eval/success_fraction/") for key in metrics)
    assert metrics["eval/direct_path_blocked_fraction"] == 1


def test_evaluation_scene_grouping_preserves_report_order(config: ppo.Config, flight_manifest: Manifest) -> None:
    manifest = replace(flight_manifest, scenarios=tuple(flight_manifest.scenarios[i] for i in (2, 0, 1)))
    config = replace(config, env=VisFlyConfig(evaluation=manifest.generation), eval_episodes=None)

    def play(*args: Any, scenario: Scenario, **kwargs: Any) -> ppo.EpisodeResult:
        return ppo.EpisodeResult(episode_return=float(scenario.seed), length=1, frames=[])

    writer = Mock()
    with (
        patch.object(ppo, "evaluation_manifest", return_value=manifest),
        patch.object(ppo, "make_env", return_value=Mock()) as make,
        patch.object(ppo, "play_episode", side_effect=play) as run,
    ):
        ppo.log_evaluation(Mock(), config, writer, 0, 0, initial_carry=())
    assert [call.kwargs["scenario"].seed for call in run.call_args_list] == [1, 2, 3]
    report = json.loads(writer.log_text.call_args.args[0])
    assert [episode["return"] for episode in report["episodes"]] == [3, 1, 2]
    make.return_value.close.assert_called_once()


def test_evaluation_shared_environment_closes_on_failure(config: ppo.Config, flight_manifest: Manifest) -> None:
    config = replace(config, env=VisFlyConfig(evaluation=flight_manifest.generation), eval_episodes=None)
    with (
        patch.object(ppo, "evaluation_manifest", return_value=flight_manifest),
        patch.object(ppo, "make_env", return_value=Mock()) as make,
        patch.object(ppo, "play_episode", side_effect=RuntimeError("step failed")),
        pytest.raises(RuntimeError, match="step failed"),
    ):
        ppo.log_evaluation(Mock(), config, Mock(), 0, 0, initial_carry=())
    make.return_value.close.assert_called_once()


@pytest.mark.parametrize(
    ("content", "error"),
    [(None, FileNotFoundError), (b"{}", ValidationError)],
    ids=["missing", "invalid"],
)
def test_training_checks_manifest_before_initialization(
    config: ppo.Config, tmp_path: Path, content: bytes | None, error: type[Exception]
) -> None:
    path = tmp_path / "manifest.json"
    if content is not None:
        path.write_bytes(content)
    config = replace(
        config,
        env=VisFlyConfig(evaluation=EvaluationConfig(manifest_path=str(path))),
        eval_episodes=None,
        eval_interval=1,
    )
    with (
        patch.object(ppo, "configure_compilation_cache") as cache,
        patch.object(ppo, "MLflowLogger") as logger,
        patch.object(ppo, "make_env") as make_env,
        pytest.raises(error),
    ):
        ppo.train(config)
    cache.assert_not_called()
    logger.assert_not_called()
    make_env.assert_not_called()


def test_offline_manifest_cli_uses_training_config(config: ppo.Config, flight_manifest: Manifest) -> None:
    config = replace(
        config,
        env=VisFlyConfig(evaluation=flight_manifest.generation),
        eval_episodes=None,
        eval_seed=123,
    )
    with (
        patch.object(manifest_generation, "load_config", return_value=config) as load,
        patch.object(manifest_generation, "generate_manifest", return_value=flight_manifest) as generate,
    ):
        manifest_generation.main(["--config", "fixture.yaml", "--workers", "2"])
    load.assert_called_once_with("fixture.yaml", ppo.Config)
    generate.assert_called_once_with(config.env, config.eval_seed, workers=2)


def test_offline_manifest_cli_rejects_other_environments(config: ppo.Config) -> None:
    with (
        patch.object(manifest_generation, "load_config", return_value=config),
        patch.object(manifest_generation, "generate_manifest") as generate,
        pytest.raises(SystemExit, match="2"),
    ):
        manifest_generation.main(["--config", "fixture.yaml", "--workers", "2"])
    generate.assert_not_called()
