"""Host rollout contracts, VisFly timeout semantics, and compiled PPO integration."""

import json
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
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
from pydantic import TypeAdapter

from rl2 import train_ppo_gym as ppo
from rl2.actions import ContinuousSpace, TruncatedGaussianPolicy
from rl2.envs import GymStyleEnv, GymStyleStep
from rl2.envs.gym_env import GymConfig, GymEnv
from rl2.envs.visfly_env import VisFlyConfig, VisFlyEnv
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

    def __init__(self, num_envs: int = 4) -> None:
        self.num_envs = num_envs
        self.action_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.closed = False
        self.steps = 0
        self.state = torch.zeros((num_envs, 1))  # (B, 1)
        self.depth = torch.zeros((num_envs, 1, 4, 4))  # (B, C, H, W)

    def reset(self) -> dict[str, torch.Tensor]:
        return {"state": self.state, "depth": self.depth}

    def step(self, actions: NDArray[np.float32]) -> tuple[Any, ...]:  # (B, 4)
        assert isinstance(actions, np.ndarray)
        assert actions.shape == (self.num_envs, 4)
        assert np.all((actions >= -1) & (actions <= 1))
        self.steps += 1
        self.state[:] = -1
        self.depth[:] = -1
        infos = []
        for index in range(self.num_envs):
            infos.append(
                {
                    "episode_done": index in (0, 2),
                    "TimeLimit.truncated": index in (1, 2),
                    "terminal_observation": {
                        "state": torch.tensor([10.0 * (index + 1)]),  # (1,)
                        "depth": torch.full((1, 4, 4), float(index + 1)),  # (C, H, W)
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
        return [np.full((4, 4, 4), self.steps, dtype=np.uint8)]  # one (H, W, RGBA) scene

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
        assert all(isinstance(leaf, np.ndarray) for leaf in result)
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
    backend = VisFlyBatch()
    settings = VisFlyConfig(num_scenes=2, kwargs={"device": "cpu", "visual": True})
    with (
        patch("rl2.envs.visfly_env.load_env_creator", return_value=Mock(return_value=backend)) as loader,
        closing(VisFlyEnv(config=settings, num_envs=4, seed=17)) as env,
    ):
        initial = env.reset()
        result = env.step(np.zeros((4, 4), dtype=np.float32))
        assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves((initial, result)))
        assert result.observations["depth"].shape == (4, 4, 4, 1)
        assert result.final_observations["state"].shape == (4, 1)
        np.testing.assert_array_equal(result.terminated, [True, False, True, False])
        np.testing.assert_array_equal(result.truncated, [False, True, True, False])
        np.testing.assert_array_equal(result.final_observations["state"][:, 0], [10, 20, 30, -1])
        np.testing.assert_array_equal(result.observations["state"], -1)
        np.testing.assert_array_equal(initial["state"], 0)  # simulator buffer reuse cannot mutate a rollout
        loader.return_value.assert_called_once_with(
            num_scene=2,
            num_agent_per_scene=2,
            seed=17,
            requires_grad=False,
            device="cpu",
            visual=True,
        )
        assert env.render().shape == (4, 4, 3)
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
            config=env_config,
            num_envs=1 if evaluation else config.num_envs,
            seed=17,
            evaluation=evaluation,
        )
        gym_env.assert_not_called()
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


def test_rollout_bootstraps_timeout_with_final_observation_and_memory(config: ppo.Config) -> None:
    config = replace(config, env=VisFlyConfig(), num_envs=4)
    backend = VisFlyBatch()
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
        advanced, rollout = ppo.collect_rollout(runner, state, env, config, jax.random.key(9))
        batch = rollout.batch
        assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves((advanced, batch)))
        assert batch.actions.shape == (4, 2, 4)
        assert batch.observations["depth"].shape == (4, 2, 4, 4, 1)
        assert batch.returns.shape == (4, 2)
        # Bootstrap sees the terminal observation (20), and memory advanced twice (2).
        np.testing.assert_allclose(batch.returns[:3], [[1, 1], [13, 13], [3, 3]])
        np.testing.assert_array_equal(batch.episode_starts, [[True, True]] * 3 + [[True, False]])
        np.testing.assert_array_equal(advanced.model_carry[0][0][:, 0], [1, 1, 1, 2])
        assert int(rollout.episode_count) == 6
        assert float(rollout.episode_return_sum) == 12.0  # raw rewards, excluding timeout correction
        assert int(rollout.episode_length_sum) == 6
        np.testing.assert_array_equal(advanced.episode_returns, [0, 0, 0, 8])
        np.testing.assert_array_equal(batch.observations["state"][:, 0], 0)


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


def test_mixed_image_vector_encoder_and_sequence_shapes(config: ppo.Config) -> None:
    settings = replace(
        config.model, encoder=ppo.Conv2DEncoderConfig(stages=(ConvStage(channels=2, blocks=0),), embedding_size=4)
    )
    model = ppo.ActorCritic(action_space=ppo.DiscreteSpace(n=2), model=settings)
    obs = {"depth": jnp.zeros((2, 2, 4, 4, 1)), "state": jnp.zeros((2, 2, 3))}  # (B, T, *O)
    memory = model.initial_carry(2)  # leaves: (B, H)
    starts = jnp.zeros((2, 2), dtype=bool)  # (B, T)
    params = model.init(jax.random.key(1), obs, memory, starts)
    output = jax.jit(model.apply)(params, obs, memory, starts)
    assert output.values.shape == (2, 2)
    assert output.policy.logits.shape == (2, 2, 2)
    assert all(leaf.shape == (2, 4) for leaf in jax.tree.leaves(output.carry))


def test_evaluation_uses_fresh_environment_and_closes_on_failure(config: ppo.Config) -> None:
    backend = VisFlyBatch(num_envs=1)
    config = replace(config, env=VisFlyConfig(), video_max_frames=1)
    state = TrainState.create(apply_fn=value_policy, params={}, tx=optax.sgd(0.1))
    memory = ((jnp.zeros((1, 1)), jnp.zeros((1, 1))),)  # leaves: (1, 1)
    with patch("rl2.envs.visfly_env.load_env_creator", return_value=Mock(return_value=backend)):
        result = ppo.play_episode(state, config, jax.random.key(0), memory, record=True, stop_at_frame_limit=True)
        assert len(result.frames) == 1
        assert result.length == 0
        assert backend.steps == 0
        assert backend.closed
        backend.closed = False
        with (
            patch.object(backend, "render", side_effect=RuntimeError("renderer failed")),
            pytest.raises(RuntimeError, match="renderer failed"),
        ):
            ppo.play_episode(state, config, jax.random.key(0), memory, record=True)
        assert backend.closed


def test_configuration_round_trip(config: ppo.Config) -> None:
    values = asdict(config)
    values["env"] = {"id": "VisFly", "num_scenes": 2, "kwargs": {"visual": True}}
    parsed = TypeAdapter(ppo.Config).validate_json(json.dumps(values))
    assert isinstance(parsed.env, VisFlyConfig)
    assert parsed.env.kwargs == {"visual": True}
    hash(parsed)  # Configurations with arbitrary environment kwargs can specialize JIT updates.


def test_evaluation_scores_without_rendering(config: ppo.Config) -> None:
    config = replace(config, env=VisFlyConfig(), eval_episodes=2)
    backends = []

    def create(**kwargs: Any) -> VisFlyBatch:
        assert kwargs["num_scene"] == kwargs["num_agent_per_scene"] == 1
        backend = VisFlyBatch(num_envs=1)
        backends.append(backend)
        return backend

    state = TrainState.create(apply_fn=value_policy, params={}, tx=optax.sgd(0.1))
    memory = ((jnp.zeros((1, 1)), jnp.zeros((1, 1))),)  # leaves: (1, 1)
    writer = Mock()
    with patch("rl2.envs.visfly_env.load_env_creator", return_value=create):
        ppo.log_evaluation(state, config, writer, 0, 4, initial_carry=memory)
    assert len(backends) == 2
    assert all(backend.closed and backend.steps == 1 for backend in backends)
    scalars, step = writer.log_metrics.call_args.args
    assert step == 4
    assert scalars["eval/return"] == 1
    assert scalars["eval/return_std"] == scalars["eval/return_sem"] == 0
    writer.log_video.assert_not_called()


def test_train_orchestration_uses_host_env_and_closes_resources(config: ppo.Config) -> None:
    config = replace(config, total_steps=4, run_id="host-test")
    env = Mock(action_space=gym.spaces.Discrete(2))
    env.reset.return_value = np.zeros((2, 3), dtype=np.float32)  # (B, O)
    model = Mock()
    model.initial_carry.side_effect = lambda batch: ((jnp.zeros((batch, 1)), jnp.zeros((batch, 1))),)
    model.init.return_value = {"params": {"weight": jnp.ones(1)}}
    manager = MagicMock()
    manager.latest_step.return_value = None
    writer = MagicMock()

    def iteration(
        runner: ppo.RolloutCarry,
        state: TrainState,
        envs: ppo.PPOEnv,
        settings: ppo.Config,
        key: jax.Array,  # ()
        metrics: ppo.TrainingMetrics,
        schedule_index: int,
    ) -> ppo.TrainStepResult:
        assert envs is env
        assert schedule_index == 0
        window = metrics.window._replace(
            minibatch_count=jnp.asarray(1),
            rollout_count=jnp.asarray(1),
            update_count=jnp.asarray(1),
        )
        return ppo.TrainStepResult(
            train_state=state.replace(step=1),
            rollout_carry=runner,
            key=key,
            metrics=metrics._replace(window=window),
        )

    with (
        patch.object(ppo, "make_env", return_value=env),
        patch.object(ppo, "ActorCritic", return_value=model),
        patch.object(ppo, "checkpoint_manager", return_value=manager),
        patch.object(ppo, "MLflowLogger", return_value=writer),
        patch.object(ppo, "train_step", side_effect=iteration),
    ):
        result = ppo.train(config)
    assert int(result.step) == 1
    env.reset.assert_called_once_with()
    env.close.assert_called_once_with()
    manager.close.assert_called_once_with()
    manager.save.assert_called_once()
    scalars = {key: value for call in writer.log_metrics.call_args_list for key, value in call.args[0].items()}
    assert scalars["train/env_steps_total"] == 4
    assert scalars["train/iterations_total"] == 1
    assert "rollout/return" not in scalars
    assert "train/return" not in scalars
