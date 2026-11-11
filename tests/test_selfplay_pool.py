"""Pool lifecycle, learner-only rollouts, and checkpoint integration."""

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import tokamax  # noqa: F401 - load Triton before environment helpers import TensorFlow
import yaml
from flax.training.train_state import TrainState
from jax.sharding import AxisType, Mesh, PartitionSpec
from ppo_helpers import metric_history
from test_train_mappo import config as config  # noqa: PLC0414

from rl2 import train_mappo as ppo
from rl2.envs.multi_agent import AgentObservation, BaseMultiAgentEnv, MultiAgentStep
from rl2.selfplay.pool import MatchAssignment, SelfPlayConfig, init_pool, insert_snapshot, resize_pool, sample_opponents

pytestmark = pytest.mark.usefixtures("ppo_mesh")


class Duel(BaseMultiAgentEnv[jax.Array, jax.Array]):
    num_agents = 2
    action_space = ppo.DiscreteSpace(7)
    supports_selfplay = True

    def observation(self, state: jax.Array) -> AgentObservation[jax.Array]:  # () -> (N, O)/(N, A)
        return AgentObservation(
            observation=jnp.stack((jnp.full(2, state, dtype=jnp.float32), jnp.arange(2)), axis=-1),  # (N=2, O=2)
            actions_mask=jnp.ones((2, 7), bool),  # (N, A)
        )

    def reset(self, key: jax.Array) -> tuple[AgentObservation[jax.Array], jax.Array]:  # () -> obs; ()
        state = jnp.asarray(0, jnp.int32)  # ()
        return self.observation(state), state

    def step(self, key: jax.Array, state: jax.Array, action: jax.Array) -> MultiAgentStep[jax.Array, jax.Array]:
        # key/state: (); action: (N,)
        final = state + 1  # ()
        done = final == 3  # ()
        next_state = jnp.where(done, 0, final)  # ()
        return MultiAgentStep(
            observation=self.observation(next_state),
            state=next_state,
            reward=(action + 10 * action[::-1]).astype(jnp.float32),  # (N,), exposes both policies' commands
            done=done,
            agent_done=jnp.full(2, done),
            agent_inactive=jnp.zeros(2, bool),  # (); (N,); (N,)
            terminated=done,
            truncated=jnp.asarray(False),
            final_observation=self.observation(final),
            final_state=final,
        )

    def render(self, state: jax.Array) -> jax.Array:  # () -> (H, W, C)
        return jnp.zeros((1, 1, 3), jnp.uint8)


def counting_policy(
    variables: dict[str, Any],
    observation: jax.Array,  # (B, O)
    memory: ppo.RecurrentCarry,  # leaves: (B, H)
    starts: jax.Array,  # (B,)
    actions_mask: jax.Array,  # (B, A)
    *,
    method: str,
) -> ppo.ActorCriticOutput:
    carry = jax.tree.map(
        lambda leaf: jnp.where(starts[:, None], 0, leaf) + variables["params"]["increment"], memory
    )  # leaves: (B, H)
    logits = jnp.where(jnp.arange(7) == carry[0][0], 0.0, -jnp.inf)  # (B, A=7)
    return ppo.ActorCriticOutput(
        carry=carry,
        policy=ppo.CategoricalPolicy(logits=logits),
        value=jnp.zeros(starts.shape),  # (B,)
    )


def test_pool_protects_initial_and_active_snapshots_and_resizes() -> None:
    params = {"w": jnp.array([0.0, 1.0])}  # (F=2,)
    pool = init_pool(params, 3)
    pool = insert_snapshot(pool, {"w": params["w"] + 1}, 10, jnp.array([-1]))
    pool = insert_snapshot(pool, {"w": params["w"] + 2}, 20, jnp.array([1]))
    np.testing.assert_array_equal(pool.iteration, [0, 10, 20])
    chex.assert_trees_all_equal(pool, insert_snapshot(pool, params, 30, jnp.array([1, 2])))
    replaced = insert_snapshot(pool, {"w": params["w"] + 3}, 30, jnp.array([2]))
    np.testing.assert_array_equal(replaced.iteration, [0, 30, 20])
    np.testing.assert_array_equal(replaced.params["w"], [[0, 1], [3, 4], [2, 3]])
    np.testing.assert_array_equal(pool.params["w"][1], [1, 2])  # Previous pool remains immutable.
    smaller = resize_pool(replaced, 2)
    np.testing.assert_array_equal(smaller.iteration, [0, 30])
    larger = resize_pool(smaller, 4)
    assert larger.params["w"].shape == (4, 2)
    np.testing.assert_array_equal(larger.iteration, [0, 30, -1, -1])


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0], ids=["historical", "mixed", "current"])
def test_sampling_excludes_empty_slots_and_balances_sides(fraction: float) -> None:
    pool = init_pool({"w": jnp.zeros(1)}, 3)
    pool = insert_snapshot(pool, {"w": jnp.ones(1)}, 1, jnp.array([-1]))
    settings = SelfPlayConfig(current_policy_fraction=fraction)
    assignment = sample_opponents(jax.random.key(1), pool, settings, 128)
    chex.assert_shape(assignment, (128,))
    assert set(np.asarray(assignment.learner_team)) == {0, 1}
    expected = {0, 1} if fraction == 0 else {-1} if fraction == 1 else {-1, 0, 1}
    assert set(np.asarray(assignment.opponent)) == expected
    chex.assert_trees_all_equal(assignment, sample_opponents(jax.random.key(1), pool, settings, 128))


@pytest.mark.parametrize("settings", [{"pool_size": 1}, {"snapshot_interval": 0}, {"current_policy_fraction": 1.1}])
def test_invalid_pool_settings(settings: dict[str, int | float]) -> None:
    with pytest.raises(ValueError):
        SelfPlayConfig(**settings)


def test_opt_in_config_and_environment_validation(config: ppo.Config, tmp_path: Path) -> None:
    assert config.selfplay is None
    enabled = replace(config, selfplay=SelfPlayConfig())
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(asdict(enabled)))
    assert ppo.load_config(path, ppo.Config).selfplay == SelfPlayConfig()
    with pytest.raises(ValueError, match="supports_selfplay"):
        ppo.make_env(enabled)
    env = Duel()
    with patch.object(ppo, "SMAXEnv", return_value=env):
        assert ppo.make_env(enabled) is env
        with pytest.raises(ValueError, match="num_envs must be divisible"):
            ppo.make_env(replace(enabled, num_minibatches=4))
        env.num_agents = 4
        with pytest.raises(ValueError, match="two controlled agents"):
            ppo.make_env(enabled)


def test_rollout_preserves_opponent_memory_and_learner_only_gae(config: ppo.Config) -> None:
    config = replace(config, selfplay=SelfPlayConfig(current_policy_fraction=0), gamma=1.0)
    env = Duel()
    observation, env_state = jax.vmap(env.reset)(jax.random.split(jax.random.key(0), 2))
    state = TrainState.create(apply_fn=counting_policy, params={"increment": jnp.asarray(1.0)}, tx=optax.sgd(0.1))
    pool = init_pool({"increment": jnp.asarray(2.0)}, 2)
    memory = ((jnp.zeros((4, 1)), jnp.zeros((4, 1))),)  # leaves: (E*N, H)
    assignment = MatchAssignment(opponent=jnp.array([0, -1]), learner_team=jnp.array([0, 1]))  # (E,) each
    runner = ppo.RolloutCarry(
        env_state=env_state,
        observation=observation,
        model_carry=memory,
        episode_start=jnp.ones(4, bool),
        episode_return=jnp.zeros(2),
        episode_step_count=jnp.zeros(2, jnp.int32),
        selfplay=assignment,
    )
    collect = jax.jit(ppo.collect_rollout, static_argnames=("env", "config"))
    advanced, rollout = collect(runner, state, env, config, jax.random.key(7), pool)
    chex.assert_trees_all_equal(advanced.selfplay, assignment)  # No full game has ended.
    batch = rollout.batch
    assert batch.action.shape == (2, 2) and batch.observation.shape == (2, 2, 2)
    assert batch.model_carry[0][0].shape == (2, 1)
    np.testing.assert_array_equal(batch.action, [[1, 2], [1, 2]])
    np.testing.assert_array_equal(batch.old_log_prob, 0)
    np.testing.assert_array_equal(batch.return_target, [[63, 42], [33, 22]])
    np.testing.assert_array_equal(advanced.model_carry[0][0][:, 0], [2, 4, 2, 2])
    # Cross an iteration boundary, then a game boundary. Old opponents act through the final step.
    finished, next_rollout = collect(advanced, state, env, config, rollout.key, pool)
    np.testing.assert_array_equal(next_rollout.batch.action, [[3, 1], [3, 1]])
    np.testing.assert_array_equal(next_rollout.batch.episode_start, [[False, True], [False, True]])
    np.testing.assert_array_equal(next_rollout.batch.return_target, [[63, 21], [33, 21]])
    np.testing.assert_array_equal(next_rollout.batch.model_carry[0][0], [[2], [2]])
    assert int(next_rollout.episode_count) == 2
    assert float(next_rollout.episode_return_sum) == 192
    np.testing.assert_array_equal(finished.selfplay.opponent, 0)
    np.testing.assert_array_equal(finished.episode_step_count, 1)


def test_pool_checkpoint_restore_resize_and_disable(config: ppo.Config, tmp_path: Path) -> None:
    config = replace(config, selfplay=SelfPlayConfig(pool_size=3))
    state = TrainState.create(apply_fn=counting_policy, params={"increment": jnp.asarray(1.0)}, tx=optax.sgd(0.1))
    pool = init_pool(state.params, 3)
    pool = insert_snapshot(pool, {"increment": jnp.asarray(2.0)}, 10, jnp.array([-1]))
    pool = insert_snapshot(pool, {"increment": jnp.asarray(3.0)}, 20, jnp.array([-1]))
    progress = ppo.TrainingProgress(
        train_state=state,
        key=jax.random.key(7),
        rollout_count=20,
        episode_count=3,
        episode_return_sum=5.0,
        episode_length_sum=9,
        environment_step_count=9,
        observation_count=18,
        opponent_pool=pool,
    )
    with ppo.checkpoint_manager(str(tmp_path)) as manager:
        ppo.save_checkpoint(manager, progress, config)
        manager.wait_until_finished()
        restored = ppo.restore_checkpoint(manager, state, config)
        chex.assert_trees_all_equal(restored, progress)
        smaller = ppo.restore_checkpoint(manager, state, replace(config, selfplay=SelfPlayConfig(pool_size=2)))
        np.testing.assert_array_equal(smaller.opponent_pool.iteration, [0, 20])
        np.testing.assert_array_equal(smaller.opponent_pool.params["increment"], [1, 3])
        disabled = ppo.restore_checkpoint(manager, state, replace(config, selfplay=None))
        assert disabled.opponent_pool is None
        # Older checkpoints without pool data can enable the feature on resume.
        ppo.save_checkpoint(
            manager, progress._replace(rollout_count=21, opponent_pool=None), replace(config, selfplay=None)
        )
        manager.wait_until_finished()
        assert ppo.restore_checkpoint(manager, state, config).opponent_pool is None


def test_pool_training_matches_replicated_and_sharded_updates(config: ppo.Config) -> None:
    config = replace(config, selfplay=SelfPlayConfig(pool_size=2), num_minibatches=1)
    env = Duel()
    model = ppo.make_actor_critic(action_space=env.action_space, model=config.model)
    observation, env_state = jax.vmap(env.reset)(jax.random.split(jax.random.key(0), 2))
    memory = model.initial_carry(4)  # leaves: (B=4, H)
    params = model.init(
        jax.random.key(1),
        observation.observation,
        model.initial_carry(2),
        jnp.ones((2, 2), bool),
        observation.actions_mask,
    )["params"]
    optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=config.learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    pool = init_pool(params, 2)
    assignment = MatchAssignment(opponent=jnp.array([0, -1]), learner_team=jnp.array([0, 1]))  # leaves: (E,)
    runner = ppo.RolloutCarry(
        env_state=env_state,
        observation=observation,
        model_carry=memory,
        episode_start=jnp.ones(4, bool),
        episode_return=jnp.zeros(2),
        episode_step_count=jnp.zeros(2, jnp.int32),
        selfplay=assignment,
    )
    key = jax.random.key(2)  # ()
    metrics = ppo.init_metrics()
    reference = jax.jit(ppo.train_step, static_argnames=("env", "config"))(
        runner, state, env, config, key, metrics, jnp.asarray(0.0), pool
    )
    mesh = Mesh(devices=jax.devices()[:2], axis_names=("batch",), axis_types=(AxisType.Auto,))
    with jax.set_mesh(mesh):
        runner = jax.device_put(runner, PartitionSpec("batch"))
        state, pool, key, metrics = jax.device_put((state, pool, key, metrics), PartitionSpec())
        actual = ppo.train_step_with_pool_donated(runner, state, env, config, key, metrics, jnp.asarray(0.0), pool)
        chex.assert_trees_all_close(
            (actual.train_state, actual.rollout_carry, actual.metrics),
            (reference.train_state, reference.rollout_carry, reference.metrics),
            atol=2e-6,
            rtol=2e-6,
        )
        np.testing.assert_array_equal(jax.random.key_data(actual.key), jax.random.key_data(reference.key))
        assert int(actual.train_state.step) == 1
        assert actual.rollout_carry.selfplay.opponent.sharding.spec == PartitionSpec("batch")
        np.testing.assert_array_equal(pool.iteration, [0, -1])  # Donation never consumes snapshot storage.


def test_lux_pool_preserves_structured_observations_and_match_memory(config: ppo.Config) -> None:
    config = replace(
        config,
        env=ppo.LuxConfig(map_size=8, max_units=1, max_steps_in_match=2, match_count=2, randomize_params=False),
        model=ppo.LuxModelConfig(
            embedding_size=4,
            channels=4,
            recurrent=config.model.recurrent,
            ship_recurrent=ppo.LSTMConfig(hidden_size=4),
            attention_implementation="xla",
        ),
        num_envs=1,
        num_steps=4,
        num_minibatches=1,
        selfplay=SelfPlayConfig(pool_size=2, current_policy_fraction=0),
    )
    env = ppo.make_env(config)
    observation, env_state = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.key(0), 1))
    model = ppo.make_actor_critic(action_space=env.action_space, model=config.model)
    memory = model.initial_carry(2)  # leaves: (E*N=2, H)
    sequence = jax.tree.map(lambda leaf: jnp.swapaxes(leaf, 0, 1), observation)  # leaves: (B=2, T=1, ...)
    params = model.init(jax.random.key(1), sequence.observation, memory, jnp.ones((2, 1), bool), sequence.actions_mask)[
        "params"
    ]
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.sgd(0.1))
    pool = init_pool(params, 2)
    runner = ppo.RolloutCarry(
        env_state=env_state,
        observation=observation,
        model_carry=memory,
        episode_start=jnp.ones(2, bool),
        episode_return=jnp.zeros(1),
        episode_step_count=jnp.zeros(1, jnp.int32),
        selfplay=MatchAssignment(opponent=jnp.array([0]), learner_team=jnp.array([1])),
    )
    advanced, result = jax.jit(ppo.collect_rollout, static_argnames=("env", "config"))(
        runner, state, env, config, jax.random.key(2), pool
    )
    batch = result.batch
    assert batch.action.shape == (1, 4, 1)
    assert batch.actions_mask.shape == (1, 4, 1, 230)
    assert batch.observation.world.sensor_mask.shape == (1, 4, 8, 8)
    assert batch.observation.self.units.position.dtype == jnp.int16
    np.testing.assert_array_equal(batch.episode_start, [[True, False, False, False]])
    chex.assert_trees_all_equal(advanced.selfplay, runner.selfplay)
    replay = model.apply(
        {"params": params}, batch.observation, batch.model_carry, batch.episode_start, batch.actions_mask
    )
    np.testing.assert_allclose(replay.policy.log_prob(batch.action), batch.old_log_prob, atol=1e-5)


def test_training_restores_pool_inserts_snapshots_and_saves(config: ppo.Config) -> None:
    config = replace(
        config,
        selfplay=SelfPlayConfig(pool_size=3, snapshot_interval=1, current_policy_fraction=1),
        num_minibatches=1,
        total_env_steps=20,
        log_interval=1,
    )
    calls = []

    def restore(manager: Any, state: TrainState, settings: ppo.Config) -> ppo.TrainingProgress:
        pool = init_pool(jax.tree.map(lambda leaf: leaf - 1, state.params), 3)
        return ppo.TrainingProgress(
            train_state=state,
            key=jax.random.key(1),
            rollout_count=2,
            episode_count=0,
            episode_return_sum=0.0,
            episode_length_sum=0,
            environment_step_count=8,
            observation_count=16,
            opponent_pool=pool,
        )

    def step(
        runner: ppo.RolloutCarry,
        state: TrainState,
        env: ppo.PPOEnv,
        settings: ppo.Config,
        key: jax.Array,
        metrics: ppo.TrainingMetrics,
        environment_step_count: jax.Array,
        pool: ppo.OpponentPool,
    ) -> ppo.TrainStepResult:
        # runner leaves: (E, ...)/(E*N, H); key/step/metric leaves: (); pool leaves: (K, ...)
        calls.append(int(environment_step_count))
        np.testing.assert_array_equal(pool.iteration, ([0, -1, -1], [0, 3, -1], [0, 3, 4])[len(calls) - 1])
        np.testing.assert_array_equal(runner.selfplay.opponent, -1)
        if len(calls) == 1:
            chex.assert_trees_all_close(jax.tree.map(lambda leaf: leaf[0] + 1, pool.params), state.params, atol=1e-7)
        with jax.enable_x64():
            metrics = metrics._replace(window=jax.tree.map(lambda leaf: leaf + 1, metrics.window))
        return ppo.TrainStepResult(train_state=state, rollout_carry=runner, key=key, metrics=metrics)

    with (
        patch.object(ppo, "make_env", return_value=Duel()),
        patch.object(ppo, "checkpoint_manager"),
        patch.object(ppo, "restore_checkpoint", side_effect=restore),
        patch.object(ppo, "MLflowLogger") as logger,
        patch.object(ppo, "train_step_with_pool_donated", side_effect=step),
        patch.object(ppo, "train_step_donated") as ordinary,
        patch.object(ppo, "save_checkpoint", return_value=0.0) as save,
    ):
        ppo.train(config)
    assert calls == [8, 12, 16]
    history = metric_history(logger.return_value)
    assert history["train/pool_param_iteration_min"] == [(0.0, 12), (0.0, 16), (0.0, 20)]
    assert history["train/pool_param_iteration_max"] == [(3.0, 12), (4.0, 16), (5.0, 20)]
    ordinary.assert_not_called()
    progress = save.call_args.args[1]
    assert progress.environment_step_count == 20 and progress.observation_count == 40
    np.testing.assert_array_equal(progress.opponent_pool.iteration, [0, 5, 4])
