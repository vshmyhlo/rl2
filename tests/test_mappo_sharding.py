"""Run multi-device checks with JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=2."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec
from test_train_mappo import ScriptedSMAX, tiny_policy
from test_train_mappo import config as config  # noqa: PLC0414

from rl2 import train_mappo as ppo

pytestmark = pytest.mark.usefixtures("ppo_mesh")


@pytest.fixture
def mesh() -> Mesh:
    if jax.device_count() < 2:
        pytest.skip("Requires two devices; use JAX_NUM_CPU_DEVICES=2 on CPU")
    return Mesh(devices=jax.devices()[:2], axis_names=("batch",), axis_types=(AxisType.Auto,))


def test_training_mesh_uses_all_devices_and_allows_agent_minibatches(config: ppo.Config) -> None:
    config = replace(config, num_envs=jax.device_count(), num_minibatches=3, total_iterations=1)
    mesh = ppo.training_mesh(config, num_agents=3)
    assert list(mesh.devices.flat) == jax.devices()
    assert mesh.axis_names == ("batch",)


@pytest.mark.parametrize("num_envs,num_minibatches", [(3, 1), (2, 4)], ids=["rollout", "minibatch"])
def test_training_mesh_rejects_indivisible_batches(config: ppo.Config, num_envs: int, num_minibatches: int) -> None:
    config = replace(config, num_envs=num_envs, num_minibatches=num_minibatches)
    with (
        patch.object(ppo.jax, "devices", return_value=[jax.devices()[0]] * 2),
        pytest.raises(ValueError, match=r"must be divisible by .*device_count \(2\)"),
    ):
        ppo.training_mesh(config, num_agents=2)


def test_train_places_restored_state_and_fresh_environments_on_all_devices(config: ppo.Config, mesh: Mesh) -> None:
    config = replace(config, num_envs=jax.device_count(), total_iterations=2)
    devices = set(jax.devices())

    def restore(manager: ppo.ocp.CheckpointManager, state: TrainState, config: ppo.Config) -> ppo.TrainingProgress:
        assert all(leaf.sharding.device_set == devices for leaf in jax.tree.leaves(state))
        return ppo.TrainingProgress(
            train_state=jax.device_put(state, jax.devices()[0]).replace(step=7),
            key=jax.random.key(42),  # ()
            rollout_count=1,
            episode_count=3,
            episode_return_sum=5.0,
            episode_length_sum=6,
        )

    def inspect_step(
        runner: ppo.RolloutCarry,  # state: (E, ...); memory: (B, H)
        state: TrainState,
        env: ppo.PPOEnv,
        config: ppo.Config,
        key: jax.Array,  # ()
        metrics: ppo.TrainingMetrics,  # scalar leaves
        iteration_count: jax.Array,  # ()
    ) -> ppo.TrainStepResult:
        assert int(iteration_count) == 1
        assert int(state.step) == 7
        assert int(metrics.total.episode_count) == 3
        np.testing.assert_array_equal(jax.random.key_data(key), jax.random.key_data(jax.random.key(42)))
        for leaf in jax.tree.leaves((state, key, metrics)):
            assert leaf.sharding.device_set == devices
            assert leaf.sharding.is_fully_replicated
        for leaf in jax.tree.leaves(runner):  # (E, ...)/(B, ...)
            assert leaf.sharding.device_set == devices
            assert leaf.sharding.spec == PartitionSpec("batch")
            assert leaf.addressable_shards[0].data.shape[0] == leaf.shape[0] // len(devices)
        chex.assert_shape(runner.observation.observation, (config.num_envs, 2, 2))
        chex.assert_shape(jax.tree.leaves(runner.model_carry), (config.num_envs * 2, 4))
        np.testing.assert_array_equal(runner.episode_start, True)
        np.testing.assert_array_equal(runner.episode_return, 0)
        raise RuntimeError("inspected sharded inputs")

    with (
        patch.object(ppo, "make_env", return_value=ScriptedSMAX()),
        patch.object(ppo, "checkpoint_manager"),
        patch.object(ppo, "restore_checkpoint", side_effect=restore),
        patch.object(ppo, "MLflowLogger"),
        patch.object(ppo, "train_step_donated", side_effect=inspect_step) as step,
        pytest.raises(RuntimeError, match="inspected sharded inputs"),
    ):
        ppo.train(config)
    step.assert_called_once()


@pytest.mark.parametrize("target_kl,updates", [(0.4, 1), (0.3, 0)], ids=["global-kl-accepts", "global-kl-rejects"])
def test_sharded_update_normalizes_advantages_and_checks_kl_globally(
    config: ppo.Config, mesh: Mesh, target_kl: float, updates: int
) -> None:
    config = replace(config, target_kl=target_kl)
    state = TrainState.create(
        apply_fn=tiny_policy,
        params={"logits": jnp.zeros(2), "value": jnp.asarray(0.0)},  # (A,); ()
        tx=optax.sgd(0.1),
    )
    batch = ppo.PPOBatch(
        observation=jnp.zeros((2, 1, 1)),  # (B=2, T=1, O=1)
        actions_mask=jnp.ones((2, 1, 2), bool),  # (B, T, A=2)
        action=jnp.array([[0], [1]]),  # (B, T)
        old_log_prob=jnp.log(0.5) - jnp.array([[0.0], [1.0]]),  # (B, T); per-shard KL: 0 and exp(1)-2
        value=jnp.zeros((2, 1)),  # (B, T)
        advantage=jnp.array([[-1.0], [1.0]]),  # (B, T); per-shard centering would erase the gradient
        return_target=jnp.zeros((2, 1)),  # (B, T)
        model_carry=((jnp.zeros((2, 1)), jnp.zeros((2, 1))),),  # leaves: (B, H=1)
        episode_start=jnp.zeros((2, 1), bool),  # (B, T)
    )
    run_update = jax.jit(
        ppo.update,
        static_argnames="config",
        in_shardings=(PartitionSpec(), PartitionSpec("batch")),
        out_shardings=PartitionSpec(),
    )
    expected = run_update(state, batch, config)
    with jax.set_mesh(mesh):
        actual = run_update(
            jax.device_put(state, PartitionSpec()), jax.device_put(batch, PartitionSpec("batch")), config
        )
    chex.assert_trees_all_close(actual, expected, atol=1e-6)
    assert int(actual[0].step) == updates
    assert all(leaf.sharding.is_fully_replicated for leaf in jax.tree.leaves(actual))
    if updates:
        assert not np.array_equal(actual[0].params["logits"], state.params["logits"])
    else:
        chex.assert_trees_all_equal(actual[0], state)


def test_sharded_training_matches_single_device_and_restores_checkpoint(
    config: ppo.Config, mesh: Mesh, tmp_path: Path
) -> None:
    config = replace(config, update_epochs=2)
    replicated = NamedSharding(mesh, PartitionSpec())
    batched = NamedSharding(mesh, PartitionSpec("batch"))
    env = ScriptedSMAX()
    observation, env_state = jax.vmap(env.reset)(jax.random.split(jax.random.key(0), 2))  # (E=2, N=2, O=2)/(E,N,A=3)
    env_state = env_state.replace(enemy_policy_state=jnp.array([False, True]))  # (E,); timeout and termination
    model = ppo.make_actor_critic(action_space=env.action_space, model=config.model)
    memory = model.initial_carry(4)  # leaves: (B=4, H=4)
    starts = jnp.ones(4, bool)  # (B,)
    state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(
            jax.random.key(1),
            observation.observation.reshape(4, 1, 2),  # (B, T=1, O)
            memory,
            starts[:, None],  # (B, T)
            observation.actions_mask.reshape(4, 1, 3),  # (B, T, A)
        )["params"],
        tx=optax.inject_hyperparams(
            lambda learning_rate: optax.chain(optax.clip_by_global_norm(0.1), optax.adam(learning_rate, eps=1e-5))
        )(learning_rate=config.learning_rate),
    )
    state = jax.device_put(state, PartitionSpec())
    runner = ppo.RolloutCarry(
        env_state=env_state,  # leaves: (E, ...)
        observation=observation,  # (E, N, O)/(E, N, A)
        model_carry=memory,  # leaves: (B, H)
        episode_start=starts,  # (B,)
        episode_return=jnp.arange(2, dtype=jnp.float32),  # (E,)
        episode_step_count=jnp.zeros(2, jnp.int32),  # (E,)
    )
    key = jax.random.key(2)  # ()
    metrics = ppo.init_metrics()  # scalar leaves
    expected = jax.jit(
        ppo.train_step, static_argnames=("env", "config"), in_shardings=PartitionSpec(), out_shardings=PartitionSpec()
    )(runner, state, env, config, key, metrics, 1)
    with jax.set_mesh(mesh):
        actual = ppo.train_step_donated(
            jax.device_put(runner, batched),
            jax.device_put(state, replicated),
            env,
            config,
            jax.device_put(key, replicated),
            jax.device_put(metrics, replicated),
            1,
        )
    chex.assert_trees_all_close(
        (actual.train_state, actual.metrics), (expected.train_state, expected.metrics), rtol=2e-5, atol=2e-6
    )
    chex.assert_trees_all_close(actual.rollout_carry, expected.rollout_carry, atol=2e-6)
    chex.assert_trees_all_equal(actual.key, expected.key)
    assert int(actual.train_state.step) == config.num_minibatches * config.update_epochs
    assert int(actual.metrics.total.episode_count) == config.num_envs
    for leaf in jax.tree.leaves((actual.train_state, actual.key, actual.metrics)):
        assert leaf.sharding == replicated
        assert len(leaf.addressable_shards) == 2
    for leaf, reference in zip(jax.tree.leaves(actual.rollout_carry), jax.tree.leaves(runner), strict=True):
        assert leaf.shape == reference.shape  # (E, ...)/(B, ...)
        assert leaf.sharding == batched
        assert leaf.addressable_shards[0].data.shape == (leaf.shape[0] // 2, *leaf.shape[1:])

    progress = ppo.TrainingProgress(
        train_state=actual.train_state,
        key=actual.key,  # ()
        rollout_count=2,
        episode_count=config.num_envs,
        episode_return_sum=float(actual.metrics.total.episode_return_sum),
        episode_length_sum=int(actual.metrics.total.episode_length_sum),
        environment_step_count=8,
    )
    with ppo.checkpoint_manager(str(tmp_path)) as manager:
        ppo.save_checkpoint(manager, progress, config)
        restored = ppo.restore_checkpoint(manager, state, replace(config, num_envs=1))
    assert restored is not None
    chex.assert_trees_all_equal(restored, progress)
    assert all(leaf.sharding.num_devices == 1 for leaf in jax.tree.leaves(restored.train_state))
    with jax.set_mesh(mesh):
        episode = ppo.play_episode(
            actual.train_state, config, jax.device_put(key, replicated), model.initial_carry(2), record=False, env=env
        )
    assert episode.length == 2


def test_hierarchical_sharded_update_preserves_sequences_and_matches_single_device(
    config: ppo.Config, mesh: Mesh
) -> None:
    model = ppo.make_actor_critic(
        action_space=ppo.HierarchicalMultiDiscreteSpace(nvec=(3, 3), group_sizes=(1, 2)), model=config.model
    )
    observation = jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 2)  # (B=2, T=2, O=2)
    memory = model.initial_carry(2)  # leaves: (B, H=4)
    starts = jnp.array([[True, False], [False, True]])  # (B, T)
    mask = jnp.ones((2, 2, 2, 3), bool).at[..., 0, 2].set(False)  # (B, T, D=2, A=3)
    params = model.init(jax.random.key(0), observation, memory, starts, mask)["params"]
    output, _ = model.apply({"params": params}, observation, memory, starts, mask)  # policy/value batch: (B, T)
    action = jnp.array([[[0, 1], [1, 2]], [[1, 0], [0, 2]]])  # (B, T, D)
    batch = ppo.PPOBatch(
        observation=observation,  # (B, T, O)
        actions_mask=mask,  # (B, T, D, A)
        action=action,  # (B, T, D)
        old_log_prob=output.policy.log_prob(action),  # (B, T)
        value=output.value,  # (B, T)
        advantage=jnp.array([[1.0, 2.0], [-1.0, -2.0]]),  # (B, T)
        return_target=jnp.ones((2, 2)),  # (B, T)
        model_carry=memory,  # leaves: (B, H)
        episode_start=starts,  # (B, T)
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.sgd(0.1))
    run_update = jax.jit(
        ppo.update,
        static_argnames="config",
        in_shardings=(PartitionSpec(), PartitionSpec("batch")),
        out_shardings=PartitionSpec(),
    )
    expected = run_update(state, batch, config)
    with jax.set_mesh(mesh):
        sharded = jax.device_put(batch, PartitionSpec("batch"))  # (B, T, ...); memory: (B, H)
        actual = run_update(jax.device_put(state, PartitionSpec()), sharded, config)
    for leaf, reference in zip(jax.tree.leaves(sharded), jax.tree.leaves(batch), strict=True):
        assert leaf.shape == reference.shape
        assert leaf.addressable_shards[0].data.shape == (1, *reference.shape[1:])
        np.testing.assert_array_equal(leaf, reference)
    chex.assert_trees_all_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert int(actual[0].step) == 1
    assert all(leaf.sharding.is_fully_replicated for leaf in jax.tree.leaves(actual))
