"""Run multi-device checks with JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=2."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pgx
import pytest
from flax.training.train_state import TrainState
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

from rl2.alphazero import train as az
from rl2.alphazero.config import ModelConfig


@pytest.fixture
def mesh() -> Mesh:
    if jax.device_count() < 2:
        pytest.skip("Requires two devices; use JAX_NUM_CPU_DEVICES=2 on CPU")
    return Mesh(devices=jax.devices()[:2], axis_names=("batch",), axis_types=(AxisType.Auto,))


def test_training_mesh_uses_all_devices() -> None:
    config = az.Config(num_envs=jax.device_count(), batch_size=jax.device_count())
    mesh = az.training_mesh(config)
    assert list(mesh.devices.flat) == jax.devices()
    assert mesh.axis_names == ("batch",)
    assert mesh.axis_types == (AxisType.Auto,)


@pytest.mark.parametrize("field", ["num_envs", "batch_size"])
def test_training_mesh_rejects_indivisible_sizes(field: str) -> None:
    config = replace(az.Config(num_envs=2, batch_size=2), **{field: 1})
    with (
        patch.object(az.jax, "devices", return_value=[jax.devices()[0]] * 2),
        pytest.raises(ValueError, match=rf"{field} \(1\) must be divisible by device_count \(2\)"),
    ):
        az.training_mesh(config)


def test_selfplay_shards_games_and_preserves_global_results(mesh: Mesh) -> None:
    env = pgx.make("tic_tac_toe")
    config = az.Config(num_envs=4, max_moves=2, num_simulations=2, batch_size=4)
    model = az.PolicyValueNet(num_actions=env.num_actions, channels=2, num_blocks=1)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 3, 3, 2)))["params"]  # Parameter shapes.
    key = jax.random.PRNGKey(1)  # (2,)
    single = Mesh(devices=jax.devices()[:1], axis_names=("batch",), axis_types=(AxisType.Auto,))
    with jax.set_mesh(single):
        expected = az.collect_selfplay(params, key, env=env, apply_fn=model.apply, config=config)
    with jax.set_mesh(mesh):
        actual = az.collect_selfplay(
            jax.device_put(params, PartitionSpec()),
            jax.device_put(key, PartitionSpec()),
            env=env,
            apply_fn=model.apply,
            config=config,
        )
    chex.assert_trees_all_close(actual, expected, atol=1e-6)
    batch, metrics = actual
    chex.assert_shape(batch.observation, (4, 2, 3, 3, 2))
    chex.assert_shape(batch.action_weights, (4, 2, 9))
    chex.assert_shape((batch.value_targets, batch.policy_mask, batch.value_mask), (4, 2))
    for leaf in jax.tree.leaves(batch):  # (B, T, ...)
        assert leaf.sharding == NamedSharding(mesh, PartitionSpec("batch"))
        assert leaf.addressable_shards[0].data.shape == (2, *leaf.shape[1:])
    for leaf in jax.tree.leaves(metrics):  # ()
        assert leaf.shape == ()
        assert leaf.sharding == NamedSharding(mesh, PartitionSpec())
    assert metrics.environment_steps == 8
    assert metrics.completed_games == 4


def test_sampling_uses_global_valid_positions_with_empty_shard(mesh: Mesh) -> None:
    labels = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)  # (B, T)
    batch = az.Batch(
        observation=labels[..., None, None, None],  # (B, T, 1, 1, 1)
        action_weights=labels[..., None],  # (B, T, 1)
        value_targets=labels,  # (B, T)
        policy_mask=jnp.array([[False, False, False], [True, False, True]]),  # (B, T)
        value_mask=jnp.array([[False, False, False], [True, False, False]]),  # (B, T)
    )
    key = jax.random.PRNGKey(2)  # (2,)
    single = Mesh(devices=jax.devices()[:1], axis_names=("batch",), axis_types=(AxisType.Auto,))
    with jax.set_mesh(single):
        expected = az.sample_batch(batch, key, batch_size=8)
    with jax.set_mesh(mesh):
        sampled = az.sample_batch(
            jax.device_put(batch, PartitionSpec("batch")), jax.device_put(key, PartitionSpec()), batch_size=8
        )
    chex.assert_trees_all_equal(sampled, expected)
    np.testing.assert_array_equal(np.unique(sampled.value_targets), [3, 5])
    np.testing.assert_array_equal(sampled.value_mask, sampled.value_targets == 3)
    assert sampled.policy_mask.all()
    for leaf in jax.tree.leaves(sampled):  # (B, ...)
        assert leaf.shape[0] == 8
        assert leaf.sharding == NamedSharding(mesh, PartitionSpec("batch"))
        assert leaf.addressable_shards[0].data.shape == (4, *leaf.shape[1:])


def test_update_uses_global_masked_means(mesh: Mesh) -> None:
    model = az.PolicyValueNet(num_actions=2, channels=2, num_blocks=1)
    observations = jax.random.normal(jax.random.PRNGKey(0), (4, 1, 1, 1))  # (B * T, *O)
    params = model.init(jax.random.PRNGKey(1), observations)["params"]  # Parameter shapes.
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.chain(optax.clip_by_global_norm(0.1), optax.adamw(1e-3)),
    )
    batch = az.Batch(
        observation=observations,  # (B * T, *O)
        action_weights=jnp.array([[1, 0], [1, 0], [0, 1], [0, 1]], jnp.float32),  # (B * T, A)
        value_targets=jnp.array([0.0, 0.0, -1.0, 1.0]),  # (B * T,)
        policy_mask=jnp.array([True, False, True, True]),  # (B * T,); unequal counts across shards.
        value_mask=jnp.array([False, False, True, True]),  # (B * T,); one shard has no value targets.
    )

    def as_rollout(x: jax.Array) -> jax.Array:  # (B * T, ...) -> (B, T, ...)
        return x.reshape((2, 2, *x.shape[1:]))  # (B, T, ...)

    batch = jax.tree.map(as_rollout, batch)  # Leaves: (B, T, ...).
    key = jax.random.PRNGKey(2)  # (2,)
    single = Mesh(devices=jax.devices()[:1], axis_names=("batch",), axis_types=(AxisType.Auto,))

    @jax.jit
    def update_once(
        state: TrainState,
        minibatch: az.Batch,  # Leaves: (M, ...).
    ) -> tuple[TrainState, az.UpdateMetrics]:
        (_, metrics), grads = jax.value_and_grad(az.loss_fn, has_aux=True)(state.params, state.apply_fn, minibatch)
        return state.apply_gradients(grads=grads), metrics  # Metric leaves: ().

    updates_per_iteration = 3
    expected_state = state
    expected_key = key  # (2,)
    metrics = []
    with jax.set_mesh(single):
        for _ in range(updates_per_iteration):
            expected_key, sample_key = jax.random.split(expected_key)  # Each: (2,)
            minibatch = az.sample_batch(batch, sample_key, batch_size=8)  # Leaves: (M, ...).
            expected_state, update_metrics = update_once(expected_state, minibatch)
            metrics.append(update_metrics)
        expected = az.TrainStepResult(
            train_state=expected_state,
            key=expected_key,  # (2,)
            update_metrics=jax.tree.map(lambda *values: sum(values), *metrics),  # Scalar leaves: ().
        )
    with jax.set_mesh(mesh):
        template_state = jax.device_put(state, PartitionSpec())
        actual = az.train_step(
            template_state,
            jax.device_put(key, PartitionSpec()),
            jax.device_put(batch, PartitionSpec("batch")),
            batch_size=8,
            updates_per_iteration=updates_per_iteration,
        )
    chex.assert_trees_all_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert isinstance(actual, az.TrainStepResult)
    actual_state, actual_key, actual_metrics = actual  # Key: (2,); metric leaves: ().
    assert actual_state.step == updates_per_iteration
    assert actual_metrics.value_update_count == updates_per_iteration
    chex.assert_shape(actual_metrics, ())
    chex.assert_shape(actual_key, (2,))
    np.testing.assert_array_equal(actual_key, expected_key)
    for leaf in jax.tree.leaves(actual):  # Parameter/optimizer shapes and scalar metrics.
        assert leaf.sharding == NamedSharding(mesh, PartitionSpec())
        assert len(leaf.addressable_shards) == 2


def test_checkpoint_restores_replication_with_different_device_counts(mesh: Mesh, tmp_path: Path) -> None:
    def apply(params: az.Parameters) -> jax.Array:  # weight: (2,)
        return params["weight"]  # (2,)

    state = TrainState.create(apply_fn=apply, params={"weight": jnp.ones(2)}, tx=optax.adam(1e-3))
    single = Mesh(devices=jax.devices()[:1], axis_names=("batch",), axis_types=(AxisType.Auto,))
    with jax.set_mesh(mesh):
        template_state = jax.device_put(state, PartitionSpec())
        key = jax.device_put(jax.random.PRNGKey(3), PartitionSpec())  # (2,)
    checkpoint = az.Checkpoint(train_state=template_state, key=key, iteration=1, steps=3, completed_games=1)
    config = az.Config(model=ModelConfig(channels=2, num_blocks=1))
    with az.checkpoint_manager(str(tmp_path)) as manager:
        az.save_checkpoint(manager, checkpoint, config)
        with jax.set_mesh(single):
            template = checkpoint._replace(
                train_state=jax.device_put(state, PartitionSpec()), key=jax.device_put(key, PartitionSpec())
            )
        restored = az.restore_checkpoint(manager, template)
        assert restored is not None
        chex.assert_trees_all_equal(restored, checkpoint)
        for leaf in jax.tree.leaves((restored.train_state, restored.key)):
            assert leaf.sharding == NamedSharding(single, PartitionSpec())
        az.save_checkpoint(manager, restored._replace(iteration=2), config)
        restored = az.restore_checkpoint(manager, checkpoint._replace(train_state=template_state))
        assert restored is not None
        chex.assert_trees_all_equal(restored, checkpoint._replace(iteration=2))
        for leaf in jax.tree.leaves((restored.train_state, restored.key)):
            assert leaf.sharding == NamedSharding(mesh, PartitionSpec())


def test_train_initializes_and_restores_replicated_state(mesh: Mesh) -> None:
    config = az.Config(
        env_id="tic_tac_toe",
        num_envs=jax.device_count(),
        batch_size=jax.device_count(),
        iterations=1,
        model=ModelConfig(channels=2, num_blocks=1),
    )
    replicated = NamedSharding(az.training_mesh(config), PartitionSpec())

    def restore(manager: ocp.CheckpointManager, checkpoint: az.Checkpoint) -> az.Checkpoint:
        for leaf in jax.tree.leaves((checkpoint.train_state, checkpoint.key)):
            assert leaf.sharding == replicated
        return checkpoint._replace(iteration=1, train_state=checkpoint.train_state.replace(step=7))

    with (
        patch.object(az, "checkpoint_manager"),
        patch.object(az, "MLflowLogger"),
        patch.object(az, "restore_checkpoint", side_effect=restore) as restore_mock,
    ):
        state = az.train(config)
    restore_mock.assert_called_once()
    assert state.step == 7
    for leaf in jax.tree.leaves(state):
        assert leaf.sharding == replicated
