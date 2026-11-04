"""Run multi-device checks with JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=2."""

from dataclasses import replace
from functools import partial
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

from rl2 import ppo
from rl2.envs.gymnax_env import GymnaxEnv
from rl2.model.encoders.conv2d import ConvStage

pytestmark = pytest.mark.usefixtures("ppo_mesh")


@pytest.fixture
def config() -> ppo.Config:
    return replace(
        ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml", ppo.Config),
        num_envs=4,
        num_steps=2,
        num_minibatches=2,
        update_epochs=2,
        target_kl=1.0,
        bf16=False,
        model=ppo.ModelConfig(
            encoder=ppo.Conv2DEncoderConfig(stages=(ConvStage(4, blocks=0),), embedding_size=4),
            recurrent=ppo.LSTMConfig(hidden_size=4),
        ),
    )


@pytest.fixture
def mesh() -> Mesh:
    if jax.device_count() < 2:
        pytest.skip("Requires two devices; use JAX_NUM_CPU_DEVICES=2 on CPU")
    return Mesh(devices=jax.devices()[:2], axis_names=("batch",), axis_types=(AxisType.Auto,))


def test_training_mesh_uses_all_devices(config: ppo.Config) -> None:
    config = replace(config, num_envs=2 * jax.device_count())
    mesh = ppo.training_mesh(config)
    assert list(mesh.devices.flat) == jax.devices()
    assert mesh.axis_names == ("batch",)


@pytest.mark.parametrize("num_envs,num_minibatches", [(3, 1), (2, 2)], ids=["rollout", "minibatch"])
def test_training_mesh_rejects_indivisible_batches(config: ppo.Config, num_envs: int, num_minibatches: int) -> None:
    config = replace(config, num_envs=num_envs, num_minibatches=num_minibatches)
    with (
        patch.object(ppo.jax, "devices", return_value=[jax.devices()[0]] * 2),
        pytest.raises(ValueError, match=r"num_envs .* num_minibatches .* device_count \(2\)"),
    ):
        ppo.training_mesh(config)


def test_sequence_sharding_preserves_observation_pytrees_and_memory(mesh: Mesh) -> None:
    values = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)  # (B=4, T=2)
    memory = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)  # (B=4, H=3)
    batch = ppo.PPOBatch(
        observations={"vector": values[..., None], "image": values[..., None, None]},  # (B,T,1); (B,T,1,1)
        actions=jnp.zeros((4, 2), jnp.int32),  # (B,T)
        old_log_probs=values,  # (B,T)
        values=values,  # (B,T)
        advantages=values,  # (B,T)
        returns=values,  # (B,T)
        model_carry=((memory, memory + 1),),  # leaves: (B,H)
        episode_starts=jnp.zeros((4, 2), bool),  # (B,T)
    )
    with jax.set_mesh(mesh):
        sharded = jax.jit(
            partial(jax.lax.with_sharding_constraint, shardings=PartitionSpec("batch")),
            in_shardings=PartitionSpec(),
            out_shardings=PartitionSpec("batch"),
        )(jax.device_put(batch, PartitionSpec()))
    for actual, expected in zip(jax.tree.leaves(sharded), jax.tree.leaves(batch), strict=True):
        assert actual.sharding == NamedSharding(mesh, PartitionSpec("batch"))
        assert actual.shape == expected.shape  # memory: (B,H); sequence leaves: (B,T,...)
        assert actual.addressable_shards[0].data.shape == (2, *expected.shape[1:])
        np.testing.assert_array_equal(actual, expected)


def test_train_places_restored_state_and_fresh_environments_on_all_devices(config: ppo.Config, mesh: Mesh) -> None:
    config = replace(config, num_envs=2 * jax.device_count(), total_steps=8 * jax.device_count())
    devices = set(jax.devices())

    def restore(manager: ppo.ocp.CheckpointManager, state: TrainState, config: ppo.Config) -> ppo.TrainingProgress:
        assert all(leaf.sharding.device_set == devices for leaf in jax.tree.leaves(state))
        return ppo.TrainingProgress(
            train_state=state.replace(step=7),
            key=jax.random.key(42),  # ()
            rollout_count=1,
            episode_count=3,
            episode_return_sum=5.0,
            episode_length_sum=6,
        )

    def inspect_step(
        runner: ppo.RolloutCarry,  # leaves: (B,...)
        state: TrainState,
        env: ppo.PPOEnv,
        config: ppo.Config,
        key: jax.Array,  # ()
        metrics: ppo.TrainingMetrics,  # scalar leaves
        rollout_index: int,
    ) -> ppo.TrainStepResult:
        assert rollout_index == 1
        assert int(state.step) == 7
        assert int(metrics.total.episode_count) == 3
        np.testing.assert_array_equal(jax.random.key_data(key), jax.random.key_data(jax.random.key(42)))
        for leaf in jax.tree.leaves((state, key, metrics)):
            assert leaf.sharding.device_set == devices
            assert leaf.sharding.is_fully_replicated
        for leaf in jax.tree.leaves(runner):  # (B,...)
            assert leaf.shape[0] == config.num_envs
            assert leaf.sharding.device_set == devices
            assert leaf.sharding.spec == PartitionSpec("batch")
        np.testing.assert_array_equal(runner.episode_starts, True)
        np.testing.assert_array_equal(runner.episode_returns, 0)
        raise RuntimeError("inspected sharded inputs")

    with (
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

    def apply(
        variables: optax.Params,
        observations: jax.Array,  # (B=2,T=1,O=1)
        memory: ppo.RecurrentCarry,  # leaves: (B,H=1)
        starts: jax.Array,  # (B,T)
    ) -> ppo.ActorCriticOutput:
        return ppo.ActorCriticOutput(
            carry=memory,  # leaves: (B,H)
            policy=ppo.CategoricalPolicy(
                jnp.broadcast_to(variables["params"]["logits"], (*starts.shape, 2))
            ),  # logits: (B,T,A=2)
            values=jnp.zeros_like(starts, dtype=jnp.float32),  # (B,T)
        )

    state = TrainState.create(apply_fn=apply, params={"logits": jnp.zeros(2)}, tx=optax.adam(0.01))
    batch = ppo.PPOBatch(
        observations=jnp.zeros((2, 1, 1)),  # (B=2,T=1,O=1)
        actions=jnp.array([[0], [1]]),  # (B,T)
        old_log_probs=jnp.log(0.5) - jnp.array([[0.0], [1.0]]),  # (B,T); per-shard KL: 0 and exp(1)-2
        values=jnp.zeros((2, 1)),  # (B,T)
        advantages=jnp.array([[-1.0], [1.0]]),  # (B,T); per-shard centering would erase the gradient
        returns=jnp.zeros((2, 1)),  # (B,T)
        model_carry=((jnp.zeros((2, 1)), jnp.zeros((2, 1))),),  # leaves: (B,H)
        episode_starts=jnp.zeros((2, 1), bool),  # (B,T)
    )
    run_update = jax.jit(
        ppo.update,
        static_argnames="config",
        in_shardings=(PartitionSpec(), PartitionSpec("batch")),
        out_shardings=PartitionSpec(),
    )
    expected = run_update(state, batch, config)
    with jax.set_mesh(mesh):
        batch = jax.device_put(batch, PartitionSpec("batch"))  # all leaves: (B,...)
        actual = run_update(jax.device_put(state, PartitionSpec()), batch, config)
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
    replicated = NamedSharding(mesh, PartitionSpec())
    batched = NamedSharding(mesh, PartitionSpec("batch"))
    env = GymnaxEnv(config.env.id, max_steps_in_episode=2)
    observations, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(jax.random.key(0), config.num_envs), env.params
    )  # observations: (B=4,10,10,4); state leaves: (B,...)
    model = ppo.ActorCritic(action_space=env.action_space, model=config.model)
    memory = model.initial_carry(config.num_envs)  # leaves: (B,H=4)
    starts = jnp.ones(config.num_envs, bool)  # (B,)
    state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(jax.random.key(1), observations[:, None], memory, starts[:, None])["params"],
        tx=optax.inject_hyperparams(
            lambda learning_rate: optax.chain(optax.clip_by_global_norm(0.1), optax.adam(learning_rate, eps=1e-5))
        )(learning_rate=config.learning_rate),
    )
    state = jax.device_put(state, PartitionSpec())
    runner = ppo.RolloutCarry(
        env_state=env_state,  # leaves: (B,...)
        observations=observations,  # (B,10,10,4)
        model_carry=memory,  # leaves: (B,H)
        episode_starts=starts,  # (B,)
        episode_returns=jnp.arange(config.num_envs, dtype=jnp.float32),  # (B,)
        episode_step_counts=jnp.zeros(config.num_envs, jnp.int32),  # (B,)
    )
    key = jax.random.key(2)  # ()
    metrics = ppo.init_metrics()  # scalar leaves
    expected = jax.jit(
        ppo.train_step, static_argnames=("envs", "config"), in_shardings=PartitionSpec(), out_shardings=PartitionSpec()
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
    chex.assert_trees_all_equal((actual.rollout_carry, actual.key), (expected.rollout_carry, expected.key))
    assert int(actual.train_state.step) == config.num_minibatches * config.update_epochs
    assert int(actual.metrics.total.episode_count) == config.num_envs
    for leaf in jax.tree.leaves((actual.train_state, actual.key, actual.metrics)):
        assert leaf.sharding == replicated
        assert len(leaf.addressable_shards) == 2
    for leaf, reference in zip(jax.tree.leaves(actual.rollout_carry), jax.tree.leaves(runner), strict=True):
        assert leaf.shape == reference.shape  # (B,...)
        assert leaf.sharding == batched
        assert leaf.addressable_shards[0].data.shape == (config.num_envs // 2, *leaf.shape[1:])

    progress = ppo.TrainingProgress(
        train_state=actual.train_state,
        key=actual.key,  # ()
        rollout_count=2,
        episode_count=config.num_envs,
        episode_return_sum=float(actual.metrics.total.episode_return_sum),
        episode_length_sum=int(actual.metrics.total.episode_length_sum),
    )
    with ppo.checkpoint_manager(str(tmp_path)) as manager:
        ppo.save_checkpoint(manager, progress, config)
        restored = ppo.restore_checkpoint(manager, state, config)
    assert restored is not None
    chex.assert_trees_all_equal(restored, progress)
    assert all(leaf.sharding.num_devices == 1 for leaf in jax.tree.leaves(restored.train_state))
    # A replicated policy can still run single-environment evaluation.
    with jax.set_mesh(mesh):
        episode = ppo.play_episode(
            actual.train_state, config, jax.device_put(key, replicated), model.initial_carry(1), record=False, env=env
        )
    assert episode.length == 2
