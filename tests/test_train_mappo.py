from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml
from flax.training.train_state import TrainState
from jaxmarl.environments.smax.heuristic_enemy_smax_env import State
from jaxmarl.environments.smax.smax_env import State as BattleState
from ppo_helpers import metric_history

from rl2 import train_mappo as ppo
from rl2.envs.multi_agent import AgentObservation, MultiAgentStep
from rl2.envs.smax_env import SMAXConfig, SMAXEnv
from rl2.playback import PlaybackState, advance_episode


@pytest.fixture(scope="module")
def config() -> ppo.Config:
    return ppo.Config(
        run_id="mappo_test",
        env=SMAXConfig(),
        seed=1,
        num_envs=2,
        num_steps=2,
        num_minibatches=2,
        update_epochs=1,
        total_steps=8,
        gamma=0.5,
        gae_lambda=1.0,
        learning_rate=0.001,
        anneal_lr=True,
        clip_coef=0.2,
        target_kl=None,
        entropy_coef=0.01,
        value_coef=1.0,
        max_grad_norm=0.5,
        video_fps=5.0,
        bf16=False,
        video_interval=0,
        eval_interval=0,
        model=ppo.ModelConfig(encoder=ppo.MLPEncoderConfig(embedding_size=4), recurrent=ppo.LSTMConfig(hidden_size=4)),
    )


class ScriptedSMAX(SMAXEnv):
    """Two allies: one dies at step one; step two ends or times out the battle."""

    def __init__(self) -> None:
        self.num_agents = 2
        self.action_space = ppo.DiscreteSpace(3)

    def observation(self, state: State) -> AgentObservation:  # state leaves: (...)
        step = state.state.step  # ()
        return AgentObservation(
            jnp.stack([jnp.full(2, 10.0 * step), jnp.arange(2, dtype=jnp.float32)], axis=-1),  # (N=2, O=2)
            jax.nn.one_hot(jnp.array([0, 1]), 3, dtype=bool),  # (N, A=3)
        )

    def reset(self, key: jax.Array) -> tuple[AgentObservation, State]:  # key: ()
        battle = BattleState(
            done=jnp.asarray(False),
            step=jnp.asarray(0),
            unit_positions=jnp.zeros((3, 2)),
            unit_alive=jnp.ones(3, bool),  # (units, 2); (units,)
            unit_teams=jnp.array([0, 0, 1]),
            unit_health=jnp.ones(3),  # each (units,)
            unit_types=jnp.zeros(3, jnp.int32),
            unit_weapon_cooldowns=jnp.zeros(3),  # each (units,)
            prev_movement_actions=jnp.zeros((3, 2)),
            prev_attack_actions=jnp.zeros(3, jnp.int32),
        )
        state = State(state=battle, enemy_policy_state=jnp.asarray(False))
        return self.observation(state), state

    def step(self, key: jax.Array, state: State, action: jax.Array) -> MultiAgentStep[State]:  # key: (); action: (N,)
        step = state.state.step + 1  # ()
        done = step == 2  # ()
        terminated = done & state.enemy_policy_state  # ()
        final = state.replace(
            state=state.state.replace(
                step=step,
                done=done,
                unit_alive=jnp.array([False, True, ~terminated]),  # (units,)
            )
        )
        reset = self.reset(key)[1].replace(enemy_policy_state=state.enemy_policy_state)
        advanced = jax.tree.map(lambda a, b: jnp.where(done, a, b), reset, final)
        return MultiAgentStep(
            self.observation(advanced),
            advanced,
            jnp.full(2, step, jnp.float32),  # reward: (N,)
            done,
            jnp.array([True, done]),
            terminated,
            done & ~terminated,
            self.observation(final),
            final,
        )


def scripted_policy(
    variables: dict[str, Any],
    observation: jax.Array,  # (B, O)
    memory: ppo.RecurrentCarry,  # leaves: (B, H)
    starts: jax.Array,  # (B,)
    actions_mask: jax.Array,  # (B, A)
    *,
    method: str,
) -> ppo.ActorCriticOutput:
    def advance(leaf: jax.Array) -> jax.Array:  # (B, H) -> (B, H)
        return jnp.where(starts[:, None], 0, leaf) + 1  # (B, H)

    return ppo.ActorCriticOutput(
        jax.tree.map(advance, memory),
        ppo.CategoricalPolicy(jnp.where(actions_mask, 0.0, -1e10)),
        observation[:, 0],
    )


def test_rollout_team_gae_death_timeout_and_sequence_selection(config: ppo.Config) -> None:
    env = ScriptedSMAX()
    observation, state = jax.vmap(env.reset)(jax.random.split(jax.random.key(0), 2))
    state = state.replace(enemy_policy_state=jnp.array([False, True]))
    model = ppo.ActorCritic(env.action_space, config.model)
    memory = model.initial_carry(4)  # leaves: (B=4, H=4)
    runner = ppo.RolloutCarry(state, observation, memory, jnp.ones(4, bool), jnp.zeros(2), jnp.zeros(2, jnp.int32))
    train_state = TrainState.create(apply_fn=scripted_policy, params={}, tx=optax.sgd(0.1))
    advanced, result = ppo.collect_rollout(runner, train_state, env, config, jax.random.key(1))
    chex.assert_shape(result.batch.observation, (2, 4, 2))
    chex.assert_shape(result.batch.actions_mask, (2, 4, 3))
    chex.assert_shape((result.batch.action, result.batch.value, result.batch.return_target), (2, 4))
    np.testing.assert_array_equal(result.batch.observation[0, :, 1], [0, 1, 0, 1])
    np.testing.assert_array_equal(result.batch.action, [[0, 1, 0, 1]] * 2)
    # Team reward survives death. Timeout adds gamma * V(final)=10, never V(reset)=0.
    np.testing.assert_allclose(result.batch.return_target, [[7, 7, 2, 2], [12, 12, 2, 2]])
    np.testing.assert_array_equal(result.batch.episode_start, [[True] * 4, [True, False, True, False]])
    for leaf in jax.tree.leaves(advanced.model_carry):  # each (B, H)
        np.testing.assert_array_equal(leaf[:, 0], [1, 2, 1, 2])
    assert int(result.episode_count) == 2
    assert float(result.episode_return_sum) == 6.0
    assert int(result.episode_length_sum) == 4
    np.testing.assert_array_equal(advanced.episode_return, 0)
    np.testing.assert_array_equal(advanced.episode_start, True)
    selected = result.batch.select_sequences(jnp.array([3, 0]))
    np.testing.assert_array_equal(selected.return_target, [[2, 7], [2, 12]])
    chex.assert_shape(jax.tree.leaves(selected.model_carry), (2, 4))
    repeated, duplicate = ppo.collect_rollout(advanced, train_state, env, config, result.key)
    chex.assert_trees_all_equal(duplicate.batch.model_carry, advanced.model_carry)
    chex.assert_trees_all_equal(result.batch, duplicate.batch.replace(model_carry=memory))
    assert not np.array_equal(jax.random.key_data(result.key), jax.random.key_data(duplicate.key))
    np.testing.assert_array_equal(repeated.episode_step_count, 0)


def test_local_model_mask_replay_and_shared_memory(config: ppo.Config) -> None:
    model = ppo.ActorCritic(ppo.DiscreteSpace(3), config.model)
    observation = jnp.arange(16, dtype=jnp.float32).reshape(2, 2, 4) / 16  # (T, B, O)
    starts = jnp.array([[True, True], [True, False]])  # (T, B)
    actions_mask = jnp.array([[[True, False, True], [False, True, False]]] * 2)  # (T, B, A)
    memory = model.initial_carry(2)  # leaves: (B, H)
    assert len(memory) == config.model.recurrent.num_layers
    chex.assert_shape(jax.tree.leaves(memory), (2, 4))
    params = model.init(jax.random.key(0), observation, memory, starts, actions_mask)
    chex.assert_shape(params["params"]["encoder"]["kernel"], (4, config.model.encoder.embedding_size))
    # Keep the tiny ReLU encoder active so the test exercises recurrent features.
    params["params"]["encoder"]["kernel"] = jnp.eye(4)  # (O, F)
    output = model.apply(params, observation, memory, starts, actions_mask)
    first = model.apply(params, observation[0], memory, starts[0], actions_mask[0], method="step")
    chex.assert_shape(first.policy.logits, (2, 3))
    chex.assert_shape(first.value, (2,))
    second = model.apply(params, observation[1], first.carry, starts[1], actions_mask[1], method="step")
    chex.assert_trees_all_close(second.carry, output.carry, atol=1e-6)
    np.testing.assert_allclose(output.value, jnp.stack([first.value, second.value]), atol=1e-6)
    np.testing.assert_allclose(output.policy.logits, jnp.stack([first.policy.logits, second.policy.logits]), atol=1e-6)
    action, log_prob = output.policy.sample_and_log_prob(seed=jax.random.key(2))  # each (T, B)
    assert bool(jnp.take_along_axis(actions_mask, action[..., None], axis=-1).all())
    assert bool(jnp.isfinite(output.policy.entropy()).all())
    np.testing.assert_allclose(output.policy.log_prob(action), log_prob, atol=1e-6)
    chex.assert_shape(output.value, (2, 2))
    chex.assert_shape(output.policy.logits, (2, 2, 3))
    # Both heads depend on the shared encoder; changing only the value head leaves policy and memory intact.
    changed = dict(params["params"], encoder=jax.tree.map(jnp.zeros_like, params["params"]["encoder"]))
    no_features = model.apply({"params": changed}, observation, memory, starts, actions_mask)
    assert not np.array_equal(output.policy.logits, no_features.policy.logits)
    assert not np.array_equal(output.value, no_features.value)
    changed = dict(params["params"], value_output=jax.tree.map(jnp.zeros_like, params["params"]["value_output"]))
    changed["value_output"]["bias"] = jnp.ones(1)  # (1,)
    new_output = model.apply({"params": changed}, observation, memory, starts, actions_mask)
    chex.assert_trees_all_equal((output.policy, output.carry), (new_output.policy, new_output.carry))
    assert not np.array_equal(output.value, new_output.value)


def tiny_policy(
    variables: dict[str, dict[str, jax.Array]],  # params: logits (A,), value ()
    observation: jax.Array,  # (T, B, O)
    memory: ppo.RecurrentCarry,  # leaves: (B, H)
    starts: jax.Array,  # (T, B)
    actions_mask: jax.Array,  # (T, B, A)
) -> ppo.ActorCriticOutput:
    logits = jnp.where(actions_mask, variables["params"]["logits"], -1e10)  # (T, B, A)
    return ppo.ActorCriticOutput(
        carry=memory,
        policy=ppo.CategoricalPolicy(logits),
        value=jnp.broadcast_to(variables["params"]["value"], starts.shape),  # (T, B)
    )


@pytest.fixture
def tiny_batch() -> ppo.PPOBatch:
    return ppo.PPOBatch(
        observation=jnp.zeros((2, 2, 1)),  # (T, B, O)
        actions_mask=jnp.ones((2, 2, 2), dtype=bool),  # (T, B, A)
        action=jnp.array([[0, 1], [0, 1]]),  # (T, B)
        old_log_prob=jnp.full((2, 2), -jnp.log(2.0)),  # (T, B)
        value=jnp.array([[0.0, 1.0], [0.0, 1.0]]),  # (T, B)
        advantage=jnp.array([[1.0, -1.0], [1.0, -1.0]]),  # (T, B)
        return_target=jnp.array([[0.0, 1.0], [0.0, 1.0]]),  # (T, B)
        model_carry=((jnp.zeros((2, 1)), jnp.zeros((2, 1))),),  # leaves: (B, H)
        episode_start=jnp.ones((2, 2), dtype=bool),  # (T, B)
    )


def test_update_losses_single_sample_and_kl_rejection(config: ppo.Config, tiny_batch: ppo.PPOBatch) -> None:
    state = TrainState.create(
        apply_fn=tiny_policy,
        params={"logits": jnp.zeros(2), "value": jnp.asarray(0.0)},  # (A,); ()
        tx=optax.adam(config.learning_rate),
    )
    updated, metrics = ppo.update(state, tiny_batch, config)
    assert int(updated.step) == 1
    chex.assert_shape(jax.tree.leaves(metrics), ())
    assert float(metrics.policy_loss) == pytest.approx(0.0)
    assert float(metrics.value_loss) == pytest.approx(0.25)
    assert float(metrics.entropy) == pytest.approx(np.log(2.0))
    assert float(metrics.total_loss) == pytest.approx(0.25 - config.entropy_coef * np.log(2.0))
    assert float(metrics.approx_kl) == pytest.approx(0.0)
    assert float(metrics.clip_fraction) == pytest.approx(0.0)
    assert float(updated.params["logits"][0]) > 0

    # One transition must retain its nonzero advantage rather than normalize to zero.
    singleton = jax.tree.map(lambda leaf: leaf[:1], tiny_batch.select_sequences(jnp.array([0])))
    # Sequence leaves: (T=1, B=1, ...); memory leaves remain (B=1, H=1).
    single_updated, single_metrics = ppo.update(state, singleton, config)
    assert float(single_metrics.policy_loss) == pytest.approx(-1.0)
    assert float(single_updated.params["logits"][0]) > 0

    rejected, rejected_metrics = jax.jit(ppo.update, static_argnames="config")(
        state,
        tiny_batch.replace(old_log_prob=tiny_batch.old_log_prob + 1.0),
        replace(config, target_kl=0.1),
    )
    chex.assert_trees_all_equal(rejected, state)
    assert float(rejected_metrics.approx_kl) > 0.1
    assert float(rejected_metrics.clip_fraction) == 1.0


def test_train_step_omits_undefined_variance_and_stops_updates(
    config: ppo.Config, tiny_batch: ppo.PPOBatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(config, num_minibatches=1, update_epochs=3, target_kl=0.1)
    state = TrainState.create(
        apply_fn=tiny_policy,
        params={"logits": jnp.zeros(2), "value": jnp.asarray(0.0)},  # (A,); ()
        tx=optax.inject_hyperparams(optax.adam)(learning_rate=config.learning_rate),
    )
    rollout = ppo.RolloutResult(
        batch=tiny_batch.replace(old_log_prob=tiny_batch.old_log_prob + 1.0),
        key=jax.random.key(0),  # ()
        episode_return_sum=jnp.asarray(0.0),  # ()
        episode_length_sum=jnp.asarray(0),  # ()
        episode_count=jnp.asarray(0),  # ()
    )
    runner = Mock(spec=ppo.RolloutCarry)
    collect = Mock(
        return_value=(runner, rollout._replace(batch=rollout.batch.replace(return_target=jnp.zeros((2, 2)))))
    )
    monkeypatch.setattr(ppo, "collect_rollout", collect)
    result = ppo.train_step(runner, state, ScriptedSMAX(), config, rollout.key, ppo.init_metrics(), 4)
    assert int(result.train_state.step) == 0
    assert int(result.metrics.window.minibatch_count) == 1
    assert int(result.metrics.window.update_count) == 0
    assert int(result.metrics.window.early_stop_count) == 1
    assert int(result.metrics.window.explained_variance_count) == 0
    assert float(result.metrics.window.explained_variance_sum) == 0.0
    assert float(result.train_state.opt_state.hyperparams["learning_rate"]) == pytest.approx(config.learning_rate / 2)
    # A later valid variance must survive the earlier undefined value.
    collect.return_value = runner, rollout
    result = ppo.train_step(runner, state, ScriptedSMAX(), config, rollout.key, result.metrics)
    assert int(result.metrics.window.explained_variance_count) == 1
    assert float(result.metrics.window.explained_variance_sum) == pytest.approx(1.0)


def test_schedules_use_environment_steps(config: ppo.Config) -> None:
    config = replace(config, total_steps=16, entropy_decay="cosine")
    lr = ppo.learning_rate_schedule(config)
    entropy = ppo.entropy_coef_schedule(config)
    assert float(lr(4)) == pytest.approx(config.learning_rate * 0.75)
    assert float(entropy(4)) == pytest.approx(config.entropy_coef * (1 + np.cos(np.pi / 4)) / 2)
    assert float(lr(16)) == pytest.approx(0.0)
    assert float(entropy(16)) == pytest.approx(0.0)
    changed = replace(config, num_envs=1, num_steps=1, lr_decay="cosine")
    assert float(ppo.learning_rate_schedule(changed)(4)) == pytest.approx(
        config.learning_rate * (1 + np.cos(np.pi / 4)) / 2
    )
    constant = replace(config, anneal_lr=False, entropy_decay="constant")
    assert float(ppo.learning_rate_schedule(constant)(16)) == config.learning_rate
    assert float(ppo.entropy_coef_schedule(constant)(16)) == config.entropy_coef
    large = replace(config, total_steps=2**33)
    step = jnp.asarray(2**32, dtype=jnp.float32)  # (); counters can exceed int32
    assert float(jax.jit(ppo.learning_rate_schedule(large))(step)) == pytest.approx(config.learning_rate / 2)


def test_rejects_checkpoint_with_separate_networks(config: ppo.Config) -> None:
    manager = Mock()
    manager.latest_step.return_value = 1
    manager.restore.return_value.metadata = {"version": 1, "algorithm": "smax_ippo"}
    state = TrainState.create(apply_fn=scripted_policy, params={}, tx=optax.sgd(0.1))
    with pytest.raises(ValueError, match="shared encoder/LSTM requires a new run directory"):
        ppo.restore_checkpoint(manager, state, config)


@pytest.mark.parametrize("legacy", [False, True], ids=["transition-counters", "legacy"])
def test_checkpoint_ignores_saved_config_and_preserves_progress(config: ppo.Config, legacy: bool) -> None:
    state = TrainState.create(apply_fn=scripted_policy, params={}, tx=optax.sgd(0.1))
    progress = ppo.TrainingProgress(
        train_state=state,
        key=jax.random.key(0),  # ()
        rollout_count=1,
        episode_count=2,
        episode_return_sum=3.0,
        episode_length_sum=4,
        environment_step_count=None if legacy else 2**33,
        observation_count=None if legacy else 3 * 2**33,
    )
    manager = Mock()
    ppo.save_checkpoint(manager, progress, config)
    saved = manager.save.call_args.kwargs["args"]
    metadata = saved.metadata.item
    assert metadata["config"] == asdict(config)
    metadata.update(config={"retired_setting": True}, algorithm="retired_environment")
    if legacy:
        del metadata["environment_steps"], metadata["observations"]
    else:
        assert metadata["environment_steps"] == 2**33
        assert metadata["observations"] == 3 * 2**33
    manager.latest_step.return_value = 1
    manager.restore.return_value.metadata = metadata
    manager.restore.return_value.state = saved.state.item
    changed = replace(config, env=ppo.JaxNavConfig(), num_envs=1, num_steps=1)
    chex.assert_trees_all_equal(ppo.restore_checkpoint(manager, state, changed), progress)


def test_real_smax_single_step_update_and_checkpoint(config: ppo.Config, tmp_path: Path) -> None:
    config = replace(config, num_envs=1, num_steps=1, num_minibatches=3, env=SMAXConfig(max_steps=1))
    env = ppo.make_env(config)  # Three agent sequences can form three minibatches in one environment.
    observation, state = jax.vmap(env.reset)(jax.random.split(jax.random.key(0), 1))
    model = ppo.ActorCritic(env.action_space, config.model)
    memory = model.initial_carry(3)  # leaves: (B=3, H)
    starts = jnp.ones(3, bool)  # (B,)
    params = model.init(jax.random.key(1), observation.observation, memory, starts[None], observation.actions_mask)
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params["params"],
        tx=optax.inject_hyperparams(optax.adam)(learning_rate=config.learning_rate),
    )
    runner = ppo.RolloutCarry(state, observation, memory, starts, jnp.zeros(1), jnp.zeros(1, jnp.int32))
    result = jax.jit(ppo.train_step, static_argnames=("env", "config"))(
        runner, train_state, env, config, jax.random.key(2), ppo.init_metrics()
    )
    assert int(result.train_state.step) == 3
    assert int(result.metrics.total.episode_count) == 1
    assert int(result.metrics.total.episode_length_sum) == 1
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(result.train_state.params))
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(
            jax.tree.leaves(train_state.params),
            jax.tree.leaves(result.train_state.params),
        )
    )
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(result.metrics.window.minibatch_metrics))
    # Evaluation uses all allies and records the actual terminal frame.
    episode_return, length, frames = ppo.play_episode(
        result.train_state,
        config,
        jax.random.key(3),
        memory,
        record=True,
        env=env,
    )
    assert np.isfinite(episode_return)
    assert length == 1 and len(frames) == 2
    assert frames[-1].shape == (400, 400, 3)
    progress = ppo.TrainingProgress(
        train_state=result.train_state,
        key=result.key,  # ()
        rollout_count=1,
        episode_count=1,
        episode_return_sum=episode_return,
        episode_length_sum=1,
        environment_step_count=1,
        observation_count=3,
    )
    with ppo.checkpoint_manager(str(tmp_path)) as manager:
        ppo.save_checkpoint(manager, progress, config)
        metadata = manager.restore(1, args=ppo.ocp.args.Composite(metadata=ppo.ocp.args.JsonRestore())).metadata
        assert metadata["config"] == asdict(config)
        assert metadata["environment_steps"] == 1 and metadata["observations"] == 3
        changed = replace(config, env=SMAXConfig(map_name="8m"), num_envs=2, num_steps=2)
        restored = ppo.restore_checkpoint(manager, train_state, changed)
        assert restored is not None
        chex.assert_trees_all_equal(restored, progress)


def test_playback_reuses_compiled_sampling_with_updated_inputs() -> None:
    memory = ((jnp.zeros((2, 1)), jnp.zeros((2, 1))),)  # leaves: (B=2, H=1)
    observation = jnp.zeros((2, 1))  # (B, O=1)
    starts = jnp.array([True, False])  # (B,)
    actions_mask = jnp.array([[True, False, True], [False, True, False]])  # (B, A=3)
    params = {"logits": jnp.array([[100.0, 0.0, -100.0], [0.0, 0.0, 0.0]])}  # (B, A)

    def policy(
        variables: dict[str, dict[str, jax.Array]],
        observation: jax.Array,  # (B, O)
        carry: ppo.RecurrentCarry,  # leaves: (B, H)
        episode_start: jax.Array,  # (B,)
        actions_mask: jax.Array,  # (B, A)
        *,
        method: str,
    ) -> ppo.ActorCriticOutput:
        carry = jax.tree.map(lambda leaf: jnp.where(episode_start[:, None], 0, leaf) + 1, carry)  # (B, H)
        logits = jnp.where(actions_mask, variables["params"]["logits"], -1e10)  # (B, A)
        return ppo.ActorCriticOutput(carry, ppo.CategoricalPolicy(logits), jnp.zeros(2))  # (B, H); (B, A); (B,)

    with patch.object(
        ppo.CategoricalPolicy, "sample", autospec=True, side_effect=ppo.CategoricalPolicy.sample
    ) as sample:
        memory, action = ppo.sample_policy_step(
            policy, params, observation, memory, starts, jax.random.key(0), actions_mask
        )
        chex.assert_shape(action, (2,))
        np.testing.assert_array_equal(action, [0, 1])
        params = {"logits": -params["logits"]}  # (B, A)
        actions_mask = actions_mask.at[1].set(jnp.array([True, False, False]))  # (B, A)
        memory, action = ppo.sample_policy_step(
            policy, params, observation, memory, starts, jax.random.key(1), actions_mask
        )
        np.testing.assert_array_equal(action, [2, 0])
    assert sample.call_count == 1
    chex.assert_shape(jax.tree.leaves(memory), (2, 1))
    for leaf in jax.tree.leaves(memory):  # (B, H)
        np.testing.assert_array_equal(leaf, [[1], [2]])


@pytest.mark.parametrize(
    "frame_limit,stop,expected_length,expected_frames",
    [(1, True, 0, 1), (2, True, 1, 2), (1, False, 2, 1), (None, True, 2, 3)],
    ids=["initial-frame-only", "capped-video", "evaluation-finishes", "uncapped-video"],
)
def test_episode_frame_limits(
    config: ppo.Config,
    frame_limit: int | None,
    stop: bool,
    expected_length: int,
    expected_frames: int,
) -> None:
    env = ScriptedSMAX()
    state = TrainState.create(apply_fn=scripted_policy, params={}, tx=optax.sgd(0.1))
    memory = ((jnp.zeros((2, 1)), jnp.zeros((2, 1))),)  # leaves: (N=2, H=1)

    def render(state: State) -> np.ndarray:  # state leaves: (...)
        return np.full((2, 2, 3), int(state.state.step), np.uint8)  # (P, Q, 3)

    with patch.object(env, "render", side_effect=render):
        result = ppo.play_episode(
            state,
            replace(config, video_max_frames=frame_limit),
            jax.random.key(0),
            memory,
            record=True,
            stop_at_frame_limit=stop,
            env=env,
        )
    assert result.length == expected_length and len(result.frames) == expected_frames
    assert result.episode_return == sum(range(1, expected_length + 1))
    assert all(frame.shape == (2, 2, 3) and frame.dtype == np.uint8 for frame in result.frames)
    # The final frame must precede autoreset, which returns the step counter to zero.
    np.testing.assert_array_equal(result.frames[-1], expected_frames - 1)


def test_compiled_playback_resets_dead_agent_memory() -> None:
    env = ScriptedSMAX()
    key, reset_key = jax.random.split(jax.random.key(0))  # each ()
    observation, env_state = env.reset(reset_key)  # (N, O)/(N, A); state leaves: (...)
    state = PlaybackState(
        key=key,  # ()
        observation=observation,  # (N=2, O=2)/(N, A=3)
        env_state=env_state,  # leaves: (...)
        final_state=env_state,  # leaves: (...)
        memory=((jnp.full((2, 1), 9.0), jnp.full((2, 1), 9.0)),),  # leaves: (N, H=1)
        episode_starts=jnp.ones(2, dtype=bool),  # (N,)
        episode_return=jnp.asarray(0.0),  # ()
        length=jnp.asarray(0, dtype=jnp.int32),  # ()
        done=jnp.asarray(False),  # ()
    )
    result, saved = advance_episode(ppo.playback_step, env, scripted_policy, {}, state, 100, record=False)
    assert saved is None and int(result.length) == 2 and float(result.episode_return) == 3.0
    assert result.observation.observation.shape == (2, 2)
    assert result.observation.actions_mask.shape == (2, 3)
    for leaf in jax.tree.leaves(result.memory):  # (N, H)
        np.testing.assert_array_equal(leaf, [[1.0], [2.0]])
    np.testing.assert_array_equal(result.episode_starts, [True, True])
    assert int(result.env_state.state.step) == 0 and int(result.final_state.state.step) == 2


def test_config_and_cli(config: ppo.Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "smax.yaml"
    path.write_text(yaml.safe_dump(asdict(config)))
    assert ppo.load_config(path, ppo.Config) == config
    assert hash(config)
    with pytest.raises(ValueError, match="num_agents"):
        ppo.make_env(replace(config, num_minibatches=4))
    with pytest.raises(ValueError, match="at least one rollout"):
        replace(config, total_steps=1)
    monkeypatch.setattr("sys.argv", ["mappo", "--config", str(path)])
    train = Mock()
    monkeypatch.setattr(ppo, "train", train)
    ppo.main()
    train.assert_called_once_with(config)


@pytest.mark.parametrize("interval", [1, 3, 0], ids=["final-on-interval", "final-before-interval", "disabled"])
def test_training_resume_wiring_and_environment_step_logging(
    config: ppo.Config,
    monkeypatch: pytest.MonkeyPatch,
    interval: int,
) -> None:
    config = replace(
        config,
        total_steps=10,
        video_interval=interval,
        eval_interval=interval,
        checkpoint_interval=interval or 3,
        log_interval=1,
    )
    env = ScriptedSMAX()
    writer, manager = MagicMock(), Mock()
    video, evaluation, save = Mock(), Mock(), Mock(return_value=2.5)
    monkeypatch.setattr(ppo, "make_env", lambda _: env)
    monkeypatch.setattr(ppo, "MLflowLogger", Mock(return_value=writer))
    monkeypatch.setattr(ppo, "checkpoint_manager", lambda _: manager)
    monkeypatch.setattr(ppo, "log_video", video)
    monkeypatch.setattr(ppo, "log_evaluation", evaluation)
    monkeypatch.setattr(ppo, "save_checkpoint", save)

    def restore(manager: Any, state: TrainState, config: ppo.Config) -> ppo.TrainingProgress:
        return ppo.TrainingProgress(
            train_state=state.replace(step=5),
            key=jax.random.key(7),  # ()
            rollout_count=1,
            episode_count=1,
            episode_return_sum=3.0,
            episode_length_sum=2,
            environment_step_count=6,  # The previous iteration used a different rollout size.
            observation_count=18,  # The previous environment had three agents.
        )

    def train_step(
        runner: ppo.RolloutCarry,
        state: TrainState,
        env: SMAXEnv,
        config: ppo.Config,
        key: jax.Array,
        metrics: ppo.TrainingMetrics,
        environment_step_count: jax.Array,  # (); key: (); metrics: scalar leaves
    ) -> ppo.TrainStepResult:
        assert environment_step_count == 6
        chex.assert_shape(runner.observation.observation, (2, 2, 2))
        chex.assert_shape(jax.tree.leaves(runner.model_carry), (4, 4))
        np.testing.assert_array_equal(runner.episode_start, True)
        np.testing.assert_array_equal(runner.episode_return, 0)
        with jax.enable_x64():
            metrics = jax.tree.map(lambda x: x + 1, metrics)  # scalar leaves
        metrics = metrics._replace(
            window=metrics.window._replace(
                explained_variance_count=jnp.asarray(0),  # ()
                explained_variance_sum=jnp.asarray(0.0),  # ()
            )
        )
        return ppo.TrainStepResult(
            train_state=state.replace(step=state.step + 1),
            rollout_carry=runner,
            key=key,  # ()
            metrics=metrics,
        )

    monkeypatch.setattr(ppo, "restore_checkpoint", restore)
    monkeypatch.setattr(ppo, "train_step_donated", train_step)
    state = ppo.train(config)
    writer.log_params.assert_called_once_with(asdict(config))
    for logger in (video, evaluation):
        if not interval:
            logger.assert_not_called()
            continue
        logger.assert_called_once()
        assert logger.call_args.args[0] is state
        assert logger.call_args.args[3:] == (2, 10)
        chex.assert_shape(jax.tree.leaves(logger.call_args.kwargs["initial_carry"]), (2, 4))
        assert logger.call_args.kwargs["env"] is env
    scalars = {tag: value[-1] for tag, value in metric_history(writer).items()}
    assert scalars["train/env_steps_total"] == (10.0, 10)
    assert scalars["train/observations_total"] == (26.0, 10)
    assert scalars["train/episodes_total"] == (2.0, 10)
    assert scalars["train/iterations_total"] == (2.0, 10)
    assert scalars["train/updates_total"] == (6.0, 10)
    assert scalars["train/progress_fraction"] == (1.0, 10)
    assert scalars["train/learning_rate"] == pytest.approx((0.0004, 10))
    assert scalars["checkpoint/save_seconds"] == (2.5, 10)
    assert "train/explained_variance_mean" not in scalars
    assert "train/kl_early_stop_fraction" not in scalars
    for name in ("iterations", "env_steps", "observations", "episodes"):
        assert scalars[f"train/{name}_per_second"][0] > 0
    save.assert_called_once()
    assert save.call_args.args[1].train_state is state
    assert save.call_args.args[1].rollout_count == 2
    assert save.call_args.args[1].environment_step_count == 10
    assert save.call_args.args[1].observation_count == 26
    writer.__exit__.assert_called_once()
    manager.close.assert_called_once()


def test_mlflow_resume_without_checkpoint(config: ppo.Config) -> None:
    model = Mock()
    model.init.return_value = {"params": {}}
    with (
        patch.object(ppo, "configure_compilation_cache"),
        patch.object(ppo, "make_env", return_value=ScriptedSMAX()),
        patch.object(ppo, "checkpoint_manager"),
        patch.object(ppo, "ActorCritic", return_value=model),
        patch.object(ppo.TrainState, "create"),
        patch.object(ppo, "restore_checkpoint", return_value=None),
        patch.object(ppo, "MLflowLogger", side_effect=RuntimeError("stop before training")) as logger,
        pytest.raises(RuntimeError, match="stop before training"),
    ):
        ppo.train(config)
    logger.assert_called_once_with(experiment="mappo", run_name=config.run_id)
