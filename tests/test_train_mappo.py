import json
from contextlib import ExitStack, closing, nullcontext
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from threading import Event
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
from rl2.actions import MultiCategoricalPolicy
from rl2.envs.lux_env import LuxConfig, LuxEnv
from rl2.envs.multi_agent import AgentObservation, MultiAgentStep
from rl2.envs.smax_env import SMAXConfig, SMAXEnv

pytestmark = pytest.mark.usefixtures("ppo_mesh")


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
        total_env_steps=8,
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
        model=ppo.GenericModelConfig(
            encoder=ppo.MLPEncoderConfig(embedding_size=4), recurrent=ppo.LSTMConfig(hidden_size=4)
        ),
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

    def step(
        self, key: jax.Array, state: State, action: jax.Array
    ) -> MultiAgentStep[State, jax.Array]:  # key: (); action: (N,)
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
            observation=self.observation(advanced),  # (N, O)/(N, A)
            state=advanced,  # leaves: (...)
            reward=jnp.full(2, step, jnp.float32),  # (N,)
            done=done,  # ()
            agent_done=jnp.array([True, done]),  # (N,)
            agent_inactive=jnp.array([True, False]),  # (N,)
            terminated=terminated,  # ()
            truncated=done & ~terminated,  # ()
            final_observation=self.observation(final),  # (N, O)/(N, A)
            final_state=final,  # leaves: (...)
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
    model = ppo.make_actor_critic(env.action_space, config.model)
    memory = model.initial_carry(4)  # leaves: (B=4, H=4)
    runner = ppo.RolloutCarry(state, observation, memory, jnp.ones(4, bool), jnp.zeros(2), jnp.zeros(2, jnp.int32))
    train_state = TrainState.create(apply_fn=scripted_policy, params={}, tx=optax.sgd(0.1))
    advanced, result = ppo.collect_rollout(runner, train_state, env, config, jax.random.key(1))
    chex.assert_shape(result.batch.observation, (4, 2, 2))
    chex.assert_shape(result.batch.actions_mask, (4, 2, 3))
    chex.assert_shape(
        (
            result.batch.action,
            result.batch.old_log_prob,
            result.batch.value,
            result.batch.advantage,
            result.batch.return_target,
            result.batch.episode_start,
        ),
        (4, 2),
    )
    np.testing.assert_array_equal(result.batch.observation[:, 0, 1], [0, 1, 0, 1])
    np.testing.assert_array_equal(result.batch.action, [[0, 0], [1, 1], [0, 0], [1, 1]])
    # Team reward survives death. Timeout adds gamma * V(final)=10, never V(reset)=0.
    np.testing.assert_allclose(result.batch.return_target, [[7, 12], [7, 12], [2, 2], [2, 2]])
    np.testing.assert_array_equal(
        result.batch.episode_start, [[True, True], [True, False], [True, True], [True, False]]
    )
    for leaf in jax.tree.leaves(advanced.model_carry):  # each (B, H)
        np.testing.assert_array_equal(leaf[:, 0], [1, 2, 1, 2])
    assert int(result.episode_count) == 2
    chex.assert_shape(result.terminated_episode_count, ())
    assert int(result.terminated_episode_count) == 1
    assert float(result.episode_return_sum) == 6.0
    assert int(result.episode_length_sum) == 4
    np.testing.assert_array_equal(advanced.episode_return, 0)
    np.testing.assert_array_equal(advanced.episode_start, True)
    selected = result.batch.select_sequences(jnp.array([3, 0]))
    np.testing.assert_array_equal(selected.return_target, [[2, 2], [7, 12]])
    chex.assert_shape(jax.tree.leaves(selected.model_carry), (2, 4))
    repeated, duplicate = ppo.collect_rollout(advanced, train_state, env, config, result.key)
    chex.assert_trees_all_equal(duplicate.batch.model_carry, advanced.model_carry)
    chex.assert_trees_all_equal(result.batch, duplicate.batch.replace(model_carry=memory))
    assert not np.array_equal(jax.random.key_data(result.key), jax.random.key_data(duplicate.key))
    np.testing.assert_array_equal(repeated.episode_step_count, 0)

    # Simultaneous flags still count a termination once; unfinished episodes do not count.
    step = env.step

    def step_with_both_flags(
        key: jax.Array,
        state: State,
        action: jax.Array,  # key: (); state leaves: (...); action: (N,)
    ) -> MultiAgentStep[State, jax.Array]:
        result = step(key, state, action)
        return result._replace(truncated=result.done)  # ()

    env.step = step_with_both_flags
    partial, unfinished = ppo.collect_rollout(advanced, train_state, env, replace(config, num_steps=1), result.key)
    assert int(unfinished.episode_count) == 0
    assert int(unfinished.terminated_episode_count) == 0
    _, completed = ppo.collect_rollout(partial, train_state, env, replace(config, num_steps=1), unfinished.key)
    assert int(completed.episode_count) == 2
    assert int(completed.terminated_episode_count) == 1


def test_rollout_normalizes_weak_episode_flags(config: ppo.Config) -> None:
    env = ScriptedSMAX()
    original_step = env.step

    def step(key: jax.Array, state: State, action: jax.Array) -> MultiAgentStep:  # (); state leaves: (...); (N,)
        result = original_step(key, state, action)
        return result._replace(agent_done=jnp.full(2, result.done))  # (N=2,) weak bool, as emitted by Lux

    observation, env_state = jax.vmap(env.reset)(jax.random.split(jax.random.key(0), 2))
    # observation: (E=2, N=2, O=2)/(E, N, A=3); state leaves: (E, ...)
    runner = ppo.RolloutCarry(
        env_state=env_state,
        observation=observation,
        model_carry=((jnp.zeros((4, 1)), jnp.zeros((4, 1))),),  # leaves: (B=4, H=1)
        episode_start=jnp.ones(4, dtype=bool),  # (B,)
        episode_return=jnp.zeros(2),  # (E,)
        episode_step_count=jnp.zeros(2, dtype=jnp.int32),  # (E,)
    )
    state = TrainState.create(apply_fn=scripted_policy, params={}, tx=optax.sgd(0.1))
    with patch.object(env, "step", side_effect=step):
        advanced, _ = jax.jit(ppo.collect_rollout, static_argnames=("env", "config"))(
            runner, state, env, config, jax.random.key(1)
        )
    assert advanced.episode_start.shape == (4,)
    assert advanced.episode_start.dtype == jnp.bool_ and not advanced.episode_start.weak_type
    np.testing.assert_array_equal(advanced.episode_start, True)


def test_timeout_bootstrap_resets_only_inactive_agent_memory(config: ppo.Config) -> None:
    def memory_policy(
        variables: dict[str, Any],
        observation: jax.Array,  # (B, O)
        memory: ppo.RecurrentCarry,  # leaves: (B, H)
        starts: jax.Array,  # (B,)
        actions_mask: jax.Array,  # (B, A)
        *,
        method: str,
    ) -> ppo.ActorCriticOutput:
        output = scripted_policy(variables, observation, memory, starts, actions_mask, method=method)
        return output._replace(value=output.carry[0][0][:, 0])  # value: (B,)

    env = ScriptedSMAX()
    _, state = jax.vmap(env.reset)(jax.random.split(jax.random.key(0), 1))  # leaves: (E=1, ...)
    state = state.replace(state=state.state.replace(step=jnp.ones(1, jnp.int32)))
    observation = jax.vmap(env.observation)(state)  # (E, N=2, O=2)/(E, N, A=3)
    runner = ppo.RolloutCarry(
        env_state=state,
        observation=observation,
        model_carry=((jnp.full((2, 1), 7.0), jnp.full((2, 1), 7.0)),),  # leaves: (B=2, H=1)
        episode_start=jnp.zeros(2, bool),  # (B,)
        episode_return=jnp.zeros(1),  # (E,)
        episode_step_count=jnp.ones(1, jnp.int32),  # (E,)
    )
    train_state = TrainState.create(apply_fn=memory_policy, params={}, tx=optax.sgd(0.1))
    advanced, result = ppo.collect_rollout(
        runner, train_state, env, replace(config, num_envs=1, num_steps=1, num_minibatches=1), jax.random.key(1)
    )
    # Final values use cleared memory for the dead agent (1), retained memory for the living one (9).
    np.testing.assert_allclose(result.batch.return_target, [[2.5], [6.5]])
    np.testing.assert_array_equal(advanced.episode_start, True)
    for leaf in jax.tree.leaves(advanced.model_carry):  # (B, H)
        np.testing.assert_array_equal(leaf, 8.0)  # Bootstrap evaluation does not advance rollout memory.


def test_local_model_mask_replay_and_shared_memory(config: ppo.Config) -> None:
    model = ppo.make_actor_critic(ppo.DiscreteSpace(3), config.model)
    observation = jnp.arange(24, dtype=jnp.float32).reshape(3, 2, 4) / 24  # (B, T, O)
    starts = jnp.array([[True, True], [True, False], [False, False]])  # (B, T)
    actions_mask = jnp.array(
        [[[True, False, True]] * 2, [[False, True, False]] * 2, [[True, True, False]] * 2]
    )  # (B, T, A)
    memory = model.initial_carry(3)  # leaves: (B, H)
    assert len(memory) == config.model.recurrent.num_layers
    chex.assert_shape(jax.tree.leaves(memory), (3, 4))
    params = model.init(jax.random.key(0), observation, memory, starts, actions_mask)
    chex.assert_shape(params["params"]["encoder"]["kernel"], (4, config.model.encoder.embedding_size))
    # Keep the tiny ReLU encoder active so the test exercises recurrent features.
    params["params"]["encoder"]["kernel"] = jnp.eye(4)  # (O, F)
    output = model.apply(params, observation, memory, starts, actions_mask)
    first = model.apply(params, observation[:, 0], memory, starts[:, 0], actions_mask[:, 0], method="step")
    chex.assert_shape(first.policy.logits, (3, 3))
    chex.assert_shape(first.value, (3,))
    second = model.apply(params, observation[:, 1], first.carry, starts[:, 1], actions_mask[:, 1], method="step")
    chex.assert_trees_all_close(second.carry, output.carry, atol=1e-6)
    np.testing.assert_allclose(output.value, jnp.stack([first.value, second.value], axis=1), atol=1e-6)
    np.testing.assert_allclose(
        output.policy.logits, jnp.stack([first.policy.logits, second.policy.logits], axis=1), atol=1e-6
    )
    action, log_prob = output.policy.sample_and_log_prob(key=jax.random.key(2))  # each (B, T)
    assert bool(jnp.take_along_axis(actions_mask, action[..., None], axis=-1).all())
    assert bool(jnp.isfinite(output.policy.entropy()).all())
    np.testing.assert_allclose(output.policy.log_prob(action), log_prob, atol=1e-6)
    chex.assert_shape(output.value, (3, 2))
    chex.assert_shape(output.policy.logits, (3, 2, 3))
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
    observation: jax.Array,  # (B, T, O)
    memory: ppo.RecurrentCarry,  # leaves: (B, H)
    starts: jax.Array,  # (B, T)
    actions_mask: jax.Array,  # (B, T, A)
) -> ppo.ActorCriticOutput:
    logits = jnp.where(actions_mask, variables["params"]["logits"], -1e10)  # (B, T, A)
    return ppo.ActorCriticOutput(
        carry=memory,
        policy=ppo.CategoricalPolicy(logits),
        value=jnp.broadcast_to(variables["params"]["value"], starts.shape),  # (B, T)
    )


@pytest.fixture
def tiny_batch() -> ppo.PPOBatch:
    return ppo.PPOBatch(
        observation=jnp.zeros((2, 3, 1)),  # (B, T, O)
        actions_mask=jnp.ones((2, 3, 2), dtype=bool),  # (B, T, A)
        action=jnp.array([[0, 0, 0], [1, 1, 1]]),  # (B, T)
        old_log_prob=jnp.full((2, 3), -jnp.log(2.0)),  # (B, T)
        value=jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),  # (B, T)
        advantage=jnp.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]),  # (B, T)
        return_target=jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),  # (B, T)
        model_carry=((jnp.zeros((2, 1)), jnp.zeros((2, 1))),),  # leaves: (B, H)
        episode_start=jnp.ones((2, 3), dtype=bool),  # (B, T)
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
    singleton = jax.tree.map(lambda leaf: leaf[:, :1], tiny_batch.select_sequences(jnp.array([0])))
    # Sequence leaves: (B=1, T=1, ...); memory leaves remain (B=1, H=1).
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
        params={"logits": jnp.zeros(2), "value": jnp.asarray(0.0, dtype=jnp.float32)},  # (A,); ()
        tx=optax.inject_hyperparams(optax.adam)(learning_rate=config.learning_rate),
    )
    rollout = ppo.RolloutResult(
        batch=tiny_batch.replace(old_log_prob=tiny_batch.old_log_prob + 1.0),
        key=jax.random.key(0),  # ()
        episode_return_sum=jnp.asarray(0.0),  # ()
        episode_length_sum=jnp.asarray(0),  # ()
        episode_count=jnp.asarray(0),  # ()
        terminated_episode_count=jnp.asarray(0),  # ()
    )
    env = ScriptedSMAX()
    observation, env_state = jax.vmap(env.reset)(jax.random.split(jax.random.key(0), 2))
    runner = ppo.RolloutCarry(
        env_state=env_state,  # leaves: (E=2, ...)
        observation=observation,  # (E, N=2, O=2)/(E, N, A=3)
        model_carry=ppo.make_actor_critic(env.action_space, config.model).initial_carry(4),  # leaves: (B=4, H=4)
        episode_start=jnp.ones(4, bool),  # (B,)
        episode_return=jnp.zeros(2),  # (E,)
        episode_step_count=jnp.zeros(2, jnp.int32),  # (E,)
    )
    collect = Mock(
        return_value=(
            runner,
            rollout._replace(batch=rollout.batch.replace(return_target=jnp.zeros_like(tiny_batch.return_target))),
        )
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
    collect.return_value = (
        runner,
        rollout._replace(
            episode_count=jnp.asarray(2),
            terminated_episode_count=jnp.asarray(1),  # each ()
        ),
    )
    result = ppo.train_step(runner, state, ScriptedSMAX(), config, rollout.key, result.metrics)
    assert int(result.metrics.window.explained_variance_count) == 1
    assert float(result.metrics.window.explained_variance_sum) == pytest.approx(1.0)
    assert int(result.metrics.window.terminated_episode_count) == 1

    # Window resets and changing step counts must reuse the compiled learner on every device.
    with jax.set_mesh(ppo.training_mesh(config, env.num_agents)):
        state, key, metrics = jax.device_put((state, rollout.key, ppo.init_metrics()), ppo.PartitionSpec())
        runner = jax.device_put(runner, ppo.PartitionSpec("batch"))  # state: (E, ...); memory: (B, H)
        collect.return_value = (runner, collect.return_value[1])
        result = ppo.train_step_donated(runner, state, env, config, key, metrics, jnp.asarray(4, dtype=jnp.float32))
        traces = collect.call_count
        with jax.enable_x64():
            metrics = result.metrics._replace(window=ppo.RunningSum.init(result.metrics.window))  # scalar leaves
        result = ppo.train_step_donated(
            result.rollout_carry,
            result.train_state,
            env,
            config,
            result.key,
            metrics,
            jnp.asarray(8, dtype=jnp.float32),  # ()
        )
        assert collect.call_count == traces
        assert int(result.metrics.window.rollout_count) == 1
        assert int(result.metrics.total.episode_count) == 4


def test_schedules_use_environment_steps(config: ppo.Config) -> None:
    config = replace(config, total_env_steps=16, entropy_decay="cosine")
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
    large = replace(config, total_env_steps=2**33)
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


@pytest.mark.parametrize("fail", [False, True], ids=["completed", "background-failure"])
def test_async_checkpoint_finishes_on_close(config: ppo.Config, tmp_path: Path, fail: bool) -> None:
    release = Event()

    def finalize() -> None:
        assert release.wait(timeout=10), "save blocked on background finalization"
        if fail:
            raise OSError("background save failed")

    state = TrainState.create(apply_fn=scripted_policy, params={}, tx=optax.sgd(0.1))
    progress = ppo.TrainingProgress(
        train_state=state,
        key=jax.random.key(0),  # ()
        rollout_count=1,
        episode_count=2,
        episode_return_sum=3.0,
        episode_length_sum=4,
        environment_step_count=4,
        observation_count=8,
    )
    with (
        patch.object(
            ppo.ocp,
            "CheckpointManagerOptions",
            partial(
                ppo.ocp.CheckpointManagerOptions,
                async_options=ppo.ocp.AsyncOptions(post_finalization_callback=finalize),
            ),
        ),
        pytest.raises(OSError, match="background save failed") if fail else nullcontext(),
        ExitStack() as resources,
    ):
        manager = resources.enter_context(closing(ppo.checkpoint_manager(str(tmp_path))))
        resources.callback(release.set)
        ppo.save_checkpoint(manager, progress, config)
        assert manager.is_saving_in_progress()
    if not fail:
        with ppo.checkpoint_manager(str(tmp_path)) as manager:
            chex.assert_trees_all_equal(ppo.restore_checkpoint(manager, state, config), progress)


@pytest.mark.parametrize("selfplay", [False, True], ids=["heuristic_enemy", "selfplay"])
def test_real_smax_single_step_update_and_checkpoint(config: ppo.Config, tmp_path: Path, selfplay: bool) -> None:
    config = replace(config, num_envs=1, num_steps=1, num_minibatches=3, env=SMAXConfig(max_steps=1, selfplay=selfplay))
    env = ppo.make_env(config)  # Three or six agent sequences form three minibatches.
    observation, state = jax.vmap(env.reset)(jax.random.split(jax.random.key(0), 1))  # (E=1, N, O)/(E, N, A); (...)
    model = ppo.make_actor_critic(env.action_space, config.model)
    memory = model.initial_carry(env.num_agents)  # leaves: (B=N, H)
    starts = jnp.ones(env.num_agents, bool)  # (B,)
    params = model.init(
        jax.random.key(1),
        observation.observation[0, :, None],  # (B, T=1, O)
        memory,
        starts[:, None],  # (B, T=1)
        observation.actions_mask[0, :, None],  # (B, T=1, A)
    )
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
    # Evaluation uses all controlled agents and records the actual terminal frame.
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
        observation_count=env.num_agents,
    )
    with ppo.checkpoint_manager(str(tmp_path)) as manager:
        ppo.save_checkpoint(manager, progress, config)
        manager.wait_until_finished()
        metadata = manager.restore(1, args=ppo.ocp.args.Composite(metadata=ppo.ocp.args.JsonRestore())).metadata
        assert metadata["config"] == asdict(config)
        assert metadata["environment_steps"] == 1 and metadata["observations"] == env.num_agents
        changed = replace(config, env=SMAXConfig(map_name="8m"), num_envs=2, num_steps=2)
        restored = ppo.restore_checkpoint(manager, train_state, changed)
        assert restored is not None
        chex.assert_trees_all_equal(restored, progress)


def test_playback_reuses_compiled_sampling_with_updated_inputs() -> None:
    memory = ((jnp.zeros((2, 1)), jnp.zeros((2, 1))),)  # leaves: (N=2, H=1)
    observation = AgentObservation(
        observation=jnp.zeros((2, 1)),  # (N, O=1)
        actions_mask=jnp.array([[True, False, True], [False, True, False]]),  # (N, A=3)
    )
    params = {"logits": jnp.array([[100.0, 0.0, -100.0], [0.0, 0.0, 0.0]])}  # (N, A)

    def policy(
        variables: dict[str, dict[str, jax.Array]],
        observation: jax.Array,  # (N, O)
        carry: ppo.RecurrentCarry,  # leaves: (N, H)
        episode_start: jax.Array,  # (N,)
        actions_mask: jax.Array,  # (N, A)
        *,
        method: str,
    ) -> ppo.ActorCriticOutput:
        logits = jnp.where(actions_mask, variables["params"]["logits"], -1e10)  # (N, A)
        return ppo.ActorCriticOutput(carry=carry, policy=ppo.CategoricalPolicy(logits), value=jnp.zeros(2))

    def step(key: jax.Array, state: jax.Array, action: jax.Array) -> MultiAgentStep:  # (); (N,); (N,)
        return MultiAgentStep(
            observation=observation,
            state=action,  # (N,)
            reward=action.astype(jnp.float32),  # (N,)
            done=jnp.asarray(True),  # ()
            agent_done=jnp.ones(2, dtype=bool),  # (N,)
            agent_inactive=jnp.zeros(2, dtype=bool),  # (N,)
            terminated=jnp.asarray(True),  # ()
            truncated=jnp.asarray(False),  # ()
            final_observation=observation,
            final_state=action,  # (N,)
        )

    env = Mock(num_agents=2)
    env.reset.return_value = observation, jnp.zeros(2, dtype=jnp.int32)  # (N, O)/(N, A); (N,)
    env.step.side_effect = step
    with patch.object(
        ppo.CategoricalPolicy, "sample", autospec=True, side_effect=ppo.CategoricalPolicy.sample
    ) as sample:
        first = ppo.run_episode(
            env, policy, params, jax.random.key(0), memory, recording_capacity=2, stop_at_frame_limit=False
        )
        second = ppo.run_episode(
            env,
            policy,
            {"logits": -params["logits"]},
            jax.random.key(1),
            memory,
            recording_capacity=2,
            stop_at_frame_limit=False,
        )
    assert sample.call_count == 1
    assert first.states.shape == second.states.shape == (2, 2)
    np.testing.assert_array_equal(first.states[1], [0, 1])
    np.testing.assert_array_equal(second.states[1], [2, 1])
    assert float(first.episode_return) == 0.5 and float(second.episode_return) == 1.5


def test_playback_reuses_compiled_episode(config: ppo.Config) -> None:
    env = ScriptedSMAX()
    state = TrainState.create(apply_fn=scripted_policy, params={}, tx=optax.sgd(0.1))
    memory = ((jnp.zeros((2, 1)), jnp.zeros((2, 1))),)  # leaves: (N=2, H=1)
    # Count class-method traces without replacing the bound-method cache key.
    with patch.object(ScriptedSMAX, "observation", autospec=True, side_effect=ScriptedSMAX.observation) as observe:
        first = ppo.play_episode(state, config, jax.random.key(0), memory, record=False, env=env)
        traces = observe.call_count
        second = ppo.play_episode(state, config, jax.random.key(1), memory, record=False, env=env)
        assert observe.call_count == traces
    assert first.length == second.length == 2
    assert first.episode_return == second.episode_return == 3.0


@pytest.mark.parametrize("recording_capacity", [0, 4], ids=["no-recording", "recorded-states"])
def test_compiled_episode_state_buffer(recording_capacity: int) -> None:
    env = ScriptedSMAX()
    memory = ((jnp.zeros((2, 1)), jnp.zeros((2, 1))),)  # leaves: (N=2, H=1)
    result = ppo.run_episode(
        env,
        scripted_policy,
        {},
        jax.random.key(0),
        memory,
        recording_capacity=recording_capacity,
        stop_at_frame_limit=True,
    )
    assert result.episode_return.shape == result.length.shape == ()
    assert float(result.episode_return) == 3.0 and int(result.length) == 2
    if recording_capacity == 0:
        assert result.states is None
        return
    assert result.states.state.step.shape == (4,)
    assert result.states.state.unit_positions.shape == (4, 3, 2)
    assert result.states.state.unit_alive.shape == (4, 3)
    assert result.states.state.unit_alive.dtype == jnp.bool_
    np.testing.assert_array_equal(result.states.state.step[:3], [0, 1, 2])
    np.testing.assert_array_equal(result.states.state.unit_alive[:3, 0], [True, False, False])


@pytest.mark.parametrize(
    "frame_limit,stop,expected_length,expected_frames",
    [(1, True, 0, 1), (2, True, 1, 2), (1, False, 2, 1), (2, False, 2, 2), (None, True, 2, 3)],
    ids=["initial-frame-only", "capped-video", "evaluation-finishes", "finish-after-recording", "uncapped-video"],
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
    memory = ((jnp.full((2, 1), 9.0), jnp.full((2, 1), 9.0)),)  # leaves: (N=2, H=1)

    def policy(
        variables: dict[str, Any],
        observation: jax.Array,  # (N, O=2)
        carry: ppo.RecurrentCarry,  # leaves: (N, H)
        starts: jax.Array,  # (N,)
        mask: jax.Array,  # (N, A=3)
        *,
        method: str,
    ) -> ppo.ActorCriticOutput:
        carry = jax.tree.map(lambda leaf: jnp.where(starts[:, None], 0, leaf) + 1, carry)  # leaves: (N, H)
        logits = jnp.where(jnp.arange(3) == carry[0][0], 0.0, -jnp.inf)  # (N, A)
        return ppo.ActorCriticOutput(carry=carry, policy=ppo.CategoricalPolicy(logits), value=jnp.zeros(2))

    def step(key: jax.Array, state: State, action: jax.Array) -> MultiAgentStep:  # (); leaves: (...); (N,)
        result = ScriptedSMAX.step(env, key, state, action)
        final = result.final_state.replace(
            state=result.final_state.state.replace(
                prev_attack_actions=jnp.concatenate((action, jnp.zeros(1, jnp.int32)))
            )
        )  # prev_attack_actions: (units=3,)
        return result._replace(final_state=final)

    with patch.object(env, "step", side_effect=step):
        result = ppo.run_episode(
            env, policy, {}, jax.random.key(0), memory, recording_capacity=3, stop_at_frame_limit=False
        )
    assert int(result.length) == 2 and float(result.episode_return) == 3.0
    np.testing.assert_array_equal(result.states.state.step, [0, 1, 2])
    # Actions expose memory: both reset initially, then only the dead agent resets.
    np.testing.assert_array_equal(result.states.state.prev_attack_actions[1:, :2], [[1, 1], [1, 2]])


@pytest.mark.parametrize("selfplay", [False, True], ids=["heuristic_enemy", "selfplay"])
def test_config_and_cli(config: ppo.Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selfplay: bool) -> None:
    config = replace(config, env=SMAXConfig(selfplay=selfplay))
    path = tmp_path / "smax.yaml"
    path.write_text(yaml.safe_dump(asdict(config)))
    assert ppo.load_config(path, ppo.Config) == config
    assert hash(config)
    with pytest.raises(ValueError, match="num_agents"):
        ppo.make_env(replace(config, num_minibatches=5))
    with pytest.raises(ValueError, match="at least one rollout"):
        replace(config, total_env_steps=1)
    monkeypatch.setattr("sys.argv", ["mappo", "--config", str(path)])
    train = Mock()
    monkeypatch.setattr(ppo, "train", train)
    ppo.main()
    train.assert_called_once_with(config)


@pytest.mark.parametrize(
    "interval,checkpoint_fails",
    [(1, False), (3, False), (0, False), (0, True)],
    ids=["final-on-interval", "final-before-interval", "disabled", "checkpoint-close-failure"],
)
def test_training_resume_wiring_and_environment_step_logging(
    config: ppo.Config,
    monkeypatch: pytest.MonkeyPatch,
    interval: int,
    checkpoint_fails: bool,
) -> None:
    config = replace(
        config,
        total_env_steps=10,
        video_interval=interval,
        eval_interval=interval,
        checkpoint_interval=interval or 3,
        log_interval=1,
    )
    env = ScriptedSMAX()
    writer, manager = MagicMock(), Mock()
    checkpoint_error = OSError("final checkpoint failed")
    if checkpoint_fails:
        manager.close.side_effect = checkpoint_error
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
    if checkpoint_fails:
        with pytest.raises(OSError, match="final checkpoint failed"):
            ppo.train(config)
        state = save.call_args.args[1].train_state
    else:
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
    # ExitStack passes the context manager itself before the exception arguments.
    assert writer.__exit__.call_args.args[-3:-1] == ((OSError, checkpoint_error) if checkpoint_fails else (None, None))
    manager.close.assert_called_once()


def test_episode_end_fractions_use_completed_window_episodes(config: ppo.Config) -> None:
    config = replace(config, total_env_steps=20, log_interval=2)
    writer = MagicMock()
    model = Mock()
    model.init.return_value = {"params": {}}

    def initial_carry(batch_size: int) -> ppo.RecurrentCarry:
        return ((jnp.zeros((batch_size, 1)), jnp.zeros((batch_size, 1))),)  # leaves: (B, H=1)

    model.initial_carry.side_effect = initial_carry
    episode_counts = iter([(1, 1), (3, 0), (2, 2), (0, 0), (0, 0)])

    def train_step(
        runner: ppo.RolloutCarry,
        state: TrainState,
        env: ppo.PPOEnv,
        config: ppo.Config,
        key: jax.Array,  # ()
        metrics: ppo.TrainingMetrics,  # scalar leaves
        environment_step_count: jax.Array,  # ()
    ) -> ppo.TrainStepResult:
        completed, terminated = next(episode_counts)
        with jax.enable_x64():
            increment = jax.tree.map(jnp.ones_like, ppo.init_metrics())  # scalar leaves
            episodes = ppo.EpisodeMetrics(
                episode_count=jnp.asarray(completed),  # ()
                episode_return_sum=jnp.asarray(0.0),  # ()
                episode_length_sum=jnp.asarray(completed * 2),  # ()
            )
            increment = increment._replace(
                window=increment.window._replace(
                    episodes=episodes,
                    terminated_episode_count=jnp.asarray(terminated),  # ()
                ),
                total=episodes,
            )
            metrics = ppo.RunningSum.update(metrics, increment)  # scalar leaves
        return ppo.TrainStepResult(train_state=state, rollout_carry=runner, key=key, metrics=metrics)

    with (
        patch.object(ppo, "configure_compilation_cache"),
        patch.object(ppo, "make_env", return_value=ScriptedSMAX()),
        patch.object(ppo, "checkpoint_manager"),
        patch.object(ppo, "make_actor_critic", return_value=model),
        patch.object(ppo, "restore_checkpoint", return_value=None),
        patch.object(ppo, "save_checkpoint", return_value=1.0),
        patch.object(ppo, "MLflowLogger", return_value=writer),
        patch.object(ppo, "train_step_donated", side_effect=train_step),
    ):
        ppo.train(config)

    history = metric_history(writer)
    assert history["rollout/episode_count"] == [(4.0, 8), (2.0, 16), (0.0, 20)]
    assert history["rollout/episode_end_fraction/terminated"] == [(0.25, 8), (1.0, 16)]
    assert history["rollout/episode_end_fraction/truncated"] == [(0.75, 8), (0.0, 16)]


def test_mlflow_resume_without_checkpoint(config: ppo.Config) -> None:
    with (
        patch.object(ppo, "configure_compilation_cache"),
        patch.object(ppo, "make_env", return_value=ScriptedSMAX()),
        patch.object(ppo, "checkpoint_manager"),
        patch.object(ppo, "restore_checkpoint", return_value=None),
        patch.object(ppo, "MLflowLogger", side_effect=RuntimeError("stop before training")) as logger,
        pytest.raises(RuntimeError, match="stop before training"),
    ):
        ppo.train(config)
    logger.assert_called_once_with(experiment="mappo", run_name=config.run_id)


def test_hierarchical_model_masks_and_sequence_replay(config: ppo.Config) -> None:
    model = ppo.make_actor_critic(
        action_space=ppo.HierarchicalMultiDiscreteSpace(nvec=(3, 3), group_sizes=(1, 2)), model=config.model
    )
    observation = jnp.ones((2, 2, 4))  # (B, T, O)
    starts = jnp.array([[True, False], [True, False]])  # (B, T)
    masks = jnp.broadcast_to(jnp.array([[True, True, False], [True, False, True]]), (2, 2, 2, 3))  # (B, T, D, A)
    memory = model.initial_carry(2)  # leaves: (B, H)
    params = model.init(jax.random.key(0), observation, memory, starts, masks)
    output = model.apply(params, observation, memory, starts, masks)
    assert isinstance(output.policy, ppo.HierarchicalMultiCategoricalPolicy)
    chex.assert_shape(output.policy.logits, (2, 2, 2, 3))
    chex.assert_shape(output.value, (2, 2))
    action, log_prob = output.policy.sample_and_log_prob(key=jax.random.key(1))  # (B, T, D); (B, T)
    assert action.shape == (2, 2, 2) and log_prob.shape == (2, 2)
    assert bool((action[..., 0] < 2).all())
    assert bool((action[..., 1] != 1).all())
    np.testing.assert_allclose(output.policy.log_prob(action), log_prob, atol=1e-6)
    first = model.apply(params, observation[:, 0], memory, starts[:, 0], masks[:, 0], method="step")
    second = model.apply(params, observation[:, 1], first.carry, starts[:, 1], masks[:, 1], method="step")
    chex.assert_trees_all_close(second.carry, output.carry, atol=1e-6)
    np.testing.assert_allclose(
        output.policy.logits, jnp.stack((first.policy.logits, second.policy.logits), axis=1), atol=1e-6
    )


@pytest.mark.parametrize("lux_env", [False, True], ids=["lux-model-other-env", "lux-env-generic-model"])
def test_lux_model_requires_matching_environment(config: ppo.Config, lux_env: bool) -> None:
    with pytest.raises(ValueError, match="lux model"):
        replace(
            config,
            env=LuxConfig() if lux_env else config.env,
            model=config.model if lux_env else ppo.LuxModelConfig(recurrent=config.model.recurrent),
        )


@pytest.mark.parametrize("channels", [0, 3], ids=["nonpositive", "incompatible_attention_width"])
def test_lux_model_requires_positive_even_channels(channels: int) -> None:
    with pytest.raises(ValueError, match="channels"):
        ppo.LuxModelConfig(channels=channels)


def test_lux_config_rollout_update_and_playback(config: ppo.Config, tmp_path: Path) -> None:
    config = replace(
        config,
        env=LuxConfig(map_size=128, max_units=2, max_steps_in_match=2, match_count=1, randomize_params=False),
        model=ppo.LuxModelConfig(
            embedding_size=4, channels=4, recurrent=config.model.recurrent, attention_implementation="xla"
        ),
        num_envs=1,
        num_steps=3,
        num_minibatches=1,
        total_env_steps=3,
    )
    path = tmp_path / "lux.yaml"
    path.write_text(yaml.safe_dump(asdict(config)))
    loaded = ppo.load_config(path, ppo.Config)
    env = ppo.make_env(loaded)
    assert isinstance(env, LuxEnv) and env.num_agents == 2
    assert isinstance(loaded.model, ppo.LuxModelConfig)
    with pytest.raises(ValueError, match="num_agents"):
        ppo.make_env(replace(loaded, num_minibatches=3))
    observation, state = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.key(0), 1))
    # observation: (E, N, ...)/(E, N, U, A); state leaves: (E, ...)
    model = ppo.make_actor_critic(action_space=env.action_space, model=config.model)
    memory = model.initial_carry(2)  # leaves: (B, H)
    starts = jnp.ones(2, dtype=bool)  # (B,)
    sequence = jax.tree.map(lambda leaf: jnp.swapaxes(leaf, 0, 1), observation)  # (B, T=1, ...)
    params = model.init(jax.random.key(1), sequence.observation, memory, starts[:, None], sequence.actions_mask)
    train_state = TrainState.create(apply_fn=model.apply, params=params["params"], tx=optax.adam(0.001))
    runner = ppo.RolloutCarry(
        env_state=state,
        observation=observation,
        model_carry=memory,
        episode_start=starts,
        episode_return=jnp.zeros(1),
        episode_step_count=jnp.zeros(1, dtype=jnp.int32),
    )  # state leaves: (E, ...); observation: (E, N, ...)/(E, N, U, A); memory: (B, H); counts: (E,)
    advanced, rollout = jax.jit(ppo.collect_rollout, static_argnames=("env", "config"))(
        runner, train_state, env, config, jax.random.key(2)
    )
    batch = rollout.batch  # batch-major array leaves: (B, T, ...)
    chex.assert_shape(batch.action, (2, 3, 2))
    chex.assert_shape(batch.observation.self.units.position, (2, 3, 2, 2))
    chex.assert_shape(batch.observation.world.steps, (2, 3))
    chex.assert_shape(batch.observation.world.sensor_mask, (2, 3, 128, 128))
    assert batch.observation.self.units.position.dtype == jnp.int16
    selected = batch.select_sequences(jnp.array([1]))  # leaves: (M=1, T=3, ...)
    chex.assert_trees_all_equal(selected.observation, jax.tree.map(lambda leaf: leaf[1:2], batch.observation))
    chex.assert_shape(batch.old_log_prob, (2, 3))
    chex.assert_shape(batch.actions_mask, (2, 3, 2, 230))
    assert int(rollout.episode_count) == 1 and int(rollout.episode_length_sum) == 3
    assert float(rollout.episode_return_sum) == 0.5
    np.testing.assert_array_equal(advanced.episode_start, True)
    replay = model.apply(
        {"params": train_state.params}, batch.observation, memory, batch.episode_start, batch.actions_mask
    )
    np.testing.assert_allclose(replay.policy.log_prob(batch.action), batch.old_log_prob, atol=1e-5)
    updated, metrics = jax.jit(ppo.update, static_argnames="config")(train_state, batch, config)
    assert int(updated.step) == 1
    assert float(metrics.entropy) > 0
    assert float(metrics.approx_kl) == pytest.approx(0.0, abs=1e-6)
    assert any(
        not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(train_state.params), jax.tree.leaves(updated.params))
    )
    episode = ppo.play_episode(updated, config, jax.random.key(3), memory, record=True, env=env)
    assert episode.length == 3 and episode.episode_return == 0.5
    assert len(episode.frames) == 4
    assert all(frame.shape == (416, 640, 3) and frame.dtype == np.uint8 for frame in episode.frames)


def test_grouped_model_balances_types_and_masks(config: ppo.Config) -> None:
    model = ppo.make_actor_critic(
        action_space=ppo.HierarchicalMultiDiscreteSpace(nvec=(6, 6), group_sizes=(1, 2, 3)), model=config.model
    )
    observation = jnp.zeros((2, 1, 4))  # (B, T, O)
    starts = jnp.ones((2, 1), dtype=bool)  # (B, T)
    mask = jnp.ones((2, 1, 2, 6), dtype=bool).at[1, :, :, 3:].set(False)  # (B, T, D, A), no sap for team 1
    memory = model.initial_carry(2)  # leaves: (B, H)
    params = model.init(jax.random.key(0), observation, memory, starts, mask)
    params["params"]["policy_output"] = jax.tree.map(jnp.zeros_like, params["params"]["policy_output"])
    output = model.apply(params, observation, memory, starts, mask)
    assert isinstance(output.policy, ppo.HierarchicalMultiCategoricalPolicy)
    assert output.policy.raw_logits.shape == (2, 1, 2, 8)
    assert output.policy.logits.shape == (2, 1, 2, 6)
    probs = jax.nn.softmax(output.policy.logits, axis=-1)  # (B, T, D, A)
    np.testing.assert_allclose(probs[0, 0, 0], [1 / 3, 1 / 6, 1 / 6, 1 / 9, 1 / 9, 1 / 9], rtol=1e-6)
    np.testing.assert_allclose(probs[1, 0, 0], [1 / 2, 1 / 4, 1 / 4, 0, 0, 0], rtol=1e-6)
    step = model.apply(params, observation[:, 0], memory, starts[:, 0], mask[:, 0], method="step")
    np.testing.assert_allclose(step.policy.logits, output.policy.logits[:, 0], atol=1e-6)
    np.testing.assert_allclose(step.policy.entropy(), output.policy.entropy()[:, 0], atol=1e-6)


@pytest.mark.parametrize("start", [0, 137], ids=["fresh", "resume-without-history"])
def test_eval_snapshot_ages_and_final_schedule(start: int) -> None:
    history: dict[int, optax.Params] = {}
    final_iteration = 651
    for iteration in range(start, final_iteration):
        params = {"weight": np.asarray([iteration], dtype=np.float32)}  # (1,)
        ppo.retain_eval_snapshot(history, params, iteration, 60, final_iteration)
        params["weight"][0] = -1  # Snapshots must own storage even on CPU.
        completed = iteration + 1
        if completed % 60 == 0 or completed == final_iteration:
            available = {age: history[completed - age] for age in ppo.EVAL_OPPONENT_AGES if completed - age in history}
            assert set(available) == {age for age in ppo.EVAL_OPPONENT_AGES if completed - age >= start}
            for age, snapshot in available.items():
                np.testing.assert_array_equal(snapshot["weight"], [completed - age])
        assert all(previous >= iteration - 400 for previous in history)


class ScriptedMatch(ScriptedSMAX):
    def step(
        self,
        key: jax.Array,  # ()
        state: State,  # leaves: (...)
        action: jax.Array,  # (N=2,)
    ) -> MultiAgentStep[State, jax.Array]:
        result = super().step(key, state, action)
        return result._replace(reward=action.astype(jnp.float32))  # (N=2,)


def test_historical_playback_uses_separate_weights_and_current_team_reward(config: ppo.Config) -> None:
    env = ScriptedMatch()

    def policy(
        variables: dict[str, Any],
        observation: jax.Array,  # (B, O)
        memory: ppo.RecurrentCarry,  # leaves: (B, H)
        starts: jax.Array,  # (B,)
        actions_mask: jax.Array,  # (B, A)
        *,
        method: str,
    ) -> ppo.ActorCriticOutput:
        memory = jax.tree.map(
            lambda leaf: jnp.where(starts[:, None], 0, leaf) + variables["params"]["increment"], memory
        )  # leaves: (B, H)
        # Deterministic actions expose both policy selection and each team's memory.
        logits = jnp.where(jnp.arange(3) == memory[0][0], 0.0, -jnp.inf)  # (B, A=3)
        return ppo.ActorCriticOutput(
            carry=memory, policy=ppo.CategoricalPolicy(logits=logits), value=jnp.zeros(starts.shape)
        )  # carry: (B, H); logits: (B, A); value: (B,)

    state = TrainState.create(apply_fn=policy, params={"increment": jnp.asarray(1.0)}, tx=optax.sgd(0.1))
    memory = ((jnp.zeros((2, 1)), jnp.zeros((2, 1))),)  # leaves: (N=2, H=1)
    for team, expected_return in ((0, 2.0), (1, 3.0)):
        result = ppo.play_episode(
            state,
            config,
            jax.random.key(0),
            memory,
            record=False,
            env=env,
            opponent_params={"increment": jnp.asarray(0.0)},  # ()
            current_team=team,
        )
        assert result.episode_return == expected_return
        assert result.length == 2


@pytest.mark.parametrize(
    "returns,best_index",
    [([0.0], 0), ([-2.0, 3.0, 1.0], 1), ([0.0, 0.0], 0)],
    ids=["no-uncertainty", "best-on-opposite-side", "first-tied-zero"],
)
def test_historical_eval_metrics_and_artifacts(config: ppo.Config, returns: list[float], best_index: int) -> None:
    episodes = len(returns)
    config = replace(
        config, env=LuxConfig(), model=ppo.LuxModelConfig(recurrent=config.model.recurrent), eval_episodes=episodes
    )
    writer = Mock()
    memory = ((jnp.zeros((2, 1)), jnp.zeros((2, 1))),)  # leaves: (N=2, H=1)
    opponent = {"weight": jnp.asarray(7.0)}  # ()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)  # (H, W, C)
    results = [ppo.EpisodeResult(episode_return=value, length=2, frames=[]) for value in returns]
    # The capped replay must not contribute to score or episode-length statistics.
    results.append(ppo.EpisodeResult(episode_return=99.0, length=1, frames=[frame]))
    with patch.object(ppo, "play_episode", side_effect=results * len(ppo.EVAL_OPPONENT_AGES)) as play:
        for age in ppo.EVAL_OPPONENT_AGES:
            ppo.log_evaluation(
                Mock(),
                config,
                writer,
                5,
                1600,
                initial_carry=memory,
                env=Mock(),
                opponent_params=opponent,
                opponent_age=age,
            )
    for index, call in enumerate(play.call_args_list):
        position = index % (episodes + 1)
        recorded = position == episodes
        episode_index = best_index if recorded else position
        assert call.kwargs["opponent_params"] is opponent
        assert call.kwargs["current_team"] == episode_index % 2
        assert call.kwargs["record"] is recorded
        if recorded:
            assert call.kwargs["stop_at_frame_limit"] is True
        np.testing.assert_array_equal(
            jax.random.key_data(call.args[2]),
            jax.random.key_data(jax.random.fold_in(jax.random.key(config.eval_seed), episode_index)),
        )
    for age, call in zip(ppo.EVAL_OPPONENT_AGES, writer.log_metrics.call_args_list, strict=True):
        metrics, step = call.args
        assert step == 1600
        assert metrics[f"eval/return_mean/age_{age}"] == pytest.approx(sum(returns) / episodes)
        assert metrics[f"eval/training_episodes_total/age_{age}"] == 5
        assert (f"eval/return_std/age_{age}" in metrics) == (episodes > 1)
        assert (f"eval/return_sem/age_{age}" in metrics) == (episodes > 1)
        assert all(tag.endswith(f"/age_{age}") for tag in metrics)
    for call in writer.log_text.call_args_list:
        report = json.loads(call.args[0])
        assert report["video_episode_index"] == best_index
        assert report["episodes"] == [
            {"return": value, "length": 2, "current_team": index % 2} for index, value in enumerate(returns)
        ]
    assert [call.args[1] for call in writer.log_text.call_args_list] == [
        f"eval/report/age_{age}/1600.json" for age in ppo.EVAL_OPPONENT_AGES
    ]
    assert [call.args[1] for call in writer.log_video.call_args_list] == [
        f"eval/gameplay/age_{age}/1600.gif" for age in ppo.EVAL_OPPONENT_AGES
    ]


def test_training_evaluates_exact_historical_iterations(config: ppo.Config) -> None:
    config = replace(
        config,
        env=LuxConfig(),
        model=ppo.LuxModelConfig(recurrent=config.model.recurrent),
        total_env_steps=401 * 4,
        eval_interval=100,
        log_interval=1000,
    )
    model = Mock()
    model.init.return_value = {"params": {"weight": jnp.asarray(0.0)}}  # ()
    model.initial_carry.side_effect = lambda size: ((jnp.zeros((size, 1)), jnp.zeros((size, 1))),)  # (B, H=1)
    metrics = jax.tree.map(lambda leaf: leaf + 1, ppo.init_metrics())  # scalar leaves

    def train_step(
        runner: ppo.RolloutCarry,
        state: TrainState,
        env: ppo.PPOEnv,
        config: ppo.Config,
        key: jax.Array,  # ()
        old_metrics: ppo.TrainingMetrics,  # scalar leaves
        step: jax.Array,  # ()
    ) -> ppo.TrainStepResult:
        iteration = int(state.step) // 7 + 1
        return ppo.TrainStepResult(
            train_state=state.replace(step=iteration * 7, params={"weight": np.asarray(float(iteration))}),  # ()
            rollout_carry=runner,
            key=key,
            metrics=metrics,
        )

    with (
        patch.object(ppo, "make_env", return_value=ScriptedSMAX()),
        patch.object(ppo, "make_actor_critic", return_value=model),
        patch.object(ppo, "MLflowLogger"),
        patch.object(ppo, "checkpoint_manager"),
        patch.object(ppo, "restore_checkpoint", return_value=None),
        patch.object(ppo, "save_checkpoint", return_value=1.0),
        patch.object(ppo, "train_step_donated", side_effect=train_step),
        patch.object(ppo, "log_evaluation") as evaluate,
    ):
        ppo.train(config)
    expected = [
        (iteration, age)
        for iteration in (100, 200, 300, 400, 401)
        for age in ppo.EVAL_OPPONENT_AGES
        if age <= iteration
    ]
    assert len(evaluate.call_args_list) == len(expected)
    for call, (iteration, age) in zip(evaluate.call_args_list, expected, strict=True):
        assert call.args[4] == iteration * 4
        assert call.kwargs["opponent_age"] == age
        assert float(call.args[0].params["weight"]) == iteration
        assert float(call.kwargs["opponent_params"]["weight"]) == iteration - age


def test_historical_playback_pytree_observations_and_multidiscrete_actions() -> None:
    observation = AgentObservation(
        observation={"map": jnp.zeros((2, 2, 2), dtype=jnp.int16), "score": jnp.zeros((2, 1))},
        actions_mask=jnp.ones((2, 2, 3), dtype=bool),
    )  # observation leaves: (N=2, ...); mask: (N, D=2, A=3)
    memory = ((jnp.zeros((2, 1)), jnp.zeros((2, 1))),)  # leaves: (N=2, H=1)

    def policy(
        variables: dict[str, Any],
        observation: dict[str, jax.Array],  # leaves: (B=1, ...)
        memory: ppo.RecurrentCarry,  # leaves: (B, H=1)
        starts: jax.Array,  # (B,)
        mask: jax.Array,  # (B, D, A)
        *,
        method: str,
    ) -> ppo.ActorCriticOutput:
        assert observation["map"].shape == (1, 2, 2) and observation["map"].dtype == jnp.int16
        assert observation["score"].shape == (1, 1)
        logits = jnp.where(mask & (jnp.arange(3) == variables["params"]["action"]), 0.0, -jnp.inf)  # (B, D, A)
        memory = jax.tree.map(lambda leaf: leaf + variables["params"]["action"] + 1, memory)  # (B, H)
        return ppo.ActorCriticOutput(carry=memory, policy=MultiCategoricalPolicy(logits=logits), value=jnp.zeros(1))

    def step(key: jax.Array, old_state: jax.Array, action: jax.Array) -> MultiAgentStep:  # (); (N, D); (N, D)
        assert action.shape == (2, 2)
        return MultiAgentStep(
            observation=observation,
            state=action,
            reward=action.sum(axis=-1).astype(jnp.float32),
            done=jnp.asarray(True),
            agent_done=jnp.ones(2, dtype=bool),
            agent_inactive=jnp.zeros(2, dtype=bool),
            terminated=jnp.asarray(True),
            truncated=jnp.asarray(False),
            final_observation=observation,
            final_state=action,
        )  # observation: (N, ...); state: (N, D); reward/done flags: (N,) or ()

    env = Mock(num_agents=2)
    env.reset.return_value = observation, jnp.zeros((2, 2), dtype=jnp.int32)  # state: (N, D)
    env.step.side_effect = step
    result = ppo.run_episode(
        env,
        policy,
        ppo.EvaluationMatch(
            current={"action": jnp.asarray(0)}, opponent={"action": jnp.asarray(1)}, current_team=jnp.asarray(1)
        ),
        jax.random.key(0),
        memory,
        recording_capacity=2,
        stop_at_frame_limit=False,
    )
    assert result.states.shape == (2, 2, 2)
    np.testing.assert_array_equal(result.states[1], [[1, 1], [0, 0]])
    assert float(result.episode_return) == 0.0 and int(result.length) == 1
