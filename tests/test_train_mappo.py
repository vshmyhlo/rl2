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
    return replace(
        ppo.load_config(Path(__file__).resolve().parents[1] / "configs/mappo/smax.yaml"),
        num_envs=2,
        num_steps=2,
        num_minibatches=2,
        update_epochs=1,
        total_steps=8,
        gamma=0.5,
        gae_lambda=1.0,
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

    def step(self, key: jax.Array, state: State, actions: jax.Array) -> MultiAgentStep[State]:  # key: (); actions: (N,)
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
            jnp.full(2, step, jnp.float32),  # rewards: (N,)
            done,
            jnp.array([True, done]),
            terminated,
            done & ~terminated,
            self.observation(final),
            final,
        )


def scripted_policy(
    variables: dict[str, Any],
    observations: jax.Array,  # (B, O)
    memory: ppo.RecurrentCarry,  # leaves: (B, H)
    starts: jax.Array,  # (B,)
    masks: jax.Array,  # (B, A)
    *,
    method: str,
) -> ppo.ActorCriticOutput:
    def advance(leaf: jax.Array) -> jax.Array:  # (B, H) -> (B, H)
        return jnp.where(starts[:, None], 0, leaf) + 1  # (B, H)

    return ppo.ActorCriticOutput(
        jax.tree.map(advance, memory),
        ppo.CategoricalPolicy(jnp.where(masks, 0.0, -1e10)),
        observations[:, 0],
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
    chex.assert_shape(result.batch.observations, (2, 4, 2))
    chex.assert_shape(result.batch.action_masks, (2, 4, 3))
    chex.assert_shape((result.batch.actions, result.batch.values, result.batch.returns), (2, 4))
    np.testing.assert_array_equal(result.batch.observations[0, :, 1], [0, 1, 0, 1])
    np.testing.assert_array_equal(result.batch.actions, [[0, 1, 0, 1]] * 2)
    # Team reward survives death. Timeout adds gamma * V(final)=10, never V(reset)=0.
    np.testing.assert_allclose(result.batch.returns, [[7, 7, 2, 2], [12, 12, 2, 2]])
    np.testing.assert_array_equal(result.batch.episode_starts, [[True] * 4, [True, False, True, False]])
    for leaf in jax.tree.leaves(advanced.model_carry):  # each (B, H)
        np.testing.assert_array_equal(leaf[:, 0], [1, 2, 1, 2])
    assert int(result.episode_count) == 2
    assert float(result.episode_return_sum) == 6.0
    assert int(result.episode_length_sum) == 4
    np.testing.assert_array_equal(advanced.episode_returns, 0)
    np.testing.assert_array_equal(advanced.episode_starts, True)
    selected = result.batch.select_sequences(jnp.array([3, 0]))
    np.testing.assert_array_equal(selected.returns, [[2, 7], [2, 12]])
    chex.assert_shape(jax.tree.leaves(selected.model_carry), (2, 4))
    repeated, duplicate = ppo.collect_rollout(advanced, train_state, env, config, result.key)
    chex.assert_trees_all_equal(duplicate.batch.model_carry, advanced.model_carry)
    chex.assert_trees_all_equal(result.batch, duplicate.batch.replace(model_carry=memory))
    assert not np.array_equal(jax.random.key_data(result.key), jax.random.key_data(duplicate.key))
    np.testing.assert_array_equal(repeated.episode_step_counts, 0)


def test_local_model_mask_replay_and_shared_memory(config: ppo.Config) -> None:
    model = ppo.ActorCritic(ppo.DiscreteSpace(3), config.model)
    obs = jnp.arange(16, dtype=jnp.float32).reshape(2, 2, 4) / 16  # (T, B, O)
    starts = jnp.array([[True, True], [True, False]])  # (T, B)
    masks = jnp.array([[[True, False, True], [False, True, False]]] * 2)  # (T, B, A)
    memory = model.initial_carry(2)  # leaves: (B, H)
    assert len(memory) == config.model.recurrent.num_layers
    chex.assert_shape(jax.tree.leaves(memory), (2, 4))
    params = model.init(jax.random.key(0), obs, memory, starts, masks)
    chex.assert_shape(params["params"]["encoder"]["kernel"], (4, config.model.encoder.embedding_size))
    # Keep the tiny ReLU encoder active so the test exercises recurrent features.
    params["params"]["encoder"]["kernel"] = jnp.eye(4)  # (O, F)
    output = model.apply(params, obs, memory, starts, masks)
    first = model.apply(params, obs[0], memory, starts[0], masks[0], method="step")
    chex.assert_shape(first.policy.logits, (2, 3))
    chex.assert_shape(first.values, (2,))
    second = model.apply(params, obs[1], first.carry, starts[1], masks[1], method="step")
    chex.assert_trees_all_close(second.carry, output.carry, atol=1e-6)
    np.testing.assert_allclose(output.values, jnp.stack([first.values, second.values]), atol=1e-6)
    np.testing.assert_allclose(output.policy.logits, jnp.stack([first.policy.logits, second.policy.logits]), atol=1e-6)
    actions, log_probs = output.policy.sample_and_log_prob(seed=jax.random.key(2))  # each (T, B)
    assert bool(jnp.take_along_axis(masks, actions[..., None], axis=-1).all())
    assert bool(jnp.isfinite(output.policy.entropy()).all())
    np.testing.assert_allclose(output.policy.log_prob(actions), log_probs, atol=1e-6)
    chex.assert_shape(output.values, (2, 2))
    chex.assert_shape(output.policy.logits, (2, 2, 3))
    # Both heads depend on the shared encoder; changing only the value head leaves policy and memory intact.
    changed = dict(params["params"], encoder=jax.tree.map(jnp.zeros_like, params["params"]["encoder"]))
    no_features = model.apply({"params": changed}, obs, memory, starts, masks)
    assert not np.array_equal(output.policy.logits, no_features.policy.logits)
    assert not np.array_equal(output.values, no_features.values)
    changed = dict(params["params"], value_output=jax.tree.map(jnp.zeros_like, params["params"]["value_output"]))
    changed["value_output"]["bias"] = jnp.ones(1)  # (1,)
    new_output = model.apply({"params": changed}, obs, memory, starts, masks)
    chex.assert_trees_all_equal((output.policy, output.carry), (new_output.policy, new_output.carry))
    assert not np.array_equal(output.values, new_output.values)


def test_rejects_checkpoint_with_separate_networks(config: ppo.Config) -> None:
    manager = Mock()
    manager.latest_step.return_value = 1
    manager.restore.return_value.metadata = {"version": 1, "algorithm": "smax_ippo"}
    state = TrainState.create(apply_fn=scripted_policy, params={}, tx=optax.sgd(0.1))
    with pytest.raises(ValueError, match="shared encoder/LSTM requires a new run directory"):
        ppo.restore_checkpoint(manager, state, config)


@pytest.mark.parametrize(
    "env",
    [
        ppo.SMAXConfig(),
        ppo.JaxNavConfig(),
        ppo.MPEConfig(),
        ppo.MPEConfig(id="MPE_simple_reference_v3", num_agents=2),
    ],
    ids=["smax", "jaxnav", "spread", "reference"],
)
def test_restore_legacy_environment_config(
    config: ppo.Config, env: ppo.SMAXConfig | ppo.JaxNavConfig | ppo.MPEConfig
) -> None:
    config = replace(config, env=env)
    saved_config = asdict(config)
    saved_env = saved_config.pop("env")
    saved_config["env_id"] = saved_env.pop("id")
    env_key = {"SMAX": "smax", "JaxNav": "jaxnav"}.get(env.id, "mpe")
    saved_config[env_key] = saved_env
    state = TrainState.create(apply_fn=scripted_policy, params={}, tx=optax.sgd(0.1))
    key = jax.random.key(0)  # ()
    manager = Mock()
    manager.latest_step.return_value = 1
    manager.restore.return_value.metadata = {
        "version": 2,
        "algorithm": f"{env.id.lower()}_ippo",
        "config": saved_config,
        "iteration": 1,
        "completed_episodes": 2,
        "training_return_sum": 3.0,
        "training_length_sum": 4,
    }
    manager.restore.return_value.state = {"train_state": state, "key": jax.random.key_data(key)}
    restored = ppo.restore_checkpoint(manager, state, config)
    expected = ppo.TrainingProgress(
        train_state=state,
        key=key,  # ()
        rollout_count=1,
        episode_count=2,
        episode_return_sum=3.0,
        episode_length_sum=4,
    )
    chex.assert_trees_all_equal(restored, expected)
    with pytest.raises(ValueError, match="different environment settings"):
        ppo.restore_checkpoint(manager, state, replace(config, env=replace(env, max_steps=env.max_steps + 1)))


def test_real_smax_single_step_update_and_checkpoint(config: ppo.Config, tmp_path: Path) -> None:
    config = replace(config, num_envs=1, num_steps=1, num_minibatches=3, env=SMAXConfig(max_steps=1))
    env = ppo.make_env(config)  # Three agent sequences can form three minibatches in one environment.
    observation, state = jax.vmap(env.reset)(jax.random.split(jax.random.key(0), 1))
    model = ppo.ActorCritic(env.action_space, config.model)
    memory = model.initial_carry(3)  # leaves: (B=3, H)
    starts = jnp.ones(3, bool)  # (B,)
    params = model.init(jax.random.key(1), observation.observations, memory, starts[None], observation.action_masks)
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params["params"],
        tx=optax.inject_hyperparams(optax.adam)(learning_rate=config.learning_rate),
    )
    runner = ppo.RolloutCarry(state, observation, memory, starts, jnp.zeros(1), jnp.zeros(1, jnp.int32))
    result = jax.jit(ppo.train_step, static_argnames=("envs", "config"))(
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
    progress = ppo.TrainingProgress(result.train_state, result.key, 1, 1, episode_return, 1)
    with ppo.checkpoint_manager(str(tmp_path)) as manager:
        ppo.save_checkpoint(manager, progress, config)
        manager.wait_until_finished()
        restored = ppo.restore_checkpoint(manager, train_state, config)
        assert restored is not None
        chex.assert_trees_all_equal(restored, progress)
        with pytest.raises(ValueError, match="environment settings"):
            ppo.restore_checkpoint(manager, train_state, replace(config, env=SMAXConfig(map_name="8m")))


def test_playback_reuses_compiled_sampling_with_updated_inputs() -> None:
    memory = ((jnp.zeros((2, 1)), jnp.zeros((2, 1))),)  # leaves: (B=2, H=1)
    observations = jnp.zeros((2, 1))  # (B, O=1)
    starts = jnp.array([True, False])  # (B,)
    masks = jnp.array([[True, False, True], [False, True, False]])  # (B, A=3)
    params = {"logits": jnp.array([[100.0, 0.0, -100.0], [0.0, 0.0, 0.0]])}  # (B, A)

    def policy(
        variables: dict[str, dict[str, jax.Array]],
        observations: jax.Array,  # (B, O)
        carry: ppo.RecurrentCarry,  # leaves: (B, H)
        episode_starts: jax.Array,  # (B,)
        action_masks: jax.Array,  # (B, A)
        *,
        method: str,
    ) -> ppo.ActorCriticOutput:
        carry = jax.tree.map(lambda leaf: jnp.where(episode_starts[:, None], 0, leaf) + 1, carry)  # (B, H)
        logits = jnp.where(action_masks, variables["params"]["logits"], -1e10)  # (B, A)
        return ppo.ActorCriticOutput(carry, ppo.CategoricalPolicy(logits), jnp.zeros(2))  # (B, H); (B, A); (B,)

    with patch.object(
        ppo.CategoricalPolicy, "sample", autospec=True, side_effect=ppo.CategoricalPolicy.sample
    ) as sample:
        memory, actions = ppo.sample_policy_step(policy, params, observations, memory, starts, jax.random.key(0), masks)
        chex.assert_shape(actions, (2,))
        np.testing.assert_array_equal(actions, [0, 1])
        params = {"logits": -params["logits"]}  # (B, A)
        masks = masks.at[1].set(jnp.array([True, False, False]))  # (B, A)
        memory, actions = ppo.sample_policy_step(policy, params, observations, memory, starts, jax.random.key(1), masks)
        np.testing.assert_array_equal(actions, [2, 0])
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
    assert result.observation.observations.shape == (2, 2)
    assert result.observation.action_masks.shape == (2, 3)
    for leaf in jax.tree.leaves(result.memory):  # (N, H)
        np.testing.assert_array_equal(leaf, [[1.0], [2.0]])
    np.testing.assert_array_equal(result.episode_starts, [True, True])
    assert int(result.env_state.state.step) == 0 and int(result.final_state.state.step) == 2


def test_config_and_cli(config: ppo.Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "smax.yaml"
    path.write_text(yaml.safe_dump(asdict(config)))
    assert ppo.load_config(path) == config
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
        config, video_interval=interval, eval_interval=interval, checkpoint_interval=interval or 3, log_interval=1
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
        )

    def train_step(
        runner: ppo.RolloutCarry,
        state: TrainState,
        env: SMAXEnv,
        config: ppo.Config,
        key: jax.Array,
        metrics: ppo.TrainingMetrics,
        iteration: int,  # key: (); metrics: scalar leaves
    ) -> ppo.TrainStepResult:
        assert iteration == 1
        chex.assert_shape(runner.observations.observations, (2, 2, 2))
        chex.assert_shape(jax.tree.leaves(runner.model_carry), (4, 4))
        np.testing.assert_array_equal(runner.episode_starts, True)
        np.testing.assert_array_equal(runner.episode_returns, 0)
        metrics = jax.tree.map(lambda x: x + 1, metrics)  # scalar leaves
        return ppo.TrainStepResult(
            train_state=state.replace(step=state.step + 1),
            rollout_carry=runner,
            key=key,  # ()
            metrics=metrics,
        )

    monkeypatch.setattr(ppo, "restore_checkpoint", restore)
    monkeypatch.setattr(ppo, "train_step_donated", train_step)
    state = ppo.train(config)
    for logger in (video, evaluation):
        if not interval:
            logger.assert_not_called()
            continue
        logger.assert_called_once()
        assert logger.call_args.args[0] is state
        assert logger.call_args.args[3:] == (2, 8)
        chex.assert_shape(jax.tree.leaves(logger.call_args.kwargs["initial_carry"]), (2, 4))
        assert logger.call_args.kwargs["env"] is env
    scalars = {tag: values[-1] for tag, values in metric_history(writer).items()}
    assert scalars["charts/total_environment_steps"] == (8.0, 8)
    assert scalars["charts/total_observations"] == (16.0, 8)
    assert scalars["charts/total_episodes"] == (2.0, 8)
    assert scalars["charts/total_iterations"] == (2.0, 8)
    assert scalars["charts/total_updates"] == (6.0, 8)
    assert scalars["charts/training_progress"] == (1.0, 8)
    assert scalars["time/checkpoint_seconds"] == (2.5, 8)
    save.assert_called_once()
    assert save.call_args.args[1].train_state is state
    assert save.call_args.args[1].rollout_count == 2
    writer.__exit__.assert_called_once()
    manager.close.assert_called_once()
