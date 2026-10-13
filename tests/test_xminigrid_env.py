from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import xminigrid
from flax.training.train_state import TrainState
from xminigrid.benchmarks import Benchmark
from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
from xminigrid.core.goals import AgentOnTileGoal
from xminigrid.core.rules import EmptyRule
from xminigrid.environment import EnvParams
from xminigrid.types import TimeStep

from rl2 import ppo
from rl2.envs.xminigrid_env import XMiniGridConfig, XMiniGridEnv


def test_full_render_uses_grid_and_agent_pose() -> None:
    env = XMiniGridEnv(XMiniGridConfig())
    observation, state = env.reset(jax.random.key(0), env.params)  # (7,7,2); timestep leaves: (...)
    frame = env.render(state)  # (160,160,3), from the full 5x5 world
    assert observation.shape == (7, 7, 2)
    assert frame.shape == (160, 160, 3) and frame.dtype == np.uint8
    np.testing.assert_array_equal(frame, env.env.render(env.params, state))
    agent = state.state.agent.replace(
        position=jnp.array([3, 3]), direction=jnp.array(1)
    )  # position: (2,); direction: ()
    moved = state.replace(state=state.state.replace(agent=agent))  # timestep leaves: (...)
    # An unchanged policy observation must not hide a changed world/agent state.
    assert not np.array_equal(frame, env.render(moved))
    np.testing.assert_array_equal(env.render(moved), env.env.render(env.params, moved))


def test_terminal_timeout_both_and_partial_autoreset() -> None:
    env = XMiniGridEnv(XMiniGridConfig(max_steps=2))
    key = jax.random.key(0)  # ()
    observations, initial = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(key, 4), env.params
    )  # (B=4,7,7,2); leaves: (B,...)
    chex.assert_trees_all_equal(
        (observations, initial), jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(key, 4), env.params)
    )
    assert observations.shape == (4, 7, 7, 2) and observations.dtype == jnp.uint8
    assert env.action_space.n == 6
    # Move right into the goal, timeout, both, and continuing, respectively.
    agent = initial.state.agent.replace(position=jnp.array([[3, 2], [1, 1], [3, 2], [1, 1]]))  # leaves: (B,...)
    state = initial.replace(state=initial.state.replace(agent=agent, step_num=jnp.array([0, 1, 1, 0])))
    key = jax.random.key(1)  # ()
    expected = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        jax.random.split(key, 4), state, jnp.zeros(4, jnp.int32), env.params
    )  # observation: (B, H, W, 2); scalar leaves: (B,); state leaves: (B, ...)
    result = env.step_batch(jax.random.split(key, 4), state, jnp.zeros(4, jnp.int32), env.params)
    chex.assert_trees_all_equal(result, expected)
    next_observations, next_state, rewards, done, info = result
    np.testing.assert_array_equal(info["terminated"], [True, False, True, False])
    np.testing.assert_array_equal(info["truncated"], [False, True, True, False])
    np.testing.assert_array_equal(done, [True, True, True, False])
    np.testing.assert_array_equal(next_state.state.step_num, [0, 0, 0, 1])
    np.testing.assert_array_equal(info["final_state"].state.step_num, [1, 2, 2, 1])
    np.testing.assert_array_equal(info["final_state"].observation, info["final_observation"])
    np.testing.assert_allclose(rewards, [0.55, 0, 0.1, 0], atol=1e-6)
    assert next_observations.shape == info["final_observation"].shape == observations.shape
    chex.assert_shape((rewards, done, info["terminated"], info["truncated"]), (4,))
    np.testing.assert_array_equal(next_observations, next_state.observation)
    reset_observations, reset_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(key, 4), env.params
    )  # (B,7,7,2); leaves: (B,...)
    np.testing.assert_array_equal(next_observations[:3], reset_observations[:3])

    def completed_rows(array: jax.Array) -> jax.Array:  # (B, ...) -> (3, ...)
        return array[:3]  # (3, ...)

    actual, expected = jax.tree.map(completed_rows, (next_state, reset_state))  # leaves: (3, ...)
    chex.assert_trees_all_equal(actual, expected)
    np.testing.assert_array_equal(next_observations[3], info["final_observation"][3])
    assert not np.array_equal(next_observations[0], info["final_observation"][0])
    next_observations, next_state, rewards, done, info = env.step_batch(
        jax.random.split(key, 4), initial, jnp.zeros(4, jnp.int32), env.params
    )  # (B,H,W,2); leaves: (B,...); rewards/done/flags: (B,); final_observation: (B,H,W,2)
    np.testing.assert_array_equal(next_state.state.step_num, 1)
    np.testing.assert_array_equal(next_observations, info["final_observation"])


def test_batch_step_skips_reset_work_until_an_episode_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    env = XMiniGridEnv(XMiniGridConfig(max_steps=2))
    keys = jax.random.split(jax.random.key(0), 2)  # (B,) keys
    _, initial = jax.vmap(env.reset, in_axes=(0, None))(keys, env.params)  # (B, H, W, 2); leaves: (B, ...)
    calls: list[None] = []
    original_reset = env.reset

    def record_reset() -> None:
        calls.append(None)

    def counted_reset(key: jax.Array, params: EnvParams) -> tuple[jax.Array, TimeStep]:  # key: ()
        jax.debug.callback(record_reset)
        return original_reset(key, params)  # (H, W, 2); leaves: (...)

    monkeypatch.setattr(env, "reset", counted_reset)
    actions = jnp.ones(2, jnp.int32)  # (B,) turn, without reaching a goal
    first = env.step_batch(keys, initial, actions, env.params)
    jax.block_until_ready(first)
    jax.effects_barrier()
    assert not calls
    np.testing.assert_array_equal(first.done, False)
    second = env.step_batch(keys, first.state, actions, env.params)
    jax.block_until_ready(second)
    jax.effects_barrier()
    assert len(calls) == 1
    np.testing.assert_array_equal(second.done, True)
    np.testing.assert_array_equal(second.state.state.step_num, 0)


@pytest.fixture
def benchmark(monkeypatch: pytest.MonkeyPatch) -> Benchmark:
    tiles = jnp.stack([TILES_REGISTRY[Tiles.BALL, Colors.RED], TILES_REGISTRY[Tiles.BALL, Colors.BLUE]])  # (N=2,2)
    goals = jnp.stack([AgentOnTileGoal(tile=tile).encode() for tile in tiles])  # (N,G)
    rules = jnp.broadcast_to(EmptyRule().encode(), (2, 1, 7))  # (N,R=1,7)
    benchmark = Benchmark(goals=goals, rules=rules, init_tiles=tiles[:, None], num_rules=jnp.zeros(2, jnp.int32))

    # Benchmark leaves: (N,...) with N=2 tasks; no network downloads in tests.
    def load_benchmark(name: str) -> Benchmark:
        return benchmark  # leaves: (N, ...)

    monkeypatch.setattr(xminigrid, "load_benchmark", load_benchmark)
    return benchmark


@pytest.mark.parametrize("ruleset_id", [None, 0], ids=["sample-on-reset", "fixed-task"])
def test_xland_rulesets_are_seeded_and_survive_transitions(benchmark: Benchmark, ruleset_id: int | None) -> None:
    env = XMiniGridEnv(
        XMiniGridConfig(env_id="XLand-MiniGrid-R1-9x9", benchmark="trivial-21k", ruleset_id=ruleset_id, max_steps=2),
    )
    key = jax.random.key(1)  # (); this batch samples both tasks
    observations, state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(key, 4), env.params
    )  # (B=4,5,5,2); leaves: (B,...)
    assert observations.shape == (4, 5, 5, 2)
    chex.assert_trees_all_equal(
        (observations, state), jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(key, 4), env.params)
    )
    if ruleset_id is None:
        assert np.unique(state.state.goal_encoding, axis=0).shape[0] == 2
    else:
        np.testing.assert_array_equal(state.state.goal_encoding, np.broadcast_to(benchmark.goals[0], (4, 5)))
    next_observations, next_state, _, done, info = env.step_batch(
        jax.random.split(jax.random.key(1), 4), state, jnp.ones(4, jnp.int32), env.params
    )  # (B,H,W,2); leaves: (B,...); rewards/done/flags: (B,); final_observation: (B,H,W,2)
    np.testing.assert_array_equal(next_state.state.goal_encoding, state.state.goal_encoding)
    np.testing.assert_array_equal(next_state.state.rule_encoding, state.state.rule_encoding)
    reset_key = jax.random.key(2)  # ()
    next_observations, next_state, _, done, info = env.step_batch(
        jax.random.split(reset_key, 4), next_state, jnp.ones(4, jnp.int32), env.params
    )  # (B,H,W,2); leaves: (B,...); rewards/done/flags: (B,); final_observation: (B,H,W,2)
    np.testing.assert_array_equal(info["truncated"], True)
    np.testing.assert_array_equal(done, True)
    chex.assert_trees_all_equal(
        (next_observations, next_state),
        jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(reset_key, 4), env.params),
    )


def test_ruleset_index_is_checked_before_jax_indexing(benchmark: Benchmark) -> None:
    with pytest.raises(ValueError, match="ruleset_id"):
        XMiniGridEnv(XMiniGridConfig(env_id="XLand-MiniGrid-R1-9x9", benchmark="trivial-21k", ruleset_id=2))


def test_real_rollout_update_and_video() -> None:
    config = replace(
        ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo/xminigrid.yaml"),
        xminigrid=XMiniGridConfig(max_steps=2),
        num_envs=1,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        target_kl=None,
        bf16=False,
        model=ppo.ModelConfig(
            encoder=ppo.XMiniGridEncoderConfig(token_size=2, channels=2, blocks=0, embedding_size=2),
            recurrent=ppo.LSTMConfig(hidden_size=2),
        ),
    )
    env = ppo.make_env(config)
    assert isinstance(env, XMiniGridEnv)
    observations, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(jax.random.key(0), 1), env.params
    )  # (1,7,7,2); leaves: (1,...)
    model = ppo.ActorCritic(env.action_space, config.model)
    carry = model.initial_carry(1)  # leaves: (1,2)
    starts = jnp.ones(1, bool)  # (1,)
    params = model.init(jax.random.key(1), observations[None], carry, starts[None])["params"]  # parameter pytree
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    runner = ppo.RolloutCarry(env_state, observations, carry, starts, jnp.zeros(1), jnp.zeros(1, jnp.int32))
    advanced, rollout = ppo.collect_rollout(runner, state, env, config, jax.random.key(2))
    assert rollout.batch.observations.shape == (2, 1, 7, 7, 2)
    chex.assert_shape((rollout.batch.actions, rollout.batch.returns), (2, 1))
    chex.assert_shape((rollout.episode_count, rollout.episode_return_sum, rollout.episode_length_sum), ())
    assert int(rollout.episode_count) == 1
    assert int(rollout.episode_length_sum) == 2
    np.testing.assert_array_equal(advanced.env_state.state.step_num, 0)
    _, logits, values = model.apply(
        {"params": params}, rollout.batch.observations, rollout.batch.model_carry, rollout.batch.episode_starts
    )  # carry leaves: (1,2); (T=2,B=1,A); (T=2,B=1)
    logits = logits.logits  # (..., A)
    assert logits.shape == (2, 1, env.action_space.n)
    np.testing.assert_allclose(values, rollout.batch.values, atol=1e-5)
    updated, metrics = ppo.update(state, rollout.batch, config)
    assert int(updated.step) == 1
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves((updated.params, metrics)))
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(state.params), jax.tree.leaves(updated.params)))
    score, length, frames = ppo.play_episode(state, config, jax.random.key(3), carry, record=True, env=env)
    assert np.isfinite(score) and length == 2 and len(frames) == 3
    assert all(frame.shape == (160, 160, 3) and frame.dtype == np.uint8 for frame in frames)


@pytest.mark.parametrize(
    "config,match",
    [
        (XMiniGridConfig(env_id="MiniGrid-Unknown"), "Unknown XLand-MiniGrid environment"),
        (
            XMiniGridConfig(env_id="XLand-MiniGrid-R1-9x9", benchmark="unknown"),
            "Unknown XLand-MiniGrid benchmark",
        ),
    ],
)
def test_unknown_environment_or_benchmark(config: XMiniGridConfig, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        XMiniGridEnv(config)
