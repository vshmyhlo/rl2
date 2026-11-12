"""Iteration-based map curricula and state continuity across stages."""

from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tokamax  # noqa: F401 - load Triton before environment dependencies
import yaml
from flax.training.train_state import TrainState
from ppo_helpers import metric_history
from test_train_mappo import ScriptedSMAX
from test_train_mappo import config as config  # noqa: PLC0414

from rl2 import train_mappo as ppo

pytestmark = pytest.mark.usefixtures("ppo_mesh")


def test_curriculum_loading_and_thresholds(config: ppo.Config, tmp_path: Path) -> None:
    settings = asdict(replace(config, env=ppo.LuxConfig(), model=ppo.LuxModelConfig()))
    settings["map_size_curriculum"] = [
        {"start_iteration": 0, "map_size": 8},
        {"start_iteration": 500, "map_size": 16},
        {"start_iteration": 2500, "map_size": 24},
    ]
    path = tmp_path / "curriculum.yaml"
    path.write_text(yaml.safe_dump(settings))
    loaded = ppo.load_config(path, ppo.Config)
    assert hash(loaded)
    for iteration, size in [(0, 8), (499, 8), (500, 16), (2499, 16), (2500, 24), (9999, 24)]:
        active = ppo.config_for_iteration(loaded, iteration)
        assert active.env.map_size == size
        assert active.map_size_curriculum == loaded.map_size_curriculum
    assert loaded.env.map_size == 24
    assert ppo.config_for_iteration(config, 500) is config
    fixed = replace(loaded, map_size_curriculum=())
    assert ppo.config_for_iteration(fixed, 500) is fixed


@pytest.mark.parametrize(
    "stages,match",
    [
        ((ppo.MapSizeStage(start_iteration=1, map_size=8),), "start at iteration 0"),
        ((ppo.MapSizeStage(start_iteration=0, map_size=8),) * 2, "strictly increasing"),
        (
            tuple(ppo.MapSizeStage(start_iteration=i, map_size=8) for i in (0, 2, 1)),
            "strictly increasing",
        ),
        ((ppo.MapSizeStage(start_iteration=0, map_size=9),), "multiple_of"),
    ],
    ids=["missing-initial-stage", "duplicate-threshold", "unordered-thresholds", "invalid-lux-size"],
)
def test_invalid_curriculum(config: ppo.Config, stages: tuple[ppo.MapSizeStage, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        replace(config, env=ppo.LuxConfig(), model=ppo.LuxModelConfig(), map_size_curriculum=stages)


@pytest.mark.parametrize("iteration,size", [(-1, 8), (True, 8), (0, 0), (0, 8.5)])
def test_invalid_stage(iteration: int, size: float) -> None:
    with pytest.raises(ValueError):
        ppo.MapSizeStage(start_iteration=iteration, map_size=size)


def test_curriculum_environment_support(config: ppo.Config) -> None:
    stages = (ppo.MapSizeStage(start_iteration=0, map_size=5),)
    with pytest.raises(ValueError, match="requires a Lux or JaxNav"):
        replace(config, map_size_curriculum=stages)
    nav = replace(config, env=ppo.JaxNavConfig(), map_size_curriculum=stages)
    assert ppo.make_env(ppo.config_for_iteration(nav, 0)).config.map_size == 5
    with pytest.raises(ValueError, match="map_size"):
        replace(nav, map_size_curriculum=(ppo.MapSizeStage(start_iteration=0, map_size=4),))


def test_lux_curriculum_reset_and_parameter_shapes(config: ppo.Config) -> None:
    config = replace(
        config,
        env=ppo.LuxConfig(map_size=8, max_units=1, max_steps_in_match=2, match_count=1),
        model=ppo.LuxModelConfig(conv_channels=2, heads=1, head_dim=2, num_blocks=1, attention_implementation="xla"),
        num_envs=1,
        num_minibatches=1,
    )
    env = ppo.make_env(config)
    model = ppo.make_actor_critic(env.action_space, config.model)
    carry = ppo.init_rollout_carry(env, model, config, jax.random.key(0))  # leaves: (E=1, ...) or (B=2, ...)
    sequence = jax.tree.map(lambda leaf: leaf[0, :, None], carry.observation)  # leaves: (B=2, T=1, ...)
    starts = carry.episode_start[:, None]  # (B=2, T=1)
    params = jax.eval_shape(
        model.init, jax.random.key(1), sequence.observation, carry.model_carry, starts, sequence.actions_mask
    )  # leaves: parameter shapes
    larger = replace(config, env=replace(config.env, map_size=16))
    carry = ppo.init_rollout_carry(
        ppo.make_env(larger), model, larger, jax.random.key(2)
    )  # leaves: (E=1, ...) or (B=2, ...)
    assert carry.observation.observation.world.sensor_mask.shape == (1, 2, 16, 16)
    assert carry.observation.actions_mask.shape == (1, 2, 1, 230)
    chex.assert_shape(jax.tree.leaves(carry.model_carry), (2, 2))
    np.testing.assert_array_equal(carry.episode_start, True)
    np.testing.assert_array_equal(carry.episode_return, 0)
    np.testing.assert_array_equal(carry.episode_step_count, 0)
    sequence = jax.tree.map(lambda leaf: leaf[0, :, None], carry.observation)  # leaves: (B=2, T=1, ...)
    output, memory = jax.eval_shape(
        model.apply, params, sequence.observation, carry.model_carry, starts, sequence.actions_mask
    )
    assert output.value.shape == (2, 1)
    assert output.policy.raw_logits.shape == (2, 1, 1, 232)
    chex.assert_shape(jax.tree.leaves(memory), (2, 2))


@pytest.mark.parametrize("resume_iteration,pool_enabled", [(0, True), (2, False)], ids=["transition-pool", "resume"])
def test_curriculum_training_wiring(config: ppo.Config, resume_iteration: int, pool_enabled: bool) -> None:
    config = replace(
        config,
        env=ppo.LuxConfig(),
        model=ppo.LuxModelConfig(),
        map_size_curriculum=(
            ppo.MapSizeStage(start_iteration=0, map_size=8),
            ppo.MapSizeStage(start_iteration=2, map_size=16),
        ),
        selfplay=ppo.SelfPlayConfig(pool_size=2, current_policy_fraction=0.0) if pool_enabled else None,
        total_iterations=4,
        log_interval=1,
        video_interval=1,
        eval_interval=1,
        eval_ref_policies={"baseline": "/unused"},
    )
    model, writer = Mock(), MagicMock()
    model.init.return_value = {"params": {"weight": jnp.asarray(1.0)}}  # ()

    def initial_carry(size: int) -> ppo.RecurrentCarry:
        return ((jnp.zeros((size, 1)), jnp.zeros((size, 1))),)  # leaves: (B, H=1)

    model.initial_carry.side_effect = initial_carry
    seen_sizes: list[int] = []
    pools: list[ppo.OpponentPool] = []

    def restore(manager: Mock, state: TrainState, current: ppo.Config) -> ppo.TrainingProgress | None:
        if not resume_iteration:
            return None
        return ppo.TrainingProgress(
            train_state=state.replace(step=resume_iteration * 7),
            key=jax.random.key(9),  # ()
            rollout_count=resume_iteration,
            episode_count=5,
            episode_return_sum=10.0,
            episode_length_sum=20,
        )

    def train_step(
        runner: ppo.RolloutCarry,
        state: TrainState,
        env: ppo.PPOEnv,
        active: ppo.Config,
        key: jax.Array,  # ()
        metrics: ppo.TrainingMetrics,  # scalar leaves
        step: jax.Array,  # ()
        pool: ppo.OpponentPool | None = None,
    ) -> ppo.TrainStepResult:
        iteration = int(step)
        assert int(state.step) == iteration * 7
        assert float(state.params["weight"]) == 1.0
        seen_sizes.append(active.env.map_size)
        fresh = iteration in (resume_iteration, 2)
        np.testing.assert_array_equal(runner.episode_return, 0 if fresh else 99)
        np.testing.assert_array_equal(runner.episode_step_count, 0 if fresh else 99)
        np.testing.assert_array_equal(runner.episode_start, fresh)
        for leaf in jax.tree.leaves(runner.model_carry):
            np.testing.assert_array_equal(leaf, 0 if fresh else 99)
        if pool is not None:
            pools.append(pool)
            np.testing.assert_array_equal(runner.selfplay.opponent, 0 if fresh else -1)
        with jax.enable_x64():
            metrics = jax.tree.map(lambda leaf: leaf + 1, metrics)  # scalar leaves
        dirty = runner._replace(
            episode_return=jnp.full(2, 99.0),  # (E,)
            episode_step_count=jnp.full(2, 99),  # (E,)
            episode_start=jnp.zeros(4, bool),  # (B,)
            model_carry=jax.tree.map(lambda leaf: leaf + 99, runner.model_carry),  # leaves: (B, H=1)
            selfplay=None if pool is None else runner.selfplay._replace(opponent=jnp.full(2, -1)),  # (E,)
        )
        return ppo.TrainStepResult(
            train_state=state.replace(step=state.step + 7), rollout_carry=dirty, key=key, metrics=metrics
        )

    with (
        patch.object(ppo, "configure_compilation_cache"),
        patch.object(ppo, "make_env", side_effect=lambda _: ScriptedSMAX()) as make_env,
        patch.object(ppo, "make_actor_critic", return_value=model),
        patch.object(ppo, "MLflowLogger", return_value=writer),
        patch.object(ppo, "checkpoint_manager"),
        patch.object(ppo, "restore_checkpoint", side_effect=restore),
        patch.object(ppo, "save_checkpoint", return_value=0.1) as save,
        patch.object(ppo, "train_step_donated", side_effect=train_step),
        patch.object(ppo, "train_step_with_pool_donated", side_effect=train_step),
        patch.object(ppo, "load_eval_ref_policies", return_value={"baseline": {}}),
        patch.object(ppo, "log_video") as video,
        patch.object(ppo, "log_evaluation") as evaluate,
    ):
        result = ppo.train(config)
    assert seen_sizes == [8, 8, 16, 16][resume_iteration:]
    assert [call.args[0].env.map_size for call in make_env.call_args_list] == [8, 16]
    assert model.init.call_count == 1
    assert int(result.step) == 28
    for logger in (video, evaluate):
        assert [call.args[1].env.map_size for call in logger.call_args_list] == seen_sizes
    assert metric_history(writer)["train/map_size"] == [
        (float(size), (index + resume_iteration + 1) * 4) for index, size in enumerate(seen_sizes)
    ]
    progress, saved_config = save.call_args.args[1:]
    assert saved_config is config
    assert progress.rollout_count == 4 and progress.environment_step_count == 16
    assert progress.episode_count == (5 if resume_iteration else 0) + len(seen_sizes)
    if pool_enabled:
        assert all(pool is pools[0] for pool in pools)
        assert progress.opponent_pool is pools[0]
