from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml
from flax.training.train_state import TrainState

from rl2 import train_karel_ppo_ast_ar_edit as ppo
from rl2.karel import KarelConfig
from rl2.karel_ast import ACTION_ID
from rl2.shape_checker import ShapeChecker


@pytest.fixture(scope="module")
def config() -> ppo.Config:
    return ppo.Config(
        total_updates=1,
        num_tasks=2,
        group_size=1,
        num_minibatches=1,
        update_epochs=1,
        d_model=16,
        num_layers=1,
        num_heads=2,
        max_nodes=8,
        max_depth=4,
        max_seq_len=9,
        target_kl=None,
        entropy_coef=0.01,
        log_interval=1,
        log_program_interval=1,
        env=KarelConfig(height=3, width=3, max_depth=0, max_statements=1, max_program_tokens=8),
    )


@pytest.fixture(scope="module")
def state(config: ppo.Config) -> TrainState:
    grids = np.zeros((1, 3, 3, 6), np.int32)
    state = ppo.create_state(config, grids, grids)
    params = dict(state.params)
    # Nonuniform logits make cached/replayed policy agreement meaningful.
    params["head"] = {
        **params["head"],
        "kernel": 0.1 * jax.random.normal(jax.random.key(9), params["head"]["kernel"].shape),
    }
    return state.replace(params=params)


@pytest.fixture(scope="module")
def rollout(config: ppo.Config, state: TrainState) -> ppo.Rollout:
    # STOP immediately in one episode; spend the exact sequence budget on two
    # complete edits in the other. This also exercises padding after STOP.
    script = [3, 1 + config.max_nodes + ACTION_ID["turnRight"], 3, 1 + config.max_nodes + ACTION_ID["putMarker"]]
    offset = 0

    def act(logits: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        nonlocal offset
        sc = ShapeChecker(B=2, V=1 + config.max_nodes + len(ppo.AST_ACTIONS))
        sc.check(logits, "BV", dtype=jnp.float32)
        sc.check(key, "")
        assert jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
        actions = jnp.asarray([0, script[offset]], jnp.int32)
        assert np.all(np.isfinite(np.asarray(logits)[np.arange(2), actions]))
        offset += 1
        return actions, ppo.edit.action_log_prob(logits, actions)

    with patch.object(ppo, "act", act), ppo.KarelASTEditVectorEnv(config.edit_config, 2) as envs:
        rollout = ppo.collect_rollout(state, envs, np.random.default_rng(4), jax.random.key(7), config)
    assert offset == len(script)
    return rollout


def test_cached_rollout_replays_values_and_updates_both_heads(
    config: ppo.Config, state: TrainState, rollout: ppo.Rollout
) -> None:
    batch = rollout.batch
    _, logits, values = state.apply_fn({"params": state.params}, batch.history)
    replay = ppo.edit.action_log_prob(ppo.mask_logits(logits, batch.legal), batch.actions)
    np.testing.assert_allclose(replay, batch.old_log_probs, atol=2e-6)
    np.testing.assert_allclose(np.asarray(values)[batch.mask], batch.values[batch.mask], atol=2e-6)
    np.testing.assert_array_equal(batch.mask.sum(axis=0), [1, 4])
    assert batch.actions[config.edit_config.prefill_length - 1, 0] == 0
    assert rollout.diagnostics["charts/sequence_budget_exhausted_rate"] == 0.5
    np.testing.assert_array_equal(batch.returns[~batch.mask], 0)
    np.testing.assert_array_equal(batch.advantages[~batch.mask], 0)
    np.testing.assert_array_equal(batch.values[~batch.mask], 0)
    np.testing.assert_allclose(batch.rewards.sum(axis=0), rollout.rewards)
    for index in range(2):
        last = np.flatnonzero(batch.mask[:, index])[-1]
        assert batch.returns[last, index] == pytest.approx(batch.rewards[last, index], abs=1e-6)
    updated, metrics = ppo.update(state, batch, config)
    assert int(updated.step) == int(state.step) + 1
    assert np.all(np.isfinite(metrics)) and float(metrics[3]) < 1e-6
    for name in ("head", "value_head", "backbone"):
        assert any(
            not np.array_equal(before, after)
            for before, after in zip(
                jax.tree.leaves(state.params[name]), jax.tree.leaves(updated.params[name]), strict=True
            )
        )
    selected = ppo.select_episodes(batch, np.array([1, 0], np.int64))
    for original, reordered in zip(batch[1:], selected[1:], strict=True):
        np.testing.assert_array_equal(reordered, original[:, [1, 0]])
    np.testing.assert_array_equal(selected.history.events.feedback, batch.history.events.feedback[:, [1, 0]])
    np.testing.assert_array_equal(selected.history.events.grid, batch.history.events.grid[:, [1, 0]])


def test_gae_discounts_decisions_and_stops_at_episode_boundaries() -> None:
    mask = np.array([[False, False], [True, True], [True, False], [True, False], [False, False]])
    rewards = np.full((5, 2), np.nan, np.float32)
    values = np.full((5, 2), np.nan, np.float32)
    rewards[1:4, 0], values[1:4, 0] = [0, 2, -1], [0.5, 1.5, 2]
    rewards[1, 1], values[1, 1] = 4, 1
    advantages, returns = ppo.gae(rewards, values, mask, 0.5, 0.5)
    np.testing.assert_allclose(advantages, [[0, 0], [0.4375, 3], [0.75, 0], [-3, 0], [0, 0]])
    np.testing.assert_allclose(returns, [[0, 0], [0.9375, 4], [2.25, 0], [-1, 0], [0, 0]])
    # Undiscounted lambda=1 recovers exact return-to-go, not one episode-wide advantage.
    _, returns = ppo.gae(rewards, values, mask, 1.0, 1.0)
    np.testing.assert_allclose(returns, [[0, 0], [1, 4], [1, 0], [-1, 0], [0, 0]])
    # Zero discount is a separate boundary: no credit crosses a decision.
    _, returns = ppo.gae(rewards, values, mask, 0.0, 0.5)
    np.testing.assert_allclose(returns, np.where(mask, rewards, 0))


@pytest.fixture
def synthetic(config: ppo.Config) -> tuple[TrainState, ppo.EditBatch]:
    shape = (3, 2)
    vocab = 1 + config.max_nodes + len(ppo.AST_ACTIONS)
    events = ppo.empty_events(*shape, config)
    legal = np.zeros((*shape, vocab), bool)
    legal[..., 0] = True
    legal[:2, :, 1] = True
    mask = np.array([[True, True], [True, True], [False, False]])
    probabilities = np.array([[0.25, 0.75], [0.75, 0.25], [0.5, 0.5]], np.float32)
    logits = np.zeros((*shape, vocab), np.float32)
    logits[..., 0], logits[..., 1] = np.log(probabilities), np.log(1 - probabilities)
    batch = ppo.EditBatch(
        ppo.History(events),
        np.zeros(shape, np.int32),
        np.full(shape, np.log(0.5), np.float32),
        legal,
        mask,
        np.zeros(shape, np.float32),
        np.ones(shape, np.float32),
        np.array([[-3, -1], [1, 3], [np.nan, np.nan]], np.float32),
        np.zeros(shape, np.float32),
    )

    def apply(variables: dict[str, optax.Params], history: ppo.History) -> tuple[None, jax.Array, jax.Array]:
        ppo.check_history(history)
        return None, variables["params"]["logits"], variables["params"]["values"]

    state = TrainState.create(
        apply_fn=apply, params={"logits": jnp.asarray(logits), "values": jnp.ones(shape)}, tx=optax.sgd(0.1)
    )
    return state, batch


def test_ppo_loss_clipping_normalization_and_padding(
    config: ppo.Config, synthetic: tuple[TrainState, ppo.EditBatch]
) -> None:
    state, batch = synthetic
    config = replace(config, entropy_coef=0)
    logits, values = state.params["logits"], state.params["values"]
    loss, metrics = ppo.objective(logits, values, batch, config)
    advantages = np.array([[-3, -1], [1, 3]]) / np.sqrt(5)
    ratio = np.array([[0.5, 1.5], [1.5, 0.5]])
    expected = -np.minimum(ratio * advantages, np.clip(ratio, 0.8, 1.2) * advantages).mean()
    assert float(metrics[0]) == pytest.approx(expected)
    assert float(metrics[1]) == pytest.approx(0.5)
    assert float(metrics[4]) == 1
    assert float(loss) == pytest.approx(expected + 0.25)
    updated, _ = ppo.update(state, batch, config)
    # Negative and positive advantages each cover clipped and unclipped ratios.
    np.testing.assert_array_equal(updated.params["logits"][:2, 0], logits[:2, 0])
    assert updated.params["logits"][0, 1, 0] < logits[0, 1, 0]
    assert updated.params["logits"][1, 1, 0] > logits[1, 1, 0]
    assert np.all(updated.params["values"][:2] < values[:2])
    np.testing.assert_array_equal(updated.params["values"][2], values[2])
    # Padded NaNs must not affect losses or contribute gradients.
    dirty = batch._replace(
        returns=np.where(batch.mask, batch.returns, np.nan).astype(np.float32),
        old_log_probs=np.where(batch.mask, batch.old_log_probs, np.nan).astype(np.float32),
    )
    dirty_logits = jnp.where(batch.mask[..., None], logits, jnp.nan)
    dirty_values = jnp.where(batch.mask, values, jnp.nan)
    dirty_loss, dirty_metrics = ppo.objective(dirty_logits, dirty_values, dirty, config)
    np.testing.assert_allclose(dirty_metrics, metrics)
    assert float(dirty_loss) == pytest.approx(float(loss))


@pytest.mark.parametrize("reason", ["kl", "nonfinite", "empty"])
def test_update_rejection_preserves_optimizer(
    config: ppo.Config, synthetic: tuple[TrainState, ppo.EditBatch], reason: str
) -> None:
    state, batch = synthetic
    if reason == "kl":
        config = replace(config, target_kl=1e-6)
    elif reason == "nonfinite":
        batch = batch._replace(returns=np.full_like(batch.returns, np.nan))
    else:
        batch = batch._replace(mask=np.zeros_like(batch.mask))
    updated, _ = ppo.update(state, batch, config)
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(updated), strict=True):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [2]], ids=["empty", "duplicate", "negative", "out-of-range"])
def test_episode_selection_rejects_invalid_indices(rollout: ppo.Rollout, indices: list[int]) -> None:
    with pytest.raises(ValueError, match="unique, in-range"):
        ppo.select_episodes(rollout.batch, np.asarray(indices, np.int64))


@pytest.mark.parametrize(
    "changes",
    [
        {"gamma": -0.1},
        {"gae_lambda": 1.1},
        {"gamma": float("nan")},
        {"value_coef": -1},
        {"value_coef": float("inf")},
        {"group_size": 0},
    ],
    ids=[
        "discount-negative",
        "lambda-too-large",
        "discount-nonfinite",
        "critic-negative",
        "critic-nonfinite",
        "empty-group",
    ],
)
def test_invalid_config(config: ppo.Config, changes: dict[str, float]) -> None:
    with pytest.raises((ValueError, AssertionError)):
        replace(config, **changes)


def test_config_roundtrip_and_bf16_shapes(config: ppo.Config, tmp_path: Path) -> None:
    config = replace(config, bf16=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(asdict(config)))
    assert ppo.load_config(path) == config
    shipped = ppo.load_config(Path(__file__).resolve().parents[1] / "configs/karel_ppo_ast_ar_edit.yaml")
    assert shipped.value_coef > 0 and shipped.run_id is None
    grids = jax.ShapeDtypeStruct((1, 3, 3, 6), jnp.int32)
    state = jax.eval_shape(partial(ppo.create_state, config), grids, grids)
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(state.params))


def test_explained_variance_ignores_padding() -> None:
    mask = np.array([[True, True], [False, False]])
    returns = np.array([[0, 2], [np.nan, np.nan]], np.float32)
    values = np.array([[0, 1], [np.nan, np.nan]], np.float32)
    assert ppo.explained_variance(values, returns, mask) == pytest.approx(0.75)
    assert np.isnan(ppo.explained_variance(values, np.zeros_like(returns), mask))
    assert np.isnan(ppo.explained_variance(values, returns, np.zeros_like(mask)))


@pytest.mark.parametrize("invalid", ["shape", "dtype"])
def test_batch_validation(rollout: ppo.Rollout, invalid: str) -> None:
    values = rollout.batch.values
    values = values[:-1] if invalid == "shape" else values.astype(np.int32)
    with pytest.raises(AssertionError):
        ppo.check_batch(rollout.batch._replace(values=values))


@pytest.mark.parametrize("reject", [False, True], ids=["accepted", "rejected"])
def test_training_checkpoint_resume_and_inference_wiring(
    config: ppo.Config,
    state: TrainState,
    rollout: ppo.Rollout,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reject: bool,
) -> None:
    monkeypatch.setattr(ppo, "create_state", MagicMock(return_value=state))
    collect = MagicMock(return_value=rollout)
    monkeypatch.setattr(ppo, "collect_rollout", collect)

    def update(current: TrainState, batch: ppo.EditBatch, settings: ppo.Config) -> tuple[TrainState, ppo.Metrics]:
        ppo.check_batch(batch)
        assert settings.value_coef == config.value_coef
        return (current if reject else current.replace(step=current.step + 1)), tuple(jnp.arange(5, dtype=jnp.float32))

    mocked_update = MagicMock(side_effect=update)
    monkeypatch.setattr(ppo, "update", mocked_update)
    writer = MagicMock()
    monkeypatch.setattr(ppo, "SummaryWriter", MagicMock(return_value=writer))
    config = replace(config, log_dir=str(tmp_path), run_id="ppo", update_epochs=2)
    trained = ppo.train(config)
    assert mocked_update.call_count == (1 if reject else 2)
    writer.add_scalar.assert_any_call("losses/value", 1.0, 2)
    writer.add_scalar.assert_any_call("policy/early_stop", float(reject), 2)
    writer.close.assert_called_once()
    assert (tmp_path / "ppo/checkpoint.msgpack").is_file()
    resumed = ppo.train(config)
    assert int(resumed.step) == int(trained.step)
    assert collect.call_count == 1
    loaded_config, loaded = ppo.load_model(tmp_path / "ppo", attention_implementation="xla")
    assert loaded_config == config
    for expected, actual in zip(jax.tree.leaves(trained.params), jax.tree.leaves(loaded.params), strict=True):
        np.testing.assert_array_equal(expected, actual)


def test_cli_loads_ppo_config(monkeypatch: pytest.MonkeyPatch) -> None:
    train = MagicMock()
    monkeypatch.setattr(ppo, "train", train)
    monkeypatch.setattr("sys.argv", ["train_karel_ppo_ast_ar_edit"])
    ppo.main()
    train.assert_called_once()
    assert isinstance(train.call_args.args[0], ppo.Config)
