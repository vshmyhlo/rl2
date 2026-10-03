from dataclasses import replace
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import serialization
from flax.training.train_state import TrainState
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX import SummaryWriter

from rl2 import grpo
from rl2.grpo import (
    Config,
    GRPOBatch,
    action_log_prob,
    collect_rollout,
    create_state,
    format_group_programs,
    group_advantages,
    learning_rate_schedule,
    load_config,
    load_model,
    objective,
    train,
    update,
)
from rl2.karel import REWARD_COMPONENTS, TOKEN_TO_ID, KarelConfig, KarelProgramEnv, _parse
from rl2.karel_ast import ACTION_ID, AST_ACTIONS, ASTFeatures, KarelAST, batch_features


@pytest.fixture(scope="module")
def config() -> Config:
    return Config(
        total_updates=1,
        num_tasks=1,
        group_size=2,
        num_minibatches=1,
        update_epochs=1,
        d_model=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        max_nodes=12,
        max_depth=8,
        decision_batch_size=5,
        entropy_coef=0.01,
        target_kl=None,
        env=KarelConfig(height=3, width=3, max_depth=0, max_statements=1, max_program_tokens=8),
    )


@pytest.fixture(scope="module")
def state(config: Config) -> TrainState:
    initial, target = KarelProgramEnv(config.env).reset(seed=42)
    state = create_state(config, initial[None], target[None])
    # Nonuniform logits expose replay/masking/gradient mistakes hidden by a zero head.
    params = dict(state.params)
    for name in ("constructor_head", "value_head"):
        params[name] = {
            **params[name],
            "kernel": 0.1 * jax.random.normal(jax.random.key(5), params[name]["kernel"].shape),
        }
    return state.replace(params=params)


@pytest.fixture(scope="module")
def rollout(config: Config, state: TrainState) -> tuple[GRPOBatch, np.ndarray, dict[str, float], jax.Array]:
    envs = [KarelProgramEnv(config.env) for _ in range(config.group_size)]
    return collect_rollout(state, envs, np.random.default_rng(4), jax.random.key(3), config)


def replay_logits(state: TrainState, batch: GRPOBatch, params: optax.Params | None = None) -> jax.Array:
    variables = {"params": state.params if params is None else params}

    def step(tree: ASTFeatures) -> jax.Array:
        return state.apply_fn(variables, batch.initial, batch.target, tree)

    return jax.vmap(step)(jax.tree.map(jnp.asarray, batch.tree))


def synthetic_batch(logits: jax.Array, mask: jax.Array, advantages: jax.Array) -> GRPOBatch:
    chex.assert_type(logits, jnp.float32)
    chex.assert_shape(mask, logits.shape[:-1])
    chex.assert_type(mask, jnp.bool_)
    chex.assert_shape(advantages, (logits.shape[1],))
    chex.assert_type(advantages, jnp.float32)
    actions = jnp.where(mask, ACTION_ID["move"], 0).astype(jnp.int32)
    initial = jnp.zeros((logits.shape[1], 3, 3, 6), jnp.int32)
    tree = batch_features(tuple(KarelAST.empty(4, 2).features() for _ in range(logits.shape[1])))
    snapshots = ASTFeatures(*(jnp.broadcast_to(x, (logits.shape[0], *x.shape)) for x in tree))
    return GRPOBatch(initial, initial, snapshots, actions, action_log_prob(logits, actions), mask, advantages)


def test_group_advantages() -> None:
    rewards = jnp.asarray([[0, 1, 0, 1], [0.1, 0.1, 0.1, 0.1]], jnp.float32)
    np.testing.assert_allclose(group_advantages(rewards), [[-1, 1, -1, 1], [0, 0, 0, 0]])


@pytest.mark.parametrize("group_size", [8, 16, 32, 128])
def test_fractional_equal_rewards_have_zero_advantages(group_size: int) -> None:
    rewards = jnp.broadcast_to(jnp.asarray([0.1, 1 / 3, -0.7], jnp.float32)[:, None], (3, group_size))
    np.testing.assert_array_equal(group_advantages(rewards), 0)


def test_objective_program_weighting_padding_entropy_and_clipping(config: Config) -> None:
    logits = jnp.zeros((4, 2, 4, len(AST_ACTIONS)), jnp.float32)
    mask = jnp.zeros((4, 2, 4), jnp.bool_).at[0, 0, 0].set(True)
    mask = mask.at[:, 1, 0].set(True).at[0, 1, 1].set(True)
    batch = synthetic_batch(logits, mask, jnp.asarray([1.0, -1.0], jnp.float32))
    config = replace(config, entropy_coef=0)
    loss, metrics = objective(logits, batch, config)
    np.testing.assert_allclose(loss, 0, atol=1e-6)
    np.testing.assert_allclose(metrics[1], np.log(len(AST_ACTIONS) - 1), rtol=1e-6)
    padded = batch._replace(old_log_probs=jnp.where(mask, batch.old_log_probs, -1000.0))
    changed = logits.at[1:, 0, :, ACTION_ID["move"]].set(30)

    def loss_fn(values: jax.Array) -> jax.Array:
        chex.assert_shape(values, logits.shape)
        chex.assert_type(values, jnp.float32)
        return objective(values, padded, config)[0]

    np.testing.assert_allclose(loss_fn(changed), loss, atol=1e-6)
    gradients = jax.grad(loss_fn)(changed)
    assert np.isfinite(gradients).all()
    np.testing.assert_array_equal(gradients[1:, 0], 0)
    ratios = jnp.asarray([[[2.0], [0.5]]], jnp.float32)
    clipped = batch._replace(old_log_probs=batch.old_log_probs - jnp.log(ratios))
    loss, metrics = objective(logits, clipped, config)
    np.testing.assert_allclose(loss, -config.clip_coef, atol=1e-6)
    np.testing.assert_allclose(metrics[3], 1.0, atol=1e-6)
    reference = action_log_prob(logits, clipped.actions) - 1
    penalized, kl = objective(logits, clipped, replace(config, kl_coef=0.1), reference)
    np.testing.assert_allclose(kl[4], np.exp(-1), atol=1e-6)
    np.testing.assert_allclose(penalized, loss + 0.1 * np.exp(-1), atol=1e-6)


def test_rollout_pairs_masks_replay_and_exact_environment_rewards(
    config: Config, state: TrainState, rollout: tuple
) -> None:
    batch, rewards, diagnostics, _ = rollout
    np.testing.assert_array_equal(batch.initial[0], batch.initial[1])
    np.testing.assert_array_equal(batch.target[0], batch.target[1])
    np.testing.assert_array_equal(batch.mask, batch.actions != 0)
    logits = replay_logits(state, batch)
    np.testing.assert_allclose(
        np.asarray(action_log_prob(logits, batch.actions))[batch.mask], batch.old_log_probs[batch.mask], atol=2e-6
    )
    assert (batch.mask.sum(axis=-1) > 1).any()
    assert diagnostics["charts/generation_rounds_mean"] < diagnostics["charts/episode_length_mean"]
    assert diagnostics["charts/syntax_error_rate"] == diagnostics["charts/truncation_rate"] == 0
    assert sum(diagnostics[f"charts/reward_{name}_mean"] for name in REWARD_COMPONENTS) == pytest.approx(
        diagnostics["charts/reward_mean"]
    )
    # Reproduce reset seeds and score the same printed sources through the existing env.
    seeds = np.repeat(np.random.default_rng(4).integers(0, 2**31, size=config.num_tasks), config.group_size)
    successes = []
    for column, seed in enumerate(seeds):
        tree = KarelAST.empty(config.max_nodes, config.max_depth, config.env.max_program_tokens)
        for t in np.flatnonzero(batch.mask[:, column].any(axis=-1)):
            for expected, stored in zip(tree.features(), batch.tree):
                np.testing.assert_array_equal(stored[t, column], expected)
            tree = tree.expand_round(batch.actions[t, column])
        assert tree.complete and len(tree.tokens()) <= config.env.max_program_tokens
        _parse(tree.tokens())
        env = KarelProgramEnv(config.env)
        env.reset(seed=int(seed))
        for token in tree.tokens():
            _, reward, terminated, truncated, info = env.step(TOKEN_TO_ID[token])
        assert terminated and not truncated
        np.testing.assert_allclose(rewards[column], reward, rtol=1e-6)
        successes.append(info["success"])
    assert diagnostics["charts/success_rate"] == np.mean(successes)
    assert diagnostics["charts/group_success_rate"] == float(any(successes))


@pytest.mark.parametrize("microbatch", [1, 5, 32])
def test_microbatch_update_matches_full_batch_gradient(
    config: Config, state: TrainState, rollout: tuple, microbatch: int
) -> None:
    batch = rollout[0]._replace(advantages=jnp.asarray([1.0, -1.0], jnp.float32))
    config = replace(config, decision_batch_size=microbatch, kl_coef=0.05)
    reference_params = jax.tree.map(lambda x: x * 0.99, state.params)
    reference = action_log_prob(replay_logits(state, batch, reference_params), batch.actions)

    def full_loss(params: optax.Params) -> tuple[jax.Array, tuple]:
        return objective(replay_logits(state, batch, params), batch, config, reference)

    (_, expected_metrics), grads = jax.jit(jax.value_and_grad(full_loss, has_aux=True))(state.params)
    expected = state.apply_gradients(grads=grads)
    actual, metrics = update(state, batch, config, reference_params)
    np.testing.assert_allclose(metrics, expected_metrics, atol=2e-6, rtol=2e-5)
    for a, b in zip(jax.tree.leaves(actual.params), jax.tree.leaves(expected.params)):
        np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-5)
    assert int(actual.step) == 1


def test_padding_and_rejected_kl_do_not_update_state(config: Config, state: TrainState, rollout: tuple) -> None:
    batch = rollout[0]
    padded = batch._replace(old_log_probs=np.where(batch.mask, batch.old_log_probs, -10000.0).astype(np.float32))
    one, m1 = update(state, batch, config)
    two, m2 = update(state, padded, config)
    chex.assert_trees_all_close(one, two)
    np.testing.assert_allclose(m1, m2)
    rejected = batch._replace(old_log_probs=batch.old_log_probs - 2.0)
    result, _ = update(state, rejected, replace(config, target_kl=0.001))
    chex.assert_trees_all_equal(result, state)
    with pytest.raises(ValueError, match="Frozen reference"):
        update(state, batch, replace(config, kl_coef=0.1))


@pytest.mark.parametrize("bf16", [False, True])
def test_train_checkpoint_tensorboard(
    config: Config, tmp_path: Path, capsys: pytest.CaptureFixture[str], bf16: bool
) -> None:
    config = replace(config, bf16=bf16, log_dir=str(tmp_path), log_program_count=2)
    state = train(config)
    assert int(state.step) == 1
    assert all(np.isfinite(p).all() and p.dtype == jnp.float32 for p in jax.tree.leaves(state.params))
    output = capsys.readouterr().out
    assert "DEF run" not in output and "Sample 0" not in output
    (directory,) = tmp_path.iterdir()
    _, restored = load_model(directory, attention_implementation="xla")
    chex.assert_trees_all_equal(restored.params, state.params)
    events = EventAccumulator(str(directory)).Reload()
    assert events.Scalars("charts/reward_mean")
    assert events.Scalars("charts/program_token_length_mean")
    assert "samples/generated_programs/text_summary" in events.Tags()["tensors"]


def test_gcs_checkpoint_paths(config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files: dict[str, bytes] = {}
    paths: list[str] = []

    class FakeGCS:
        def cat_file(self, path: str) -> bytes:
            if path not in files:
                raise FileNotFoundError(path)
            return files[path]

        def pipe_file(self, path: str, data: bytes) -> None:
            assert path.startswith("gs://test-bucket/grpo/")
            files[path] = data

    def writer(logdir: str, purge_step: int | None = None) -> SummaryWriter:
        paths.append(logdir)
        return SummaryWriter(logdir=str(tmp_path), purge_step=purge_step)

    monkeypatch.setattr("rl2.grpo.gcsfs.GCSFileSystem", FakeGCS)
    monkeypatch.setattr("rl2.grpo.SummaryWriter", writer)
    cloud_config = replace(config, log_dir="gs://test-bucket/grpo/", run_id="resume-test")
    state = train(cloud_config)
    assert set(files) == {f"{paths[0]}/config.yaml", f"{paths[0]}/checkpoint.msgpack"}
    _, restored = load_model(paths[0])
    chex.assert_trees_all_equal(restored.params, state.params)
    chex.assert_trees_all_equal(serialization.to_state_dict(train(cloud_config)), serialization.to_state_dict(state))
    continued = train(replace(cloud_config, total_updates=2))
    assert int(continued.step) == 2
    assert paths == ["gs://test-bucket/grpo/resume-test"] * 2
    assert serialization.msgpack_restore(files[f"{paths[0]}/checkpoint.msgpack"])["iteration"] == 2


def test_formatted_samples_replay_ast_source(config: Config, rollout: tuple) -> None:
    batch, rewards, _, _ = rollout
    output = format_group_programs(batch, rewards, config=config, group_size=2, group_index=0, count=2)
    assert "same initial/target pair" in output
    assert "actions=" in output and "tokens=" in output
    for column, source in enumerate(output.split("```text\n")[1:]):
        printed = tuple(source.split("```")[0].split())
        tree = KarelAST.empty(config.max_nodes, config.max_depth, config.env.max_program_tokens)
        for step in np.flatnonzero(batch.mask[:, column].any(axis=-1)):
            tree = tree.expand_round(batch.actions[step, column])
        assert printed == tree.tokens()
        assert "\n  " in source


def test_configuration_and_schedule(config: Config) -> None:
    loaded = load_config(Path(__file__).resolve().parents[1] / "configs/grpo_karel.yaml")
    assert loaded.max_nodes == 128 and loaded.decision_batch_size == 32
    assert loaded.bf16 and loaded.attention_implementation == "cudnn"
    assert loaded.d_model == 320 and loaded.num_layers == 7 and loaded.num_heads == 5
    schedule = learning_rate_schedule(config)
    np.testing.assert_allclose(schedule(0), config.learning_rate)
    np.testing.assert_allclose(schedule(config.total_updates), 0.0)
    with pytest.raises(AssertionError):
        replace(config, decision_batch_size=0)


@pytest.mark.skipif(not any(d.platform == "gpu" for d in jax.devices()), reason="cuDNN needs NVIDIA GPU")
def test_cudnn_ast_grpo(config: Config) -> None:
    config = replace(config, bf16=True, attention_implementation="cudnn")
    envs = [KarelProgramEnv(config.env) for _ in range(2)]
    initial, target = envs[0].reset(seed=1)
    state = create_state(config, initial[None], target[None])
    batch, _, _, _ = collect_rollout(state, envs, np.random.default_rng(1), jax.random.key(1), config)
    state, metrics = update(state, batch, config)
    assert all(np.isfinite(x) for x in metrics)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(state.params))


def test_interrupted_resume_matches_uninterrupted_training(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(config, total_updates=3, kl_coef=0.05, log_dir=str(tmp_path), log_program_interval=10)
    collect = grpo.collect_rollout
    save = grpo._save_checkpoint
    calls = 0
    interrupted = False
    now = 0.0
    saves: dict[str, list[int]] = {}

    def clock() -> float:
        nonlocal now
        now += 0.1
        return now

    def rollouts(
        state: TrainState, envs: list[KarelProgramEnv], rng: np.random.Generator, key: jax.Array, cfg: Config
    ) -> tuple[GRPOBatch, np.ndarray, dict[str, float], jax.Array]:
        nonlocal calls, now
        calls += 1
        if interrupted and calls == 3:
            raise RuntimeError("simulated interruption after an unsaved rollout")
        # First rollout crosses five minutes, the next one does not.
        now += 301 if calls == 1 else 1
        return collect(state, envs, rng, key, cfg)

    def checkpoint(directory: str, progress: grpo.TrainingProgress, rng: np.random.Generator) -> None:
        saves.setdefault(directory, []).append(progress.iteration)
        save(directory, progress, rng)

    monkeypatch.setattr(grpo, "monotonic", clock)
    monkeypatch.setattr(grpo, "collect_rollout", rollouts)
    monkeypatch.setattr(grpo, "_save_checkpoint", checkpoint)
    expected = train(replace(config, run_id="continuous"))
    assert saves[str(tmp_path / "continuous")] == [0, 1, 3]
    calls, interrupted = 0, True
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train(replace(config, run_id="resumed"))
    resumed_dir = tmp_path / "resumed"
    saved = serialization.msgpack_restore((resumed_dir / "checkpoint.msgpack").read_bytes())
    assert saved["iteration"] == 1
    assert not (resumed_dir / "params.msgpack").exists()
    assert "config" not in saved  # Configuration lives separately; no duplicate weights export.
    calls, interrupted = 0, False
    actual = train(replace(config, run_id="resumed"))
    chex.assert_trees_all_equal(
        (actual.params, actual.opt_state, actual.step), (expected.params, expected.opt_state, expected.step)
    )
    final = serialization.msgpack_restore((resumed_dir / "checkpoint.msgpack").read_bytes())
    baseline = serialization.msgpack_restore((tmp_path / "continuous" / "checkpoint.msgpack").read_bytes())
    chex.assert_trees_all_equal(final, baseline)
    # Purge the discarded second rollout's event, then continue at the saved step.
    actual_events = EventAccumulator(str(resumed_dir)).Reload().Scalars("charts/reward_mean")
    expected_events = EventAccumulator(str(tmp_path / "continuous")).Reload().Scalars("charts/reward_mean")
    assert [(event.step, event.value) for event in actual_events] == [
        (event.step, event.value) for event in expected_events
    ]


def test_resume_uses_modified_config_and_counts_prior_episodes(config: Config, tmp_path: Path) -> None:
    config = replace(config, log_dir=str(tmp_path), run_id="mutable")
    initial = train(config)
    changed = replace(
        config,
        total_updates=2,
        learning_rate=0.002,
        anneal_lr=False,
        group_size=4,
        num_minibatches=2,
        entropy_coef=0.03,
        max_grad_norm=1.0,
        seed=999,
        env=replace(config.env, marker_weight=2.0),
        kl_coef=0.05,
    )
    result = train(changed)
    assert int(result.step) == int(initial.step) + 2
    np.testing.assert_allclose(result.opt_state.hyperparams["learning_rate"], 0.002)
    directory = tmp_path / "mutable"
    payload = serialization.msgpack_restore((directory / "checkpoint.msgpack").read_bytes())
    assert payload["iteration"] == 2 and payload["episodes"] == 6
    # A newly enabled KL penalty anchors at the resumed weights.
    chex.assert_trees_all_equal(payload["reference_params"], initial.params)
    assert load_config(directory / "config.yaml") == changed


def test_completed_run_is_noop_and_new_id_starts_fresh(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(config, log_dir=str(tmp_path), run_id="first")
    initial = train(config)
    checkpoint = tmp_path / "first" / "checkpoint.msgpack"
    original = checkpoint.read_bytes()

    def no_rollout(*args: object, **kwargs: object) -> None:
        pytest.fail("A completed run must not collect another rollout")

    with monkeypatch.context() as patch:
        patch.setattr(grpo, "collect_rollout", no_rollout)
        restored = train(config)
    chex.assert_trees_all_equal(restored.params, initial.params)
    assert checkpoint.read_bytes() == original
    fresh = train(replace(config, run_id="second"))
    assert int(fresh.step) == 1
    chex.assert_trees_all_equal(fresh.params, initial.params)
    assert (tmp_path / "second" / "checkpoint.msgpack").is_file()


def test_incompatible_or_corrupt_checkpoint_is_not_overwritten(config: Config, tmp_path: Path) -> None:
    config = replace(config, log_dir=str(tmp_path), run_id="protected")
    train(config)
    directory = tmp_path / "protected"
    checkpoint = directory / "checkpoint.msgpack"
    before, old_config = checkpoint.read_bytes(), (directory / "config.yaml").read_bytes()
    with pytest.raises(ValueError, match="incompatible"):
        train(replace(config, d_model=32))
    assert checkpoint.read_bytes() == before
    assert (directory / "config.yaml").read_bytes() == old_config
    checkpoint.write_bytes(serialization.msgpack_serialize({"version": 999}))
    with pytest.raises(ValueError, match="version"):
        train(config)
    assert checkpoint.read_bytes() == serialization.msgpack_serialize({"version": 999})


def test_checkpoint_io_preserves_previous_file_and_propagates_storage_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "checkpoint.msgpack"
    destination.write_bytes(b"old")

    def failed_replace(source: Path, target: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(grpo.os, "replace", failed_replace)
    with pytest.raises(OSError, match="replace failure"):
        grpo._write_bytes(str(destination), b"new")
    assert destination.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [destination]
    assert grpo._read_optional(str(tmp_path / "missing")) is None

    def denied(path: str) -> bytes:
        raise PermissionError("storage access denied")

    monkeypatch.setattr(grpo, "_read_bytes", denied)
    with pytest.raises(PermissionError, match="denied"):
        grpo._read_optional("gs://bucket/run/checkpoint.msgpack")


@pytest.mark.parametrize("run_id", ["", " ", ".", "..", "../run", "foo/bar", "foo\\bar", 1])
def test_invalid_run_id(config: Config, run_id: str | int) -> None:
    with pytest.raises((ValueError, TypeError)):
        replace(config, run_id=run_id)


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan")])
def test_invalid_checkpoint_interval(config: Config, interval: float) -> None:
    with pytest.raises((AssertionError, ValueError)):
        replace(config, checkpoint_interval_seconds=interval)


def test_legacy_weights_load_for_inference_but_cannot_resume(config: Config, state: TrainState, tmp_path: Path) -> None:
    config = replace(config, log_dir=str(tmp_path), run_id="legacy")
    directory = tmp_path / "legacy"
    directory.mkdir()
    (directory / "config.yaml").write_text(grpo.yaml.safe_dump(grpo.asdict(config)))
    (directory / "params.msgpack").write_bytes(serialization.to_bytes(state.params))
    _, loaded = load_model(directory)
    chex.assert_trees_all_equal(loaded.params, state.params)
    with pytest.raises(ValueError, match="only inference weights"):
        train(config)
    assert not (directory / "checkpoint.msgpack").exists()
