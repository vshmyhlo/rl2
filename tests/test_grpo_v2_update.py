import logging
from dataclasses import replace
from functools import partial
from typing import Any
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from rl2 import train_grpo_v2 as training
from rl2.qwen3 import Qwen3Config, Qwen3LM


def rollout(tokens: list[int], mask: list[bool], advantage: float) -> training.Rollout:
    return training.Rollout(
        sequence=training.TrainingSequence(
            token_ids=jnp.asarray(tokens, dtype=jnp.int32),  # (T,)
            loss_mask=jnp.asarray(mask, dtype=jnp.bool_),  # (T,)
            old_log_probs=jnp.zeros(len(tokens), dtype=jnp.float32),  # (T,)
        ),
        reward=float(advantage > 0),
        advantage=advantage,
        completion="",
        num_turns=1,
        num_input_tokens=len(tokens) - sum(mask),
        num_output_tokens=sum(mask),
        rollout_seconds=1.0,
        stop_condition="agent_completed",
    )


@pytest.fixture
def batcher() -> Any:
    return training.Batcher.__ray_metadata__.modified_class(actor=Mock())


@pytest.fixture
def groups() -> list[training.RolloutGroup]:
    return [
        training.RolloutGroup(
            task_id="0",
            prompt="test",
            rollouts=[
                rollout([1, 2, 3, 4, 5], [False, True, False, True, True], 1.0),
                rollout([1, 2, 6], [False, False, True], -1.0),
            ],
        )
    ]


@pytest.mark.parametrize("context_length,width", [(512, 128), (5, 5)], ids=["bucketed", "context-boundary"])
def test_prepare_batch(batcher: Any, groups: list[training.RolloutGroup], context_length: int, width: int) -> None:
    batch = batcher.prepare_batch(groups, context_length=context_length)
    assert batch.token_ids.shape == batch.attention_mask.shape == batch.loss_mask.shape == (2, width)
    assert batch.old_log_probs.shape == (2, width)
    assert batch.advantages.shape == (2,)
    assert all(array.device.platform == "cpu" for array in batch)
    assert batch.token_ids.dtype == jnp.int32
    assert batch.attention_mask.dtype == batch.loss_mask.dtype == jnp.bool_
    assert batch.old_log_probs.dtype == batch.advantages.dtype == jnp.float32
    np.testing.assert_array_equal(batch.token_ids[0, :5], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(batch.token_ids[1, :3], [1, 2, 6])
    np.testing.assert_array_equal(batch.attention_mask.sum(axis=1), [5, 3])
    np.testing.assert_array_equal(batch.loss_mask[0, :5], [False, True, False, True, True])
    np.testing.assert_array_equal(batch.loss_mask.sum(axis=1), [3, 1])
    np.testing.assert_array_equal(batch.advantages, [1.0, -1.0])
    assert not np.asarray(batch.attention_mask[1, 3:]).any()
    assert not np.asarray(batch.loss_mask[1, 3:]).any()
    np.testing.assert_array_equal(batch.token_ids[1, 3:], 0)
    np.testing.assert_array_equal(batch.old_log_probs[1, 3:], 0)


@pytest.mark.parametrize(
    "length,context_length,width",
    [(256, 512, 256), (257, 512, 384), (257, 300, 300)],
    ids=["exact-bucket", "next-bucket", "context-cap"],
)
def test_prepare_batch_bucket_boundaries(batcher: Any, length: int, context_length: int, width: int) -> None:
    sample = rollout([1] * length, [False] + [True] * (length - 1), 1.0)
    groups = [training.RolloutGroup(task_id="0", prompt="test", rollouts=[sample])]
    batch = batcher.prepare_batch(groups, context_length=context_length)  # (B=1, T=width), (B,) leaves.
    assert all(array.shape == (1, width) for array in batch[:-1])
    assert batch.advantages.shape == (1,)
    assert int(batch.attention_mask.sum()) == length
    assert int(batch.loss_mask.sum()) == length - 1
    np.testing.assert_array_equal(batch.token_ids[0, :length], sample.sequence.token_ids)


def test_prepare_minimal_sequences_and_multiple_groups(batcher: Any) -> None:
    groups = [
        training.RolloutGroup(task_id=str(i), prompt="test", rollouts=[rollout([1, 2], [False, True], advantage)])
        for i, advantage in enumerate([0.0, 1.0])
    ]
    batch = batcher.prepare_batch(groups, context_length=8)
    assert batch.token_ids.shape == (2, 8)
    np.testing.assert_array_equal(batch.advantages, [0.0, 1.0])


@pytest.mark.parametrize("failure", ["empty", "overlong", "no-context", "no-samples", "single-token"])
def test_prepare_batch_rejects_unusable_sequences(
    batcher: Any, failure: str, groups: list[training.RolloutGroup]
) -> None:
    if failure == "empty":
        groups = []
    elif failure in {"no-context", "no-samples", "single-token"}:
        sample = {
            "no-context": rollout([1, 2], [True, True], 1.0),
            "no-samples": rollout([1, 2], [False, False], 1.0),
            "single-token": rollout([1], [True], 1.0),
        }[failure]
        groups = [replace(groups[0], rollouts=[sample])]
    with pytest.raises(ValueError):
        batcher.prepare_batch(groups, context_length=4 if failure == "overlong" else 8)


def test_objective_clipping_masking_and_sequence_weighting() -> None:
    batch = training.GRPOBatch(
        token_ids=jnp.zeros((2, 4), dtype=jnp.int32),  # (B=2, T=4)
        attention_mask=jnp.ones((2, 4), dtype=jnp.bool_),  # (B, T)
        loss_mask=jnp.array([[False, True, False, False], [False, True, True, False]]),  # (B, T)
        old_log_probs=jnp.full((2, 4), -0.7),  # (B, T)
        advantages=jnp.array([1.0, -1.0]),  # (B,)
    )
    # Unused old probabilities must not overflow the exponential or affect gradients.
    batch = batch._replace(old_log_probs=jnp.where(batch.loss_mask, batch.old_log_probs, -1000))

    def loss_fn(log_probs: jax.Array) -> jax.Array:  # (B, T-1) -> ()
        return training.objective(log_probs, batch, clip_coef=0.2)[0]

    log_probs = jnp.array([[jnp.log(1.5) - 0.7, 1000, 1000], [jnp.log(0.5) - 0.7] * 2 + [1000]])  # (B, T-1)
    loss, metrics = training.objective(log_probs, batch, clip_coef=0.2)
    assert loss.shape == () and all(value.shape == () for value in metrics.values())
    assert float(loss) == pytest.approx(-0.2)
    assert float(metrics["policy/clip_fraction"]) == 1.0
    np.testing.assert_array_equal(jax.grad(loss_fn)(log_probs), 0)
    adverse = jnp.array([[jnp.log(0.5) - 0.7, 1000, 1000], [jnp.log(1.5) - 0.7] * 2 + [1000]])  # (B, T-1)
    assert float(loss_fn(adverse)) == pytest.approx(0.5)
    np.testing.assert_allclose(jax.grad(loss_fn)(adverse), [[-0.25, 0, 0], [0.375, 0.375, 0]], atol=1e-6)


@pytest.fixture(scope="module")
def model() -> Qwen3LM:
    return Qwen3LM(
        config=Qwen3Config(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            eos_token_id=7,
            max_position_embeddings=8,
        ),
        dtype="float32",
    )


@pytest.fixture(scope="module")
def state(model: Qwen3LM) -> TrainState:
    params = model.init(jax.random.key(0), jnp.ones((1, 7), dtype=jnp.int32))["params"]  # HF parameter shapes.
    params = jax.tree.map(lambda value: value.astype(jnp.bfloat16), params)  # Emulate checkpoint storage.
    return training.create_state(model, params, training.Config(learning_rate=0.01, max_grad_norm=0.1))


@pytest.fixture
def batch(batcher: Any, groups: list[training.RolloutGroup], state: TrainState) -> training.GRPOBatch:
    batch = batcher.prepare_batch(groups, context_length=8)
    old = training.token_log_probs(state, state.params, batch, temperature=0.7)  # (B=2, T-1=7)
    return batch._replace(old_log_probs=jnp.pad(old, ((0, 0), (1, 0))))


def test_token_log_probs_shift_temperature_and_padding(state: TrainState, batch: training.GRPOBatch) -> None:
    log_probs = training.token_log_probs(state, state.params, batch, temperature=0.7)  # (B, T-1)
    assert log_probs.shape == (2, 7)
    logits = state.apply_fn({"params": state.params}, batch.token_ids).logits  # (B, T, V)
    expected = jax.nn.log_softmax(logits / 0.7)  # (B, T, V)
    np.testing.assert_allclose(log_probs[0, [0, 2, 3]], expected[0, [0, 2, 3], [2, 4, 5]], atol=1e-6)
    np.testing.assert_allclose(log_probs[1, 1], expected[1, 1, 6], atol=1e-6)
    padded = batch._replace(token_ids=jnp.where(batch.attention_mask, batch.token_ids, 7))
    changed = training.token_log_probs(state, state.params, padded, temperature=0.7)  # (B, T-1)
    np.testing.assert_allclose(changed[batch.loss_mask[:, 1:]], log_probs[batch.loss_mask[:, 1:]], atol=1e-6)


def test_accumulated_update_matches_full_batch_and_improves_policy(
    state: TrainState, batch: training.GRPOBatch
) -> None:
    accumulated, metrics = training.update(state, batch, grad_accum_steps=2, temperature=0.7, clip_coef=0.2)
    full, full_metrics = training.update(state, batch, grad_accum_steps=1, temperature=0.7, clip_coef=0.2)
    assert int(accumulated.step) == int(full.step) == 1
    for actual, expected in zip(jax.tree.leaves(accumulated), jax.tree.leaves(full), strict=True):
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
    assert all(value.dtype == jnp.float32 for value in accumulated.params.values())
    assert all(value.shape == () for value in metrics.values())
    assert float(metrics["policy/grad_norm"]) > 0.1  # Exercise gradient clipping.
    for name in metrics:
        np.testing.assert_allclose(metrics[name], full_metrics[name], atol=1e-6)
    after = training.token_log_probs(accumulated, accumulated.params, batch, temperature=0.7)  # (B, T-1)
    loss, _ = training.objective(after, batch, clip_coef=0.2)
    assert float(loss) < float(metrics["losses/policy"])
    assert any(not np.array_equal(state.params[name], value) for name, value in accumulated.params.items())


def test_update_logs_only_new_traces(
    state: TrainState, batch: training.GRPOBatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A unique callable isolates this test's JIT cache without clearing other tests' entries.
    state = state.replace(apply_fn=partial(state.apply_fn))
    trace = partial(training.update.lower, state, grad_accum_steps=2, temperature=0.7, clip_coef=0.2)
    caplog.set_level(logging.INFO, logger=training.__name__)
    trace(batch)
    trace(batch)
    assert caplog.messages == ["JIT trace update: batch_shape=(2, 8) grad_accum_steps=2 temperature=0.7 clip_coef=0.2"]
    caplog.clear()
    shorter = jax.tree.map(lambda value: value[:, :5] if value.ndim == 2 else value, batch)  # (B, 5), (B,).
    trace(shorter)
    trace(batch)
    assert caplog.messages == ["JIT trace update: batch_shape=(2, 5) grad_accum_steps=2 temperature=0.7 clip_coef=0.2"]


def test_zero_advantages_leave_fresh_optimizer_parameters_unchanged(
    state: TrainState, batch: training.GRPOBatch
) -> None:
    batch = batch._replace(advantages=jnp.zeros_like(batch.advantages))
    updated, metrics = training.update(state, batch, grad_accum_steps=2, temperature=0.7, clip_coef=0.2)
    assert float(metrics["policy/grad_norm"]) == float(metrics["losses/policy"]) == 0.0
    for name in state.params:
        np.testing.assert_array_equal(updated.params[name], state.params[name])


@pytest.mark.parametrize("grad_accum_steps", [0, 3], ids=["nonpositive", "indivisible"])
def test_update_rejects_invalid_accumulation(
    state: TrainState, batch: training.GRPOBatch, grad_accum_steps: int
) -> None:
    with pytest.raises(ValueError, match="grad_accum_steps must be positive and divide the minibatch size"):
        training.update.lower(state, batch, grad_accum_steps=grad_accum_steps, temperature=0.7, clip_coef=0.2)


@pytest.mark.parametrize("epochs,minibatches", [(1, 1), (2, 2)], ids=["default", "multiple-updates"])
def test_learner_averages_all_update_metrics(
    state: TrainState,
    batch: training.GRPOBatch,
    monkeypatch: pytest.MonkeyPatch,
    epochs: int,
    minibatches: int,
) -> None:
    learner = training.Learner.__ray_metadata__.modified_class(
        batcher=Mock(),
        actor=Mock(),
        config=training.Config(group_size=2, update_epochs=epochs, minibatches_per_epoch=minibatches),
    )
    learner.mesh = Mesh(devices=np.array([jax.devices()[0]]), axis_names=("batch",))
    learner.state = state

    def update(
        current: TrainState, minibatch: training.GRPOBatch, **kwargs: Any
    ) -> tuple[TrainState, training.Metrics]:
        assert minibatch.token_ids.shape == (2 // minibatches, 8)
        assert minibatch.advantages.shape == (2 // minibatches,)
        assert kwargs["grad_accum_steps"] == 1
        return current.replace(step=current.step + 1), {"losses/policy": jnp.asarray(current.step)}  # () metric.

    monkeypatch.setattr(training, "update", update)
    metrics = learner.optimize(batch)
    steps = epochs * minibatches
    assert int(learner.state.step) == steps
    assert metrics == {"losses/policy": (steps - 1) / 2}


def test_learner_rejects_indivisible_rollout_batch(state: TrainState, batch: training.GRPOBatch) -> None:
    learner = training.Learner.__ray_metadata__.modified_class(
        batcher=Mock(), actor=Mock(), config=training.Config(group_size=6, minibatches_per_epoch=3)
    )
    learner.mesh = Mesh(devices=np.array([jax.devices()[0]]), axis_names=("batch",))
    learner.state = state
    with pytest.raises(ValueError, match="minibatches_per_epoch must divide the rollout batch size"):
        learner.optimize(batch)
    assert int(learner.state.step) == 0


def test_learner_optimizes_and_publishes_updated_bf16_weights(
    batcher: Any,
    model: Qwen3LM,
    state: TrainState,
    groups: list[training.RolloutGroup],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = training.Learner.__ray_metadata__.modified_class(
        batcher=Mock(),
        actor=Mock(),
        config=training.Config(temperature=0.7, grad_accum_steps=2, minibatches_per_epoch=2, update_epochs=2),
    )
    learner.model = model.clone(dtype="bfloat16", param_dtype=jnp.bfloat16)
    learner.mesh = Mesh(devices=np.array([jax.devices()[0]]), axis_names=("batch",))
    learner.state = training.create_state(learner.model, state.params, training.Config(learning_rate=0.01))
    learner._transfer_group = Mock()
    groups = groups + [replace(groups[0], rollouts=list(reversed(groups[0].rollouts)))]
    batch = batcher.prepare_batch(groups, context_length=8)  # (B=4, T=8), (B,) leaves.
    old = training.token_log_probs(learner.state, learner.state.params, batch, temperature=0.7)  # (B, T-1)
    assert old.shape == (4, 7)
    assert old.dtype == jnp.float32
    batch = batch._replace(old_log_probs=jnp.pad(old, ((0, 0), (1, 0))))
    update = Mock(wraps=training.update)
    monkeypatch.setattr(training, "update", update)
    metrics = learner.optimize(batch)
    assert update.call_count == 4
    for index, call in enumerate(update.call_args_list):
        assert int(call.args[0].step) == index
        assert call.kwargs["grad_accum_steps"] == 2
        placed_batch = call.args[1]  # (B/N=2, T=8), (B/N,) leaves.
        start = (index % 2) * 2
        for placed, original in zip(placed_batch, batch, strict=True):
            assert placed.shape == (2, *original.shape[1:])
            assert placed.sharding == NamedSharding(learner.mesh, PartitionSpec("batch"))
            np.testing.assert_array_equal(placed, original[start : start + 2])
    assert int(learner.state.step) == 4
    assert metrics["policy/grad_norm"] > 0
    assert all(isinstance(value, float) for value in metrics.values())
    for name, value in learner.state.params.items():  # HF parameter shapes.
        assert value.shape == state.params[name].shape
        assert value.dtype == jnp.float32
    adam = learner.state.opt_state[1][0]
    for moments in (adam.mu, adam.nu):  # HF parameter shapes.
        for name, value in moments.items():
            assert value.shape == learner.state.params[name].shape
            assert value.dtype == jnp.float32
    assert optax.global_norm(jax.tree.map(jnp.subtract, learner.state.params, state.params)) > 0

    def send(iterator: Any, trainer_args: training.NCCLTrainerSendWeightsArgs) -> None:
        weights = dict(iterator)  # HF parameter shapes.
        for name, value in learner.state.params.items():
            assert str(weights[name].dtype) == "torch.bfloat16"
            np.testing.assert_array_equal(
                weights[name].float().cpu().numpy(), np.asarray(value.astype(jnp.bfloat16), dtype=np.float32)
            )

    monkeypatch.setattr(training.NCCLWeightTransferEngine, "trainer_send_weights", send)
    monkeypatch.setattr(training.torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(training.ray, "get", Mock())
    learner.publish_weights()
