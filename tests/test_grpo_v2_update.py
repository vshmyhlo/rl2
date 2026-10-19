from dataclasses import replace
from typing import Any
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from rl2 import train_grpo_v2 as training
from rl2.qwen3 import Qwen3Config, Qwen3LM


def rollout(tokens: list[int], mask: list[bool], advantage: float) -> training.Rollout:
    return training.Rollout(
        sequence=training.TrainingSequence(
            token_ids=np.asarray(tokens, dtype=np.int32),  # (T,)
            loss_mask=np.asarray(mask, dtype=np.bool_),  # (T,)
            old_log_probs=np.zeros(len(tokens), dtype=np.float32),  # (T,)
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


@pytest.mark.parametrize("context_length,width", [(8, 8), (5, 5)], ids=["bucketed", "context-boundary"])
def test_prepare_batch(groups: list[training.RolloutGroup], context_length: int, width: int) -> None:
    batch = training.prepare_batch(groups, context_length=context_length)
    assert batch.token_ids.shape == batch.attention_mask.shape == batch.loss_mask.shape == (2, width)
    assert batch.old_log_probs.shape == (2, width)
    assert batch.advantages.shape == (2,)
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


def test_prepare_minimal_sequences_and_multiple_groups() -> None:
    groups = [
        training.RolloutGroup(task_id=str(i), prompt="test", rollouts=[rollout([1, 2], [False, True], advantage)])
        for i, advantage in enumerate([0.0, 1.0])
    ]
    batch = training.prepare_batch(groups, context_length=8)
    assert batch.token_ids.shape == (2, 2)
    np.testing.assert_array_equal(batch.advantages, [0.0, 1.0])


@pytest.mark.parametrize("failure", ["empty", "overlong", "no-context", "no-samples", "single-token"])
def test_prepare_batch_rejects_unusable_sequences(failure: str, groups: list[training.RolloutGroup]) -> None:
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
        training.prepare_batch(groups, context_length=4 if failure == "overlong" else 8)


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
def batch(groups: list[training.RolloutGroup], state: TrainState) -> training.GRPOBatch:
    batch = training.prepare_batch(groups, context_length=8)
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
    accumulated, metrics = training.update(state, batch, microbatch_size=1, temperature=0.7, clip_coef=0.2)
    full, full_metrics = training.update(state, batch, microbatch_size=2, temperature=0.7, clip_coef=0.2)
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


def test_zero_advantages_leave_fresh_optimizer_parameters_unchanged(
    state: TrainState, batch: training.GRPOBatch
) -> None:
    batch = batch._replace(advantages=jnp.zeros_like(batch.advantages))
    updated, metrics = training.update(state, batch, microbatch_size=1, temperature=0.7, clip_coef=0.2)
    assert float(metrics["policy/grad_norm"]) == float(metrics["losses/policy"]) == 0.0
    for name in state.params:
        np.testing.assert_array_equal(updated.params[name], state.params[name])


def test_learner_optimizes_and_publishes_updated_bf16_weights(
    model: Qwen3LM,
    state: TrainState,
    groups: list[training.RolloutGroup],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = training.Learner.__ray_metadata__.modified_class(
        batcher=Mock(),
        actor=Mock(),
        config=training.Config(temperature=0.7, update_epochs=2),
    )
    learner.model = model.clone(dtype="bfloat16", param_dtype=jnp.bfloat16)
    learner.state = training.create_state(learner.model, state.params, training.Config(learning_rate=0.01))
    learner._transfer_group = Mock()
    batch = training.prepare_batch(groups, context_length=8)
    old = training.token_log_probs(learner.state, learner.state.params, batch, temperature=0.7)  # (B, T-1)
    batch = batch._replace(old_log_probs=jnp.pad(old, ((0, 0), (1, 0))))
    for index, sample in enumerate(groups[0].rollouts):
        sample.sequence.old_log_probs[:] = np.asarray(batch.old_log_probs[index, : len(sample.sequence.token_ids)])
    metrics = learner.optimize(groups)
    assert int(learner.state.step) == 2
    assert metrics["policy/grad_norm"] > 0
    assert all(isinstance(value, float) for value in metrics.values())
    assert optax.global_norm(jax.tree.map(jnp.subtract, learner.state.params, state.params)) > 0

    def send(iterator: Any, trainer_args: training.NCCLTrainerSendWeightsArgs) -> None:
        weights = dict(iterator)  # HF parameter shapes.
        for name, value in learner.state.params.items():
            assert str(weights[name].dtype) == "torch.bfloat16"
            np.testing.assert_array_equal(
                weights[name].float().numpy(), np.asarray(value.astype(jnp.bfloat16), dtype=np.float32)
            )

    monkeypatch.setattr(training.NCCLWeightTransferEngine, "trainer_send_weights", send)
    monkeypatch.setattr(training.torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(training.ray, "get", Mock())
    learner.publish_weights()
