import logging
import pickle
from collections.abc import Generator
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from safetensors.numpy import load_file

from rl2 import train_grpo_v2 as training
from rl2.qwen3 import Qwen3Config, Qwen3LM, Qwen3Output, Variables


@pytest.fixture
def mesh_context() -> Generator[None]:
    with jax.set_mesh(Mesh(devices=jax.devices()[:1], axis_names=("batch",))):
        yield


def rollout(tokens: list[int], mask: list[bool], advantage: float) -> training.Rollout:
    return training.Rollout(
        prompt="Environment prompt",
        sequence=training.TrainingSequence(
            token_ids=np.asarray(tokens, dtype=np.int32),  # (T,)
            loss_mask=np.asarray(mask, dtype=np.bool_),  # (T,)
            old_log_probs=np.zeros(len(tokens), dtype=np.float32),  # (T,)
        ),
        reward=float(advantage > 0),
        advantage=advantage,
        completion="",
        interaction="",
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
            rollouts=[
                rollout([1, 2, 3, 4, 5], [False, True, False, True, True], 1.0),
                rollout([1, 2, 6], [False, False, True], -1.0),
            ],
        )
    ]


@pytest.mark.parametrize("context_length,width", [(512, 256), (5, 5)], ids=["bucketed", "context-boundary"])
def test_prepare_batch(batcher: Any, groups: list[training.RolloutGroup], context_length: int, width: int) -> None:
    for index, sample in enumerate(groups[0].rollouts):
        sequence = replace(
            sample.sequence,
            old_log_probs=-np.arange(len(sample.sequence.token_ids), dtype=np.float32) / 10,  # (L,)
        )
        groups[0].rollouts[index] = replace(sample, sequence=sequence)
    batch = batcher.prepare_batch(groups, context_length=context_length)  # (B=2, T=width), (B,) leaves.
    result = training.build_rollout_batch(groups=groups, training_batch=batch, transfer_metrics={})
    assert result.rollout_metrics["batcher/token_utilization_fraction"] == pytest.approx(8 / (2 * width))
    assert result.rollout_metrics["rollout/total_tokens_mean"] == 4.0
    assert result.rollout_metrics["batcher/padded_sequence_tokens"] == width
    assert batch.token_ids.shape == batch.attention_mask.shape == batch.loss_mask.shape == (2, width)
    assert batch.old_log_probs.shape == (2, width)
    assert batch.advantages.shape == (2,)
    assert all(isinstance(array, np.ndarray) for array in batch)
    assert batch.token_ids.dtype == np.int32
    assert batch.attention_mask.dtype == batch.loss_mask.dtype == np.bool_
    assert batch.old_log_probs.dtype == batch.advantages.dtype == np.float32
    restored = pickle.loads(pickle.dumps(batch))  # (B, T), (B,) leaves.
    for original, value in zip(batch, restored, strict=True):
        assert isinstance(value, np.ndarray)
        np.testing.assert_array_equal(value, original)
    np.testing.assert_array_equal(batch.token_ids[0, :5], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(batch.token_ids[1, :3], [1, 2, 6])
    np.testing.assert_array_equal(batch.attention_mask.sum(axis=1), [5, 3])
    np.testing.assert_array_equal(batch.loss_mask[0, :5], [False, True, False, True, True])
    np.testing.assert_array_equal(batch.loss_mask.sum(axis=1), [3, 1])
    np.testing.assert_array_equal(batch.advantages, [1.0, -1.0])
    for index, sample in enumerate(groups[0].rollouts):
        length = len(sample.sequence.token_ids)
        np.testing.assert_array_equal(batch.old_log_probs[index, :length], sample.sequence.old_log_probs)
    assert not np.asarray(batch.attention_mask[1, 3:]).any()
    assert not np.asarray(batch.loss_mask[1, 3:]).any()
    np.testing.assert_array_equal(batch.token_ids[1, 3:], 0)
    np.testing.assert_array_equal(batch.old_log_probs[1, 3:], 0)


@pytest.mark.parametrize(
    "length,context_length,max_tokens,multiple,width",
    [(4, 16, 16, 4, 4), (5, 16, 16, 4, 8), (5, 6, 16, 4, 6), (5, 16, 6, 4, 6), (5, 16, 5, 4, 5), (5, 16, 16, 1, 5)],
    ids=["exact-bucket", "next-bucket", "context-cap", "token-cap", "exact-token-cap", "no-rounding"],
)
def test_prepare_batch_bucket_boundaries(
    length: int, context_length: int, max_tokens: int, multiple: int, width: int
) -> None:
    batcher = training.Batcher.__ray_metadata__.modified_class(
        actor=Mock(), config=training.Config(padding_multiple=multiple, max_tokens=max_tokens)
    )
    sample = rollout([1] * length, [False] + [True] * (length - 1), 1.0)
    groups = [training.RolloutGroup(task_id="0", rollouts=[sample])]
    batch = batcher.prepare_batch(groups, context_length=context_length)  # (B=1, T=width), (B,) leaves.
    result = training.build_rollout_batch(groups=groups, training_batch=batch, transfer_metrics={})
    assert result.rollout_metrics["batcher/token_utilization_fraction"] == pytest.approx(length / width)
    assert result.rollout_metrics["batcher/padded_sequence_tokens"] == width
    assert all(array.shape == (1, width) for array in batch[:-1])
    assert batch.advantages.shape == (1,)
    assert int(batch.attention_mask.sum()) == length
    assert int(batch.loss_mask.sum()) == length - 1
    np.testing.assert_array_equal(batch.token_ids[0, :length], sample.sequence.token_ids)


def test_prepare_minimal_sequences_and_multiple_groups(batcher: Any) -> None:
    groups = [
        training.RolloutGroup(task_id="0", rollouts=[rollout([0, 2], [False, True], 0.0)]),
        training.RolloutGroup(task_id="1", rollouts=[rollout([3, 4, 5], [False, True, True], 1.0)]),
    ]
    batch = batcher.prepare_batch(groups, context_length=8)  # (B=2, T=8), (B,) leaves.
    result = training.build_rollout_batch(groups=groups, training_batch=batch, transfer_metrics={})
    assert result.rollout_metrics["batcher/token_utilization_fraction"] == 5 / 16
    assert result.rollout_metrics["rollout/total_tokens_mean"] == 2.5
    assert batch.token_ids.shape == (2, 8)
    np.testing.assert_array_equal(batch.token_ids[:, :3], [[0, 2, 0], [3, 4, 5]])
    np.testing.assert_array_equal(batch.attention_mask[:, :3], [[True, True, False], [True, True, True]])
    np.testing.assert_array_equal(batch.loss_mask[:, :3], [[False, True, False], [False, True, True]])
    np.testing.assert_array_equal(batch.advantages, [0.0, 1.0])


@pytest.mark.parametrize(
    "failure", ["empty", "overlong", "over-budget", "no-context", "no-samples", "single-token", "empty-transcript"]
)
def test_prepare_batch_rejects_unusable_sequences(
    batcher: Any, failure: str, groups: list[training.RolloutGroup]
) -> None:
    if failure == "over-budget":
        batcher.config = training.Config(max_tokens=4)
    if failure == "empty":
        groups = []
    elif failure in {"no-context", "no-samples", "single-token", "empty-transcript"}:
        sample = {
            "no-context": rollout([1, 2], [True, True], 1.0),
            "no-samples": rollout([1, 2], [False, False], 1.0),
            "single-token": rollout([1], [True], 1.0),
            "empty-transcript": rollout([], [], 1.0),
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
    assert float(metrics["train/clip_fraction"]) == 1.0
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
        attention_backend="xla",
    )


@pytest.fixture(scope="module")
def state(model: Qwen3LM) -> training.LoRATrainState:
    params = model.init(jax.random.key(0), jnp.ones((1, 7), dtype=jnp.int32))["params"]  # HF parameter shapes.
    params = jax.tree.map(lambda value: value.astype(jnp.bfloat16), params)  # Emulate checkpoint storage.
    return training.create_state(
        model, params, training.Config(learning_rate=0.01, max_grad_norm=0.1, lora_rank=2, lora_alpha=4)
    )


@pytest.fixture
def batch(
    batcher: Any, groups: list[training.RolloutGroup], state: training.LoRATrainState
) -> training.GRPOBatch[jax.Array]:
    batch = jax.device_put(batcher.prepare_batch(groups, context_length=8))  # (B, T), (B,) leaves.
    old = training.token_log_probs(state, state.params, batch, temperature=0.7)  # (B=2, T-1=7)
    return batch._replace(old_log_probs=jnp.pad(old, ((0, 0), (1, 0))))


def test_token_log_probs_shift_temperature_and_padding(
    state: training.LoRATrainState, batch: training.GRPOBatch
) -> None:
    log_probs = training.token_log_probs(state, state.params, batch, temperature=0.7)  # (B, T-1)
    assert log_probs.shape == (2, 7)
    logits = state.apply_fn(
        {"params": state.base_params}, batch.token_ids, lora=state.params, lora_scale=state.lora_scale
    ).logits  # (B, T, V)
    expected = jax.nn.log_softmax(logits / 0.7)  # (B, T, V)
    np.testing.assert_allclose(log_probs[0, [0, 2, 3]], expected[0, [0, 2, 3], [2, 4, 5]], atol=1e-6)
    np.testing.assert_allclose(log_probs[1, 1], expected[1, 1, 6], atol=1e-6)
    padded = batch._replace(token_ids=jnp.where(batch.attention_mask, batch.token_ids, 7))
    changed = training.token_log_probs(state, state.params, padded, temperature=0.7)  # (B, T-1)
    np.testing.assert_allclose(changed[batch.loss_mask[:, 1:]], log_probs[batch.loss_mask[:, 1:]], atol=1e-6)


def test_token_log_probs_gradients_match_gather(state: training.LoRATrainState, batch: training.GRPOBatch) -> None:
    def apply_fn(
        variables: Variables,
        tokens: jax.Array,  # (B, T-1)
        *,
        attention_mask: jax.Array,  # (B, T-1)
        lora: training.LoRAParameters,  # Fake logits in the A factor: (B, T-1, V).
        lora_scale: float,
    ) -> Qwen3Output:
        return Qwen3Output(logits=lora["logits"].a, cache=None)  # (B, T-1, V)

    state = state.replace(apply_fn=apply_fn)
    batch = batch._replace(token_ids=batch.token_ids.at[:, 1].set(jnp.array([0, 7])))  # (B=2, T=8)
    params = {
        "logits": training.LoRAWeights(
            a=jnp.arange(112, dtype=jnp.float32).reshape(2, 7, 8) / 10,  # (B, T-1, V=8) fake logits.
            b=jnp.zeros((1,), dtype=jnp.float32),  # (1,) unused factor.
        )
    }

    def selected_loss(weights: training.LoRAParameters) -> jax.Array:  # Logits: (B, T-1, V).
        log_probs = training.token_log_probs(state, weights, batch, temperature=0.7)  # (B, T-1)
        return jnp.where(batch.loss_mask[:, 1:], log_probs, 0).sum()  # ()

    def reference_loss(weights: training.LoRAParameters) -> jax.Array:  # Logits: (B, T-1, V).
        log_probs = jax.nn.log_softmax(weights["logits"].a / 0.7)  # (B, T-1, V)
        selected = jnp.take_along_axis(log_probs, batch.token_ids[:, 1:, None], axis=-1)[..., 0]  # (B, T-1)
        return jnp.where(batch.loss_mask[:, 1:], selected, 0).sum()  # ()

    actual = jax.value_and_grad(selected_loss)(params)  # (), (B, T-1, V) gradient leaf.
    expected = jax.value_and_grad(reference_loss)(params)  # (), (B, T-1, V) gradient leaf.
    for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert value.shape == reference.shape
        np.testing.assert_allclose(value, reference, atol=2e-6)
    np.testing.assert_array_equal(actual[1]["logits"].a[~batch.loss_mask[:, 1:]], 0)


def test_lora_initialization_and_optimizer_shapes(state: training.LoRATrainState) -> None:
    assert set(state.params) == {
        f"model.layers.0.{projection}.weight"
        for projection in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    }
    for weight in state.base_params.values():  # HF parameter shapes.
        assert weight.dtype == jnp.bfloat16
    adam = state.opt_state[1][0]
    for tree in (state.params, adam.mu, adam.nu):  # LoRA factor shapes.
        assert set(tree) == set(state.params)
        for name, adapter in tree.items():
            output_size, input_size = state.base_params[name].shape
            assert adapter.a.shape == (2, input_size)
            assert adapter.b.shape == (output_size, 2)
            assert adapter.a.dtype == adapter.b.dtype == jnp.float32
            np.testing.assert_array_equal(adapter.b, 0)
    assert sum(value.size for value in jax.tree.leaves(state.params)) < sum(
        value.size for value in jax.tree.leaves(state.base_params)
    )


def test_lora_initialization_keeps_fp32_with_x64_enabled() -> None:
    name = "model.layers.0.self_attn.q_proj.weight"
    with jax.enable_x64():
        state = training.create_state(
            Mock(), {name: jnp.ones((2, 3), dtype=jnp.bfloat16)}, training.Config(lora_rank=1)
        )  # Base (O=2, I=3); A (1, I), B (O, 1).
    assert state.step.dtype == jnp.int32
    assert state.base_params[name].dtype == jnp.bfloat16
    for tree in (state.params, state.opt_state[1][0].mu, state.opt_state[1][0].nu):
        assert all(value.dtype == jnp.float32 for value in jax.tree.leaves(tree))


def test_lora_rank_one_and_seed() -> None:
    name = "model.layers.0.self_attn.q_proj.weight"
    params = {name: jnp.ones((2, 3), dtype=jnp.bfloat16)}  # (O=2, I=3)
    config = training.Config(lora_rank=1, lora_alpha=3)
    state = training.create_state(Mock(), params, config)
    repeated = training.create_state(Mock(), params, config)
    np.testing.assert_array_equal(state.params[name].a, repeated.params[name].a)
    changed_seed = training.create_state(Mock(), params, replace(config, seed=2))
    assert not np.array_equal(state.params[name].a, changed_seed.params[name].a)
    assert state.params[name].a.shape == (1, 3)
    assert state.params[name].b.shape == (2, 1)
    assert state.lora_scale == 3
    np.testing.assert_array_equal(state.base_params[name], 1)


@pytest.mark.parametrize("setting", [{"lora_rank": 0}, {"lora_rank": 513}, {"lora_alpha": 0}])
def test_lora_config_rejects_invalid_settings(setting: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="lora_"):
        training.Config(**setting)


def test_lora_requires_projection_weights() -> None:
    with pytest.raises(ValueError, match="No attention or MLP projection"):
        training.create_state(Mock(), {"model.norm.weight": jnp.ones(3)}, training.Config())  # (D=3,)


@pytest.mark.parametrize("mixed", [False, True], ids=["nonzero", "zero-first-microbatch"])
@pytest.mark.usefixtures("mesh_context")
def test_accumulated_update_matches_full_batch_and_improves_policy(
    state: training.LoRATrainState, batch: training.GRPOBatch[jax.Array], mixed: bool
) -> None:
    if mixed:
        batch = batch._replace(advantages=batch.advantages.at[0].set(0))  # (B, T), (B,) leaves.
    accumulated, metrics = training.update(jax.tree.map(jnp.copy, state), batch, 2, 0.7, 0.2)
    full, full_metrics = training.update(jax.tree.map(jnp.copy, state), batch, 1, 0.7, 0.2)
    assert int(accumulated.step) == int(full.step) == 1
    for actual, expected in zip(jax.tree.leaves(accumulated), jax.tree.leaves(full), strict=True):
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
    assert all(value.dtype == jnp.float32 for value in jax.tree.leaves(accumulated.params))
    assert all(value.shape == () for value in metrics.values())
    assert float(metrics["train/gradient_norm"]) > 0.1  # Exercise gradient clipping.
    assert float(metrics["train/skipped_update_fraction"]) == 0.0
    for name in metrics:
        np.testing.assert_allclose(metrics[name], full_metrics[name], atol=1e-6)
    after = training.token_log_probs(accumulated, accumulated.params, batch, temperature=0.7)  # (B, T-1)
    loss, _ = training.objective(after, batch, clip_coef=0.2)
    assert float(loss) < float(metrics["train/policy_loss"])
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(jax.tree.leaves(state.params), jax.tree.leaves(accumulated.params), strict=True)
    )
    for name, value in accumulated.base_params.items():  # HF parameter shapes.
        np.testing.assert_array_equal(value, state.base_params[name])


@pytest.mark.usefixtures("mesh_context")
def test_update_logs_only_new_traces(
    state: training.LoRATrainState, batch: training.GRPOBatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A unique callable isolates this test's JIT cache without clearing other tests' entries.
    state = jax.device_put(state.replace(apply_fn=partial(state.apply_fn)), PartitionSpec())
    batch = jax.device_put(batch, PartitionSpec("batch"))  # (B, T), (B,) leaves.

    def trace(
        current: training.LoRATrainState,
        minibatch: training.GRPOBatch[jax.Array],  # (B, T), (B,) leaves.
    ) -> jax.stages.Lowered:
        return training.update.lower(current, minibatch, 2, 0.7, 0.2)

    caplog.set_level(logging.INFO, logger=training.__name__)
    trace(state, batch)
    trace(state, batch)
    # The device scalar returned by the first update must reuse the initial trace.
    trace(state.replace(step=jnp.asarray(1, dtype=jnp.int32)), batch)
    assert caplog.messages == ["JIT trace update: batch_shape=(2, 8) grad_accum_steps=2 temperature=0.7 clip_coef=0.2"]
    caplog.clear()
    shorter = jax.tree.map(lambda value: value[:, :5] if value.ndim == 2 else value, batch)  # (B, 5), (B,).
    trace(state, shorter)
    trace(state, batch)
    assert caplog.messages == ["JIT trace update: batch_shape=(2, 5) grad_accum_steps=2 temperature=0.7 clip_coef=0.2"]


@pytest.mark.parametrize("has_momentum", [False, True], ids=["fresh", "with-momentum"])
@pytest.mark.usefixtures("mesh_context")
def test_zero_advantages_preserve_optimizer_state(
    state: training.LoRATrainState, batch: training.GRPOBatch[jax.Array], has_momentum: bool
) -> None:
    if has_momentum:
        state = state.apply_gradients(grads=jax.tree.map(jnp.ones_like, state.params))
    batch = batch._replace(advantages=jnp.zeros_like(batch.advantages))  # (B, T), (B,) leaves.
    updated, metrics = training.update(jax.tree.map(jnp.copy, state), batch, 2, 0.7, 0.2)
    assert float(metrics["train/gradient_norm"]) == float(metrics["train/policy_loss"]) == 0.0
    assert all(value.shape == () for value in metrics.values())
    assert {name: float(value) for name, value in metrics.items()} == {
        "train/policy_loss": 0.0,
        "train/approx_kl": 0.0,
        "train/clip_fraction": 0.0,
        "train/gradient_norm": 0.0,
        "train/skipped_update_fraction": 1.0,
    }
    for actual, expected in zip(jax.tree.leaves(updated), jax.tree.leaves(state), strict=True):
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("grad_accum_steps", [0, 3], ids=["nonpositive", "indivisible"])
@pytest.mark.usefixtures("mesh_context")
def test_update_rejects_invalid_accumulation(
    state: training.LoRATrainState, batch: training.GRPOBatch, grad_accum_steps: int
) -> None:
    with pytest.raises(ValueError, match="grad_accum_steps must be positive and divide the minibatch size"):
        training.update.lower(state, batch, grad_accum_steps, 0.7, 0.2)


@pytest.mark.parametrize("skip", [False, True], ids=["zero-advantage-replica", "all-zero"])
@pytest.mark.usefixtures("mesh_context")
def test_data_parallel_update_matches_single_device(
    state: training.LoRATrainState, batch: training.GRPOBatch[jax.Array], skip: bool
) -> None:
    if len(jax.devices()) < 2:
        pytest.skip("Requires two devices; use JAX_NUM_CPU_DEVICES=2 on CPU")
    mesh = Mesh(devices=jax.devices()[:2], axis_names=("batch",))
    replicated = NamedSharding(mesh, PartitionSpec())
    batch = jax.tree.map(lambda value: jnp.concatenate([value, value]), batch)  # (B=4, T=8), (B,).
    batch = batch._replace(
        advantages=jnp.array([0.0, 0.0 if skip else -1.0, 0.0, 0.0 if skip else 1.0])  # (B=4,)
    )
    expected, expected_metrics = training.update(jax.tree.map(jnp.copy, state), batch, 2, 0.7, 0.2)
    with jax.set_mesh(mesh):
        actual, metrics = training.update(
            jax.device_put(jax.tree.map(jnp.copy, state), replicated),
            jax.device_put(batch, NamedSharding(mesh, PartitionSpec("batch"))),
            2,
            0.7,
            0.2,
        )
    for result, reference in zip(
        jax.tree.leaves((actual, metrics)), jax.tree.leaves((expected, expected_metrics)), strict=True
    ):
        assert result.shape == reference.shape
        assert result.sharding.is_fully_replicated
        assert result.devices() == set(mesh.devices.flat)
        for shard in result.addressable_shards:
            np.testing.assert_allclose(shard.data, reference, atol=2e-6, rtol=2e-5)


def test_data_parallel_update_rejects_indivisible_microbatch(
    state: training.LoRATrainState, batch: training.GRPOBatch[jax.Array]
) -> None:
    if len(jax.devices()) < 2:
        pytest.skip("Requires two devices; use JAX_NUM_CPU_DEVICES=2 on CPU")
    mesh = Mesh(devices=jax.devices()[:2], axis_names=("batch",))
    with (
        jax.set_mesh(mesh),
        pytest.raises(ValueError, match="learner device count must divide the microbatch size"),
    ):
        training.update.lower(state, batch, 2, 0.7, 0.2)


@pytest.mark.parametrize(
    "settings",
    [{"learner_gpus": 0}, {"learner_gpus": 2}, {"learner_gpus": 2, "minibatches_per_epoch": 2, "grad_accum_steps": 2}],
    ids=["nonpositive", "small-microbatch", "accumulated-small-microbatch"],
)
def test_data_parallel_config_rejects_invalid_settings(settings: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="learner_gpus"):
        training.Config(**settings)


@pytest.mark.parametrize(
    "epochs,minibatches", [(1, 1), (1, 4), (2, 2)], ids=["single-minibatch", "default", "multiple-epochs"]
)
def test_learner_shuffles_each_epoch_and_averages_metrics(
    state: training.LoRATrainState,
    batch: training.GRPOBatch,
    monkeypatch: pytest.MonkeyPatch,
    epochs: int,
    minibatches: int,
) -> None:
    learner = training.Learner.__ray_metadata__.modified_class(
        batcher=Mock(),
        actor=Mock(),
        config=training.Config(group_size=8, update_epochs=epochs, minibatches_per_epoch=minibatches),
    )
    learner.mesh = Mesh(devices=np.array([jax.devices()[0]]), axis_names=("batch",))
    learner.state = state
    batch = jax.tree.map(lambda value: jnp.concatenate([value] * 4), batch)  # (B=8, T=8), (B,) leaves.
    batch = batch._replace(advantages=jnp.arange(8, dtype=jnp.float32))  # (B,) unique rollout IDs.
    observed: list[training.GRPOBatch] = []  # Minibatches: (B/N, T), (B/N,) leaves.

    def update(
        current: training.LoRATrainState,
        minibatch: training.GRPOBatch,  # (B/N, T), (B/N,) leaves.
        grad_accum_steps: int,
        temperature: float,
        clip_coef: float,
    ) -> tuple[training.LoRATrainState, training.Metrics]:
        assert minibatch.token_ids.shape == (8 // minibatches, 8)
        assert minibatch.advantages.shape == (8 // minibatches,)
        assert grad_accum_steps == 1
        assert jax.sharding.get_mesh() == learner.mesh
        observed.append(minibatch)
        return current.replace(step=current.step + 1), {"train/policy_loss": jnp.asarray(current.step)}  # () metric.

    monkeypatch.setattr(training, "update", update)
    host_batch = jax.tree.map(np.asarray, batch)  # (B, T), (B,) leaves.
    with jax.set_mesh(learner.mesh):
        metrics = learner.optimize(host_batch)
    steps = epochs * minibatches
    assert int(learner.state.step) == steps
    assert metrics == {"train/policy_loss": (steps - 1) / 2}
    with jax.set_mesh(learner.mesh):
        learner.optimize(host_batch)
    orders: list[np.ndarray] = []  # Each epoch: (B,).
    for start in range(0, len(observed), minibatches):
        epoch = jax.tree.map(
            lambda *values: jnp.concatenate(values), *observed[start : start + minibatches]
        )  # (B, T), (B,) leaves.
        order = np.asarray(epoch.advantages, dtype=np.int32)  # (B,)
        np.testing.assert_array_equal(np.sort(order), np.arange(8))
        for actual, original in zip(epoch, batch, strict=True):
            np.testing.assert_array_equal(actual, original[order])
        orders.append(order)
    assert all(not np.array_equal(orders[0], order) for order in orders[1:])
    learner.state = state
    observed.clear()
    with jax.set_mesh(learner.mesh):
        learner.optimize(host_batch)
    replay = jnp.concatenate([minibatch.advantages for minibatch in observed])  # (epochs * B,)
    np.testing.assert_array_equal(replay, np.concatenate(orders[:epochs]))


def test_learner_rejects_indivisible_rollout_batch(state: training.LoRATrainState, batch: training.GRPOBatch) -> None:
    learner = training.Learner.__ray_metadata__.modified_class(
        batcher=Mock(), actor=Mock(), config=training.Config(group_size=6, minibatches_per_epoch=3)
    )
    learner.mesh = Mesh(devices=np.array([jax.devices()[0]]), axis_names=("batch",))
    learner.state = state
    with pytest.raises(ValueError, match="minibatches_per_epoch must divide the rollout batch size"):
        learner.optimize(jax.tree.map(np.asarray, batch))
    assert int(learner.state.step) == 0


def test_learner_optimizes_and_publishes_updated_adapters(
    batcher: Any,
    model: Qwen3LM,
    state: training.LoRATrainState,
    groups: list[training.RolloutGroup],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = training.Learner.__ray_metadata__.modified_class(
        batcher=Mock(),
        actor=Mock(),
        config=training.Config(temperature=0.7, minibatches_per_epoch=2, update_epochs=2, lora_rank=2, lora_alpha=4),
    )
    learner.model = model.clone(dtype="bfloat16", param_dtype=jnp.bfloat16)
    learner.mesh = Mesh(devices=jax.devices()[:2], axis_names=("batch",))
    learner.state = training.create_state(
        learner.model,
        jax.tree.map(jnp.copy, state.base_params),
        training.Config(learning_rate=0.01, lora_rank=2, lora_alpha=4),
    )
    learner.state = jax.device_put(learner.state, NamedSharding(learner.mesh, PartitionSpec()))
    groups = groups + [replace(groups[0], rollouts=list(reversed(groups[0].rollouts)))]
    batch = batcher.prepare_batch(groups, context_length=8)  # (B=4, T=8), (B,) leaves.
    old = training.token_log_probs(
        learner.state, learner.state.params, jax.device_put(batch), temperature=0.7
    )  # (B, T-1)
    assert old.shape == (4, 7)
    assert old.dtype == jnp.float32
    batch = batch._replace(old_log_probs=np.pad(np.asarray(old), ((0, 0), (1, 0))))  # (B, T), (B,) leaves.
    assert all(isinstance(value, np.ndarray) for value in batch)
    actual_update = training.update
    steps: list[int] = []

    def record_update(
        current: training.LoRATrainState,
        minibatch: training.GRPOBatch,  # (B/N, T), (B/N,) leaves.
        grad_accum_steps: int,
        temperature: float,
        clip_coef: float,
    ) -> tuple[training.LoRATrainState, training.Metrics]:
        steps.append(int(current.step))
        return actual_update(current, minibatch, grad_accum_steps, temperature, clip_coef)

    update = Mock(wraps=record_update)
    monkeypatch.setattr(training, "update", update)
    permutations = [jnp.array([2, 0, 3, 1]), jnp.array([1, 3, 0, 2])]  # Each: (B,).
    permutation = Mock(side_effect=permutations)
    monkeypatch.setattr(training.jax.random, "permutation", permutation)
    with jax.set_mesh(learner.mesh):
        metrics = learner.optimize(batch)
    assert update.call_count == 4
    assert permutation.call_count == 2
    assert steps == [0, 1, 2, 3]
    for index, call in enumerate(update.call_args_list):
        assert jax.tree.leaves(call.args[0].params)[0].is_deleted()
        assert call.args[2] == 1
        placed_batch = call.args[1]  # (B/N=2, T=8), (B/N,) leaves.
        start = (index % 2) * 2
        indices = permutations[index // 2][start : start + 2]  # (B/N,)
        for placed, original in zip(placed_batch, batch, strict=True):
            assert placed.shape == (2, *original.shape[1:])
            assert placed.sharding == NamedSharding(learner.mesh, PartitionSpec("batch"))
            np.testing.assert_array_equal(placed, original[indices])
    assert int(learner.state.step) == 4
    assert metrics["train/gradient_norm"] > 0
    assert all(isinstance(value, float) for value in metrics.values())
    for value in jax.tree.leaves(learner.state):  # HF/LoRA parameter shapes and scalar optimizer counters.
        assert value.sharding.is_fully_replicated
        assert value.devices() == set(learner.mesh.devices.flat)
    adam = learner.state.opt_state[1][0]
    for tree in (learner.state.params, adam.mu, adam.nu):  # LoRA factor shapes.
        for value, initial in zip(jax.tree.leaves(tree), jax.tree.leaves(state.params), strict=True):
            assert value.shape == initial.shape
            assert value.dtype == jnp.float32
    for name, value in learner.state.base_params.items():  # HF parameter shapes.
        np.testing.assert_array_equal(value, state.base_params[name])
    assert optax.global_norm(jax.tree.map(jnp.subtract, learner.state.params, state.params)) > 0

    def receive(path: str) -> None:
        weights = load_file(Path(path) / "adapter_model.safetensors")  # (R, I) and (O, R) leaves.
        assert len(weights) == 2 * len(learner.state.params)
        for name, adapter in learner.state.params.items():
            for factor, value in (("A", adapter.a), ("B", adapter.b)):  # (R, I) or (O, R)
                key = f"base_model.model.{name.removesuffix('.weight')}.lora_{factor}.weight"
                np.testing.assert_array_equal(weights[key], np.asarray(value))

    learner.actor.update_weights.remote.side_effect = receive
    monkeypatch.setattr(training.ray, "get", Mock())
    learner.publish_weights()
    learner.actor.update_weights.remote.assert_called_once()
    assert not Path(learner.actor.update_weights.remote.call_args.args[0]).exists()

    def fail_load(result: Any) -> None:
        path = Path(learner.actor.update_weights.remote.call_args.args[0])
        assert (path / "adapter_config.json").is_file()
        assert (path / "adapter_model.safetensors").is_file()
        raise RuntimeError("Adapter load failed")

    monkeypatch.setattr(training.ray, "get", fail_load)
    with pytest.raises(RuntimeError, match="Adapter load failed"):
        learner.publish_weights()
    assert not Path(learner.actor.update_weights.remote.call_args.args[0]).exists()
