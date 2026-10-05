from dataclasses import replace
from pathlib import Path
from typing import NamedTuple
from unittest.mock import patch

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.training.train_state import TrainState

from rl2.ast_transformer import ASTTransformer, _ASTBlock
from rl2.karel import KarelConfig, _parse
from rl2.karel_ast import ACTION_ID, AST_ACTIONS, ASTFeatures, KarelAST, batch_features, teacher_forcing
from rl2.train_karel_ast_grpo import (
    Config,
    bucket_tree,
    create_state,
    generate,
    load_config,
    predict,
)
from rl2.transformer import AttentionImplementation
from rl2.tree_attention import Relation, tree_relations


class ModelBatch(NamedTuple):
    initial: np.ndarray
    target: np.ndarray
    tree: ASTFeatures
    actions: np.ndarray


@pytest.fixture(scope="module")
def batch() -> ModelBatch:
    tokens = (
        "DEF",
        "run",
        "m(",
        "REPEAT",
        "R=2",
        "r(",
        "IF",
        "c(",
        "frontIsClear",
        "c)",
        "i(",
        "move",
        "i)",
        "r)",
        "m)",
    )
    snapshots, actions = teacher_forcing(tokens, 24, 16)
    # Include structural and value choices plus forced steps.
    initial = np.zeros((len(actions), 4, 4, 6), np.int32)
    initial[:, 1, 1, 0] = 1
    target = initial.copy()
    target[:, 1, 1, 5] = 1
    return ModelBatch(initial, target, batch_features(snapshots), actions)


@pytest.fixture(scope="module")
def config() -> Config:
    return Config(
        total_updates=2,
        num_tasks=1,
        group_size=2,
        num_minibatches=1,
        update_epochs=1,
        d_model=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        max_nodes=24,
        max_depth=16,
        learning_rate=0.01,
        log_program_interval=2,
        log_program_count=2,
        env=KarelConfig(height=4, width=4, max_depth=0, max_statements=2),
    )


@pytest.fixture(scope="module")
def state(config: Config, batch: ModelBatch) -> TrainState:
    return create_state(config, batch.initial[:1], batch.target[:1])


def test_trimmed_sequence_matches_full_logits_and_gradients(config: Config, batch: ModelBatch) -> None:
    config = replace(config, max_nodes=64)
    initial, target = batch.initial[:4], batch.target[:4]
    tree = ASTFeatures(
        *(np.pad(field[:4], ((0, 0), (0, 64 - field.shape[1])) + ((0, 0),) * (field.ndim - 2)) for field in batch.tree)
    )
    trimmed = bucket_tree(tree)
    assert trimmed.node_mask.shape == (4, 32)
    state = create_state(config, initial, target)
    params = dict(state.params)
    for name in ("constructor_head", "value_head"):
        params[name] = {
            **params[name],
            "kernel": jax.random.normal(jax.random.key(5), params[name]["kernel"].shape),
        }
    params["layers_0"] = {
        **params["layers_0"],
        "tree_bias": jax.tree.map(
            lambda value: jax.random.normal(jax.random.key(value.size), value.shape), params["layers_0"]["tree_bias"]
        ),
    }

    def loss(parameters: dict, features: ASTFeatures) -> tuple[jax.Array, jax.Array]:
        logits = state.apply_fn({"params": parameters}, initial, target, features)
        # Ignore impossible productions in both the objective and its gradient.
        return jnp.where(features.action_mask, logits, 0).sum(), logits

    (full_loss, full_logits), full_grads = jax.jit(jax.value_and_grad(loss, has_aux=True))(params, tree)
    (small_loss, small_logits), small_grads = jax.jit(jax.value_and_grad(loss, has_aux=True))(params, trimmed)
    np.testing.assert_allclose(small_logits, full_logits[:, :32], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(small_loss, full_loss, atol=3e-5, rtol=3e-5)
    chex.assert_trees_all_close(small_grads, full_grads, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize(
    "use_bias,zero_qk",
    [
        pytest.param(False, False, id="content"),
        pytest.param(True, False, id="content-and-bias"),
        pytest.param(True, True, id="bias-only"),
    ],
)
def test_bidirectional_attention_matches_numpy_and_excludes_padding(use_bias: bool, zero_qk: bool) -> None:
    block = _ASTBlock(16, 2, 1, 32, 1, jnp.float32, "xla")
    x = jax.random.normal(jax.random.key(12), (2, 5, 16))
    present = jnp.asarray([[True, True, True, False, False], [True, True, True, True, True]])
    relations = tree_relations(jnp.asarray([[0, 1, 0, 0], [0, 1, 2, 1]], jnp.int32), present[:, 1:])
    params = block.init(jax.random.key(13), x, present, relations)["params"]
    params = {**params, "down": {"kernel": jnp.zeros_like(params["down"]["kernel"])}}
    params["query_norm"]["scale"] = jnp.linspace(0.5, 1.5, 8)
    params["key_norm"]["scale"] = jnp.linspace(1.5, 0.5, 8)
    if zero_qk:
        for name in ("query", "key"):
            params[name]["kernel"] = jnp.zeros_like(params[name]["kernel"])
    if use_bias:
        params["tree_bias"] = jax.tree.map(
            lambda v: jax.random.normal(jax.random.key(v.size), v.shape), params["tree_bias"]
        )
    actual = block.apply({"params": params}, x, present, relations)
    tables = {name: np.asarray(value) for name, value in params["tree_bias"].items()}
    bias = tables["relation"][relations.kind]
    bias += np.where(
        (relations.kind <= Relation.OTHER)[..., None],
        tables["distance"][relations.distance] + tables["relative_depth"][relations.relative_depth],
        0,
    )
    bias = np.where((relations.kind != Relation.PADDING)[..., None], bias, 0)
    normalized = np.asarray(x) / np.sqrt(np.mean(np.asarray(x) ** 2, axis=-1, keepdims=True) + 1e-6)
    normalized *= np.asarray(params["attention_norm"]["scale"])
    query = (normalized @ params["query"]["kernel"]).reshape(2, 5, 2, 8)
    key = (normalized @ params["key"]["kernel"]).reshape(2, 5, 1, 8)
    value = (normalized @ params["value"]["kernel"]).reshape(2, 5, 1, 8)
    query = np.asarray(query) / np.sqrt(np.mean(np.asarray(query) ** 2, axis=-1, keepdims=True) + 1e-6)
    key = np.asarray(key) / np.sqrt(np.mean(np.asarray(key) ** 2, axis=-1, keepdims=True) + 1e-6)
    query *= np.asarray(params["query_norm"]["scale"])
    key *= np.asarray(params["key_norm"]["scale"])
    attended = np.zeros((2, 5, 2, 8), np.float32)
    for b in range(2):
        for t in range(5):
            for h in range(2):
                scores = np.asarray(key[b, :, 0] @ query[b, t, h]) / np.sqrt(8)
                scores += bias[b, t, :, h]
                scores = np.where(present[b], scores, -np.inf)
                weights = np.exp(scores - scores.max())
                attended[b, t, h] = weights / weights.sum() @ value[b, :, 0]
    expected = x + attended.reshape(2, 5, 16) @ params["attention_out"]["kernel"]
    np.testing.assert_allclose(actual[present], expected[present], rtol=2e-5, atol=2e-6)
    # With query lengths, padded attention outputs are zero (the residual remains).
    np.testing.assert_array_equal(actual[~present], x[~present])

    def first_output(inputs: jax.Array) -> jax.Array:
        chex.assert_shape(inputs, x.shape)
        chex.assert_type(inputs, jnp.float32)
        return block.apply({"params": params}, inputs, present, relations)[0, 0].sum()

    gradients = jax.grad(first_output)(x)
    assert np.linalg.norm(gradients[0, 2]) > 1e-5  # A later present node influences the first position.
    np.testing.assert_array_equal(gradients[0, 3:], 0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_sequence_lengths_match_explicit_mask_outputs_and_gradients(dtype: jax.typing.DTypeLike) -> None:
    block = _ASTBlock(16, 2, 1, 32, 1, dtype, "xla")
    x = jax.random.normal(jax.random.key(31), (3, 6, 16))
    # Prefix-only, partially filled, and completely filled sequences in one batch.
    present = jnp.arange(6)[None] < jnp.asarray([1, 3, 6])[:, None]
    depths = jnp.asarray([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 2, 1, 2]], jnp.int32)
    relations = tree_relations(depths, present[:, 1:])
    params = block.init(jax.random.key(32), x, present, relations)["params"]
    params["tree_bias"] = jax.tree.map(
        lambda v: jax.random.normal(jax.random.key(v.size), v.shape), params["tree_bias"]
    )
    attention = jax.nn.dot_product_attention

    def masked_attention(
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        *,
        bias: jax.Array,
        query_seq_lengths: jax.Array,
        key_value_seq_lengths: jax.Array,
        is_causal: bool,
        implementation: AttentionImplementation,
    ) -> jax.Array:
        chex.assert_rank((query, key, value, bias), 4)
        chex.assert_type((query, key, value, bias), jnp.floating)
        chex.assert_shape((query_seq_lengths, key_value_seq_lengths), (3,))
        chex.assert_type((query_seq_lengths, key_value_seq_lengths), jnp.int32)
        return attention(
            query,
            key,
            value,
            bias=bias,
            mask=present[:, None, None, :],
            is_causal=is_causal,
            implementation=implementation,
        )

    def loss(parameters: dict, inputs: jax.Array) -> tuple[jax.Array, jax.Array]:
        chex.assert_shape(inputs, x.shape)
        chex.assert_type(inputs, jnp.float32)
        output = block.apply({"params": parameters}, inputs, present, relations)
        live = jnp.where(present[..., None], output, 0)
        return jnp.square(live).sum(), live

    (_, actual), actual_grads = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(params, x)
    with patch("jax.nn.dot_product_attention", side_effect=masked_attention):
        (_, expected), expected_grads = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(params, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    for actual_grad, expected_grad in zip(jax.tree.leaves(actual_grads), jax.tree.leaves(expected_grads)):
        assert np.isfinite(actual_grad).all()
        np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(actual_grads[1][~present], 0)


@pytest.mark.skipif(not any(device.platform == "gpu" for device in jax.devices()), reason="cuDNN requires a GPU")
def test_cudnn_tree_bias_matches_xla_forward_and_gradients() -> None:
    # Odd length exercises the cuDNN padding path; different trees exercise
    # per-example bias gradients rather than a bias shared across the batch.
    x = jax.random.normal(jax.random.key(3), (2, 5, 32))
    present = jnp.asarray([[True, True, True, False, False], [True, True, True, True, True]])
    relations = tree_relations(jnp.asarray([[0, 1, 0, 0], [0, 1, 1, 2]], jnp.int32), present[:, 1:])
    xla = _ASTBlock(32, 2, 1, 32, 1, jnp.bfloat16, "xla")
    cudnn = _ASTBlock(32, 2, 1, 32, 1, jnp.bfloat16, "cudnn")
    params = xla.init(jax.random.key(4), x, present, relations)["params"]
    params["tree_bias"] = jax.tree.map(
        lambda v: jax.random.normal(jax.random.key(v.size), v.shape), params["tree_bias"]
    )

    def loss(parameters: dict, block: _ASTBlock) -> jax.Array:
        output = block.apply({"params": parameters}, x, present, relations)
        return jnp.where(present[..., None], output**2, 0).mean()

    expected, expected_grad = jax.jit(jax.value_and_grad(lambda p: loss(p, xla)))(params)
    actual, actual_grad = jax.jit(jax.value_and_grad(lambda p: loss(p, cudnn)))(params)
    np.testing.assert_allclose(actual, expected, rtol=0.02, atol=0.01)
    for name in params["tree_bias"]:
        assert np.isfinite(actual_grad["tree_bias"][name]).all()
        assert np.linalg.norm(actual_grad["tree_bias"][name]) > 0
        np.testing.assert_allclose(
            actual_grad["tree_bias"][name], expected_grad["tree_bias"][name], rtol=0.1, atol=0.005
        )


def test_initial_policy_is_uniform_over_typed_actions(config: Config, batch: ModelBatch, state: TrainState) -> None:
    logits = predict(state, batch.initial, batch.target, batch.tree)
    chex.assert_shape(logits, (len(batch.actions), config.max_nodes, len(AST_ACTIONS)))
    logits = logits[jnp.arange(len(batch.actions)), batch.tree.action_mask.any(axis=-1).argmax(axis=-1)]
    chex.assert_type(logits, jnp.float32)
    masks = batch.tree.action_mask[np.arange(len(batch.actions)), batch.tree.action_mask.any(axis=-1).argmax(axis=-1)]
    expected = masks / masks.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(jax.nn.softmax(logits), expected, rtol=1e-6)
    assert np.isneginf(np.asarray(logits)[:, 0]).all()
    count_row = list(batch.actions).index(ACTION_ID["R=2"])
    assert batch.tree.action_mask[count_row].sum() == 20


def test_padded_features_do_not_affect_predictions(config: Config, batch: ModelBatch, state: TrainState) -> None:
    params = dict(state.params)
    for name in ("constructor_head", "value_head"):
        params[name] = {**params[name], "kernel": jax.random.normal(jax.random.key(5), params[name]["kernel"].shape)}
    params["layers_0"] = dict(params["layers_0"])
    params["layers_0"]["tree_bias"] = jax.tree.map(
        lambda v: jax.random.normal(jax.random.key(v.size), v.shape), params["layers_0"]["tree_bias"]
    )
    state = state.replace(params=params)
    dirty = ASTFeatures(
        *(np.where(batch.tree.node_mask, array, 999999).astype(np.int32) for array in batch.tree[:5]),
        batch.tree.node_mask,
        batch.tree.action_mask,
    )
    np.testing.assert_array_equal(
        predict(state, batch.initial, batch.target, batch.tree), predict(state, batch.initial, batch.target, dirty)
    )


@pytest.mark.parametrize("bf16", [False, True])
def test_masked_head_gradients_are_finite_and_learnable(config: Config, batch: ModelBatch, bf16: bool) -> None:
    state = create_state(replace(config, bf16=bf16), batch.initial[:1], batch.target[:1])
    initial_params = state.params

    @jax.jit
    def train_step(state: TrainState) -> tuple[TrainState, jax.Array]:
        def loss(params: dict) -> jax.Array:
            logits = state.apply_fn({"params": params}, batch.initial, batch.target, batch.tree)
            logits = logits[jnp.arange(len(batch.actions)), batch.tree.action_mask.any(axis=-1).argmax(axis=-1)]
            return -jnp.take_along_axis(jax.nn.log_softmax(logits), jnp.asarray(batch.actions)[:, None], axis=-1).mean()

        value, grads = jax.value_and_grad(loss)(state.params)
        return state.apply_gradients(grads=grads), value

    losses = []
    for _ in range(3):
        state, value = train_step(state)
        assert np.isfinite(value)
        losses.append(float(value))
    assert losses[-1] < losses[0]
    for array in jax.tree.leaves(state.params):
        assert array.dtype == jnp.float32
        assert np.isfinite(array).all()
    for name in (
        "context_projection",
        "node_embedding",
        "value_embedding",
        "constructor_head",
        "value_head",
        "layers_0",
    ):
        assert any(
            not np.array_equal(a, b)
            for a, b in zip(jax.tree.leaves(initial_params[name]), jax.tree.leaves(state.params[name]))
        )
    for name in ("relation", "distance", "relative_depth"):
        assert np.linalg.norm(state.params["layers_0"]["tree_bias"][name]) > 0
    for name in ("query_norm", "key_norm"):
        assert not np.array_equal(initial_params["layers_0"][name]["scale"], state.params["layers_0"][name]["scale"])


def test_sampling_finishes_and_completed_trees_use_safe_dummy_logits(
    config: Config, batch: ModelBatch, state: TrainState
) -> None:
    trees = generate(state, batch.initial[:4], batch.target[:4], jax.random.key(20), config.max_nodes, config.max_depth)
    for tree in trees:
        assert tree.complete
        _parse(tree.tokens())
    logits = predict(state, batch.initial[:4], batch.target[:4], batch_features(tuple(t.features() for t in trees)))
    expected = np.zeros((4, config.max_nodes, len(AST_ACTIONS)), np.float32)
    expected[..., 0] = 1
    np.testing.assert_array_equal(jax.nn.softmax(logits), expected)


def test_example_config_and_invalid_model_dimensions(batch: ModelBatch) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs/karel_ast_grpo.yaml")
    assert config.bf16 and config.max_nodes == 128
    model = ASTTransformer(d_model=15, num_heads=2, num_layers=1, max_nodes=24)
    with pytest.raises(AssertionError):
        model.init(jax.random.key(1), batch.initial, batch.target, batch.tree)


def test_parallel_predictions_cover_every_hole_without_frontier(
    config: Config, batch: ModelBatch, state: TrainState
) -> None:
    tree = KarelAST.empty(config.max_nodes, config.max_depth)
    for name in ("Program", "ConsNonEmpty", "IFELSE"):
        tree = tree.expand(ACTION_ID[name])
    features = batch_features((tree.features(),))
    assert features.is_hole.sum() == 4
    logits = predict(state, batch.initial[:1], batch.target[:1], features)
    np.testing.assert_array_equal(np.isfinite(logits[..., 1:]), features.action_mask[..., 1:])
    probabilities = np.asarray(jax.nn.softmax(logits))
    for position in range(config.max_nodes):
        allowed = features.action_mask[0, position]
        if features.is_hole[0, position]:
            np.testing.assert_allclose(probabilities[0, position], allowed / allowed.sum(), rtol=1e-6)
        else:
            assert probabilities[0, position, 0] == 1
    assert not hasattr(features, "frontier")
