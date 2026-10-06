from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.training.train_state import TrainState

from rl2.ast_transformer import ASTTransformer
from rl2.karel import KarelConfig, _parse
from rl2.karel_ast import ACTION_ID, AST_ACTIONS, ASTFeatures, KarelAST, batch_features, teacher_forcing
from rl2.shape_checker import ShapeChecker
from rl2.train_karel_ast_grpo import (
    Config,
    bucket_tree,
    create_state,
    generate,
    load_config,
    predict,
)


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
        *(np.pad(field[:4], ((0, 0), (0, 40))) for field in batch.tree[:5]),
        batch.tree.seq_len[:4],
        np.pad(batch.tree.action_mask[:4], ((0, 0), (0, 40), (0, 0))),
    )
    trimmed = bucket_tree(tree)
    assert trimmed.node_type.shape == (4, 32)
    np.testing.assert_array_equal(trimmed.seq_len, tree.seq_len)
    state = create_state(config, initial, target)
    params = dict(state.params)
    for name in ("constructor_head", "value_head"):
        params[name] = {
            **params[name],
            "kernel": jax.random.normal(jax.random.key(5), params[name]["kernel"].shape),
        }
    for name, array in params["tree_bias_0"].items():
        params["tree_bias_0"][name] = jax.random.normal(jax.random.key(array.size), array.shape)

    def loss(parameters: dict, features: ASTFeatures) -> tuple[jax.Array, jax.Array]:
        logits = state.apply_fn({"params": parameters}, initial, target, features)
        # Ignore impossible productions in both the objective and its gradient.
        return jnp.where(features.action_mask, logits, 0).sum(), logits

    (full_loss, full_logits), full_grads = jax.jit(jax.value_and_grad(loss, has_aux=True))(params, tree)
    (small_loss, small_logits), small_grads = jax.jit(jax.value_and_grad(loss, has_aux=True))(params, trimmed)
    np.testing.assert_allclose(small_logits, full_logits[:, :32], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(small_loss, full_loss, atol=3e-5, rtol=3e-5)
    chex.assert_trees_all_close(small_grads, full_grads, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(not any(device.platform == "gpu" for device in jax.devices()), reason="cuDNN requires a GPU")
def test_cudnn_tree_bias_matches_xla_forward_and_gradients(batch: ModelBatch) -> None:
    model = ASTTransformer(d_model=16, num_heads=2, num_kv_heads=1, num_layers=1, max_nodes=24, dtype=jnp.bfloat16)
    fused = model.clone(attention_implementation="cudnn")
    initial, target = batch.initial[:4], batch.target[:4]
    tree = ASTFeatures(*(field[:4] for field in batch.tree))
    params = model.init(jax.random.key(4), initial, target, tree)["params"]
    for name in ("constructor_head", "value_head"):
        params[name]["kernel"] = jax.random.normal(jax.random.key(5), params[name]["kernel"].shape)
    for name, array in params["tree_bias_0"].items():
        params["tree_bias_0"][name] = jax.random.normal(jax.random.key(array.size), array.shape)

    def loss(parameters: dict, model: ASTTransformer) -> jax.Array:
        logits = model.apply({"params": parameters}, initial, target, tree)
        return jnp.where(tree.action_mask, logits, 0).sum()

    expected, expected_grad = jax.jit(jax.value_and_grad(partial(loss, model=model)))(params)
    actual, actual_grad = jax.jit(jax.value_and_grad(partial(loss, model=fused)))(params)
    np.testing.assert_allclose(actual, expected, rtol=0.02, atol=0.01)
    for name in params["tree_bias_0"]:
        gradient, reference = actual_grad["tree_bias_0"][name], expected_grad["tree_bias_0"][name]
        assert np.isfinite(gradient).all()
        assert np.linalg.norm(gradient) > 0
        np.testing.assert_allclose(gradient, reference, rtol=0.1, atol=0.005)


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
    params["tree_bias_0"] = {
        name: jax.random.normal(jax.random.key(array.size), array.shape)
        for name, array in params["tree_bias_0"].items()
    }
    state = state.replace(params=params)
    present = np.arange(config.max_nodes)[None, :] < batch.tree.seq_len[:, None]
    dirty = ASTFeatures(
        *(np.where(present, array, 999999).astype(np.int32) for array in batch.tree[:5]),
        batch.tree.seq_len,
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
        "backbone",
        "tree_bias_0",
    ):
        assert any(
            not np.array_equal(a, b)
            for a, b in zip(jax.tree.leaves(initial_params[name]), jax.tree.leaves(state.params[name]))
        )
    for name in ("relation", "distance", "relative_depth"):
        assert np.linalg.norm(state.params["tree_bias_0"][name]) > 0


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


def test_ast_backbone_accepts_odd_head_width_without_rope(batch: ModelBatch) -> None:
    model = ASTTransformer(d_model=6, num_heads=2, num_kv_heads=1, num_layers=2, max_nodes=24)
    initial, target = batch.initial[:1], batch.target[:1]
    tree = ASTFeatures(*(field[:1] for field in batch.tree))
    variables = model.init(jax.random.key(41), initial, target, tree)
    logits = model.apply(variables, initial, target, tree)
    sc = ShapeChecker(B=1, N=24, A=len(AST_ACTIONS))
    sc.check(logits, "BNA", jnp.float32)
    np.testing.assert_array_equal(np.isfinite(logits[..., 1:]), tree.action_mask[..., 1:])
