from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.training.train_state import TrainState

from rl2.ast_transformer import ASTTransformer, _ASTBlock
from rl2.grpo import (
    Config,
    create_state,
    generate,
    load_config,
    predict,
)
from rl2.karel import KarelConfig, _parse
from rl2.karel_ast import ACTION_ID, AST_ACTIONS, ASTFeatures, KarelAST, batch_features, teacher_forcing


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


def test_bidirectional_attention_matches_numpy_and_excludes_padding() -> None:
    block = _ASTBlock(16, 2, 1, 32, 1, jnp.float32, "xla")
    x = jax.random.normal(jax.random.key(12), (2, 5, 16))
    present = jnp.asarray([[True, True, True, False, False], [True, True, True, True, True]])
    params = block.init(jax.random.key(13), x, present)["params"]
    params = {**params, "down": {"kernel": jnp.zeros_like(params["down"]["kernel"])}}
    actual = block.apply({"params": params}, x, present)
    normalized = np.asarray(x) / np.sqrt(np.mean(np.asarray(x) ** 2, axis=-1, keepdims=True) + 1e-6)
    normalized *= np.asarray(params["attention_norm"]["scale"])
    query = (normalized @ params["query"]["kernel"]).reshape(2, 5, 2, 8)
    key = (normalized @ params["key"]["kernel"]).reshape(2, 5, 1, 8)
    value = (normalized @ params["value"]["kernel"]).reshape(2, 5, 1, 8)
    attended = np.zeros((2, 5, 2, 8), np.float32)
    for b in range(2):
        for t in range(5):
            for h in range(2):
                scores = np.asarray(key[b, :, 0] @ query[b, t, h]) / np.sqrt(8)
                scores = np.where(present[b], scores, -np.inf)
                weights = np.exp(scores - scores.max())
                attended[b, t, h] = weights / weights.sum() @ value[b, :, 0]
    expected = x + attended.reshape(2, 5, 16) @ params["attention_out"]["kernel"]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)

    def first_output(inputs: jax.Array) -> jax.Array:
        chex.assert_shape(inputs, x.shape)
        chex.assert_type(inputs, jnp.float32)
        return block.apply({"params": params}, inputs, present)[0, 0].sum()

    gradients = jax.grad(first_output)(x)
    assert np.linalg.norm(gradients[0, 2]) > 1e-5  # A later present node influences the first position.
    np.testing.assert_array_equal(gradients[0, 3:], 0)


def test_initial_policy_is_uniform_over_typed_actions(config: Config, batch: ModelBatch) -> None:
    state = create_state(config, batch.initial[:1], batch.target[:1])
    logits = predict(state, batch.initial, batch.target, batch.tree)
    chex.assert_shape(logits, (len(batch.actions), config.max_nodes, len(AST_ACTIONS)))
    logits = logits[jnp.arange(len(batch.actions)), batch.tree.frontier]
    chex.assert_type(logits, jnp.float32)
    masks = batch.tree.action_mask[np.arange(len(batch.actions)), batch.tree.frontier]
    expected = masks / masks.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(jax.nn.softmax(logits), expected, rtol=1e-6)
    assert np.isneginf(np.asarray(logits)[:, 0]).all()
    count_row = list(batch.actions).index(ACTION_ID["R=2"])
    assert batch.tree.action_mask[count_row].sum() == 20


def test_padded_features_do_not_affect_predictions(config: Config, batch: ModelBatch) -> None:
    state = create_state(config, batch.initial[:1], batch.target[:1])
    params = dict(state.params)
    for name in ("constructor_head", "value_head"):
        params[name] = {**params[name], "kernel": jax.random.normal(jax.random.key(5), params[name]["kernel"].shape)}
    state = state.replace(params=params)
    dirty = ASTFeatures(
        *(np.where(batch.tree.node_mask, array, 999999).astype(np.int32) for array in batch.tree[:5]),
        batch.tree.node_mask,
        batch.tree.frontier,
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
            logits = logits[jnp.arange(len(batch.actions)), batch.tree.frontier]
            return -jnp.take_along_axis(jax.nn.log_softmax(logits), jnp.asarray(batch.actions)[:, None], axis=-1).mean()

        value, grads = jax.value_and_grad(loss)(state.params)
        return state.apply_gradients(grads=grads), value

    losses = []
    for _ in range(8):
        state, value = train_step(state)
        assert np.isfinite(value)
        losses.append(float(value))
    assert losses[-1] < losses[0] * 0.8
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


def test_sampling_finishes_and_completed_trees_use_safe_dummy_logits(config: Config, batch: ModelBatch) -> None:
    state = create_state(config, batch.initial[:1], batch.target[:1])
    trees = generate(state, batch.initial[:4], batch.target[:4], jax.random.key(20), config.max_nodes, config.max_depth)
    for tree in trees:
        assert tree.complete
        _parse(tree.tokens())
    logits = predict(state, batch.initial[:4], batch.target[:4], batch_features(tuple(t.features() for t in trees)))
    expected = np.zeros((4, config.max_nodes, len(AST_ACTIONS)), np.float32)
    expected[..., 0] = 1
    np.testing.assert_array_equal(jax.nn.softmax(logits), expected)


def test_example_config_and_invalid_model_dimensions(batch: ModelBatch) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs/grpo_karel.yaml")
    assert config.bf16 and config.max_nodes == 128
    model = ASTTransformer(d_model=15, num_heads=2, num_layers=1, max_nodes=24)
    with pytest.raises(AssertionError):
        model.init(jax.random.key(1), batch.initial, batch.target, batch.tree)


def test_parallel_predictions_cover_every_hole_and_ignore_frontier(config: Config, batch: ModelBatch) -> None:
    tree = KarelAST.empty(config.max_nodes, config.max_depth)
    for name in ("Program", "ConsNonEmpty", "IFELSE"):
        tree = tree.expand(ACTION_ID[name])
    features = batch_features((tree.features(),))
    assert features.is_hole.sum() == 4
    state = create_state(config, batch.initial[:1], batch.target[:1])
    logits = predict(state, batch.initial[:1], batch.target[:1], features)
    np.testing.assert_array_equal(np.isfinite(logits[..., 1:]), features.action_mask[..., 1:])
    probabilities = np.asarray(jax.nn.softmax(logits))
    for position in range(config.max_nodes):
        allowed = features.action_mask[0, position]
        if features.is_hole[0, position]:
            np.testing.assert_allclose(probabilities[0, position], allowed / allowed.sum(), rtol=1e-6)
        else:
            assert probabilities[0, position, 0] == 1
    # The legacy DFS frontier cannot affect which holes receive predictions.
    changed = features._replace(frontier=np.asarray([config.max_nodes - 1], np.int32))
    np.testing.assert_array_equal(logits, predict(state, batch.initial[:1], batch.target[:1], changed))
