from functools import partial
from operator import itemgetter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.edit_transformer import ACTION_EVENT, PAD_EVENT, SEED_EVENT, UPDATE_EVENT, EditTransformer, Events, History
from rl2.karel_ast_edit import FEEDBACK_SIZE
from rl2.shape_checker import ShapeChecker
from rl2.train_karel_ppo_ast_ar_edit import EditActorCritic

type Variables = dict[str, Any]


@pytest.fixture(scope="module")
def history() -> History:
    grids = jnp.zeros((3, 2, 2, 6), jnp.int32)
    return History(
        grids,
        grids.at[:, 0, 0, 5].set(1),
        Events(
            jnp.array(
                [
                    [SEED_EVENT, SEED_EVENT, PAD_EVENT],
                    [UPDATE_EVENT, PAD_EVENT, PAD_EVENT],
                    [ACTION_EVENT, PAD_EVENT, PAD_EVENT],
                ],
                jnp.int32,
            ),
            jnp.ones((3, 3), jnp.int32),
            jnp.broadcast_to(grids, (3, *grids.shape)),
            jnp.ones((3, 3, FEEDBACK_SIZE), jnp.float32),
        ),
    )


def initialize(model: EditTransformer, history: History) -> Variables:
    variables = model.init(jax.random.key(0), history)
    # Exercise real backbone predictions and padding after learned head biases.
    head = variables["params"]["head"]
    head["kernel"] = jax.random.normal(jax.random.key(1), head["kernel"].shape) * 0.1
    head["bias"] = jnp.ones_like(head["bias"])
    if isinstance(model, EditActorCritic):
        variables["params"]["value_head"]["bias"] = jnp.ones((1,), jnp.float32)
    return variables


@pytest.mark.parametrize("model_class", [EditTransformer, EditActorCritic], ids=["policy", "actor_critic"])
def test_padding_prefill_and_cached_steps(model_class: type[EditTransformer], history: History) -> None:
    model = model_class(d_model=8, num_layers=1, num_heads=2, max_nodes=2, max_seq_len=3)
    variables = initialize(model, history)
    full_carry, *full_outputs = jax.jit(model.apply)(variables, history)
    np.testing.assert_array_equal(full_carry.transformer[0].position, [3, 1, 0])
    padding = np.asarray(history.events.kind == PAD_EVENT)
    for output in full_outputs:
        np.testing.assert_array_equal(np.asarray(output)[padding], 0)

    prefill = jax.jit(partial(model.apply, method=model.prefill))
    carry, *outputs = prefill(variables, history)
    for output, full in zip(outputs, full_outputs, strict=True):
        np.testing.assert_allclose(output[:2], full[jnp.array([2, 0]), jnp.arange(2)], atol=1e-6)
        np.testing.assert_array_equal(output[2], 0)

    prefix = history._replace(events=jax.tree.map(itemgetter(slice(2)), history.events))
    prefix_carry, *_ = prefill(variables, prefix)
    event = jax.tree.map(itemgetter(2), history.events)
    step = jax.jit(partial(model.apply, method=model.step))
    step_carry, *step_outputs = step(variables, event, prefix_carry)
    for actual, expected in zip(jax.tree.leaves(step_carry), jax.tree.leaves(carry), strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-6)
    for actual, expected in zip(step_outputs, full_outputs, strict=True):
        np.testing.assert_allclose(actual, expected[2], atol=1e-6)

    # Padding remains a no-op even when a member has filled its cache.
    event = event._replace(kind=jnp.full((3,), PAD_EVENT, jnp.int32))
    padded_carry, *padded_outputs = step(variables, event, step_carry)
    for actual, expected in zip(jax.tree.leaves(padded_carry), jax.tree.leaves(step_carry), strict=True):
        np.testing.assert_array_equal(actual, expected)
    for output in padded_outputs:
        np.testing.assert_array_equal(output, 0)


def test_unused_event_fields_do_not_contaminate_gradients(history: History) -> None:
    model = EditTransformer(d_model=8, num_layers=1, num_heads=2, max_nodes=2, max_seq_len=3)
    variables = initialize(model, history)
    events = history.events
    has_token = (events.kind == SEED_EVENT) | (events.kind == ACTION_EVENT)
    has_feedback = (events.kind == UPDATE_EVENT) | (events.kind == ACTION_EVENT)
    dirty = events._replace(
        value=jnp.where(has_token, events.value, 10000),
        feedback=jnp.where(has_feedback[..., None], events.feedback, jnp.nan),
    )

    def loss(params: Variables, inputs: Events) -> jax.Array:
        encoded = model.apply({"params": params}, inputs, history.initial, history.target, method=model.encode_events)
        sc = ShapeChecker(T=3, B=3, D=8)
        sc.check(encoded, "TBD", jnp.float32)
        return jnp.square(encoded).sum()

    evaluate = jax.jit(jax.value_and_grad(loss))
    expected_loss, expected_grad = evaluate(variables["params"], events)
    actual_loss, actual_grad = evaluate(variables["params"], dirty)
    np.testing.assert_allclose(actual_loss, expected_loss, atol=1e-6)
    for actual, expected in zip(jax.tree.leaves(actual_grad), jax.tree.leaves(expected_grad), strict=True):
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, atol=1e-6)
