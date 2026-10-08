from functools import partial
from operator import itemgetter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.edit_transformer import EDIT_EVENT, FEEDBACK_EVENT, PAD_EVENT, EditTransformer, Event
from rl2.karel_ast_edit import FEEDBACK_SIZE
from rl2.shape_checker import ShapeChecker

type Variables = dict[str, Any]


@pytest.fixture(scope="module")
def event() -> Event:
    grids = jnp.zeros((3, 2, 2, 6), jnp.int32)
    grid = jnp.stack((grids, grids.at[:, 0, 0, 5].set(1), grids), axis=1)
    return Event(
        jnp.array(
            [
                [EDIT_EVENT, EDIT_EVENT, PAD_EVENT],
                [FEEDBACK_EVENT, PAD_EVENT, PAD_EVENT],
                [EDIT_EVENT, PAD_EVENT, PAD_EVENT],
            ],
            jnp.int32,
        ),
        jnp.ones((3, 3), jnp.int32),
        jnp.broadcast_to(grid, (3, *grid.shape)),
        jnp.ones((3, 3, FEEDBACK_SIZE), jnp.float32),
    )


def initialize(model: EditTransformer, event: Event) -> Variables:
    variables = model.init(jax.random.key(0), event)
    # Exercise real backbone predictions and padding after learned head biases.
    head = variables["params"]["head"]
    head["kernel"] = jax.random.normal(jax.random.key(1), head["kernel"].shape) * 0.1
    head["bias"] = jnp.ones_like(head["bias"])
    return variables


def test_padding_prefill_and_cached_steps(event: Event) -> None:
    model = EditTransformer(d_model=8, num_layers=1, num_heads=2, max_nodes=2, max_seq_len=3)
    variables = initialize(model, event)
    full_carry, *full_outputs = jax.jit(model.apply)(variables, event)
    np.testing.assert_array_equal(full_carry[0].position, [3, 1, 0])
    padding = np.asarray(event.kind == PAD_EVENT)
    for output in full_outputs:
        np.testing.assert_array_equal(np.asarray(output)[padding], 0)

    prefill = jax.jit(partial(model.apply, method=model.prefill))
    carry, *outputs = prefill(variables, event)
    for output, full in zip(outputs, full_outputs, strict=True):
        np.testing.assert_allclose(output[:2], full[jnp.array([2, 0]), jnp.arange(2)], atol=1e-6)
        np.testing.assert_array_equal(output[2], 0)

    prefix = jax.tree.map(itemgetter(slice(2)), event)
    prefix_carry, *_ = prefill(variables, prefix)
    event = jax.tree.map(itemgetter(2), event)
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


def test_unused_event_fields_do_not_contaminate_gradients(event: Event) -> None:
    model = EditTransformer(d_model=8, num_layers=1, num_heads=2, max_nodes=2, max_seq_len=3)
    variables = initialize(model, event)
    has_token = event.kind == EDIT_EVENT
    has_feedback = event.kind == FEEDBACK_EVENT
    dirty = event._replace(
        action=jnp.where(has_token, event.action, 10000),
        feedback=jnp.where(has_feedback[..., None], event.feedback, jnp.nan),
        grid=jnp.where((event.kind != PAD_EVENT)[..., None, None, None, None], event.grid, 10000),
    )

    def loss(params: Variables, event: Event) -> jax.Array:
        encoded = model.apply({"params": params}, event, method=model.encode_event)
        sc = ShapeChecker(T=3, B=3, D=8)
        sc.check(encoded, "TBD", jnp.float32)
        return jnp.square(encoded).sum()

    evaluate = jax.jit(jax.value_and_grad(loss))
    expected_loss, expected_grad = evaluate(variables["params"], event)
    actual_loss, actual_grad = evaluate(variables["params"], dirty)
    np.testing.assert_allclose(actual_loss, expected_loss, atol=1e-6)
    for actual, expected in zip(jax.tree.leaves(actual_grad), jax.tree.leaves(expected_grad), strict=True):
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("invalid", ["image_count", "channels", "dtype"])
def test_event_grid_validation(event: Event, invalid: str) -> None:
    grid = event.grid
    if invalid == "image_count":
        grid = grid[:, :, :2]
    elif invalid == "channels":
        grid = grid[..., :5]
    else:
        grid = grid.astype(jnp.float32)
    event = event._replace(grid=grid)
    model = EditTransformer(d_model=8, num_layers=1, num_heads=2, max_nodes=2, max_seq_len=3)
    with pytest.raises(AssertionError, match="TBIHWC"):
        model.init(jax.random.key(0), event, method=model.encode_event)
