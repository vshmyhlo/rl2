from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from rl2.mamba3 import Mamba3, Mamba3Carry, Mamba3Stack, Mamba3StackCarry
from rl2.transformer import Transformer, TransformerCarry, TransformerStack, TransformerStackCarry

type Model = Mamba3 | Mamba3Stack | Transformer | TransformerStack
type Carry = Mamba3Carry | Mamba3StackCarry | TransformerCarry | TransformerStackCarry
type Parameters = dict[str, Any]
type LossOutput = tuple[jax.Array, tuple[Carry, jax.Array]]


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(Mamba3(8, d_state=8, headdim=4, dtype=jnp.bfloat16), id="mamba-siso"),
        pytest.param(
            Mamba3(8, d_state=8, headdim=4, mimo_rank=2, outproj_norm=True, dtype=jnp.bfloat16), id="mamba-mimo"
        ),
        pytest.param(Transformer(8, num_heads=2, num_kv_heads=1, max_seq_len=6, dtype=jnp.bfloat16), id="transformer"),
        *[
            pytest.param(
                Mamba3Stack(
                    8,
                    2,
                    d_state=8,
                    headdim=4,
                    mimo_rank=2,
                    mlp_multiple_of=8,
                    dtype=jnp.bfloat16,
                    residual_in_fp32=residual_fp32,
                ),
                id=f"mamba-stack-residual-fp32-{residual_fp32}",
            )
            for residual_fp32 in (True, False)
        ],
        pytest.param(
            TransformerStack(8, 2, num_heads=2, num_kv_heads=1, max_seq_len=6, dtype=jnp.bfloat16),
            id="transformer-stack",
        ),
        pytest.param(
            TransformerStack(
                16,
                2,
                num_heads=2,
                num_kv_heads=1,
                max_seq_len=6,
                dtype=jnp.bfloat16,
                attention_implementation="cudnn",
            ),
            id="transformer-stack-cudnn",
            marks=pytest.mark.skipif(
                not any(device.platform == "gpu" for device in jax.devices()),
                reason="cuDNN attention requires an NVIDIA GPU",
            ),
        ),
    ],
)
def test_bfloat16_chunked_gradients_and_adam_training(model: Model) -> None:
    """Train with BF16 inputs/compute and FP32 parameters, gradients and Adam state."""
    x = jax.random.normal(jax.random.key(20), (6, 2, model.d_model)).astype(jnp.bfloat16)
    target = jax.random.normal(jax.random.key(21), x.shape)
    starts = jnp.zeros((6, 2), jnp.bool_).at[4, 0].set(True)
    params = model.init(jax.random.key(22), x)["params"]

    def loss(parameters: Parameters, inputs: jax.Array, chunked: bool) -> LossOutput:
        chex.assert_shape(inputs, (6, 2, model.d_model))
        chex.assert_type(inputs, jnp.bfloat16)
        chex.assert_type(jax.tree.leaves(parameters), jnp.float32)
        if chunked:
            carry, first = model.apply({"params": parameters}, inputs[:3], episode_starts=starts[:3])
            carry, second = model.apply({"params": parameters}, inputs[3:], carry, starts[3:])
            output = jnp.concatenate((first, second))
        else:
            carry, output = model.apply({"params": parameters}, inputs, episode_starts=starts)
        chex.assert_type(output, jnp.bfloat16)
        return jnp.mean(jnp.square(output.astype(jnp.float32) - target)), (carry, output)

    (initial_loss, (carry, output)), full_grad = jax.jit(
        jax.value_and_grad(partial(loss, chunked=False), argnums=(0, 1), has_aux=True)
    )(params, x)
    (chunk_loss, (_, chunk_output)), chunk_grad = jax.jit(
        jax.value_and_grad(partial(loss, chunked=True), argnums=(0, 1), has_aux=True)
    )(params, x)
    np.testing.assert_allclose(chunk_loss, initial_loss, rtol=0.01, atol=1e-4)
    np.testing.assert_allclose(chunk_output.astype(jnp.float32), output.astype(jnp.float32), rtol=0.03, atol=0.02)
    chex.assert_type(jax.tree.leaves(full_grad[0]), jnp.float32)
    chex.assert_type(full_grad[1], jnp.bfloat16)
    chex.assert_trees_all_equal_shapes_and_dtypes(full_grad, chunk_grad)
    for full, chunk in zip(jax.tree.leaves(full_grad), jax.tree.leaves(chunk_grad)):
        a, b = np.asarray(full, np.float32), np.asarray(chunk, np.float32)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert np.linalg.norm(a) > 0
        # Different reduction groupings introduce BF16 rounding across chunks.
        assert np.linalg.norm(a - b) <= 0.05 * np.linalg.norm(a)

    if isinstance(model, (Mamba3, Mamba3Stack)):
        chex.assert_type(jax.tree.leaves(carry), jnp.float32)
    else:
        states = (carry,) if isinstance(carry, TransformerCarry) else carry
        for state in states:
            chex.assert_type((state.key, state.value), jnp.bfloat16)
            chex.assert_type(state.position, jnp.int32)

    def train_step(state: TrainState, inputs: jax.Array) -> tuple[TrainState, jax.Array]:
        chex.assert_shape(inputs, (6, 2, model.d_model))
        chex.assert_type(inputs, jnp.bfloat16)
        (value, _), grads = jax.value_and_grad(partial(loss, chunked=True), has_aux=True)(state.params, inputs)
        chex.assert_type(jax.tree.leaves(grads), jnp.float32)
        return state.apply_gradients(grads=grads), value

    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(learning_rate=0.01))
    update = jax.jit(train_step)
    for _ in range(2):
        state, value = update(state, x)
        assert np.isfinite(value)
    chex.assert_type(jax.tree.leaves(state.params), jnp.float32)
    for leaf in jax.tree.leaves((state.params, state.opt_state)):
        assert np.isfinite(leaf).all()
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            chex.assert_type(leaf, jnp.float32)
    final_loss, _ = jax.jit(partial(loss, chunked=True))(state.params, x)
    assert final_loss < initial_loss
