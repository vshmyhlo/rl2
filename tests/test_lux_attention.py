"""Set attention symmetry, bidirectional mixing, padding, and Triton routing."""

from typing import Any
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tokamax

from rl2.model.lux import _SetAttentionBlock


@pytest.fixture(scope="module")
def tokens() -> jax.Array:  # (B=1, E=4, D=8)
    return jax.random.normal(jax.random.key(2), (1, 4, 8))  # (B, E, D)


@pytest.fixture(scope="module")
def valid() -> jax.Array:  # (B=1, E=4)
    return jnp.array([[True, True, False, True]])  # (B, E), last token is global


@pytest.fixture(scope="module")
def block() -> _SetAttentionBlock:
    return _SetAttentionBlock(features=8, implementation="xla", dtype=jnp.float32)


@pytest.fixture(scope="module")
def params(block: _SetAttentionBlock, tokens: jax.Array, valid: jax.Array) -> dict[str, Any]:
    # tokens: (B, E, D); valid: (B, E)
    return block.init(jax.random.key(3), tokens, valid)


def test_bidirectional_mixing_masking_and_permutation(
    block: _SetAttentionBlock, params: dict[str, Any], tokens: jax.Array, valid: jax.Array
) -> None:
    # tokens: (B, E, D); valid: (B, E)
    apply = jax.jit(block.apply)
    expected = apply(params, tokens, valid)  # (B, E, D)
    later_changed = apply(params, tokens.at[:, 1].add(jnp.arange(8)), valid)  # (B, E, D)
    earlier_changed = apply(params, tokens.at[:, 0].add(jnp.arange(8)), valid)  # (B, E, D)
    assert float(jnp.max(jnp.abs(later_changed[:, 0] - expected[:, 0]))) > 1e-4
    assert float(jnp.max(jnp.abs(earlier_changed[:, 1] - expected[:, 1]))) > 1e-4
    masked_changed = apply(params, tokens.at[:, 2].set(30000), valid)  # (B, E, D)
    np.testing.assert_array_equal(masked_changed, expected)
    np.testing.assert_array_equal(expected[:, 2], 0)
    permutation = jnp.array([2, 1, 3, 0])  # (E,)
    reordered = apply(params, tokens[:, permutation], valid[:, permutation])  # (B, E, D)
    np.testing.assert_allclose(reordered, expected[:, permutation], atol=2e-6)


@pytest.mark.parametrize("real_triton", [False, True], ids=["cpu_routing", "gpu_kernel"])
def test_triton_forward_and_backward_match_reference(
    block: _SetAttentionBlock,
    params: dict[str, Any],
    tokens: jax.Array,  # (B, E, D)
    valid: jax.Array,  # (B, E)
    monkeypatch: pytest.MonkeyPatch,
    real_triton: bool,
) -> None:
    if real_triton and jax.default_backend() != "gpu":
        pytest.skip("Tokamax Triton kernel verification requires a GPU")

    def evaluate(module: _SetAttentionBlock) -> tuple[tuple[jax.Array, jax.Array], dict[str, Any]]:
        def loss(variables: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
            output = module.apply(variables, tokens, valid)  # (B, E, D)
            return jnp.square(output).mean(), output  # (); (B, E, D)

        return jax.jit(jax.value_and_grad(loss, has_aux=True))(params)

    expected = evaluate(block)
    original = tokamax.dot_product_attention

    def reference(
        query: jax.Array,  # (B, E, 4, 16), narrow logical heads are padded
        key: jax.Array,  # (B, E, 4, 16)
        value: jax.Array,  # (B, E, 4, 16)
        *,
        mask: jax.Array | None,  # (B, 1, 1, E)
        is_causal: bool,
        scale: float,
        implementation: str,
    ) -> jax.Array:  # (B, E, 4, 16)
        assert implementation == "triton" and not is_causal
        assert query.shape[-1] == key.shape[-1] == value.shape[-1] == 16
        assert scale == 2**-0.5  # Preserve the logical head size, not the padded width.
        return original(query, key, value, mask=mask, is_causal=is_causal, scale=scale, implementation="xla")

    attention = Mock(wraps=original if real_triton else reference)
    monkeypatch.setattr(tokamax, "dot_product_attention", attention)
    actual = evaluate(block.clone(implementation="triton"))
    assert attention.called
    assert attention.call_args.kwargs["implementation"] == "triton"
    assert attention.call_args.kwargs["is_causal"] is False
    assert attention.call_args.kwargs["mask"].shape == (1, 1, 1, 4)
    assert attention.call_args.kwargs["mask"].dtype == jnp.bool_
    for result, target in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(result, target, atol=3e-5, rtol=3e-4)
