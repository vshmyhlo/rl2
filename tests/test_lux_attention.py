"""Set attention symmetry, bidirectional mixing, padding, and Triton routing."""

from typing import Any
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tokamax

from rl2.model.lux import _Attention, _SetAttentionBlock


@pytest.fixture(scope="module")
def tokens() -> jax.Array:  # (B=1, E=4, D=8)
    return jax.random.normal(jax.random.key(2), (1, 4, 8))  # (B, E, D)


@pytest.fixture(scope="module")
def valid() -> jax.Array:  # (B=1, E=4)
    return jnp.array([[True, True, False, True]])  # (B, E), last token is global


@pytest.fixture(scope="module")
def block() -> _SetAttentionBlock:
    return _SetAttentionBlock(heads=4, head_dim=2, implementation="xla", dtype=jnp.float32)


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
    assert expected.shape == (1, 4, block.heads * block.head_dim)
    assert params["params"]["ffn_in"]["kernel"].shape == (8, 32)
    assert params["params"]["ffn_out"]["kernel"].shape == (32, 8)
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


def test_qk_rmsnorm_precedes_padding_and_preserves_values(
    params: dict[str, Any], tokens: jax.Array, monkeypatch: pytest.MonkeyPatch
) -> None:
    # tokens: (B=1, Q=4, D=8); constant projections distinguish RMSNorm from centering.
    variables = jax.tree.map(lambda leaf: leaf, params["params"]["attention"])
    for name in ("query", "key", "value"):
        variables[name]["kernel"] = jnp.zeros_like(variables[name]["kernel"])  # (D, H=4, d=2)
        variables[name]["bias"] = jnp.broadcast_to(jnp.array([3.0, 4.0]), (4, 2))  # (H, d)
    variables["query_norm"]["scale"] = jnp.array([2.0, 0.5])  # (d,), learned scale shared across heads
    attention = Mock(wraps=tokamax.dot_product_attention)
    monkeypatch.setattr(tokamax, "dot_product_attention", attention)
    output = _Attention(heads=4, head_dim=2, implementation="xla", dtype=jnp.float32).apply(
        {"params": variables}, tokens, tokens[:, :2]
    )  # (B, Q, D)
    query, key, value = attention.call_args.args  # (B, Q/K, H, padded_d=16)
    assert output.shape == (1, 4, 8)
    assert query.shape == (1, 4, 4, 16) and key.shape == value.shape == (1, 2, 4, 16)
    expected = jnp.array([3.0, 4.0]) / jnp.sqrt(12.5 + 1e-6)  # (d,)
    np.testing.assert_allclose(
        query[..., :2], jnp.broadcast_to(expected * jnp.array([2.0, 0.5]), (1, 4, 4, 2)), atol=1e-6
    )
    np.testing.assert_allclose(key[..., :2], jnp.broadcast_to(expected, (1, 2, 4, 2)), atol=1e-6)
    np.testing.assert_array_equal(value[..., :2], jnp.broadcast_to(jnp.array([3.0, 4.0]), (1, 2, 4, 2)))
    for array in (query, key, value):
        np.testing.assert_array_equal(array[..., 2:], 0)


@pytest.mark.parametrize("head_dim", [2, 16], ids=["padded", "unpadded"])
def test_attention_head_dimensions(tokens: jax.Array, head_dim: int) -> None:
    # tokens: (B=1, Q=4, D=8); context has K=2 tokens.
    attention = _Attention(heads=2, head_dim=head_dim, implementation="xla", dtype=jnp.float32)
    output, variables = attention.init_with_output(jax.random.key(3), x=tokens, context=tokens[:, :2])
    # output: (B, Q, D); projection kernels: (D, H, d).
    assert output.shape == tokens.shape
    for name in ("query", "key", "value"):
        assert variables["params"][name]["kernel"].shape == (8, 2, head_dim)
    assert variables["params"]["out"]["kernel"].shape == (2, head_dim, 8)


def test_bfloat16_normalized_attention_matches_float32(
    block: _SetAttentionBlock, params: dict[str, Any], tokens: jax.Array, valid: jax.Array
) -> None:
    # tokens: (B, E, D); valid: (B, E). Check normalization numerics independently of recurrence.
    expected = jax.jit(block.apply)(params, tokens, valid)  # (B, E, D)
    reduced = jax.jit(block.clone(dtype=jnp.bfloat16).apply)(params, tokens, valid)  # (B, E, D)
    assert reduced.dtype == tokens.dtype
    np.testing.assert_allclose(reduced.astype(jnp.float32), expected, atol=0.04, rtol=0.02)
    np.testing.assert_array_equal(reduced[:, 2], 0)


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
