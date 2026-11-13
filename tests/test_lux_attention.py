"""Set attention symmetry, bidirectional mixing, padding, and cuDNN sharding."""

from typing import Any
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

from rl2.model.lux import _Attention, _SetAttentionBlock


@pytest.fixture(scope="module")
def tokens() -> jax.Array:  # (B=1, E=4, D=8)
    return jax.random.normal(jax.random.key(2), (1, 4, 8))  # (B, E, D)


@pytest.fixture(scope="module")
def tokens_mask() -> jax.Array:  # (B=1, E=4)
    return jnp.array([[True, True, False, True]])  # (B, E), last token is global


@pytest.fixture(scope="module")
def block() -> _SetAttentionBlock:
    return _SetAttentionBlock(attention_implementation="xla", heads=4, head_dim=2, dtype=jnp.float32)


@pytest.fixture(scope="module")
def params(block: _SetAttentionBlock, tokens: jax.Array, tokens_mask: jax.Array) -> dict[str, Any]:
    # tokens: (B, E, D); tokens_mask: (B, E)
    return block.init(jax.random.key(3), tokens=tokens, tokens_mask=tokens_mask)


def test_bidirectional_mixing_masking_and_permutation(
    block: _SetAttentionBlock, params: dict[str, Any], tokens: jax.Array, tokens_mask: jax.Array
) -> None:
    # tokens: (B, E, D); tokens_mask: (B, E)
    apply = jax.jit(block.apply)
    expected = apply(params, tokens, tokens_mask)  # (B, E, D)
    assert expected.shape == (1, 4, block.heads * block.head_dim)
    assert params["params"]["ffn_in"]["kernel"].shape == (8, 32)
    assert params["params"]["ffn_out"]["kernel"].shape == (32, 8)
    later_changed = apply(params, tokens.at[:, 1].add(jnp.arange(8)), tokens_mask)  # (B, E, D)
    earlier_changed = apply(params, tokens.at[:, 0].add(jnp.arange(8)), tokens_mask)  # (B, E, D)
    assert float(jnp.max(jnp.abs(later_changed[:, 0] - expected[:, 0]))) > 1e-4
    assert float(jnp.max(jnp.abs(earlier_changed[:, 1] - expected[:, 1]))) > 1e-4
    masked_changed = apply(params, tokens.at[:, 2].set(30000), tokens_mask)  # (B, E, D)
    np.testing.assert_array_equal(masked_changed, expected)
    np.testing.assert_array_equal(expected[:, 2], 0)
    permutation = jnp.array([2, 1, 3, 0])  # (E,)
    reordered = apply(params, tokens[:, permutation], tokens_mask[:, permutation])  # (B, E, D)
    np.testing.assert_allclose(reordered, expected[:, permutation], atol=2e-6)


def test_qk_rmsnorm_preserves_values(
    params: dict[str, Any], tokens: jax.Array, monkeypatch: pytest.MonkeyPatch
) -> None:
    # tokens: (B=1, Q=4, D=8); constant projections distinguish RMSNorm from centering.
    variables = jax.tree.map(lambda leaf: leaf, params["params"]["attention"])
    for name in ("query", "key", "value"):
        variables[name]["kernel"] = jnp.zeros_like(variables[name]["kernel"])  # (D, H=4, d=2)
        variables[name]["bias"] = jnp.broadcast_to(jnp.array([3.0, 4.0]), (4, 2))  # (H, d)
    variables["query_norm"]["scale"] = jnp.array([2.0, 0.5])  # (d,), learned scale shared across heads
    attention = Mock(wraps=jax.nn.dot_product_attention)
    monkeypatch.setattr(jax.nn, "dot_product_attention", attention)
    output = _Attention(attention_implementation="xla", heads=4, head_dim=2, dtype=jnp.float32).apply(
        {"params": variables}, tokens, tokens[:, :2]
    )  # (B, Q, D)
    query, key, value = attention.call_args.args  # (B, Q/K, H, d=2)
    assert output.shape == (1, 4, 8)
    assert query.shape == (1, 4, 4, 2) and key.shape == value.shape == (1, 2, 4, 2)
    expected = jnp.array([3.0, 4.0]) / jnp.sqrt(12.5 + 1e-6)  # (d,)
    np.testing.assert_allclose(
        query[..., :2], jnp.broadcast_to(expected * jnp.array([2.0, 0.5]), (1, 4, 4, 2)), atol=1e-6
    )
    np.testing.assert_allclose(key[..., :2], jnp.broadcast_to(expected, (1, 2, 4, 2)), atol=1e-6)
    np.testing.assert_array_equal(value[..., :2], jnp.broadcast_to(jnp.array([3.0, 4.0]), (1, 2, 4, 2)))


@pytest.mark.parametrize("head_dim", [2, 16], ids=["narrow", "wide"])
def test_attention_head_dimensions(tokens: jax.Array, head_dim: int) -> None:
    # tokens: (B=1, Q=4, Dx=8); context has K=2 tokens and Dc=8.
    attention = _Attention(attention_implementation="xla", heads=2, head_dim=head_dim, dtype=jnp.float32)
    output, variables = attention.init_with_output(jax.random.key(3), tokens=tokens, context=tokens[:, :2])
    # output: (B, Q, D=H*d); projection kernels: (Dx/Dc, H, d).
    assert output.shape == (*tokens.shape[:-1], 2 * head_dim)
    for name in ("query", "key", "value"):
        assert variables["params"][name]["kernel"].shape == (8, 2, head_dim)
    assert variables["params"]["out"]["kernel"].shape == (2, head_dim, 2 * head_dim)


def test_bfloat16_normalized_attention_matches_float32(
    block: _SetAttentionBlock, params: dict[str, Any], tokens: jax.Array, tokens_mask: jax.Array
) -> None:
    # tokens: (B, E, D); tokens_mask: (B, E). Check normalization numerics independently of recurrence.
    expected = jax.jit(block.apply)(params, tokens, tokens_mask)  # (B, E, D)
    reduced = jax.jit(block.clone(dtype=jnp.bfloat16).apply)(params, tokens, tokens_mask)  # (B, E, D)
    assert reduced.dtype == tokens.dtype
    np.testing.assert_allclose(reduced.astype(jnp.float32), expected, atol=0.04, rtol=0.02)
    np.testing.assert_array_equal(reduced[:, 2], 0)


def test_cudnn_rejects_unaligned_head_width(tokens: jax.Array) -> None:  # (B=1, Q=4, Dx=8)
    module = _Attention(attention_implementation="cudnn", heads=2, head_dim=2, dtype=jnp.bfloat16)
    with pytest.raises(AssertionError, match="head_dim divisible by 8"):
        module.init(jax.random.key(5), tokens, tokens)


def test_cudnn_token_padding_preserves_masks(monkeypatch: pytest.MonkeyPatch) -> None:
    tokens = jax.random.normal(jax.random.key(4), (1, 3, 8))  # (B=1, Q=K=3, Dx=8)
    mask = jnp.array([[[[True, False, True]]]])  # (B, 1, 1, K)
    module = _Attention(attention_implementation="cudnn", heads=2, head_dim=8, dtype=jnp.bfloat16)
    original = jax.nn.dot_product_attention

    def reference(
        query: jax.Array,  # (B, Qp=4, Nh=2, Dh=8)
        key: jax.Array,  # (B, Kp=4, Nh, Dh)
        value: jax.Array,  # (B, Kp, Nh, Dh)
        *,
        mask: jax.Array,  # (B, 1, Qp, Kp)
        is_causal: bool,
        scale: float,
        implementation: str,
    ) -> jax.Array:  # (B, Qp, Nh, Dh)
        assert implementation == "cudnn" and not is_causal
        assert query.shape == key.shape == value.shape == (1, 4, 2, 8)
        assert query.dtype == key.dtype == value.dtype == jnp.bfloat16
        assert scale == 8**-0.5
        np.testing.assert_array_equal(mask, jnp.broadcast_to(jnp.array([True, False, True, False]), (1, 1, 4, 4)))
        return original(query, key, value, mask=mask, is_causal=is_causal, scale=scale, implementation="xla")

    attention = Mock(wraps=reference)
    monkeypatch.setattr(jax.nn, "dot_product_attention", attention)
    actual, variables = module.init_with_output(jax.random.key(5), tokens, tokens, mask)  # (B, Q, D=16); parameters
    assert attention.called and actual.shape == (1, 3, 16)
    monkeypatch.setattr(jax.nn, "dot_product_attention", original)
    expected = module.clone(attention_implementation="xla").apply(variables, tokens, tokens, mask)  # (B, Q, D)
    np.testing.assert_allclose(actual.astype(jnp.float32), expected.astype(jnp.float32), atol=0.02, rtol=0.02)


@pytest.mark.parametrize(
    "key_count,masked",
    [(2, False), (3, False), (3, True)],
    ids=["cross_attention", "unmasked_odd_keys", "masked_self_attention"],
)
def test_cudnn_batch_sharding_preserves_outputs_and_gradients(key_count: int, masked: bool) -> None:
    if jax.default_backend() != "gpu" or jax.device_count() < 2:
        pytest.skip("Requires two GPUs for cuDNN partitioning verification")
    mesh = Mesh(devices=jax.devices()[:2], axis_names=("batch",), axis_types=(AxisType.Auto,))
    tokens = jax.random.normal(jax.random.key(4), (2, 3, 8))  # (B=2, Q=3, Dx=8)
    context = tokens[:, :key_count]  # (B, K=3/2, Dc=8)
    mask = jnp.array([[[[True, False, True]]], [[[False, True, True]]]]) if masked else None
    # mask: (B, 1, 1, K) or None; distinct masks exercise per-device masking.
    attention = _Attention(attention_implementation="cudnn", heads=2, head_dim=8, dtype=jnp.bfloat16)
    reference = attention.clone(attention_implementation="xla")
    variables = reference.init(jax.random.key(5), tokens, context, mask)

    def loss(
        variables: dict[str, Any],
        tokens: jax.Array,  # (B, Q, Dx)
        context: jax.Array,  # (B, K, Dc)
        mask: jax.Array | None,  # (B, 1, 1, K)
        module: _Attention,
    ) -> tuple[jax.Array, jax.Array]:  # (); (B, Q, D=16)
        output = module.apply(variables, tokens, context, mask).astype(jnp.float32)  # (B, Q, D)
        return jnp.square(output).mean(), output  # (); (B, Q, D)

    evaluate = jax.value_and_grad(loss, has_aux=True)
    expected = jax.jit(evaluate, static_argnames="module")(variables, tokens, context, mask, reference)
    with jax.set_mesh(mesh):
        compiled = (
            jax.jit(
                evaluate,
                static_argnames="module",
                in_shardings=(PartitionSpec(), PartitionSpec("batch"), PartitionSpec("batch"), PartitionSpec("batch")),
                out_shardings=((PartitionSpec(), PartitionSpec("batch")), PartitionSpec()),
            )
            .lower(variables, tokens, context, mask, attention)
            .compile()
        )
        actual = compiled(variables, tokens, context, mask)
    assert actual[0][1].shape == (2, 3, 16)
    assert actual[0][1].sharding == NamedSharding(mesh, PartitionSpec("batch"))
    for result, target in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert result.shape == target.shape
        np.testing.assert_allclose(result, target, atol=0.02, rtol=0.03)
    hlo = compiled.as_text()
    assert "__cudnn$fmha" in hlo
    assert "all-gather" not in hlo
