import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.lora.lora_weights import LoRALayerWeights

from rl2 import qwen3


@pytest.mark.parametrize(("rank", "scale"), [(1, 1.0), (2, 2.3)], ids=["rank-one", "non-binary-scale"])
def test_lora_linear_matches_vllm_rounding(rank: int, scale: float) -> None:
    inputs = jax.random.normal(jax.random.key(0), (2, 4))  # (T=2, I=4)
    weight = jax.random.normal(jax.random.key(1), (3, 4))  # (O=3, I)
    adapter = qwen3.LoRAWeights(
        a=jax.random.normal(jax.random.key(2), (rank, 4)),  # (R, I)
        b=jax.random.normal(jax.random.key(3), (3, rank)),  # (O, R)
    )
    loaded = LoRALayerWeights(
        module_name="q_proj",
        rank=rank,
        lora_alpha=rank,
        lora_a=torch.tensor(np.asarray(adapter.a)).bfloat16(),  # (R, I)
        lora_b=torch.tensor(np.asarray(adapter.b)).bfloat16(),  # (O, R)
        scaling=scale,
    ).optimize()
    x = torch.tensor(np.asarray(inputs)).bfloat16()  # (T, I)
    w = torch.tensor(np.asarray(weight)).bfloat16()  # (O, I)
    base = (x.float() @ w.float().T).bfloat16()  # (T, O)
    hidden = (x.float() @ loaded.lora_a.float().T).bfloat16()  # (T, R)
    delta = (hidden.float() @ loaded.lora_b.float().T).bfloat16()  # (T, O)
    expected = (base + delta).float().numpy()  # (T, O)

    def loss(factors: qwen3.LoRAParameters) -> tuple[jax.Array, jax.Array]:
        output = qwen3._linear(inputs, weight, jnp.bfloat16, adapter=factors["q"], lora_scale=scale)  # (T, O)
        return output.astype(jnp.float32).sum(), output  # (), (T, O)

    (_, actual), gradients = jax.jit(jax.value_and_grad(loss, has_aux=True))({"q": adapter})
    assert actual.shape == (2, 3)
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(actual, dtype=np.float32), expected)
    assert gradients["q"].a.shape == (rank, 4)
    assert gradients["q"].b.shape == (3, rank)
    assert all(value.dtype == jnp.float32 for value in jax.tree.leaves(gradients))
    rounded = jax.tree.map(lambda value: value.astype(jnp.bfloat16).astype(jnp.float32), {"q": adapter})
    (_, rounded_output), rounded_gradients = jax.jit(jax.value_and_grad(loss, has_aux=True))(rounded)
    for value, reference in zip(
        jax.tree.leaves((actual, gradients)), jax.tree.leaves((rounded_output, rounded_gradients)), strict=True
    ):
        np.testing.assert_array_equal(value, reference)
    merged = qwen3._linear(inputs, weight + scale * (adapter.b @ adapter.a), jnp.bfloat16)  # (T, O)
    assert not np.array_equal(actual, merged), "Merged weights do not reproduce native BF16 LoRA"


@pytest.mark.parametrize(("scan", "remat"), [(True, True), (False, False)], ids=["scan-remat", "unrolled"])
def test_lora_model_projection_wiring_and_gradients(scan: bool, remat: bool) -> None:
    config = qwen3.Qwen3Config(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        eos_token_id=7,
        max_position_embeddings=8,
    )
    model = qwen3.Qwen3LM(
        config=config, dtype="float32", attention_backend="xla", scan_layers=scan, gradient_checkpointing=remat
    )
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)  # (B=1, T=4)
    params = model.init(jax.random.key(0), tokens)["params"]  # HF parameter shapes.
    adapters = {
        name: qwen3.LoRAWeights(
            a=jax.random.normal(jax.random.key(index), (2, value.shape[1])) * 0.03,  # (R=2, I)
            b=jax.random.normal(jax.random.key(index + 100), (value.shape[0], 2)) * 0.03,  # (O, R)
        )
        for index, (name, value) in enumerate(params.items())
        if name.endswith("_proj.weight")
    }
    scale = 1.5

    def native(factors: qwen3.LoRAParameters) -> tuple[jax.Array, jax.Array]:
        logits = model.apply({"params": params}, tokens, lora=factors, lora_scale=scale).logits  # (B, T, V)
        return jnp.square(logits).sum(), logits  # (), (B, T, V)

    def reference(factors: qwen3.LoRAParameters) -> tuple[jax.Array, jax.Array]:
        merged = dict(params)
        for name, factor in factors.items():
            merged[name] = params[name] + scale * (factor.b @ factor.a)  # (O, I)
        logits = model.apply({"params": merged}, tokens).logits  # (B, T, V)
        return jnp.square(logits).sum(), logits  # (), (B, T, V)

    actual = jax.jit(jax.value_and_grad(native, has_aux=True))(adapters)
    expected = jax.jit(jax.value_and_grad(reference, has_aux=True))(adapters)
    assert actual[0][1].shape == (1, 4, 8)
    assert len(actual[1]) == 14  # Seven projections in each of two layers.
    for value, target in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert value.shape == target.shape
        np.testing.assert_allclose(value, target, atol=3e-7, rtol=2e-4)
    assert all(np.any(np.asarray(value) != 0) for value in jax.tree.leaves(actual[1]))
    zeros = {
        name: factor._replace(b=jnp.zeros_like(factor.b)) for name, factor in adapters.items()
    }  # (O, R) B factors.
    np.testing.assert_array_equal(native(zeros)[1], model.apply({"params": params}, tokens).logits)
    first = model.apply(
        {"params": params}, tokens[:, :2], cache=model.init_cache(1, 4), lora=adapters, lora_scale=scale
    )
    second = model.apply({"params": params}, tokens[:, 2:], cache=first.cache, lora=adapters, lora_scale=scale)
    np.testing.assert_allclose(
        jnp.concatenate((first.logits, second.logits), axis=1), actual[0][1], atol=2e-7, rtol=2e-5
    )
