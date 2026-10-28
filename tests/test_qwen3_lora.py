import jax
import jax.numpy as jnp
import numpy as np
import optax
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


@pytest.fixture(scope="module")
def model() -> qwen3.Qwen3LM:
    return qwen3.Qwen3LM(
        config=qwen3.Qwen3Config(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            eos_token_id=7,
            max_position_embeddings=8,
        ),
        dtype="float32",
        attention_backend="xla",
    )


@pytest.fixture(scope="module")
def params(model: qwen3.Qwen3LM) -> qwen3.Parameters:
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)  # (B=1, T=4)
    return model.init(jax.random.key(0), tokens)["params"]  # HF parameter shapes.


@pytest.mark.parametrize("targets", ["all", "qv", "last_layer"])
def test_lora_model_projection_wiring_and_gradients(
    model: qwen3.Qwen3LM, params: qwen3.Parameters, targets: str
) -> None:
    model = model.clone(scan_layers=targets != "last_layer", gradient_checkpointing=targets == "all")
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)  # (B=1, T=4)
    adapters = {
        name: qwen3.LoRAWeights(
            a=jax.random.normal(jax.random.key(index), (2, value.shape[1])) * 0.03,  # (R=2, I)
            b=jax.random.normal(jax.random.key(index + 100), (value.shape[0], 2)) * 0.03,  # (O, R)
        )
        for index, (name, value) in enumerate(params.items())
        if name.endswith("_proj.weight")
        and (targets != "qv" or name.endswith(("q_proj.weight", "v_proj.weight")))
        and (targets != "last_layer" or name.startswith("model.layers.1."))
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
    assert len(actual[1]) == {"all": 14, "qv": 4, "last_layer": 7}[targets]
    for value, target in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert value.shape == target.shape
        np.testing.assert_allclose(value, target, atol=3e-7, rtol=2e-4)
    assert all(np.any(np.asarray(value) != 0) for value in jax.tree.leaves(actual[1]))
    zeros = {
        name: factor._replace(b=jnp.zeros_like(factor.b)) for name, factor in adapters.items()
    }  # (O, R) B factors.
    # GPU fusion can change FP32 rounding by one ULP even with a zero delta.
    np.testing.assert_allclose(native(zeros)[1], model.apply({"params": params}, tokens).logits, atol=1e-8, rtol=2e-7)
    first = model.apply(
        {"params": params}, tokens[:, :2], cache=model.init_cache(1, 4), lora=adapters, lora_scale=scale
    )
    second = model.apply({"params": params}, tokens[:, 2:], cache=first.cache, lora=adapters, lora_scale=scale)
    np.testing.assert_allclose(
        jnp.concatenate((first.logits, second.logits), axis=1), actual[0][1], atol=2e-7, rtol=2e-5
    )


@pytest.mark.parametrize("name", ["model.layers.2.self_attn.q_proj.weight", "model.embed_tokens.weight"])
def test_lora_rejects_unsupported_targets(model: qwen3.Qwen3LM, params: qwen3.Parameters, name: str) -> None:
    adapter = qwen3.LoRAWeights(a=jnp.ones((2, 8)), b=jnp.ones((8, 2)))  # (R=2, I=8), (O=8, R)
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)  # (B=1, T=4)
    with pytest.raises(ValueError, match="Unsupported LoRA target"):
        jax.eval_shape(model.apply, {"params": params}, tokens, lora={name: adapter})


@pytest.mark.parametrize(
    ("a_shape", "b_shape"),
    [((2, 8), (1, 2)), ((2, 1), (8, 2)), ((2, 8), (8, 1)), ((0, 8), (8, 0)), ((1, 2, 8), (8, 2))],
    ids=["broadcast-output", "input-width", "rank-mismatch", "zero-rank", "batched-factor"],
)
def test_lora_rejects_invalid_factor_shapes(
    model: qwen3.Qwen3LM, params: qwen3.Parameters, a_shape: tuple[int, ...], b_shape: tuple[int, ...]
) -> None:
    adapter = qwen3.LoRAWeights(a=jnp.ones(a_shape), b=jnp.ones(b_shape))  # Deliberately invalid factor shapes.
    adapters = {f"model.layers.{layer}.self_attn.q_proj.weight": adapter for layer in range(2)}
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)  # (B=1, T=4)
    with pytest.raises(ValueError, match="Invalid LoRA factor shapes"):
        jax.eval_shape(model.apply, {"params": params}, tokens, lora=adapters)


def test_bfloat16_lora_zero_initialization_learns_both_factors(model: qwen3.Qwen3LM, params: qwen3.Parameters) -> None:
    model = model.clone(dtype="bfloat16")
    base = jax.tree.map(lambda value: value.astype(jnp.bfloat16), params)  # Frozen HF parameter shapes.
    adapters = {
        name: qwen3.LoRAWeights(
            a=jax.random.normal(jax.random.key(index), (2, weight.shape[1])) * 0.1,  # (R=2, I)
            b=jnp.zeros((weight.shape[0], 2), dtype=jnp.float32),  # (O, R)
        )
        for index, (name, weight) in enumerate(base.items())
        if name.endswith("_proj.weight")
    }
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)  # (B=1, T=4)

    def loss(factors: qwen3.LoRAParameters) -> jax.Array:
        logits = model.apply({"params": base}, tokens, lora=factors, lora_scale=2).logits  # (B, T, V)
        return optax.softmax_cross_entropy_with_integer_labels(logits[:, :-1], tokens[:, 1:]).mean()  # ()

    evaluate = jax.jit(jax.value_and_grad(loss))
    initial_loss, gradients = evaluate(adapters)  # (), LoRA factor shapes.
    for factor in gradients.values():
        np.testing.assert_array_equal(factor.a, 0)
        assert bool(jnp.any(factor.b != 0))
    updated = jax.tree.map(lambda value, gradient: value - 0.1 * gradient, adapters, gradients)
    updated_loss, next_gradients = evaluate(updated)  # (), LoRA factor shapes.
    assert float(updated_loss) < float(initial_loss)
    assert jax.tree.map(jnp.shape, next_gradients) == jax.tree.map(jnp.shape, adapters)
    assert all(value.dtype == jnp.float32 for value in jax.tree.leaves(next_gradients))
    assert all(bool(jnp.any(value != 0)) for value in jax.tree.leaves(next_gradients))
