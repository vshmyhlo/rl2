"""Exercise the GRPO adapter with tiny official Gemma models; no downloads."""

from contextlib import closing
from dataclasses import replace
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock, Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sentencepiece as spm
import tokamax
from gemma import gm
from gemma_helpers import tiny_gemma_model

from rl2 import train_grpo_reasoning as grpo
from rl2.gdn2.model import GatedDeltaNet2Config, GatedDeltaNet2LM
from rl2.gemma3 import (
    AttentionBackend,
    Gemma3LM,
    Gemma3Tokenizer,
    Parameters,
    _attention,
    load_gemma3_checkpoint,
    save_gemma3_checkpoint,
)


@pytest.fixture(scope="module")
def tiny_gemma() -> tuple[Gemma3LM, Parameters]:
    return tiny_gemma_model()


def test_sampling_replay_eos_and_update(tiny_gemma: tuple[Gemma3LM, Parameters], tmp_path: Path) -> None:
    model, params = tiny_gemma
    prompts = jnp.array([[2, 3, 0], [2, 4, 3]], jnp.int32)
    lengths = jnp.array([2, 3], jnp.int32)
    key = jax.random.key(8)
    args = {"group_size": 2, "max_new_tokens": 3, "temperature": 0.7}
    tokens, _, _, _ = grpo.generate(model, params, prompts, lengths, key, stop_token_ids=(-1,), **args)
    eos = int(tokens[0, 0])
    tokens, old, mask, key = grpo.generate(model, params, prompts, lengths, key, stop_token_ids=(eos,), **args)
    assert tokens[0].tolist() == [eos, 0, 0]
    assert mask[0].tolist() == [True, False, False]
    batch = grpo.GRPOBatch(
        jnp.repeat(prompts, 2, axis=0),
        jnp.repeat(lengths, 2),
        tokens,
        mask,
        old,
        old,
        jnp.array([1, -1, 1, -1], jnp.float32),
    )
    replay = grpo.completion_log_probs(model, params, batch, 0.7)
    np.testing.assert_allclose(replay, old, atol=2e-6)
    for i in range(4):
        p, c = int(batch.prompt_lengths[i]), int(mask[i].sum())
        ids = jnp.concatenate([batch.prompts[i, :p], tokens[i, :c]])[None, :-1]
        _, logits = model.apply({"params": params}, ids, jnp.array([p + c - 1], jnp.int32))
        expected = grpo.policy_log_probs(logits[:, p - 1 :], tokens[i : i + 1, :c], 0.7)
        np.testing.assert_allclose(replay[i, :c], expected[0], atol=2e-6)
    config = grpo.Config(num_tasks=2, group_size=2, num_minibatches=1)
    state = grpo.create_state(model, params, config)
    updated, metrics = grpo.update(state, model, batch, temperature=0.7, clip_coef=0.2, beta=0.01)
    assert int(updated.step) == 1
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(
        not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(updated.params), strict=True)
    )
    with closing(grpo.checkpoint_manager(str(tmp_path))) as manager:
        grpo.save_training_checkpoint(manager, updated, key, 1, config, prompts_seen=2)
        progress = grpo.restore_training_checkpoint(manager, state)
        assert progress is not None
        restored, _, iteration, prompts_seen = progress
    assert (iteration, prompts_seen) == (1, 2)
    for a, b in zip(jax.tree.leaves(updated), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(a, b)


def test_cached_padding_positions_and_bfloat16(tiny_gemma: tuple[Gemma3LM, Parameters]) -> None:
    model, params = tiny_gemma
    model = replace(model, dtype="bfloat16")
    prompts = jnp.array([[2, 3, 0], [2, 4, 3]], jnp.int32)  # (B=2, T=3)
    lengths = jnp.array([2, 3], jnp.int32)  # (B,)
    carry, logits = model.prefill(params, prompts, lengths)  # logits: (B, T, V=8)
    assert logits.shape == (2, 3, model.vocab_size)
    assert logits.dtype == jnp.float32
    assert carry.lengths.shape == (2,)
    assert carry.valid.shape == (2, model.cache_length)
    for layer in carry.cache.values():
        assert layer["k"].shape == layer["v"].shape == (2, model.cache_length, 1, 4)
        assert layer["positions"].shape == (2, model.cache_length)
        assert layer["end_index"].shape == (2,)
        np.testing.assert_array_equal(layer["k"][:, 3:], 0)
        np.testing.assert_array_equal(layer["v"][:, 3:], 0)
    assert carry.cache["layer_0"]["end_index"].tolist() == [3, 3]
    # Token zero is a valid sampled vocabulary item, and must not become padding.
    carry, cached = model.step(params, jnp.array([0, 5], jnp.int32), jnp.array([True, True]), carry)  # cached: (B, V)
    assert cached.shape == (2, model.vocab_size)
    assert cached.dtype == jnp.float32
    assert carry.valid.tolist() == [[True, True, False, True, False, False], [True, True, True, True, False, False]]
    assert carry.lengths.tolist() == [3, 4]
    packed = jnp.array([[2, 3, 0, 0], [2, 4, 3, 5]], jnp.int32)  # (B, T+1)

    def to_bfloat16(value: jax.Array) -> jax.Array:  # (...) -> (...)
        return value.astype(jnp.bfloat16)  # (...)

    direct = model.model.apply(
        {"params": jax.tree.map(to_bfloat16, params)},
        tokens=packed,
        positions=jnp.broadcast_to(jnp.arange(4), packed.shape),
        attention_mask=(jnp.arange(4)[None, None, :] <= jnp.arange(4)[None, :, None])
        & (jnp.arange(4)[None, None, :] < carry.lengths[:, None, None]),
        return_last_only=False,
    )
    expected = direct.logits[jnp.arange(2), carry.lengths - 1].astype(jnp.float32)  # (B, V)
    np.testing.assert_allclose(cached, expected, atol=0.03, rtol=0.03)
    assert carry.cache["layer_0"]["k"].dtype == jnp.bfloat16
    # Finished rows do not gain valid cache entries or advance logical positions.
    carry, _ = model.step(params, jnp.array([0, 6], jnp.int32), jnp.array([False, True]), carry)
    assert carry.lengths.tolist() == [3, 5]
    assert not carry.valid[0, 4] and carry.valid[1, 4]
    # The last reserved slot is writable, including after another row finishes.
    carry, _ = model.step(params, jnp.array([0, 7], jnp.int32), jnp.array([False, True]), carry)
    assert carry.lengths.tolist() == [3, 6]
    assert not carry.valid[0, 5] and carry.valid[1, 5]
    for layer in carry.cache.values():
        assert layer["end_index"].tolist() == [6, 6]
        assert layer["positions"][1].tolist() == list(range(6))


@pytest.mark.parametrize("width", [pytest.param(1, id="single-token"), pytest.param(6, id="full-cache")])
def test_prefill_matches_uncached_at_boundaries(tiny_gemma: tuple[Gemma3LM, Parameters], width: int) -> None:
    model, params = tiny_gemma
    tokens = jnp.full((1, width), 2, dtype=jnp.int32)  # (B=1, T=width)
    lengths = jnp.array([width], dtype=jnp.int32)  # (B,)
    carry, actual = jax.jit(model.prefill)(params, tokens, lengths)  # actual: (B, T, V)
    _, expected = model.apply({"params": params}, tokens, lengths)  # expected: (B, T, V)
    assert actual.shape == expected.shape == (1, width, model.vocab_size)
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    assert carry.valid.tolist() == [[i < width for i in range(model.cache_length)]]
    for layer in carry.cache.values():
        assert layer["k"].shape == layer["v"].shape == (1, model.cache_length, 1, 4)
        assert layer["positions"].shape == (1, model.cache_length)
        assert layer["end_index"].tolist() == [width]


@pytest.mark.parametrize("width", [pytest.param(0, id="empty-prompt"), pytest.param(7, id="exceeds-cache")])
def test_prefill_rejects_invalid_width(tiny_gemma: tuple[Gemma3LM, Parameters], width: int) -> None:
    model, params = tiny_gemma
    with pytest.raises(ValueError, match="Prompt width"):
        model.prefill(params, jnp.ones((1, width), jnp.int32), jnp.array([width], jnp.int32))


def test_gemma_does_not_break_gdn2_initialization(tiny_gemma: tuple[Gemma3LM, Parameters]) -> None:
    # Gemma patches Flax's Module.param globally, forwarding an optional dtype.
    model = GatedDeltaNet2LM(
        config=GatedDeltaNet2Config(hidden_size=2, head_dim=2, num_heads=1, conv_size=2, conv_bias=True),
        num_layers=1,
        intermediate_size=3,
        vocab_size=5,
    )
    variables = model.init(jax.random.key(1), jnp.array([[1, 3]], jnp.int32), jnp.array([2], jnp.int32))
    assert all(value.dtype == jnp.float32 for value in jax.tree.leaves(variables["params"]))


def test_official_checkpoint_roundtrip(
    tiny_gemma: tuple[Gemma3LM, Parameters],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, params = tiny_gemma
    monkeypatch.setattr(gm.nn, "Gemma3_270M", Mock(return_value=model.model))
    path = tmp_path / "export"

    def to_bfloat16(value: jax.Array) -> jax.Array:
        return value.astype(jnp.bfloat16)

    weights = jax.tree.map(to_bfloat16, params)
    save_gemma3_checkpoint(path, weights)
    # Export is directly readable by the official API.
    upstream = gm.ckpts.load_params(path)
    loaded, variables = load_gemma3_checkpoint(str(path), cache_length=6, dtype="bfloat16")
    assert loaded.model == model.model
    for actual, expected in zip(jax.tree.leaves(variables["params"]), jax.tree.leaves(upstream), strict=True):
        assert actual.dtype == jnp.float32
        np.testing.assert_array_equal(actual, expected.astype(jnp.float32))


def test_checkpoint_validation(cudnn_gemma: tuple[Gemma3LM, Parameters], monkeypatch: pytest.MonkeyPatch) -> None:
    model, params = cudnn_gemma
    monkeypatch.setattr(gm.nn, "Gemma3_270M", Mock(return_value=model.model))
    path = "mock-checkpoint"
    mocked = Mock(return_value=params)
    monkeypatch.setattr(gm.ckpts, "load_params", mocked)
    loaded, variables = load_gemma3_checkpoint(path, cache_length=6, dtype="bfloat16", attention_backend="tokamax")
    assert loaded.attention_backend == "tokamax"
    for actual, expected in zip(jax.tree.leaves(variables["params"]), jax.tree.leaves(params), strict=True):
        np.testing.assert_array_equal(actual, expected)
    mocked.return_value = {}
    monkeypatch.setattr(gm.ckpts, "load_params", mocked)
    with pytest.raises(ValueError, match="structure"):
        load_gemma3_checkpoint(str(path), cache_length=6, dtype="float32")
    malformed = params | {"embedder": {"input_embedding": jnp.zeros((3, 4), jnp.float32)}}
    mocked.return_value = malformed
    with pytest.raises(AssertionError, match="shape"):
        load_gemma3_checkpoint(str(path), cache_length=6, dtype="float32")
    mocked.return_value = params | {
        "embedder": {"input_embedding": params["embedder"]["input_embedding"].astype(jnp.int8)}
    }
    with pytest.raises(ValueError, match="weight dtype"):
        load_gemma3_checkpoint(path, cache_length=6, dtype="float32")


def test_official_tokenizer_adapter(tmp_path: Path) -> None:
    writer = BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(["hello world", "hello there"]),
        model_writer=writer,
        vocab_size=16,
        hard_vocab_limit=False,
        pad_id=0,
        eos_id=1,
        bos_id=2,
        unk_id=3,
        minloglevel=2,
        num_threads=1,
    )
    path = tmp_path / "tokenizer.model"
    path.write_bytes(writer.getvalue())
    tokenizer = Gemma3Tokenizer(str(path))
    official = gm.text.Gemma3Tokenizer(path=path)
    ids = tokenizer.encode("hello world", out_type=int, add_bos=True, add_eos=True)
    assert ids == official.encode("hello world", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id() == 2
    assert ids[-1] == tokenizer.eos_id() == 1
    assert tokenizer.decode(ids[1:-1]) == "hello world"
    assert tokenizer.vocab_size() == official.vocab_size


def test_gemma_config_boundaries(tmp_path: Path) -> None:
    config = grpo.load_config("configs/grpo_reasoning/gemma3.yaml", grpo.Config)
    assert config.calculator_max_calls == 2
    assert config.checkpoint == str(gm.ckpts.CheckpointPath.GEMMA3_270M_IT)
    assert grpo.Config(attention_backend="tokamax").dtype == "float32"
    path = tmp_path / "config.yaml"
    path.write_text("model_type: unknown")
    with pytest.raises(ValueError, match="model_type"):
        grpo.load_config(path, grpo.Config)
    with pytest.raises(ValueError, match="32768-token context"):
        Gemma3LM(gm.nn.Gemma3_270M(), cache_length=32769)


def test_training_selects_gemma_loader(
    tiny_gemma: tuple[Gemma3LM, Parameters], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, params = tiny_gemma
    tokenizer = Mock(vocab_size=Mock(return_value=8), eos_id=Mock(return_value=1))
    load = Mock(return_value=(model, {"params": params}))
    monkeypatch.setattr(grpo, "Gemma3Tokenizer", Mock(return_value=tokenizer))
    monkeypatch.setattr(grpo, "load_gemma3_checkpoint", load)
    monkeypatch.setattr(grpo, "MLflowLogger", MagicMock())
    monkeypatch.setattr(grpo, "collect_rollout", Mock(side_effect=RuntimeError("stop before training")))
    with pytest.raises(RuntimeError, match="stop before training"):
        grpo.train(
            grpo.Config(
                checkpoint="gs://gemma-data/checkpoints/gemma3-270m-pt",
                ckpt_dir=str(tmp_path),
                max_prompt_tokens=3,
                max_new_tokens=3,
                dtype="bfloat16",
                attention_backend="tokamax",
            )
        )
    grpo.Gemma3Tokenizer.assert_called_once_with(None)
    load.assert_called_once_with(
        "gs://gemma-data/checkpoints/gemma3-270m-pt", cache_length=6, dtype="bfloat16", attention_backend="tokamax"
    )


@pytest.fixture(scope="module")
def cudnn_gemma(tiny_gemma: tuple[Gemma3LM, Parameters]) -> tuple[Gemma3LM, Parameters]:
    model, _ = tiny_gemma
    upstream = gm.nn.Gemma3_270M(config=replace(model.model.config, head_dim=8), dtype=jnp.float32)
    params = upstream.init(jax.random.key(12), tokens=jnp.array([[2, 3]], jnp.int32))["params"]
    return Gemma3LM(model=upstream, cache_length=5, dtype="bfloat16"), params


@pytest.mark.parametrize(
    ("backend", "real_cudnn"),
    [
        pytest.param("cudnn", False, id="cudnn-cpu-routing"),
        pytest.param("cudnn", True, id="cudnn-gpu"),
        pytest.param("tokamax", False, id="tokamax"),
    ],
)
def test_attention_matches_gemma(
    cudnn_gemma: tuple[Gemma3LM, Parameters],
    monkeypatch: pytest.MonkeyPatch,
    backend: AttentionBackend,
    real_cudnn: bool,
) -> None:
    if real_cudnn and not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("Requires an NVIDIA GPU with cuDNN")
    model, params = cudnn_gemma
    if real_cudnn:
        device = jax.devices()[0]
        capability = getattr(device, "compute_capability", "0.0")
        if tuple(int(part) for part in capability.split(".")) < (9, 0):
            pytest.skip("Gemma's 256-wide heads require Hopper or newer")
        upstream = model.model.clone(config=replace(model.model.config, head_dim=256))
        params = upstream.init(jax.random.key(12), tokens=jnp.array([[2, 3]], jnp.int32))["params"]
        model = replace(model, model=upstream)
    fused = replace(model, attention_backend=backend)
    original_attention = jax.nn.dot_product_attention
    tokamax_attention = Mock(wraps=tokamax.dot_product_attention)
    monkeypatch.setattr(tokamax, "dot_product_attention", tokamax_attention)
    calls = []

    def to_float32(value: jax.Array) -> jax.Array:  # (...)
        return value.astype(jnp.float32)  # (...)

    def attention(
        query: jax.Array,  # (B, T, N, H)
        key: jax.Array,  # (B, S, K, H)
        value: jax.Array,  # (B, S, K, H)
        **kwargs: Any,
    ) -> jax.Array:
        assert kwargs["implementation"] == "cudnn"
        assert kwargs["scale"] == 1.0
        assert query.dtype == key.dtype == value.dtype == jnp.bfloat16
        assert query.shape[1] % 2 == key.shape[1] % 2 == 0
        assert kwargs["mask"].shape == (query.shape[0], 1, query.shape[1], key.shape[1])
        calls.append(query.shape)
        kwargs["implementation"] = "xla"
        query, key, value = jax.tree.map(to_float32, (query, key, value))  # (B, T, N, H), each KV: (B, S, K, H)
        return original_attention(query, key, value, **kwargs).astype(jnp.bfloat16)  # (B, T, N, H)

    if backend == "cudnn" and not real_cudnn:
        monkeypatch.setattr(jax.nn, "dot_product_attention", attention)
    tokens = jnp.array([[2, 3, 0], [2, 4, 3]], jnp.int32)  # (B=2, T=3)
    lengths = jnp.array([2, 3], jnp.int32)  # (B,)
    actual_carry, actual = jax.jit(fused.prefill)(params, tokens, lengths)  # actual: (B, T, V)
    expected_carry, expected = model.prefill(params, tokens, lengths)  # expected: (B, T, V)
    assert actual.shape == (2, 3, model.vocab_size)
    np.testing.assert_allclose(actual, expected, atol=0.04, rtol=0.04)
    # Odd cache width and single-token queries exercise both cuDNN padding paths.
    for active in (jnp.array([True, True]), jnp.array([False, True])):  # (B,)
        next_tokens = jnp.array([5, 6], jnp.int32)  # (B,)
        actual_carry, actual = jax.jit(fused.step)(params, next_tokens, active, actual_carry)  # actual: (B, V)
        expected_carry, expected = model.step(params, next_tokens, active, expected_carry)  # expected: (B, V)
        assert actual.shape == (2, model.vocab_size)
        np.testing.assert_allclose(actual, expected, atol=0.04, rtol=0.04)
        for actual_leaf, expected_leaf in zip(
            jax.tree.leaves(actual_carry), jax.tree.leaves(expected_carry), strict=True
        ):  # matching cache leaf shapes
            assert actual_leaf.shape == expected_leaf.shape
            np.testing.assert_allclose(actual_leaf.astype(jnp.float32), expected_leaf.astype(jnp.float32), atol=0.04)

    def loss(weights: Parameters, lm: Gemma3LM) -> jax.Array:
        _, logits = lm.apply({"params": weights}, tokens, lengths)  # (B, T, V)
        return jnp.mean(jnp.square(logits))  # ()

    actual_loss, actual_grad = jax.jit(jax.value_and_grad(partial(loss, lm=fused)))(params)  # loss: ()
    expected_loss, expected_grad = jax.value_and_grad(partial(loss, lm=model))(params)  # loss: ()
    np.testing.assert_allclose(actual_loss, expected_loss, atol=0.04, rtol=0.04)
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual_grad), jax.tree.leaves(expected_grad), strict=True):
        assert actual_leaf.shape == expected_leaf.shape
        np.testing.assert_allclose(actual_leaf, expected_leaf, atol=0.06, rtol=0.06)

    # A fully masked row and a one-token prompt retain upstream output semantics.
    short_tokens = tokens[:, :1]  # (B, T=1)
    short_lengths = jnp.array([0, 1], jnp.int32)  # (B,)
    _, actual = fused.apply({"params": params}, short_tokens, short_lengths)  # (B, 1, V)
    _, expected = model.apply({"params": params}, short_tokens, short_lengths)  # (B, 1, V)
    assert actual.shape == (2, 1, model.vocab_size)
    np.testing.assert_allclose(actual, expected, atol=0.04, rtol=0.04)
    if backend == "cudnn" and not real_cudnn:
        assert calls
    if backend == "tokamax":
        assert tokamax_attention.called


def test_cudnn_validation(cudnn_gemma: tuple[Gemma3LM, Parameters]) -> None:
    model, _ = cudnn_gemma
    with pytest.raises(ValueError, match="bfloat16"):
        replace(model, attention_backend="cudnn", dtype="float32")
    with pytest.raises(ValueError, match="divisible by 8"):
        replace(
            model, attention_backend="cudnn", model=model.model.clone(config=replace(model.model.config, head_dim=4))
        )
    with pytest.raises(ValueError, match="soft caps"):
        replace(
            model,
            attention_backend="cudnn",
            model=model.model.clone(config=replace(model.model.config, attn_logits_soft_cap=30.0)),
        )
    with pytest.raises(ValueError, match="bfloat16"):
        grpo.Config(attention_backend="cudnn")


@pytest.mark.parametrize(
    ("backend", "kv_heads", "soft_cap"),
    [
        pytest.param("cudnn", 2, None, id="cudnn-grouped-query"),
        pytest.param("cudnn", 4, None, id="cudnn-multi-head"),
        pytest.param("tokamax", 2, None, id="tokamax-grouped-query"),
        pytest.param("tokamax", 4, 1.0, id="tokamax-multi-head-soft-cap"),
    ],
)
def test_projection_layouts_and_empty_rows(
    backend: Literal["cudnn", "tokamax"], kv_heads: int, soft_cap: float | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = gm.nn.Attention(
        num_heads=4,
        num_kv_heads=kv_heads,
        features=4,
        head_dim=8,
        attn_type=gm.nn.AttentionType.LOCAL_SLIDING,
        query_pre_attn_scalar=8**-0.5,
        sliding_window_size=2,
        use_qk_norm=True,
        attn_logits_soft_cap=soft_cap,
    )
    x = jax.random.normal(jax.random.key(4), (1, 4, 4))  # (B=1, T=4, D=4)
    positions = jnp.array([[0, 1, 4, 5]], jnp.int32)  # (B, T), including a logical position gap
    mask = jnp.tril(jnp.ones((1, 4, 4), jnp.bool_)).at[:, 0].set(False)  # (B, T, S=4)
    variables = module.init(jax.random.key(5), x, positions, None, mask)
    original_attention = jax.nn.dot_product_attention

    def attention(
        query: jax.Array,  # (B, T, N, H)
        key: jax.Array,  # (B, S, K, H)
        value: jax.Array,  # (B, S, K, H)
        **kwargs: Any,
    ) -> jax.Array:
        assert kwargs["implementation"] == "cudnn"
        kwargs["implementation"] = "xla"
        return original_attention(query, key, value, **kwargs)  # (B, T, N, H)

    if backend == "cudnn":
        monkeypatch.setattr(jax.nn, "dot_product_attention", attention)

    def loss(inputs: jax.Array, *, fused: bool) -> tuple[jax.Array, jax.Array]:  # inputs: (B, T, D)
        _, output = module.apply(
            variables, inputs, positions, None, mask, method=partial(_attention, backend=backend) if fused else None
        )  # output: (B, T, D)
        return jnp.square(output).sum(), output  # (), (B, T, D)

    # Float32 isolates projection/masking correctness from fused BF16 rounding.
    (_, actual), actual_grad = jax.jit(jax.value_and_grad(partial(loss, fused=True), has_aux=True))(
        x
    )  # each: (B, T, D)
    (_, expected), expected_grad = jax.value_and_grad(partial(loss, fused=False), has_aux=True)(x)  # each: (B, T, D)
    assert actual.shape == actual_grad.shape == (1, 4, 4)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(actual_grad, expected_grad, atol=2e-5, rtol=2e-5)


def test_generation_rejects_insufficient_cache(tiny_gemma: tuple[Gemma3LM, Parameters]) -> None:
    model, params = tiny_gemma
    with pytest.raises(ValueError, match="cache length"):
        grpo.generate(
            model,
            params,
            jnp.array([[2, 3, 4]], jnp.int32),
            jnp.array([3], jnp.int32),
            jax.random.key(0),
            group_size=2,
            max_new_tokens=4,
            temperature=1.0,
            stop_token_ids=(1,),
        )
