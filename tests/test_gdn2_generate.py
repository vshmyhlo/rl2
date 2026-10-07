import hashlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.gdn2 import GatedDeltaNet2Config, GatedDeltaNet2LM
from rl2.gdn2 import generate as generation
from rl2.gdn2.checkpoints import Parameters


def test_sampling_greedy_top_k_and_seed() -> None:
    logits = jnp.array([-2, 3, 1, -4], jnp.float32)
    sample = jax.jit(generation.sample_token, static_argnames=("temperature", "top_k"))
    key = jax.random.key(7)
    assert int(sample(logits, key, temperature=0.0, top_k=0)) == 1
    assert int(sample(logits, key, temperature=100.0, top_k=1)) == 1
    first = sample(logits, key, temperature=0.8, top_k=2)
    assert int(first) in (1, 2)
    np.testing.assert_array_equal(first, sample(logits, key, temperature=0.8, top_k=2))
    # Full-vocabulary categorical path; only one token has nonzero probability.
    assert int(sample(jnp.array([-1e4, 0, -1e4, -1e4], jnp.float32), key, temperature=1.0, top_k=0)) == 1


@pytest.fixture(scope="module")
def tiny_model() -> tuple[GatedDeltaNet2LM, Parameters]:
    model = GatedDeltaNet2LM(GatedDeltaNet2Config(hidden_size=2, head_dim=2, num_heads=1, conv_size=2), 1, 3, 5)
    variables = model.init(jax.random.key(1), jnp.array([[1], [3]], jnp.int32))
    return model, variables


def test_cached_generation_matches_full_context_and_eos(tiny_model: tuple[GatedDeltaNet2LM, Parameters]) -> None:
    model, variables = tiny_model
    prompt = [1, 3]
    actual = generation.generate_tokens(model, variables, prompt, max_new_tokens=2, temperature=0, eos_token_id=None)
    context = prompt.copy()
    expected = []
    for _ in range(2):
        _, logits = model.apply(variables, jnp.array(context, jnp.int32)[:, None])
        token = int(jnp.argmax(logits[-1, 0]))
        expected.append(token)
        context.append(token)
    assert actual == expected
    stopped = generation.generate_tokens(
        model, variables, prompt, max_new_tokens=4, temperature=0, eos_token_id=expected[0]
    )
    assert stopped == expected[:1]
    assert prompt == [1, 3]


def test_zero_budget_does_not_call_model() -> None:
    model = Mock(vocab_size=5)
    assert generation.generate_tokens(model, {}, [1], max_new_tokens=0) == []
    model.apply.assert_not_called()


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"max_new_tokens": -1}, "max_new_tokens"),
        ({"temperature": float("nan")}, "temperature"),
        ({"temperature": -0.1}, "temperature"),
        ({"top_k": -1}, "top_k"),
        ({"top_k": 6}, "top_k"),
        ({"seed": -1}, "seed"),
        ({"eos_token_id": 5}, "EOS"),
        ({"prompt_ids": []}, "at least one"),
        ({"prompt_ids": [5]}, "Prompt token"),
        ({"prompt_ids": [-1]}, "Prompt token"),
    ],
)
def test_invalid_generation_arguments(overrides: dict[str, Any], match: str) -> None:
    model = Mock(vocab_size=5)
    args = {"prompt_ids": [1], **overrides}
    with pytest.raises(ValueError, match=match):
        generation.generate_tokens(model, {}, **args)
    model.apply.assert_not_called()


def test_tokenizer_download_cache_checksum_and_local_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    content = b"test tokenizer bytes"
    cache = tmp_path / "tokenizer.model"
    calls = []

    def download(url: str, *, timeout: int) -> io.BytesIO:
        assert url == generation.TOKENIZER_URL and timeout == 60
        calls.append(url)
        return io.BytesIO(content)

    def processor(*, model_file: str) -> Path:
        return Path(model_file)

    monkeypatch.setitem(sys.modules, "sentencepiece", SimpleNamespace(SentencePieceProcessor=processor))
    monkeypatch.setattr(generation, "urlopen", download)
    monkeypatch.setattr(generation, "DEFAULT_TOKENIZER", cache)
    monkeypatch.setattr(generation, "TOKENIZER_SHA256", hashlib.sha256(content).hexdigest())
    assert generation.load_tokenizer() == cache
    assert cache.read_bytes() == content
    assert generation.load_tokenizer() == cache
    assert len(calls) == 1
    cache.write_bytes(b"bad cache")
    with pytest.raises(ValueError, match="checksum"):
        generation.load_tokenizer()
    # An explicit local tokenizer neither downloads nor enforces the default checksum.
    assert generation.load_tokenizer(cache) == cache
    cache.unlink()
    monkeypatch.setattr(generation, "TOKENIZER_SHA256", "incorrect")
    with pytest.raises(ValueError, match="checksum"):
        generation.load_tokenizer()
    assert not cache.exists()


class FakeTokenizer:
    def bos_id(self) -> int:
        return 1

    def eos_id(self) -> int:
        return 2

    def vocab_size(self) -> int:
        return 5

    def encode(self, prompt: str, *, out_type: type[int], add_bos: bool, add_eos: bool) -> list[int]:
        assert out_type is int and not add_eos
        return ([1] if add_bos else []) + ([3] if prompt else [])

    def decode(self, ids: list[int]) -> str:
        assert ids == [1, 3, 4]
        return "Supplied prompt completion"


def test_cli_wiring(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    model = cast(GatedDeltaNet2LM, SimpleNamespace(vocab_size=5))
    params = {"params": {}}

    def tokenizer(path: Path | None) -> FakeTokenizer:
        assert path is None
        return FakeTokenizer()

    def checkpoint(path: Path, *, dtype: jax.typing.DTypeLike) -> tuple[GatedDeltaNet2LM, Parameters]:
        assert path == generation.DEFAULT_CHECKPOINT and dtype == jnp.dtype(jnp.bfloat16)
        return model, params

    def generate(m: GatedDeltaNet2LM, p: Parameters, prompt: list[int], **kwargs: Any) -> list[int]:
        assert m is model and p is params and prompt == [1, 3]
        assert kwargs == {"max_new_tokens": 3, "temperature": 0.0, "top_k": 0, "seed": 8, "eos_token_id": 2}
        return [4]

    monkeypatch.setattr(generation, "load_tokenizer", tokenizer)
    monkeypatch.setattr(generation, "load_checkpoint", checkpoint)
    monkeypatch.setattr(generation, "generate_tokens", generate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate",
            "--prompt",
            "Supplied prompt",
            "--max-new-tokens",
            "3",
            "--temperature",
            "0",
            "--top-k",
            "0",
            "--seed",
            "8",
            "--dtype",
            "bfloat16",
        ],
    )
    generation.main()
    captured = capsys.readouterr()
    assert captured.out == "Supplied prompt completion\n"
    assert "Generating" in captured.err


def test_cli_rejects_invalid_budget_before_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = Mock(side_effect=AssertionError("should not load"))
    monkeypatch.setattr(generation, "load_checkpoint", loader)
    monkeypatch.setattr(sys, "argv", ["generate", "--prompt", "hello", "--max-new-tokens", "-1"])
    with pytest.raises(SystemExit, match="2"):
        generation.main()
    loader.assert_not_called()
