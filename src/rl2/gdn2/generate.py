"""Generate text from a converted GDN-2 checkpoint.

Example: uv run --extra gdn2 python -m rl2.gdn2.generate \
    --prompt "Once upon a time" --max-new-tokens 100
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import urlopen

import jax
import jax.numpy as jnp
import numpy as np

from rl2.gdn2.checkpoints import Parameters, load_checkpoint, sha256_file
from rl2.gdn2.model import GatedDeltaNet2LM, GatedDeltaNet2StackCarry
from rl2.shape_checker import ShapeChecker

if TYPE_CHECKING:
    from sentencepiece import SentencePieceProcessor

DEFAULT_CHECKPOINT = Path("checkpoints/gdn2/paper-matched-30b")
DEFAULT_TOKENIZER = Path("checkpoints/gdn2/tokenizer.model")
TOKENIZER_URL = (
    "https://huggingface.co/TinyLlama/TinyLlama_v1.1/resolve/ff3c701f2424c7625fdefb9dd470f45ef18b02d6/tokenizer.model"
)
TOKENIZER_SHA256 = "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"


def load_tokenizer(path: Path | None = None) -> SentencePieceProcessor:
    """Load a local SentencePiece model, or cache the pinned default on first use.

    Explicit paths are local-only. The default tokenizer is the 32K TinyLlama
    tokenizer named by the GDN-2 training campaign; no remote Python is loaded.
    """
    try:
        from sentencepiece import SentencePieceProcessor
    except ImportError as error:
        raise ImportError("Install the tokenizer dependency with: uv sync --extra gdn2") from error
    if path is None:
        path = DEFAULT_TOKENIZER
        if not path.exists():
            print(f"Downloading tokenizer to {path}", file=sys.stderr)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".tokenizer-", dir=path.parent) as temporary:
                downloaded = Path(temporary) / "tokenizer.model"
                with urlopen(TOKENIZER_URL, timeout=60) as response, downloaded.open("wb") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
                if sha256_file(downloaded) != TOKENIZER_SHA256:
                    raise ValueError("Downloaded tokenizer checksum mismatch")
                downloaded.replace(path)
        if sha256_file(path) != TOKENIZER_SHA256:
            raise ValueError(f"Default tokenizer checksum mismatch: {path}")
    return SentencePieceProcessor(model_file=str(path))


def _validate_sampling(temperature: float, top_k: int, vocab_size: int) -> None:
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative (0 means greedy)")
    if not 0 <= top_k <= vocab_size:
        raise ValueError(f"top_k must be between 0 and vocabulary size {vocab_size}")


def sample_token(logits: jax.Array, key: jax.Array, *, temperature: float, top_k: int) -> jax.Array:
    """Select one token; temperature/top_k must be static when using jax.jit.

    top_k=0 samples over the full vocabulary. Greedy decoding ignores top_k;
    top-k sampling selects exactly k candidates, including when scores tie.
    """
    sc = ShapeChecker()
    sc.check(logits, "V", jnp.float32)
    sc.check(jax.random.key_data(key), "R", jnp.uint32)
    _validate_sampling(temperature, top_k, logits.shape[0])
    if temperature == 0:
        token = jnp.argmax(logits).astype(jnp.int32)
    else:
        # Center before scaling to avoid overflow for small positive temperature.
        logits = (logits - jnp.max(logits)) / temperature
        if top_k:
            values, indices = jax.lax.top_k(logits, top_k)
            token = indices[jax.random.categorical(key, values)].astype(jnp.int32)
        else:
            token = jax.random.categorical(key, logits).astype(jnp.int32)
    sc.check(token, "", jnp.int32)
    return token


def generate_tokens(
    model: GatedDeltaNet2LM,
    variables: Parameters,
    prompt_ids: list[int],
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 0,
    seed: int = 0,
    eos_token_id: int | None = 2,
) -> list[int]:
    """Return continuation IDs, including EOS if sampled, for one prompt.

    Transfers weights to the JAX device once. Prompt prefill and single-token
    decode compile separately, passing parameters as arguments rather than
    embedding them as compilation constants. No model history is recomputed.
    """
    _validate_sampling(temperature, top_k, model.vocab_size)
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    if not prompt_ids:
        raise ValueError("The prompt must contain at least one token; enable BOS for an empty prompt")
    if any(token < 0 or token >= model.vocab_size for token in prompt_ids):
        raise ValueError("Prompt token ID is outside the model vocabulary")
    if eos_token_id is not None and not 0 <= eos_token_id < model.vocab_size:
        raise ValueError("EOS token ID is outside the model vocabulary")
    if max_new_tokens == 0:
        return []
    variables = jax.device_put(variables)
    tokens = jnp.asarray(prompt_ids, jnp.int32)[None, :]

    def prefill(params: Parameters, ids: jax.Array) -> tuple[GatedDeltaNet2StackCarry, jax.Array]:
        sc = ShapeChecker(B=1, V=model.vocab_size)
        sc.check(ids, "BT", jnp.int32)
        carry, logits = model.apply(params, ids, jnp.full((1,), ids.shape[1], jnp.int32))
        sc.check(logits, "BTV", jnp.float32)
        return carry, logits[0, -1]

    def step(
        params: Parameters,
        ids: jax.Array,
        carry: GatedDeltaNet2StackCarry,
    ) -> tuple[GatedDeltaNet2StackCarry, jax.Array]:
        sc = ShapeChecker(B=1, V=model.vocab_size)
        sc.check(ids, "B", jnp.int32)
        carry, logits = model.apply(params, ids, jnp.ones((1,), jnp.bool_), carry, method=model.step)
        sc.check(logits, "BV", jnp.float32)
        return carry, logits[0]

    carry, logits = jax.jit(prefill)(variables, tokens)
    decode = jax.jit(step)
    sample = jax.jit(partial(sample_token, temperature=temperature, top_k=top_k))
    key = jax.random.key(seed)
    generated: list[int] = []
    for index in range(max_new_tokens):
        key, sample_key = jax.random.split(key)
        token = int(sample(logits, sample_key))
        generated.append(token)
        if token == eos_token_id or index + 1 == max_new_tokens:
            break
        carry, logits = decode(variables, jnp.asarray([token], jnp.int32), carry)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="Plain-text completion prompt (no chat template)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--tokenizer", type=Path, help="Local SentencePiece model; defaults to cached TinyLlama tokenizer"
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 for greedy decoding (default: 0.8)")
    parser.add_argument("--top-k", type=int, default=40, help="0 disables top-k filtering (default: 40)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("jax", "triton"), default="jax")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--no-bos", action="store_true", help="Do not prepend the tokenizer's BOS token")
    args = parser.parse_args()
    try:
        # Reject scalar errors before loading a large checkpoint or downloading.
        _validate_sampling(args.temperature, args.top_k, np.iinfo(np.int32).max)
        if args.max_new_tokens < 0 or not 0 <= args.seed < 2**32:
            raise ValueError("max_new_tokens must be nonnegative and seed must be between 0 and 2**32 - 1")
        tokenizer = load_tokenizer(args.tokenizer)
        if not args.no_bos and tokenizer.bos_id() < 0:
            raise ValueError("Tokenizer does not define BOS; use --no-bos")
        prompt_ids = tokenizer.encode(args.prompt, out_type=int, add_bos=not args.no_bos, add_eos=False)
        if not prompt_ids:
            raise ValueError("The prompt must contain at least one token; enable BOS for an empty prompt")
        print(f"Loading {args.checkpoint} ({args.dtype})", file=sys.stderr)
        model, variables = load_checkpoint(args.checkpoint, dtype=jnp.dtype(args.dtype), backend=args.backend)
        if tokenizer.vocab_size() != model.vocab_size:
            raise ValueError("Tokenizer vocabulary size does not match checkpoint vocabulary size")
        print(f"Generating up to {args.max_new_tokens} tokens from {len(prompt_ids)} prompt tokens...", file=sys.stderr)
        continuation = generate_tokens(
            model,
            variables,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
            eos_token_id=tokenizer.eos_id() if tokenizer.eos_id() >= 0 else None,
        )
        print(tokenizer.decode(prompt_ids + continuation))
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
