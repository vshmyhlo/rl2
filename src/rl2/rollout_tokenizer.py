"""Importable tokenizer cache for Ray workers launched from training scripts."""

from functools import lru_cache

from renderers.base import Tokenizer, load_tokenizer


@lru_cache(maxsize=1)
def rollout_tokenizer(model: str) -> Tokenizer:
    return load_tokenizer(model)
