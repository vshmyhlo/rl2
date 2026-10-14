"""Shared tiny official Gemma model; no checkpoint or tokenizer downloads."""

from dataclasses import replace
from functools import cache

import jax
import jax.numpy as jnp
from gemma import gm

from rl2.gemma3 import Gemma3LM, Parameters


@cache
def tiny_gemma_model() -> tuple[Gemma3LM, Parameters]:
    config = replace(
        gm.nn.Gemma3_270M().config,
        num_embed=8,
        embed_dim=4,
        hidden_dim=6,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        sliding_window_size=2,
        attention_types=(gm.nn.AttentionType.LOCAL_SLIDING, gm.nn.AttentionType.GLOBAL),
    )
    upstream = gm.nn.Gemma3_270M(config=config, dtype=jnp.float32)
    params = upstream.init(jax.random.key(12), tokens=jnp.array([[2, 3]], jnp.int32))["params"]
    return Gemma3LM(model=upstream, cache_length=6), params
