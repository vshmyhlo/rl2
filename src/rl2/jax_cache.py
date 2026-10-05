"""Persistent compilation-cache defaults shared by training and tests."""

import os
from pathlib import Path

import jax


def configure_compilation_cache() -> None:
    """Configure before compiling; preserve an existing directory and JAX env overrides."""
    if jax.config.jax_compilation_cache_dir is None:
        root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache").expanduser()
        jax.config.update("jax_compilation_cache_dir", str(root / "rl2" / "jax"))
    # Small test graphs benefit from persistence too. JAX handles directory
    # creation, cache keys, and I/O failures; its explicit disable flag still works.
    if "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in os.environ:
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    if "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES" not in os.environ:
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
