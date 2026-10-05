"""Enable the persistent cache before test collection compiles any JAX programs."""

from rl2.jax_cache import configure_compilation_cache

configure_compilation_cache()
