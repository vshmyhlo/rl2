"""Enable the persistent cache before test collection compiles any JAX programs."""

from collections.abc import Generator

import jax
import pytest
from jax.sharding import AxisType, Mesh

from rl2.jax_cache import configure_compilation_cache

configure_compilation_cache()


@pytest.fixture
def ppo_mesh() -> Generator[None]:
    """Keep unit tests on one device; multi-device tests enter their own mesh."""
    with jax.set_mesh(Mesh(devices=jax.devices()[:1], axis_names=("batch",), axis_types=(AxisType.Auto,))):
        yield
