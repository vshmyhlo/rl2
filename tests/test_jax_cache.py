import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import jax
import pytest

from rl2.jax_cache import configure_compilation_cache


@pytest.fixture
def cache_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    defaults = {
        "jax_compilation_cache_dir": None,
        "jax_persistent_cache_min_compile_time_secs": 1.0,
        "jax_persistent_cache_min_entry_size_bytes": 0,
        "jax_enable_compilation_cache": True,
    }
    original = {name: getattr(jax.config, name) for name in defaults}
    for name, value in defaults.items():
        monkeypatch.delenv(name.upper(), raising=False)
        jax.config.update(name, value)
    try:
        yield
    finally:
        for name, value in original.items():
            jax.config.update(name, value)


@pytest.mark.usefixtures("cache_config")
@pytest.mark.parametrize("xdg", [False, True])
def test_default_cache_location_and_small_compilations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, xdg: bool
) -> None:
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    root = Path.home() / ".cache"
    if xdg:
        root = tmp_path / "cache"
        monkeypatch.setenv("XDG_CACHE_HOME", str(root))
    configure_compilation_cache()
    assert jax.config.jax_compilation_cache_dir == str(root / "rl2" / "jax")
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 0
    assert jax.config.jax_persistent_cache_min_entry_size_bytes == -1


@pytest.mark.usefixtures("cache_config")
def test_existing_cache_settings_and_disable_flag_are_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = {
        "jax_compilation_cache_dir": "gs://cache-bucket/jax",
        "jax_persistent_cache_min_compile_time_secs": 2.5,
        "jax_persistent_cache_min_entry_size_bytes": 4096,
        "jax_enable_compilation_cache": False,
    }
    for name, value in settings.items():
        jax.config.update(name, value)
        monkeypatch.setenv(name.upper(), str(value))
    for _ in range(2):
        configure_compilation_cache()
        assert {name: getattr(jax.config, name) for name in settings} == settings


def test_compiled_program_is_reused_in_a_fresh_process(tmp_path: Path) -> None:
    script = tmp_path / "cached_program.py"
    script.write_text(
        """import json
import chex
import jax
import jax.numpy as jnp
from rl2.jax_cache import configure_compilation_cache

configure_compilation_cache()
events: list[str] = []

def record(event: str, **metadata: object) -> None:
    events.append(event)

jax.monitoring.register_event_listener(record)

@jax.jit
def square_sum(x: jax.Array) -> jax.Array:
    chex.assert_shape(x, (4,))
    chex.assert_type(x, jnp.float32)
    return jnp.square(x).sum()

result = float(square_sum(jnp.asarray([1, 2, 3, 4], jnp.float32)))
print(json.dumps({"result": result, "hits": events.count("/jax/compilation_cache/cache_hits")}))
"""
    )
    env = dict(os.environ)
    for name in (
        "JAX_COMPILATION_CACHE_DIR",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
    ):
        env.pop(name, None)
    env.update(XDG_CACHE_HOME=str(tmp_path / "cache"), JAX_PLATFORMS="cpu", JAX_ENABLE_COMPILATION_CACHE="true")
    results = []
    for _ in range(2):
        process = subprocess.run(
            [sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=30, check=False
        )
        assert process.returncode == 0, process.stdout + process.stderr
        results.append(json.loads(process.stdout))
    assert results[0] == {"result": 30.0, "hits": 0}
    assert results[1]["result"] == 30.0
    assert results[1]["hits"] > 0
    assert any((tmp_path / "cache" / "rl2" / "jax").iterdir())
