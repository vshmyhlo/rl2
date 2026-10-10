import argparse
import signal

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2 import benchmark_attention as benchmark


@pytest.mark.parametrize("causal", [True, False])
def test_benchmark_attention_forward_and_gradients(causal: bool) -> None:
    q = jnp.zeros((1, 2, 2, 2), jnp.bfloat16)
    k = jnp.zeros((1, 2, 1, 2), jnp.bfloat16)
    v = jnp.broadcast_to(jnp.asarray([2, 6], jnp.bfloat16)[None, :, None, None], k.shape)
    inputs = (q, k, v, jnp.ones_like(q))
    attention = benchmark.make_attention("jax_xla", causal)
    out, dq, dk, dv = jax.jit(benchmark.make_workload(attention, True))(inputs)
    chex.assert_shape((out, dq), (1, 2, 2, 2))
    chex.assert_type((out, dq), jnp.bfloat16)
    chex.assert_shape((dk, dv), (1, 2, 1, 2))
    chex.assert_type((dk, dv), jnp.bfloat16)
    expected = jnp.asarray([2, 4] if causal else [4, 4])[None, :, None, None]
    np.testing.assert_allclose(out.astype(jnp.float32), jnp.broadcast_to(expected, out.shape))
    np.testing.assert_array_equal(dq.astype(jnp.float32), 0)
    np.testing.assert_array_equal(dk.astype(jnp.float32), 0)
    expected_dv = jnp.asarray([3, 1] if causal else [2, 2])[None, :, None, None]
    np.testing.assert_array_equal(dv.astype(jnp.float32), jnp.broadcast_to(expected_dv, dv.shape))
    forward = benchmark.make_workload(attention, False)(inputs)
    assert len(forward) == 1
    np.testing.assert_array_equal(forward[0], out)
    reference = tuple(x.astype(jnp.float32) for x in (out, dq, dk, dv))
    assert all(error["relative_l2"] == 0 for error in benchmark.check_outputs((out, dq, dk, dv), reference))


@pytest.mark.parametrize("bad_value", [float("nan"), 2.0], ids=["nonfinite", "inaccurate"])
def test_correctness_check_rejects_bad_results(bad_value: float) -> None:
    with pytest.raises(AssertionError):
        benchmark.check_outputs(
            (jnp.full((1, 1, 1, 1), bad_value, jnp.bfloat16),),
            (jnp.ones((1, 1, 1, 1), jnp.float32),),
        )


def test_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        benchmark.make_attention("missing", True)
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark.positive_int("0")
    assert benchmark.positive_int("1024") == 1024
    with pytest.raises(ValueError, match="Refusing"), benchmark.paused_process(1):
        pytest.fail("Invalid PID was accepted")


def test_paused_process_resumes_after_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, signal.Signals]] = []

    def kill(pid: int, sig: signal.Signals) -> None:
        calls.append((pid, sig))

    def sleep(seconds: float) -> None:
        pass

    monkeypatch.setattr(benchmark.os, "kill", kill)
    monkeypatch.setattr(benchmark.time, "sleep", sleep)
    with pytest.raises(RuntimeError, match="timing failed"), benchmark.paused_process(12345):
        raise RuntimeError("timing failed")
    assert calls == [(12345, signal.SIGSTOP), (12345, signal.SIGCONT)]
