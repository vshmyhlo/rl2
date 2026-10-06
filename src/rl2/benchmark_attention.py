"""Compare attention kernels (no projections, RoPE, or KV-cache updates).

Run with JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run --extra cuda12 python -m rl2.benchmark_attention.
Timings use CUPTI, exclude compilation, and include all returned Q/K/V gradients
in backward mode. Tokamax uses default kernel configurations, without autotuning.
"""

import argparse
import importlib.metadata
import json
import os
import signal
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np

from rl2.shape_checker import ShapeChecker

type Inputs = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
type Outputs = tuple[jax.Array, ...]
type Attention = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
type Workload = Callable[[Inputs], Outputs]

BACKENDS = (
    "jax_xla",
    "jax_cudnn",
    "tokamax_xla",
    "tokamax_xla_chunked",
    "tokamax_cudnn",
    "tokamax_triton",
    "tokamax_auto",
    "tokamax_mosaic",
)


def make_attention(backend: str, causal: bool) -> Attention:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}")
    provider, implementation = backend.split("_", 1)
    if provider == "tokamax":
        import tokamax

        kernel = tokamax.dot_product_attention
    else:
        kernel = jax.nn.dot_product_attention

    def attention(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        sc = ShapeChecker()
        sc.check(q, "BTHD", q.dtype)
        sc.check((k, v), "BTKD", q.dtype)
        chex.assert_type(q, jnp.floating)
        chex.assert_is_divisible(q.shape[2], k.shape[2])
        out = kernel(q, k, v, is_causal=causal, implementation=None if implementation == "auto" else implementation)
        sc.check(out, "BTHD", q.dtype)
        return out

    return attention


def make_workload(attention: Attention, backward: bool) -> Workload:
    def workload(args: Inputs) -> Outputs:
        q, k, v, cotangent = args
        sc = ShapeChecker()
        sc.check((q, cotangent), "BTHD", q.dtype)
        sc.check((k, v), "BTKD", q.dtype)
        if backward:
            out, pullback = jax.vjp(attention, q, k, v)
            dq, dk, dv = pullback(cotangent)
            sc.check(dq, "BTHD", q.dtype)
            sc.check((dk, dv), "BTKD", q.dtype)
            result = (out, dq, dk, dv)
        else:
            out = attention(q, k, v)
            result = (out,)
        sc.check(out, "BTHD", q.dtype)
        return result

    return workload


def check_outputs(actual: Outputs, reference: Outputs) -> list[dict[str, float]]:
    """Check finite results and BF16 accuracy against an FP32 oracle."""
    errors = []
    for value, expected in zip(actual, reference, strict=True):
        sc = ShapeChecker()
        sc.check(value, "BTHD", jnp.bfloat16)
        sc.check(expected, "BTHD", jnp.float32)
        a, b = np.asarray(value, np.float32), np.asarray(expected, np.float32)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError("Non-finite output or gradient")
        relative_l2 = float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-12))
        max_abs = float(np.max(np.abs(a - b)))
        if relative_l2 > 0.02:
            raise AssertionError(f"Relative L2 error {relative_l2:.6f} exceeds 2%")
        errors.append({"relative_l2": relative_l2, "max_abs": max_abs})
    return errors


@contextmanager
def paused_process(pid: int | None) -> Iterator[None]:
    """Pause an explicitly authorized process, always resuming after timing."""
    if pid is None:
        yield
        return
    if pid <= 1 or pid == os.getpid():
        raise ValueError("Refusing to pause this process or a system PID")
    try:
        os.kill(pid, signal.SIGSTOP)
        # Allow already submitted GPU work to drain before measuring.
        time.sleep(2)
        yield
    finally:
        os.kill(pid, signal.SIGCONT)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=positive_int, default=1024)
    parser.add_argument("--batches", type=positive_int, nargs="+", default=[1, 8])
    parser.add_argument("--heads", type=positive_int, default=8)
    parser.add_argument("--kv-heads", type=positive_int)
    parser.add_argument("--head-dim", type=positive_int, default=64)
    parser.add_argument("--iterations", type=positive_int, default=20)
    parser.add_argument("--noncausal", action="store_true")
    parser.add_argument("--backends", choices=BACKENDS, nargs="+", default=BACKENDS)
    parser.add_argument("--pause-pid", type=positive_int, help="Requires permission from the process owner")
    parser.add_argument("--output", type=Path, default=Path("runs/attention_benchmark.json"))
    args = parser.parse_args()
    kv_heads = args.kv_heads or args.heads
    if args.heads % kv_heads:
        parser.error("--heads must be divisible by --kv-heads")
    devices = jax.devices()
    if devices[0].platform != "gpu":
        parser.error("GPU required; refusing to report CPU timings as GPU results")

    import tokamax

    report: dict[str, Any] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "device": devices[0].device_kind,
        "versions": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "tokamax")},
        "config": {**vars(args), "output": str(args.output), "kv_heads": kv_heads, "dtype": "bfloat16"},
        "timing": "CUPTI device execution; default configs; no Tokamax autotuning; compilation excluded",
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch in args.batches:
        q_shape = (batch, args.seq_len, args.heads, args.head_dim)
        kv_shape = (batch, args.seq_len, kv_heads, args.head_dim)
        inputs = tuple(
            jax.random.normal(key, shape, dtype=jnp.bfloat16)
            for key, shape in zip(
                jax.random.split(jax.random.key(0), 4), (q_shape, kv_shape, kv_shape, q_shape), strict=True
            )
        )
        sc = ShapeChecker(B=batch, T=args.seq_len, H=args.heads, K=kv_heads, D=args.head_dim)
        sc.check((inputs[0], inputs[3]), "BTHD", jnp.bfloat16)
        sc.check(inputs[1:3], "BTKD", jnp.bfloat16)
        for backward in (False, True):
            reference_fn = make_workload(make_attention("jax_xla", not args.noncausal), backward)
            reference = jax.jit(reference_fn)(tuple(x.astype(jnp.float32) for x in inputs))
            jax.block_until_ready(reference)
            for backend in args.backends:
                row: dict[str, Any] = {
                    "batch": batch,
                    "backend": backend,
                    "mode": "forward_backward" if backward else "forward",
                }
                print(f"Running {row}", flush=True)
                try:
                    workload = make_workload(make_attention(backend, not args.noncausal), backward)
                    actual = jax.jit(workload)(inputs)
                    row["errors_output_dq_dk_dv"] = check_outputs(actual, reference)
                    runner = tokamax.benchmarking.compile_benchmark(workload, inputs)
                    with paused_process(args.pause_pid):
                        timing = runner(inputs, iterations=args.iterations, method="cupti")
                    row.update(
                        status="ok",
                        median_ms=timing.median_evaluation_time_ms,
                        min_ms=min(timing.evaluation_times_ms),
                        max_ms=max(timing.evaluation_times_ms),
                        samples_ms=timing.evaluation_times_ms,
                        compile_ms=timing.compile_time_ms,
                        peak_memory_mb=timing.peak_memory_mb,
                    )
                    del actual, runner
                    print(f"  {row['median_ms']:.4f} ms", flush=True)
                except Exception as error:  # noqa: BLE001 - Record backend failures and continue the comparison.
                    row.update(status="error", error=f"{type(error).__name__}: {error}")
                    print(f"  {row['error']}", flush=True)
                report["results"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                jax.clear_caches()
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
