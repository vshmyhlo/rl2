"""Opt-in real-checkpoint GPU parity; set QWEN3_CHECKPOINT to a local snapshot."""

import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.qwen3 import AttentionBackend, Qwen3LM, Variables, load_qwen3_checkpoint


@pytest.fixture(scope="module")
def checkpoint_source() -> Path:
    source = os.environ.get("QWEN3_CHECKPOINT")
    if source is None:
        pytest.skip("Set QWEN3_CHECKPOINT to run the real-checkpoint GPU tests")
    assert jax.default_backend() == "gpu", "Checkpoint parity must run on GPU"
    assert not jax.config.jax_disable_jit, "Checkpoint parity requires JIT"
    return Path(source)


@pytest.fixture(scope="module")
def checkpoint(checkpoint_source: Path) -> tuple[Qwen3LM, Variables]:
    return load_qwen3_checkpoint(str(checkpoint_source), dtype="float32")


@pytest.fixture(scope="module")
def audit_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = (
        Path(os.environ["QWEN3_AUDIT_DIR"]) if "QWEN3_AUDIT_DIR" in os.environ else tmp_path_factory.mktemp("qwen3")
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _errors(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:  # (..., V), (..., V)
    difference = np.abs(actual - expected)  # (..., V)
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "rms": float(np.sqrt(np.square(difference).mean())),
        "argmax_agreement": float(np.mean(actual.argmax(axis=-1) == expected.argmax(axis=-1))),
    }


@pytest.mark.parametrize(
    "dtype,backend",
    [
        ("float32", "xla"),
        ("float32", "tokamax"),
        ("bfloat16", "xla"),
        ("bfloat16", "tokamax"),
        ("bfloat16", "cudnn"),
    ],
)
def test_real_checkpoint(
    checkpoint: tuple[Qwen3LM, Variables],
    audit_directory: Path,
    dtype: str,
    backend: AttentionBackend,
) -> None:
    directory = audit_directory
    reference_path = directory / f"torch-{dtype}.npz"
    if not reference_path.exists():
        environment = dict(os.environ)
        if pythonpath := os.environ.get("QWEN3_TORCH_PYTHONPATH"):
            environment["PYTHONPATH"] = pythonpath
        environment["USE_TF"] = "0"
        subprocess.run(
            [
                os.environ.get("QWEN3_TORCH_PYTHON", sys.executable),
                str(Path(__file__).with_name("qwen3_checkpoint_reference.py")),
                "--checkpoint",
                os.environ["QWEN3_CHECKPOINT"],
                "--output",
                str(reference_path),
                "--dtype",
                dtype,
            ],
            env=environment,
            check=True,
        )
    with np.load(reference_path) as archive:
        reference = {name: archive[name] for name in archive.files}
    model, master = checkpoint
    model = model.clone(dtype=dtype, attention_backend=backend)
    variables = jax.tree.map(lambda value: value.astype(dtype), master)  # Checkpoint parameter shapes.
    forward = jax.jit(model.apply)
    metadata = json.loads(reference_path.with_suffix(".json").read_text())
    assert metadata["checkpoint"] == str(Path(os.environ["QWEN3_CHECKPOINT"]).resolve())
    assert metadata["dtype"] == dtype and metadata["torch_attention"] == "eager"
    results: dict[str, object] = {
        "dtype": dtype,
        "backend": backend,
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "reference": metadata,
    }
    measurements: list[dict[str, float]] = []
    for side in ("left", "right"):
        tokens = jnp.asarray(reference[f"{side}_tokens"], jnp.int32)  # (B=2, T)
        mask = jnp.asarray(reference[f"{side}_mask"])  # (B, T)
        cache = model.init_cache(batch_size=2, capacity=tokens.shape[1] + 8)
        start = perf_counter()
        output = forward(variables, tokens, attention_mask=mask, cache=cache)
        jax.block_until_ready(output)
        results[f"{side}_first_call_s"] = perf_counter() - start
        valid = reference[f"{side}_mask"]  # (B, T)
        errors = _errors(np.asarray(output.logits)[valid], reference[f"{side}_logits"][valid])
        results[f"{side}_prefill"] = errors
        measurements.append(errors)
        last = (mask * jnp.arange(tokens.shape[1])).max(axis=1)  # (B,)
        scores = output.logits[jnp.arange(2), last]  # (B, V)
        generated: list[np.ndarray] = []  # Each (B,).
        for step in range(8):
            next_ids = jnp.argmax(scores, axis=-1).astype(jnp.int32)  # (B,)
            generated.append(np.asarray(next_ids))
            if step == 7:
                break
            output = forward(variables, next_ids[:, None], cache=output.cache)
            scores = output.logits[:, -1]  # (B, V)
            if step < 3:
                # Logit comparisons require the same cached token history.
                np.testing.assert_array_equal(next_ids[:, None], reference[f"{side}_step_{step}_tokens"])
                errors = _errors(np.asarray(scores), reference[f"{side}_step_{step}_logits"])
                results[f"{side}_decode_{step}"] = errors
                measurements.append(errors)
        actual = np.stack(generated, axis=1)  # (B, generated T=8)
        continuations: list[list[int]] = []
        for row in range(2):
            expected_ids = reference["generated"][row].tolist()
            if model.config.eos_token_id in expected_ids:
                expected_ids = expected_ids[: expected_ids.index(model.config.eos_token_id) + 1]
            actual_ids = actual[row, : len(expected_ids)].tolist()
            continuations.append(actual_ids)
            assert actual_ids == expected_ids
        results[f"{side}_generated"] = continuations
    report_path = directory / f"jax-{dtype}-{backend}.json"
    report_path.write_text(json.dumps(results, indent=2) + "\n")
    print(report_path.read_text())
    for errors in measurements:
        # BF16 bounds reflect eager-vs-fused rounding; see docs/qwen3.md.
        assert errors["max_abs"] < (0.0002 if dtype == "float32" else 1.5)
        assert errors["mean_abs"] < (0.00002 if dtype == "float32" else 0.2)
        assert errors["argmax_agreement"] >= (1.0 if dtype == "float32" else 0.9)


def test_real_inference_script(checkpoint_source: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts" / "infer_qwen3.py"),
            "--model",
            str(checkpoint_source),
            "--attention-backend",
            "cudnn",
            "--temperature",
            "0",
            "--max-new-tokens",
            "8",
            "--prompt",
            "What is the capital of France? Answer in one word.",
        ],
        env={**os.environ, "JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        capture_output=True,
        text=True,
        check=True,
        timeout=180,
    )
    assert result.stdout == "Paris\n"  # Same completion as PyTorch generate, stopping at EOS.
