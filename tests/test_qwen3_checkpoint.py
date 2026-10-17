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
import optax
import pytest
from safetensors import safe_open

from rl2.qwen3 import AttentionBackend, Qwen3LM, Variables, load_qwen3_checkpoint


@pytest.fixture(scope="module")
def checkpoint_source() -> Path:
    source = os.environ.get("QWEN3_CHECKPOINT")
    if source is None:
        pytest.skip("Set QWEN3_CHECKPOINT to run the real-checkpoint GPU tests")
    platform = os.environ.get("QWEN3_TEST_PLATFORM", "gpu")
    assert jax.default_backend() == platform, f"Checkpoint parity requires {platform}"
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


def _errors(
    actual: np.ndarray,  # (..., V)
    expected: np.ndarray,  # (..., V)
    *,
    atol: float = 2e-4,
    rtol: float = 0.0,
) -> dict[str, float]:
    assert actual.shape == expected.shape
    actual, expected = actual.astype(np.float64), expected.astype(np.float64)  # Each (..., V).
    difference = np.abs(actual - expected)  # (..., V)
    relative = difference / np.maximum(np.abs(expected), 1e-3)  # (..., V)
    rms = float(np.sqrt(np.square(difference).mean()))
    reference_rms = float(np.sqrt(np.square(expected).mean()))
    percentiles = np.percentile(difference, (50, 95, 99))  # (3,).
    return {
        "elements": actual.size,
        "nan_count": int(np.isnan(actual).sum() + np.isnan(expected).sum()),
        "inf_count": int(np.isinf(actual).sum() + np.isinf(expected).sum()),
        "actual_nonzero": int(np.count_nonzero(actual)),
        "reference_rms": reference_rms,
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "rms": rms,
        "relative_rms": rms / max(reference_rms, 1e-3),
        "max_relative": float(relative.max()),
        "mean_relative": float(relative.mean()),
        "p50_abs": float(percentiles[0]),
        "p95_abs": float(percentiles[1]),
        "p99_abs": float(percentiles[2]),
        "fraction_outside_tolerance": float(np.mean(difference > atol + rtol * np.abs(expected))),
        "argmax_agreement": float(np.mean(actual.argmax(axis=-1) == expected.argmax(axis=-1))),
    }


def test_error_metrics() -> None:
    expected = np.array([[0.0, 2.0]], np.float32)  # (B=1, V=2)
    actual = np.array([[0.001, 3.0]], np.float32)  # (B, V)
    errors = _errors(actual, expected, atol=0.01, rtol=0.1)
    assert errors["max_abs"] == 1
    assert errors["max_relative"] == pytest.approx(1)
    assert errors["mean_abs"] == pytest.approx(0.5005)
    assert errors["rms"] == pytest.approx(np.sqrt(0.5000005))
    assert errors["relative_rms"] == pytest.approx(np.sqrt(0.25000025))
    assert errors["p50_abs"] <= errors["p95_abs"] <= errors["p99_abs"] <= errors["max_abs"]
    assert errors["fraction_outside_tolerance"] == 0.5
    assert errors["argmax_agreement"] == 1
    assert errors["mean_relative"] == pytest.approx(0.75)
    assert errors["nan_count"] == errors["inf_count"] == 0


def _reference_path(directory: Path, dtype: str, *, gradients: bool = False) -> Path:
    platform = os.environ.get("QWEN3_TEST_PLATFORM", "gpu")
    reference_path = directory / f"torch-{platform}-{dtype}-{'grad' if gradients else 'forward'}.npz"
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
                "--device",
                "cuda" if platform == "gpu" else "cpu",
                *(["--gradients"] if gradients else []),
            ],
            env=environment,
            check=True,
        )
    metadata = json.loads(reference_path.with_suffix(".json").read_text())
    assert metadata["checkpoint"] == str(Path(os.environ["QWEN3_CHECKPOINT"]).resolve())
    assert metadata["dtype"] == dtype and metadata["torch_attention"] == "eager"
    assert (metadata["device"] == "cpu") == (platform == "cpu")
    return reference_path


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
    if jax.default_backend() == "cpu" and backend != "xla":
        pytest.skip("Fused GPU backend parity requires GPU")
    directory = audit_directory
    reference_path = _reference_path(directory, dtype)
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
        "platform": jax.default_backend(),
        "elementwise_atol": 0.5 if dtype == "bfloat16" else 2e-4,
        "relative_denominator_floor": 1e-3,
        "jax": jax.__version__,
        "reference": metadata,
    }
    measurements: list[dict[str, float]] = []
    generation_matches: list[bool] = []
    atol = 0.5 if dtype == "bfloat16" else 2e-4
    for side in ("left", "right"):
        tokens = jnp.asarray(reference[f"{side}_tokens"], jnp.int32)  # (B=2, T)
        mask = jnp.asarray(reference[f"{side}_mask"])  # (B, T)
        cache = model.init_cache(batch_size=2, capacity=tokens.shape[1] + 8)
        start = perf_counter()
        output = forward(variables, tokens, attention_mask=mask, cache=cache)
        jax.block_until_ready(output)
        assert output.logits.shape == reference[f"{side}_logits"].shape
        assert output.logits.dtype == jnp.float32  # Public API widens projected logits.
        results[f"{side}_first_call_s"] = perf_counter() - start
        valid = reference[f"{side}_mask"]  # (B, T)
        errors = _errors(np.asarray(output.logits)[valid], reference[f"{side}_logits"][valid], atol=atol)
        results[f"{side}_prefill"] = errors
        measurements.append(errors)
        last = (mask * jnp.arange(tokens.shape[1])).max(axis=1)  # (B,)
        scores = output.logits[jnp.arange(2), last]  # (B, V)
        generated: list[np.ndarray] = []  # Each (B,).
        history_matches = True
        for step in range(8):
            next_ids = jnp.argmax(scores, axis=-1).astype(jnp.int32)  # (B,)
            generated.append(np.asarray(next_ids))
            if step == 7:
                break
            output = forward(variables, next_ids[:, None], cache=output.cache)
            scores = output.logits[:, -1]  # (B, V)
            if step < 3:
                # Logit comparisons require the same cached token history.
                history_matches &= np.array_equal(next_ids[:, None], reference[f"{side}_step_{step}_tokens"])
                if history_matches:
                    errors = _errors(np.asarray(scores), reference[f"{side}_step_{step}_logits"], atol=atol)
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
            generation_matches.append(actual_ids == expected_ids)
        results[f"{side}_generated"] = continuations
        results[f"{side}_decode_history_matches"] = history_matches
    results["generation_matches"] = generation_matches
    report_path = directory / f"jax-{dtype}-{backend}.json"
    report_path.write_text(json.dumps(results, indent=2) + "\n")
    print(report_path.read_text())
    assert all(generation_matches), "Greedy continuations differ; see saved parity report"
    assert all(results[f"{side}_decode_history_matches"] for side in ("left", "right"))
    for errors in measurements:
        assert errors["nan_count"] == errors["inf_count"] == 0
        # BF16 bounds reflect eager-vs-fused rounding; see docs/qwen3.md.
        assert errors["max_abs"] < (0.0002 if dtype == "float32" else 1.5)
        assert errors["mean_abs"] < (0.00002 if dtype == "float32" else 0.2)
        assert errors["argmax_agreement"] >= (1.0 if dtype == "float32" else 0.9)


@pytest.mark.parametrize(
    "dtype,backend",
    [("float32", "xla"), ("float32", "tokamax"), ("bfloat16", "xla"), ("bfloat16", "tokamax"), ("bfloat16", "cudnn")],
)
def test_real_checkpoint_gradients(
    checkpoint: tuple[Qwen3LM, Variables], audit_directory: Path, dtype: str, backend: AttentionBackend
) -> None:
    if jax.default_backend() == "cpu" and backend != "xla":
        pytest.skip("Fused GPU backend parity requires GPU")
    reference_path = _reference_path(audit_directory, dtype, gradients=True)
    with np.load(reference_path) as archive:
        reference = {name: archive[name] for name in archive.files}
    model, master = checkpoint
    model = model.clone(dtype=dtype, attention_backend=backend)
    params = jax.tree.map(lambda weight: weight.astype(dtype), master["params"])  # Parameter shapes.
    assert {name for name in reference if name.endswith(".weight")} == params.keys()
    tokens = jnp.asarray(reference["tokens"], jnp.int32)  # (B=1, T=24)

    def loss(weights: dict[str, jax.Array], ids: jax.Array) -> jax.Array:  # (B, T)
        logits = model.apply({"params": weights}, ids).logits  # (B, T, V)
        return optax.softmax_cross_entropy_with_integer_labels(logits[:, :-1], ids[:, 1:]).mean()  # ()

    value, gradients = jax.jit(jax.value_and_grad(loss))(params, tokens)  # (), parameter shapes.
    jax.block_until_ready(gradients)
    errors = {
        name: _errors(np.asarray(grad.astype(jnp.float32)), reference[name], atol=1e-4 if dtype == "float32" else 0.01)
        for name, grad in gradients.items()
    }
    results = {
        "platform": jax.default_backend(),
        "dtype": dtype,
        "backend": backend,
        "loss": float(value),
        "torch_loss": float(reference["loss"]),
        "gradients": errors,
        "parameter_count": len(gradients),
        "global_rms": float(
            np.sqrt(
                sum(item["rms"] ** 2 * item["elements"] for item in errors.values())
                / sum(item["elements"] for item in errors.values())
            )
        ),
        "global_relative_rms": float(
            np.sqrt(
                sum(item["rms"] ** 2 * item["elements"] for item in errors.values())
                / max(sum(item["reference_rms"] ** 2 * item["elements"] for item in errors.values()), 1e-6)
            )
        ),
        "worst_relative_rms": sorted(errors, key=lambda name: errors[name]["relative_rms"], reverse=True)[:10],
    }
    (audit_directory / f"jax-grad-{dtype}-{backend}.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    # BF16 is a discrepancy diagnostic; a 10% normalized gradient error is material.
    assert abs(float(value) - float(reference["loss"])) < (1e-4 if dtype == "float32" else 0.1)
    for name, grad in gradients.items():
        assert grad.shape == reference[name].shape
        assert grad.dtype == jnp.dtype(dtype)
        assert errors[name]["nan_count"] == errors[name]["inf_count"] == 0
        assert errors[name]["actual_nonzero"] > 0, name
        assert errors[name]["relative_rms"] < (1e-4 if dtype == "float32" else 0.1), name


def test_real_checkpoint_mapping(checkpoint_source: Path) -> None:
    model, variables = load_qwen3_checkpoint(str(checkpoint_source))
    params = variables["params"]
    seen: set[str] = set()
    for path in sorted(checkpoint_source.glob("*.safetensors")):
        with safe_open(path, framework="flax") as archive:
            for name in archive.keys():  # noqa: SIM118 -- safe_open is not iterable.
                mapped = (
                    "model.embed_tokens.weight"
                    if name == "lm_head.weight" and model.config.tie_word_embeddings
                    else name
                )
                expected = archive.get_tensor(name)  # Checkpoint parameter shape.
                assert params[mapped].dtype == expected.dtype
                assert params[mapped].shape == expected.shape
                assert bool(jnp.array_equal(params[mapped], expected)), name
                seen.add(mapped)
    assert seen == params.keys() == model.config.parameter_shapes().keys()


@pytest.mark.parametrize(
    "dtype,backend",
    [("float32", "xla"), ("bfloat16", "xla"), ("bfloat16", "tokamax"), ("bfloat16", "cudnn")],
)
def test_real_checkpoint_intermediates(
    checkpoint_source: Path, audit_directory: Path, dtype: str, backend: AttentionBackend
) -> None:
    output = audit_directory / f"boundaries-{dtype}-{backend}.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("qwen3_precision_probe.py")),
            "--checkpoint",
            str(checkpoint_source),
            "--output",
            str(output),
            "--dtype",
            dtype,
            "--attention-backend",
            backend,
        ],
        env={**os.environ, "PYTHONPATH": os.environ.get("QWEN3_TORCH_PYTHONPATH", ""), "USE_TF": "0"},
        check=True,
        timeout=300,
    )
    results = json.loads(output.read_text())
    assert results["platform"] == "gpu"
    for name, boundary in results["boundaries"].items():
        for errors in boundary.values():
            assert errors["nan_count"] == errors["inf_count"] == 0, name
            assert errors["relative_rms"] < (1e-4 if dtype == "float32" else 0.1), name


def test_real_inference_script(checkpoint_source: Path) -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("cuDNN inference requires GPU")
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
