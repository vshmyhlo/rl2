"""Manually verify a real converted checkpoint; not collected as a unit test.

Usage: uv run --no-sync python tests/verify_gdn2_checkpoint.py \
    --source /tmp/gdn2-paper-matched.pth --checkpoint checkpoints/gdn2/paper-matched-30b

Requires CPU PyTorch. Reports exact parameter preservation and full-vocabulary
logit agreement with the independent source-layout reference from the unit tests.
"""

import argparse
import gc
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.traverse_util import flatten_dict
from test_gdn2_checkpoints import torch_reference_logits

from rl2.gdn2.checkpoints import load_checkpoint, read_pytorch_checkpoint, sha256_file, weight_specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, variables = load_checkpoint(args.checkpoint)
    manifest = json.loads((args.checkpoint / "manifest.json").read_text())
    source_hash = sha256_file(args.source)
    if source_hash != manifest["metadata"]["source"]["sha256"]:
        raise ValueError("Source file does not match checkpoint provenance")
    source, _ = read_pytorch_checkpoint(args.source)
    flat = flatten_dict(variables["params"], sep="/")
    specs = weight_specs(model)
    for spec in specs:
        original = source[spec.source]
        if spec.transform == "linear":
            original = original.T
        elif spec.transform == "conv":
            original = original[:, 0, :].T
        np.testing.assert_array_equal(flat[spec.target], original, err_msg=spec.source)
    print(f"All {len(specs)} converted tensors exactly preserve source values.", flush=True)
    tokens = np.array([[1, 42, 100, model.vocab_size - 1]], np.int32)
    expected = torch_reference_logits(source, model, tokens)
    del source
    gc.collect()
    print("PyTorch CPU reference completed; evaluating JAX.", flush=True)
    _, logits = jax.jit(model.apply)(variables, jnp.asarray(tokens), jnp.full((1,), tokens.shape[1], jnp.int32))
    actual = np.asarray(logits)
    np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=5e-4)
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
    error = actual - expected
    report = {
        "source_sha256": source_hash,
        "params_sha256": manifest["params_sha256"],
        "parameter_count": manifest["parameter_count"],
        "exactly_preserved_tensors": len(specs),
        "tokens_batch_first": tokens.tolist(),
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "torch_version": torch.__version__,
        "reference": "Independent PyTorch CPU source-layout equations; not upstream Triton kernels.",
        "max_absolute_logit_error": float(np.abs(error).max()),
        "rms_logit_error": float(np.sqrt(np.mean(error**2))),
        "all_top1_tokens_match": True,
        "top1_tokens": actual.argmax(-1).tolist(),
        "rtol": 5e-4,
        "atol": 5e-4,
    }
    path = args.checkpoint / "verification.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
