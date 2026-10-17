"""Compare real-checkpoint RMSNorm boundaries to locate accumulated rounding errors."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import torch
from test_qwen3_checkpoint import _errors
from tokenizers import Tokenizer
from transformers import Qwen3ForCausalLM

from rl2 import qwen3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    assert jax.default_backend() == ("gpu" if args.device == "cuda" else "cpu")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = (
        Qwen3ForCausalLM.from_pretrained(
            args.checkpoint, dtype=getattr(torch, args.dtype), attn_implementation="eager", local_files_only=True
        )
        .to(args.device)
        .eval()
    )
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    prompt = "<|im_start|>user\nWhat is the capital of France? Answer in one word.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    expected: list[tuple[np.ndarray, np.ndarray]] = []  # Each pair (..., D/H).
    names: list[str] = []

    def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        names.append(next(name for name, candidate in reference.named_modules() if candidate is module))
        expected.append((inputs[0].float().cpu().numpy(), output.float().cpu().numpy()))  # Each (..., D/H).

    with ExitStack() as stack, torch.inference_mode():
        for name, module in reference.named_modules():
            if name.endswith(("input_layernorm", "post_attention_layernorm", "q_norm", "k_norm", "model.norm")):
                stack.callback(module.register_forward_hook(hook).remove)
        reference(torch.tensor([ids], device=args.device), use_cache=False)

    model, variables = qwen3.load_qwen3_checkpoint(str(args.checkpoint), dtype=args.dtype)
    original = qwen3._rms_norm
    captured: list[tuple[jax.Array, jax.Array]] = []  # Each pair (..., D/H), populated during tracing only.

    def capture(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:  # (..., D/H), (D/H,)
        output = original(x, weight, eps)  # (..., D/H)
        captured.append((x, output))
        return output  # (..., D/H)

    def forward(weights: qwen3.Variables, tokens: jax.Array) -> tuple[tuple[jax.Array, jax.Array], ...]:  # (B, T)
        captured.clear()
        model.apply(weights, tokens)
        return tuple(captured)  # Pairs of (..., D/H) inputs/outputs at each normalization.

    with patch.object(qwen3, "_rms_norm", capture):
        actual = jax.jit(forward)(variables, jnp.asarray([ids], jnp.int32))
        jax.block_until_ready(actual)
    results = {
        "platform": jax.default_backend(),
        "dtype": args.dtype,
        "boundaries": {
            name: {
                kind: _errors(np.asarray(value, dtype=np.float32), target, atol=0.01)
                for kind, value, target in zip(("input", "output"), values, targets, strict=True)
            }
            for name, values, targets in zip(names, actual, expected, strict=True)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
