"""Synchronized GPU reference benchmark matching audit_qwen3_hlo.py's workload."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
import transformers
from tokenizers import Tokenizer
from transformers import Qwen3ForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-backend", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if min(args.batch, args.warmup, args.runs) < 1 or args.length < 2:
        parser.error("batch/warmup/runs must be positive; length must be at least two")
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark requires CUDA")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = (
        Qwen3ForCausalLM.from_pretrained(
            args.checkpoint,
            dtype=getattr(torch, args.dtype),
            attn_implementation=args.attention_backend,
            local_files_only=True,
        )
        .to("cuda")
        .eval()
    )
    prompt = "Explain reinforcement learning in one sentence."
    encoded = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json")).encode(prompt, add_special_tokens=False).ids
    ids = torch.tensor(encoded, dtype=torch.long, device="cuda").repeat(
        args.batch, (args.length + len(encoded) - 1) // len(encoded)
    )[:, : args.length]  # (B, T)
    valid = torch.ones_like(ids, dtype=torch.bool)  # (B, T)

    def forward() -> torch.Tensor:
        return model(ids, attention_mask=valid, use_cache=False).logits.float()  # (B, T, V)

    def backward() -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        logits = forward()  # (B, T, V)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
        )  # ()
        loss.backward()
        return loss  # ()

    results: dict[str, object] = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(),
        "dtype": args.dtype,
        "backend": args.attention_backend,
        "batch": args.batch,
        "length": args.length,
        "prompt": prompt,
        "warmup": args.warmup,
        "checkpoint": str(args.checkpoint.resolve()),
        "tf32": False,
        "compiled": False,
    }
    for mode, function in (("forward", forward), ("backward", backward)):
        with torch.inference_mode(mode == "forward"):
            torch.cuda.reset_peak_memory_stats()
            for _ in range(args.warmup):
                function()
            torch.cuda.synchronize()
            durations: list[float] = []
            for _ in range(args.runs):
                start = perf_counter()
                function()
                torch.cuda.synchronize()
                durations.append(perf_counter() - start)
            elapsed = median(durations)
            results[mode] = {
                "median_seconds": elapsed,
                "tokens_per_second": args.batch * args.length / elapsed,
                "samples_seconds": durations,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
