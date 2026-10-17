"""Generate real-checkpoint GPU references with Transformers (test-only PyTorch)."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from tokenizers import Tokenizer
from transformers import Qwen3ForCausalLM

PROMPTS = (
    "What is the capital of France? Answer in one word.",
    "Compute 7 * 8. Answer with just the number.",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    parser.add_argument("--attention-backend", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gradients", action="store_true", help="Compare every parameter gradient instead of decoding")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Real checkpoint reference requires CUDA")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = (
        Qwen3ForCausalLM.from_pretrained(
            args.checkpoint,
            dtype=getattr(torch, args.dtype),
            attn_implementation=args.attention_backend,
            local_files_only=True,
        )
        .to(args.device)
        .eval()
    )
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    texts = [f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" for text in PROMPTS]
    encoded = [tokenizer.encode(text, add_special_tokens=False).ids for text in texts]
    width = max(map(len, encoded))
    arrays: dict[str, np.ndarray] = {}
    if args.gradients:
        model.train()  # This checkpoint has zero attention dropout.
        ids = torch.tensor([encoded[0]], device=args.device, dtype=torch.long)  # (B=1, T=24)
        logits = model(ids, use_cache=False).logits  # (B, T, V)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
        )  # ()
        loss.backward()
        arrays["tokens"] = ids.cpu().numpy()  # (B, T)
        arrays["loss"] = loss.detach().float().cpu().numpy()  # ()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"Missing gradient: {name}"
            arrays[name] = parameter.grad.float().cpu().numpy()  # Parameter shape.
    with torch.inference_mode():
        for side in () if args.gradients else ("left", "right"):
            ids = np.full((2, width), 151643, np.int32)  # (B=2, T)
            valid = np.zeros_like(ids, dtype=np.bool_)  # (B, T)
            for row, sequence in enumerate(encoded):
                start = width - len(sequence) if side == "left" else 0
                ids[row, start : start + len(sequence)] = sequence
                valid[row, start : start + len(sequence)] = True
            positions = np.maximum(valid.cumsum(axis=1) - 1, 0).astype(np.int32)  # (B, T)
            tokens = torch.tensor(ids, device=args.device, dtype=torch.long)  # (B, T)
            mask = torch.tensor(valid, device=args.device)  # (B, T)
            pos = torch.tensor(positions, device=args.device, dtype=torch.long)  # (B, T)
            output = model(tokens, attention_mask=mask, position_ids=pos, use_cache=True)
            arrays[f"{side}_tokens"] = ids  # (B, T)
            arrays[f"{side}_mask"] = valid  # (B, T)
            arrays[f"{side}_positions"] = positions  # (B, T)
            arrays[f"{side}_logits"] = output.logits.float().cpu().numpy()  # (B, T, V)
            lengths = mask.sum(dim=1)  # (B,)
            last = (mask * torch.arange(width, device=args.device)).max(dim=1).values  # (B,)
            scores = output.logits[torch.arange(2, device=args.device), last]  # (B, V)
            for step in range(3):
                next_ids = scores.argmax(dim=-1, keepdim=True)  # (B, 1)
                arrays[f"{side}_step_{step}_tokens"] = next_ids.cpu().numpy()  # (B, 1)
                mask = torch.cat(
                    (mask, torch.ones((2, 1), device=args.device, dtype=torch.bool)), dim=1
                )  # (B, T+step+1)
                output = model(
                    next_ids,
                    attention_mask=mask,
                    position_ids=(lengths + step)[:, None],
                    past_key_values=output.past_key_values,
                    use_cache=True,
                )
                scores = output.logits[:, -1]  # (B, V)
                arrays[f"{side}_step_{step}_logits"] = scores.float().cpu().numpy()  # (B, V)
            if side == "left":
                generated = model.generate(
                    tokens,
                    attention_mask=torch.tensor(valid, device=args.device),
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=151643,
                    eos_token_id=model.config.eos_token_id,
                )[:, width:]  # (B, generated T)
                arrays["generated"] = generated.cpu().numpy()  # (B, generated T)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    metadata = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dtype": args.dtype,
        "torch_attention": args.attention_backend,
        "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "gradients": args.gradients,
        "gradient_parameters": len(list(model.named_parameters())) if args.gradients else 0,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "autocast": False,
        "logits_dtype": str(logits.dtype) if args.gradients else str(output.logits.dtype),
        "tf32": False,
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "checkpoint": str(args.checkpoint.resolve()),
        "prompts": list(PROMPTS),
        "formatted_prompts": texts,
        "token_lengths": list(map(len, encoded)),
    }
    if "generated" in arrays:
        metadata["generated"] = arrays["generated"].tolist()
        metadata["generated_text"] = [
            tokenizer.decode(row.tolist(), skip_special_tokens=True) for row in arrays["generated"]
        ]
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
