"""Chat interactively with Qwen3 in JAX.

uv run python scripts/infer_qwen3.py --temperature 0
Type /exit or press Ctrl+D to quit.
"""

import argparse
from dataclasses import dataclass
from typing import Literal

from tokenizers import Tokenizer

from rl2.qwen3 import (
    ATTENTION_BACKENDS,
    DEFAULT_ATTENTION_BACKEND,
    checkpoint_directory,
    generate_tokens,
    load_qwen3_checkpoint,
)

SYSTEM = (
    "You are a helpful, knowledgeable assistant. Give clear, accurate answers and ask for clarification when needed."
)


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


def format_prompt(messages: list[Message]) -> str:
    """Format Qwen3 chat history, omitting reasoning from previous responses."""
    parts: list[str] = []
    for message in messages:
        content = message.content
        if message.role == "assistant":
            content = content.rsplit("</think>", 1)[-1].lstrip("\n")
            if "<think>" in content:
                content = ""  # Omit unfinished reasoning from a truncated response.
        parts.append(f"<|im_start|>{message.role}\n{content}<|im_end|>\n")
    # Leaving the assistant prefix open enables thinking.
    return "".join(parts) + "<|im_start|>assistant\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat interactively with Qwen3 in JAX/Flax.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--attention-backend", choices=ATTENTION_BACKENDS, default=DEFAULT_ATTENTION_BACKEND)
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Token budget per assistant response")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or not args.temperature >= 0 or args.top_k < 0:
        parser.error("--max-new-tokens must be positive; --temperature and --top-k must be nonnegative")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must fit uint32")
    if args.attention_backend == "cudnn" and args.dtype != "bfloat16":
        parser.error("--attention-backend cudnn requires --dtype bfloat16")

    directory = checkpoint_directory(args.model, args.revision)
    model, variables = load_qwen3_checkpoint(str(directory), dtype=args.dtype, attention_backend=args.attention_backend)
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    messages = [Message(role="system", content=SYSTEM)]
    response_index = 0
    print("Chat ready. Type /exit or press Ctrl+D to quit.", flush=True)
    try:
        while True:
            user = input("You: ").strip()
            if user == "/exit":
                break
            if not user:
                continue
            messages.append(Message(role="user", content=user))
            ids = tokenizer.encode(format_prompt(messages), add_special_tokens=False).ids
            generated = generate_tokens(
                model,
                variables,
                ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=(args.seed + response_index) % 2**32,
            )
            response_index += 1
            completion = tokenizer.decode(generated, skip_special_tokens=True)
            print(f"Assistant: {completion}", flush=True)
            if not generated or generated[-1] != model.config.eos_token_id:
                print("[Response reached the token limit; increase --max-new-tokens for longer answers.]", flush=True)
            messages.append(Message(role="assistant", content=completion))
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
