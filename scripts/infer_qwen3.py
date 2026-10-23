"""Generate text or run a two-turn thinking/tool example with Qwen3 in JAX.

uv run python scripts/infer_qwen3.py --demo
uv run python scripts/infer_qwen3.py --prompt 'Hello' --temperature 0
Use --demo --help for example options; --help lists the single-turn options.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Literal

from tokenizers import Tokenizer

from rl2.qwen3 import DEFAULT_ATTENTION_BACKEND, checkpoint_directory, generate_tokens, load_qwen3_checkpoint
from rl2.qwen3 import main as single_turn_main

TOOL = {
    "type": "function",
    "function": {
        "name": "multiply",
        "description": "Multiply two integers.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
}
SYSTEM = (
    "Use multiply for every calculation. Keep your reasoning brief.\n\n"
    f"# Tools\n<tools>\n{json.dumps(TOOL)}\n</tools>\n\n"
    'To use a tool, emit <tool_call>{"name": "multiply", "arguments": {"a": 2, "b": 3}}</tool_call> '
    "with the appropriate arguments. Wait for the tool response before answering."
)
USER_TURNS = ("What is 17 times 23? Use the tool.", "Now multiply that result by 3, using the tool again.")


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str


def format_prompt(messages: list[Message]) -> str:
    """Format this example's Qwen3 chat, retaining thinking only in the current turn."""
    last_user = max(index for index, message in enumerate(messages) if message.role == "user")
    parts: list[str] = []
    for index, message in enumerate(messages):
        content = message.content
        if message.role == "assistant" and index < last_user:
            content = content.rsplit("</think>", 1)[-1].lstrip("\n")
        role = "user" if message.role == "tool" else message.role
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    # Leaving the assistant prefix open enables thinking.
    return "".join(parts) + "<|im_start|>assistant\n"


def tool_responses(completion: str) -> str:
    """Execute multiply calls outside thinking and return Qwen3 tool-response blocks."""
    content = completion.rsplit("</think>", 1)[-1]
    if "<think>" in content:
        raise ValueError("Assistant did not finish thinking")
    calls = re.findall(r"<tool_call>(.*?)</tool_call>", content, flags=re.DOTALL)
    if content.count("<tool_call>") != len(calls) or content.count("</tool_call>") != len(calls):
        raise ValueError("Incomplete tool call")
    responses: list[str] = []
    for payload in calls:
        call = json.loads(payload)
        if not isinstance(call, dict) or call.get("name") != "multiply":
            raise ValueError("Only the multiply tool is supported")
        arguments = call.get("arguments")
        if (
            not isinstance(arguments, dict)
            or arguments.keys() != {"a", "b"}
            or any(type(value) is not int for value in arguments.values())
        ):
            raise ValueError("multiply requires integer arguments a and b")
        result = arguments["a"] * arguments["b"]
        responses.append(f"<tool_response>\n{json.dumps({'result': result})}\n</tool_response>")
    return "\n".join(responses)


def main() -> None:
    if "--demo" not in sys.argv[1:]:
        single_turn_main()
        return

    parser = argparse.ArgumentParser(description="Two user turns with Qwen3 thinking and a local multiply tool.")
    parser.add_argument("--demo", action="store_true", help="Run the thinking/tool example")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--attention-backend", choices=("xla", "tokamax", "cudnn"), default=DEFAULT_ATTENTION_BACKEND)
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
    for user in USER_TURNS:
        print(f"User: {user}", flush=True)
        messages.append(Message(role="user", content=user))
        for step in range(5):
            # Re-prefill the conversation each time to keep the example simple.
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
                raise RuntimeError("Assistant response was truncated; increase --max-new-tokens")
            messages.append(Message(role="assistant", content=completion))
            response = tool_responses(completion)
            if not response:
                break
            if step == 4:
                raise RuntimeError("Exceeded four tool rounds in one user turn")
            print(f"Tool: {response}", flush=True)
            messages.append(Message(role="tool", content=response))


if __name__ == "__main__":
    main()
