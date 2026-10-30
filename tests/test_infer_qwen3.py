import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rl2 import qwen3

example = SimpleNamespace(**runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "infer_qwen3.py")))


@pytest.fixture
def chat(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    tokenizer = Mock()
    tokenizer.encode.return_value.ids = [1, 4]
    tokenizer.decode.side_effect = ["<think>Consider the greeting.</think>\nHello!", "How can I help?"]
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=2))
    generate = Mock(return_value=[7, 2])
    load = Mock(return_value=(model, {}))
    monkeypatch.setattr(qwen3, "checkpoint_directory", Mock(return_value=tmp_path))
    monkeypatch.setattr(qwen3, "load_qwen3_checkpoint", load)
    monkeypatch.setattr("tokenizers.Tokenizer", Mock(from_file=Mock(return_value=tokenizer)))
    monkeypatch.setattr(qwen3, "generate_tokens", generate)
    monkeypatch.setattr("sys.argv", ["infer_qwen3", "--seed", str(2**32 - 1), "--max-new-tokens", "64"])
    user_input = Mock(side_effect=["Hi", "   ", "Can you help?", "/exit"])
    monkeypatch.setattr("builtins.input", user_input)
    return SimpleNamespace(tokenizer=tokenizer, generate=generate, load=load, user_input=user_input)


def run_chat() -> None:
    runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "infer_qwen3.py"), run_name="__main__")


def test_interactive_chat(chat: SimpleNamespace, capsys: pytest.CaptureFixture[str]) -> None:
    run_chat()
    chat.load.assert_called_once()
    assert chat.load.call_args.kwargs == {"dtype": "bfloat16", "attention_backend": "tokamax"}
    prompts = [call.args[0] for call in chat.tokenizer.encode.call_args_list]
    assert len(prompts) == 2
    assert prompts[0] == (
        f"<|im_start|>system\n{example.SYSTEM}<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    )
    assert "helpful" in example.SYSTEM and "<tools>" not in example.SYSTEM
    assert prompts[1] == (
        prompts[0] + "Hello!<|im_end|>\n<|im_start|>user\nCan you help?<|im_end|>\n<|im_start|>assistant\n"
    )
    assert [call.kwargs["seed"] for call in chat.generate.call_args_list] == [2**32 - 1, 0]
    assert all(call.kwargs["max_new_tokens"] == 64 for call in chat.generate.call_args_list)
    output = capsys.readouterr().out
    assert "Assistant: <think>Consider the greeting.</think>\nHello!" in output
    assert "Assistant: How can I help?" in output
    assert "token limit" not in output


def test_chat_truncation_continues(chat: SimpleNamespace, capsys: pytest.CaptureFixture[str]) -> None:
    chat.generate.side_effect = [[7], [7, 2]]
    chat.tokenizer.decode.side_effect = ["<think>Unfinished reasoning", "Hello!"]
    run_chat()
    assert chat.generate.call_count == 2
    second_prompt = chat.tokenizer.encode.call_args_list[1].args[0]
    assert "Unfinished reasoning" not in second_prompt
    assert "<|im_start|>user\nCan you help?<|im_end|>" in second_prompt
    output = capsys.readouterr().out
    assert output.count("token limit") == 1
    assert "Assistant: Hello!" in output


@pytest.mark.parametrize("stop", ["/exit", EOFError, KeyboardInterrupt], ids=["exit", "eof", "interrupt"])
def test_chat_exit(chat: SimpleNamespace, stop: str | type[BaseException]) -> None:
    chat.user_input.side_effect = [stop]
    run_chat()
    chat.generate.assert_not_called()


def test_chat_interrupt_generation(chat: SimpleNamespace) -> None:
    chat.generate.side_effect = KeyboardInterrupt
    run_chat()
    chat.generate.assert_called_once()
    chat.tokenizer.decode.assert_not_called()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--max-new-tokens", "0"],
        ["--temperature", "-1"],
        ["--top-k", "-1"],
        ["--seed", str(2**32)],
        ["--attention-backend", "cudnn", "--dtype", "float32"],
    ],
)
def test_chat_invalid_arguments(chat: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, arguments: list[str]) -> None:
    monkeypatch.setattr("sys.argv", ["infer_qwen3", *arguments])
    with pytest.raises(SystemExit, match="2"):
        run_chat()
    chat.load.assert_not_called()
