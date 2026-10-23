import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rl2 import qwen3

example = SimpleNamespace(**runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "infer_qwen3.py")))
CALL = '<tool_call>{"name": "multiply", "arguments": {"a": 17, "b": 23}}</tool_call>'


def test_tool_responses() -> None:
    assert example.tool_responses("<think>Ignore <tool_call>draft</tool_call></think>\nDone.") == ""
    assert example.tool_responses(f"<think>Use the tool.</think>\n{CALL}\n{CALL}") == (
        '<tool_response>\n{"result": 391}\n</tool_response>\n<tool_response>\n{"result": 391}\n</tool_response>'
    )


@pytest.mark.parametrize(
    "completion,match",
    [
        ("<think>unfinished", "finish thinking"),
        ("<tool_call>{", "Incomplete"),
        ("</tool_call>", "Incomplete"),
        ("<tool_call>invalid json</tool_call>", "Expecting value"),
        ("<tool_call>[]</tool_call>", "Only the multiply"),
        (CALL.replace('"multiply"', '"unknown"'), "Only the multiply"),
        (CALL.replace('"a": 17, ', ""), "integer arguments"),
        (CALL.replace('"a": 17', '"a": true'), "integer arguments"),
        (CALL.replace('"a": 17', '"a": "17"'), "integer arguments"),
    ],
)
def test_invalid_tool_responses(completion: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        example.tool_responses(completion)


@pytest.fixture
def demo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    tokenizer = Mock()
    tokenizer.encode.return_value.ids = [1, 4]
    tokenizer.decode.side_effect = [
        f"<think>First calculation.</think>\n{CALL}",
        "<think>Read first result.</think>\n391",
        (
            "<think>Follow-up calculation.</think>\n"
            '<tool_call>{"name": "multiply", "arguments": {"a": 391, "b": 3}}</tool_call>'
        ),
        "<think>Read second result.</think>\n1173",
    ]
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=2))
    generate = Mock(return_value=[7, 2])
    load = Mock(return_value=(model, {}))
    monkeypatch.setattr(qwen3, "checkpoint_directory", Mock(return_value=tmp_path))
    monkeypatch.setattr(qwen3, "load_qwen3_checkpoint", load)
    monkeypatch.setattr("tokenizers.Tokenizer", Mock(from_file=Mock(return_value=tokenizer)))
    monkeypatch.setattr(qwen3, "generate_tokens", generate)
    monkeypatch.setattr("sys.argv", ["infer_qwen3", "--demo", "--seed", "7", "--max-new-tokens", "64"])
    return SimpleNamespace(tokenizer=tokenizer, generate=generate, load=load)


def run_demo() -> None:
    runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "infer_qwen3.py"), run_name="__main__")


def test_demo(demo: SimpleNamespace, capsys: pytest.CaptureFixture[str]) -> None:
    run_demo()
    demo.load.assert_called_once()
    assert demo.load.call_args.kwargs == {"dtype": "bfloat16", "attention_backend": "tokamax"}
    prompts = [call.args[0] for call in demo.tokenizer.encode.call_args_list]
    assert len(prompts) == 4
    assert all(prompt.endswith("<|im_start|>assistant\n") for prompt in prompts)
    assert "<tools>" in prompts[0] and '"name": "multiply"' in prompts[0]
    assert "<think>First calculation.</think>" in prompts[1]
    assert '<|im_start|>user\n<tool_response>\n{"result": 391}\n</tool_response><|im_end|>' in prompts[1]
    assert "First calculation." not in prompts[2] and "Read first result." not in prompts[2]
    assert f"<|im_start|>assistant\n{CALL}<|im_end|>" in prompts[2]
    assert "<|im_start|>assistant\n391<|im_end|>" in prompts[2]
    assert "Now multiply that result by 3" in prompts[2]
    assert "<think>Follow-up calculation.</think>" in prompts[3]
    assert '{"result": 1173}' in prompts[3]
    assert [call.kwargs["seed"] for call in demo.generate.call_args_list] == [7, 8, 9, 10]
    assert all(call.kwargs["max_new_tokens"] == 64 for call in demo.generate.call_args_list)
    output = capsys.readouterr().out
    assert output.count("User: ") == 2 and output.count("Tool: ") == 2
    assert "<think>Read second result.</think>\n1173" in output


def test_demo_truncation(demo: SimpleNamespace, capsys: pytest.CaptureFixture[str]) -> None:
    demo.generate.return_value = [7]
    with pytest.raises(RuntimeError, match="truncated"):
        run_demo()
    demo.generate.assert_called_once()
    assert "Tool:" not in capsys.readouterr().out


def test_demo_tool_round_limit(demo: SimpleNamespace) -> None:
    demo.tokenizer.decode.side_effect = None
    demo.tokenizer.decode.return_value = CALL
    with pytest.raises(RuntimeError, match="four tool rounds"):
        run_demo()
    assert demo.generate.call_count == 5


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
def test_demo_invalid_arguments(demo: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, arguments: list[str]) -> None:
    monkeypatch.setattr("sys.argv", ["infer_qwen3", "--demo", *arguments])
    with pytest.raises(SystemExit, match="2"):
        run_demo()
    demo.load.assert_not_called()
