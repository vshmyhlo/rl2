import ast
from pathlib import Path
from typing import Any

import pytest

from rl2.lm_envs.codegym.tools import annotation_schema, extract_tools, tool_description

SOURCE = (Path(__file__).parent / "fixtures/codegym.py").read_text()


def test_public_tools_and_wire_parameters() -> None:
    source = SOURCE.replace('self.Earn(params["answer"])', 'self.Earn(params["result"])')
    tools = {tool["function"]["name"]: tool["function"] for tool in extract_tools(source, "SumEnv")}
    assert set(tools) == {"Observe", "Done", "Earn"}
    assert tools["Observe"]["parameters"] == {
        "type": "object",
        "properties": {"nested": {"type": "array", "items": {}}},
        "required": [],
        "additionalProperties": False,
    }
    assert tools["Done"]["parameters"]["properties"] == {"answer": {}}
    assert "answer (int)" in tools["Done"]["description"]
    assert tools["Earn"]["parameters"]["properties"] == {"result": {}}
    assert tools["Earn"]["parameters"]["required"] == ["result"]


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (
            "dict[str, list[int]]",
            {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "integer"}}},
        ),
        (
            "Tuple[int, str]",
            {"type": "array", "prefixItems": [{"type": "integer"}, {"type": "string"}], "minItems": 2, "maxItems": 2},
        ),
        ("tuple[int, ...]", {"type": "array", "items": {"type": "integer"}}),
        ("Optional[int]", {"anyOf": [{"type": "integer"}, {"type": "null"}]}),
        ("int | None", {"anyOf": [{"type": "integer"}, {"type": "null"}]}),
        ('"list[str]"', {"type": "array", "items": {"type": "string"}}),
        ("CustomType", {}),
    ],
)
def test_annotations(annotation: str, expected: dict) -> None:
    assert annotation_schema(ast.parse(annotation, mode="eval").body) == expected


def test_unknown_dispatch_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="dispatch branch"):
        extract_tools(SOURCE.replace("action_code == 2", "action_code > 2"), "SumEnv")


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('self.Earn(params["answer"])', 'self.result = params["answer"]'),
        ("action_code == 2", 'action_code == self.func_mapping["Earn"]'),
        (
            'self.func_mapping = {"Observe": 0, "Done": 1, "Earn": 2}',
            'EARN = 2; self.func_mapping = {"Observe": 0, "Done": 1, "Earn": EARN}',
        ),
        ("elif action_code == 2:", 'elif "answer" in params:'),
    ],
    ids=["inline-action", "mapping-lookup", "local-constant", "final-parameter-branch"],
)
def test_dispatch_variations(old: str, new: str) -> None:
    tools = extract_tools(SOURCE.replace(old, new), "SumEnv")
    assert tools[-1]["function"]["parameters"] == {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }


@pytest.mark.parametrize("iterable", ["required_params", '["answer"]'])
def test_parameter_key_comprehension(iterable: str) -> None:
    source = SOURCE.replace(
        'self.Earn(params["answer"])',
        f'required_params = ["answer"]; values = [params[p] for p in {iterable}]; self.Earn(*values)',
    )
    parameters = extract_tools(source, "SumEnv")[-1]["function"]["parameters"]
    assert parameters["properties"] == {"answer": {"type": "integer"}}
    assert parameters["required"] == ["answer"]


def test_dynamic_parameter_key_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="dynamic CodeGym parameter"):
        extract_tools(SOURCE.replace('params["answer"]', "params[unknown_key]"), "SumEnv")


def test_serialized_wire_value_overrides_method_annotation() -> None:
    source = SOURCE.replace('self.Earn(params["answer"])', 'self.Earn(json.loads(params["answer"]))')
    parameters = extract_tools(source, "SumEnv")[-1]["function"]["parameters"]
    assert parameters["properties"]["answer"]["type"] == "string"
    assert parameters["required"] == ["answer"]


@pytest.mark.parametrize(
    ("documented_type", "expected"),
    [
        ("int", {"type": "integer"}),
        ("list[int]", {"type": "array", "items": {"type": "integer"}}),
        ("CustomType", {}),
        ("list of integers", {}),
    ],
)
def test_documented_parameter_type(documented_type: str, expected: dict[str, Any]) -> None:
    source = SOURCE.replace(
        "Submit the sum. answer (int): The proposed result.",
        f"Submit the sum.\n\n        Args:\n            answer ({documented_type}): The proposed result.\n"
        '\n        Example Output:\n            "Reference answer: 3, reward=1"\n        ',
    )
    tool = extract_tools(source, "SumEnv")[1]["function"]
    assert tool["parameters"]["properties"]["answer"] == expected
    assert "The proposed result" in tool["description"]
    assert "Example Output" not in tool["description"]
    assert "Reference answer" not in tool["description"]


def test_annotation_and_wire_format_override_docstring() -> None:
    source = SOURCE.replace(
        "Award an intermediate reward.",
        "Award an intermediate reward.\n\n        Args:\n            answer (str): The result.\n        ",
    )
    assert extract_tools(source, "SumEnv")[-1]["function"]["parameters"]["properties"]["answer"] == {"type": "integer"}
    source = (
        source.replace("answer: int", "answer")
        .replace("answer (str)", "answer (int)")
        .replace('self.Earn(params["answer"])', 'self.Earn(json.loads(params["answer"]))')
    )
    assert extract_tools(source, "SumEnv")[-1]["function"]["parameters"]["properties"]["answer"]["type"] == "string"


def test_remove_examples_preserves_other_sections() -> None:
    description = (
        'Inspect the instance.\n\nExample Output:\n    "Reference answer: 3"\n\n'
        "Returns:\n    str: Current state.\n\nExamples:\n    Observe()\n"
    )
    assert tool_description(description) == "Inspect the instance.\n\nReturns:\n    str: Current state."
