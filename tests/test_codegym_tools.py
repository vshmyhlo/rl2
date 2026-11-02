import ast
from pathlib import Path

import pytest

from rl2.lm_envs.codegym.tools import annotation_schema, extract_tools

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
