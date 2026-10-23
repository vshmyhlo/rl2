from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from typing import Annotated, Literal
from unittest.mock import Mock

import pytest
from omegaconf.errors import InterpolationKeyError, OmegaConfBaseException
from pydantic import Field, ValidationError
from pydantic.dataclasses import dataclass

from rl2 import configuration
from rl2.configuration import BaseConfig, load_config


@dataclass(frozen=True)
class ExampleConfig(BaseConfig):
    required_count: int
    env_id: str = "ALE/Boxing-v5"
    iterations: int = 3
    batch_size: int = 1


@dataclass(frozen=True)
class FirstTaskConfig:
    path: str
    id: Literal["first"] = "first"


@dataclass(frozen=True)
class SecondTaskConfig:
    root: str
    id: Literal["second"] = "second"


@dataclass(frozen=True)
class TaskConfig(BaseConfig):
    task: Annotated[FirstTaskConfig | SecondTaskConfig, Field(discriminator="id")] = FirstTaskConfig(
        path="default-path"
    )


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        ("{}", FirstTaskConfig(path="default-path")),
        ("task: {id: first, path: custom}", FirstTaskConfig(path="custom")),
        ("task: {id: second, root: data}", SecondTaskConfig(root="data")),
    ],
    ids=["omitted", "same-variant", "different-variant"],
)
def test_nested_variant_defaults(tmp_path: Path, contents: str, expected: FirstTaskConfig | SecondTaskConfig) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(contents)
    assert load_config(path, TaskConfig).task == expected


def test_nested_variant_does_not_inherit_default_instance_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("task: {id: first}")
    with pytest.raises(ValidationError) as raised:
        load_config(path, TaskConfig)
    assert [(error["loc"], error["type"]) for error in raised.value.errors()] == [
        (("task", "first", "path"), "missing")
    ]


@pytest.mark.parametrize(
    "template,expected",
    [
        ("gs://bucket/${run_id}/${path_name:${env_id}}", "gs://bucket/run/ALE_Boxing-v5"),
        ("logs/${run_id}", "logs/run"),
        ("logs/${env_id}/${run_id}", "logs/ALE/Boxing-v5/run"),
        ("logs/", "logs/"),
    ],
    ids=["cloud-path-name", "run-only", "unmodified-env-id", "literal-path"],
)
def test_config_interpolation(tmp_path: Path, template: str, expected: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(f"required_count: 1\nenv_id: ALE/Boxing-v5\nrun_id: run\nckpt_dir: {template}\n")
    config = load_config(path, ExampleConfig)
    assert config.ckpt_dir == expected
    assert config.env_id == "ALE/Boxing-v5"


@pytest.mark.parametrize(
    "template",
    ["logs/${unknown}", "logs/${run_id", "${ckpt_dir}"],
    ids=["unknown-field", "malformed", "cycle"],
)
def test_config_rejects_invalid_interpolation(tmp_path: Path, template: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(f"ckpt_dir: {template}\n")
    with pytest.raises(OmegaConfBaseException):
        load_config(path, BaseConfig)


def test_load_config_applies_schema_defaults_after_resolving_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("required_count: 2\nbatch_size: ${required_count}\n")
    config = load_config(path, ExampleConfig)
    assert config.required_count == 2
    assert config.batch_size == 2
    assert config.iterations == 3
    assert type(config.batch_size) is int
    assert config.run_id is not None
    assert config.ckpt_dir == f"runs/{config.run_id}"


@pytest.mark.parametrize(
    "contents",
    [
        "required_count: 2\nbatch_size: ${iterations}\n",
        "required_count: 2\nckpt_dir: runs/${run_id}\n",
        "run_id: ${ckpt_dir}\n",
    ],
    ids=["schema-default", "generated-run-id", "default-checkpoint-dir"],
)
def test_interpolation_requires_values_in_yaml(contents: str, tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(contents)
    with pytest.raises(InterpolationKeyError):
        load_config(path, ExampleConfig)


def test_generated_identity_resumes_and_tracks_config_changes(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("required_count: 2\niterations: 3\nbatch_size: ${iterations}\n")
    original = load_config(path, ExampleConfig)
    assert load_config(path, ExampleConfig) == original
    path.write_text("batch_size: 3\nrequired_count: 2\nckpt_dir: elsewhere\n")
    equivalent = load_config(path, ExampleConfig)
    assert equivalent.run_id == original.run_id
    assert equivalent.ckpt_dir == "elsewhere"
    path.write_text("required_count: 2\nbatch_size: 2\n")
    changed = load_config(path, ExampleConfig)
    assert changed.run_id != original.run_id
    assert changed.ckpt_dir != original.ckpt_dir


@pytest.mark.parametrize(
    "directory,expected",
    [
        ("/runs", "/runs"),
        ("checkpoints/custom/", "checkpoints/custom/"),
        ("gs://bucket/checkpoints", "gs://bucket/checkpoints"),
        ("/", "/"),
    ],
    ids=["absolute", "relative-trailing-slash", "cloud", "root"],
)
def test_generated_identity_preserves_checkpoint_directory(tmp_path: Path, directory: str, expected: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(f"required_count: 1\nckpt_dir: {directory}\n")
    config = load_config(path, ExampleConfig)
    assert config.ckpt_dir == expected.format(run_id=config.run_id)
    assert load_config(path, ExampleConfig) == config


@pytest.mark.parametrize("directory", ["null", "''"], ids=["null", "empty"])
def test_generated_identity_rejects_invalid_directory(tmp_path: Path, directory: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(f"ckpt_dir: {directory}\n")
    with pytest.raises(ValidationError, match="ckpt_dir"):
        load_config(path, BaseConfig)


def test_explicit_identity_is_preserved_on_resubmission(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("required_count: 2\nrun_id: existing\nckpt_dir: runs/${run_id}\n")
    original = load_config(path, ExampleConfig)
    assert load_config(path, ExampleConfig) == original
    assert original.run_id == "existing"
    assert original.ckpt_dir == "runs/existing"


def test_load_remote_config(monkeypatch: pytest.MonkeyPatch) -> None:
    read = Mock(return_value=b"run_id: remote\nckpt_dir: gs://bucket/${run_id}\n")
    monkeypatch.setattr(configuration, "read_bytes", read)
    config = load_config("gs://bucket/config.yaml", BaseConfig)
    read.assert_called_once_with("gs://bucket/config.yaml")
    assert config.run_id is not None
    assert config.ckpt_dir == f"gs://bucket/{config.run_id}"


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- seed: 1\n")
    with pytest.raises(TypeError, match="mapping"):
        load_config(path, BaseConfig)


@pytest.mark.parametrize("contents", ["{}", "required_count: '1'"], ids=["missing", "wrong-type"])
def test_load_config_validates_required_fields(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(contents)
    with pytest.raises(ValidationError, match="required_count"):
        load_config(path, ExampleConfig)


@pytest.mark.parametrize(
    "settings,match",
    [
        ({"run_id": "../escape"}, "run_id"),
        ({"run_id": " "}, "run_id"),
        ({"run_id": 1}, "run_id"),
        ({"ckpt_dir": " "}, "ckpt_dir"),
        ({"ckpt_dir": None}, "ckpt_dir"),
        ({"unknown": 1}, "unknown"),
    ],
)
def test_base_config_validation(settings: dict[str, object], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        BaseConfig(**settings)


def test_python_config_preserves_literal_strings_and_remains_frozen() -> None:
    config = BaseConfig(ckpt_dir="gs://bucket/${run_id}")
    assert config.run_id is None
    assert config.ckpt_dir == "gs://bucket/${run_id}"
    assert BaseConfig(**asdict(config)) == config
    assert hash(BaseConfig(**asdict(config))) == hash(config)
    with pytest.raises(FrozenInstanceError):
        config.run_id = "changed"
