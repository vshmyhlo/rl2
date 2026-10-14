from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from unittest.mock import Mock

import pytest
from omegaconf.errors import OmegaConfBaseException
from pydantic import ValidationError
from pydantic.dataclasses import dataclass

from rl2 import configuration
from rl2.configuration import BaseConfig, load_config


@dataclass(frozen=True)
class ExampleConfig(BaseConfig):
    required_count: int
    env_id: str = "ALE/Boxing-v5"
    iterations: int = 3
    batch_size: int = 1


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
    path.write_text(f"required_count: 1\nrun_id: run\nckpt_dir: {template}\n")
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


def test_load_config_resolves_schema_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("required_count: 2\nbatch_size: ${iterations}\n")
    config = load_config(path, ExampleConfig)
    assert config.required_count == 2
    assert config.batch_size == 3
    assert type(config.batch_size) is int
    assert config.run_id is not None
    assert config.ckpt_dir == f"runs/{config.run_id}"


def test_load_remote_config(monkeypatch: pytest.MonkeyPatch) -> None:
    read = Mock(return_value=b"run_id: null\nckpt_dir: gs://bucket/${run_id}\n")
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


def test_base_config_resolves_run_and_remains_frozen() -> None:
    config = BaseConfig(ckpt_dir="gs://bucket/${run_id}").resolve_run()
    assert config.run_id is not None
    assert config.ckpt_dir == f"gs://bucket/{config.run_id}"
    assert config.resolve_run() == config
    assert BaseConfig(**asdict(config)) == config
    assert hash(BaseConfig(**asdict(config))) == hash(config)
    with pytest.raises(FrozenInstanceError):
        config.run_id = "changed"
