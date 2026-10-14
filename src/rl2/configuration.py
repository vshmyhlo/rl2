"""Shared training configuration and OmegaConf YAML interpolation."""

import json
from dataclasses import MISSING, asdict, fields, is_dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Self, cast

from omegaconf import DictConfig, OmegaConf
from pydantic import ConfigDict, TypeAdapter, field_validator
from pydantic.dataclasses import dataclass

from rl2.utils import read_bytes


@dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid", strict=True))
class BaseConfig:
    run_id: str | None = None
    ckpt_dir: str = "runs/${run_id}"  # Full run directory, including local or gs:// paths.

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value in (".", "..") or any(c in value for c in "/\\")):
            raise ValueError("run_id must be a nonempty directory name without slashes or traversal")
        return value

    @field_validator("ckpt_dir")
    @classmethod
    def validate_ckpt_dir(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("ckpt_dir must be a nonempty path")
        return value

    def resolve_run(self) -> Self:
        """Resolve a config constructed directly in Python."""
        return _resolve_config(OmegaConf.create(asdict(self)), type(self))


def path_name(value: str) -> str:
    """Convert an environment ID into a single path component."""
    return value.replace("/", "_")


OmegaConf.register_resolver("path_name", path_name)


def _resolve_config[T: BaseConfig](settings: DictConfig, config_type: type[T]) -> T:
    if settings.get("run_id") is None:
        settings.run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    values = OmegaConf.to_container(settings, resolve=True, throw_on_missing=True)
    # JSON validation accepts YAML lists/objects for strict tuples and nested dataclasses.
    return TypeAdapter(config_type).validate_json(json.dumps(values))


def load_config[T: BaseConfig](path: str | Path, config_type: type[T]) -> T:
    """Load local or gs:// YAML, merge schema defaults, resolve, and validate."""
    settings = OmegaConf.load(StringIO(read_bytes(str(path)).decode()))
    if not isinstance(settings, DictConfig):
        raise TypeError("The YAML config must contain a mapping of training settings")
    defaults = {}
    for field in fields(config_type):
        value = field.default_factory() if field.default_factory is not MISSING else field.default
        if value is not MISSING:
            defaults[field.name] = asdict(value) if is_dataclass(value) else value
    return _resolve_config(cast(DictConfig, OmegaConf.merge(defaults, settings)), config_type)
