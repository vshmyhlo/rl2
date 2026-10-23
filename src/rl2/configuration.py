"""Shared training configuration and OmegaConf YAML interpolation."""

import json
from dataclasses import asdict, replace
from hashlib import sha256
from io import StringIO
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pydantic import ConfigDict, TypeAdapter, field_validator
from pydantic.dataclasses import dataclass

from rl2.utils import read_bytes


@dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid", strict=True))
class BaseConfig:
    run_id: str | None = None
    ckpt_dir: str = "runs"  # Full run directory, including local or gs:// paths.

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


def path_name(value: str) -> str:
    """Convert an environment ID into a single path component."""
    return value.replace("/", "_")


OmegaConf.register_resolver("path_name", path_name)


def config_digest(config: BaseConfig) -> str:
    """Hash validated settings independently of run identity and output location."""
    values = asdict(config)
    del values["run_id"], values["ckpt_dir"]
    return sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def load_config[T: BaseConfig](path: str | Path, config_type: type[T]) -> T:
    """Resolve references within local or gs:// YAML, then validate and apply defaults."""
    settings = OmegaConf.load(StringIO(read_bytes(str(path)).decode()))
    if not isinstance(settings, DictConfig):
        raise TypeError("The YAML config must contain a mapping of training settings")
    values = OmegaConf.to_container(settings, resolve=True, throw_on_missing=True)
    # JSON validation accepts YAML lists/objects for strict tuples and nested dataclasses.
    config = TypeAdapter(config_type).validate_json(json.dumps(values))
    run_id = config.run_id if config.run_id is not None else f"run_{config_digest(config)}"
    return replace(
        config,
        run_id=run_id,
        ckpt_dir=config.ckpt_dir if "ckpt_dir" in settings else f"runs/{run_id}",
    )
