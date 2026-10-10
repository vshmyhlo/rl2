"""OmegaConf loading and interpolation for training configuration."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf


def path_name(value: str) -> str:
    """Convert an environment ID into a single path component."""
    return value.replace("/", "_")


OmegaConf.register_resolver("path_name", path_name)


def resolve_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Generate a missing run ID before resolving references across config fields."""
    config = OmegaConf.create(settings)
    if config.get("run_id") is None:
        env_name = path_name(str(config.env_id))
        config.run_id = f"{env_name}_seed{config.get('seed', 0)}_{datetime.now(UTC):%Y%m%d-%H%M%S-%f}"
    return cast(dict[str, Any], OmegaConf.to_container(config, resolve=True, throw_on_missing=True))


def load_settings(path: str | Path, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML mapping, merge defaults, and resolve before schema validation."""
    config = OmegaConf.load(path)
    if not isinstance(config, DictConfig):
        raise TypeError("The YAML config must contain a mapping of training settings")
    config = OmegaConf.merge(defaults or {}, config)
    settings = cast(dict[str, Any], OmegaConf.to_container(config, resolve=False))
    return resolve_settings(settings)
