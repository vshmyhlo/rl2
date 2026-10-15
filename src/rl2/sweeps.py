"""Expand YAML parameter grids into resumable training configurations."""

import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, ListConfig, OmegaConf

from rl2.configuration import config_digest


@dataclass(frozen=True)
class Sweep:
    name: str
    trials: list[DictConfig]


def load_sweep(path: Path, root: Path) -> Sweep:
    """Resolve base config paths relative to the repository root."""
    spec = OmegaConf.load(path)
    if not isinstance(spec, DictConfig) or set(spec) - {"name", "config", "parameters", "ckpt_dir"}:
        raise ValueError("Sweep must be a mapping containing name, config, parameters, and optional ckpt_dir")
    name = spec.get("name", path.stem)
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?", name):
        raise ValueError(
            "Sweep name must be 1..63 lowercase letters, digits, underscores or hyphens, with alphanumeric ends"
        )
    config = spec.get("config")
    if not isinstance(config, str) or not config:
        raise ValueError("Sweep config must be a YAML file path")
    parameters = spec.get("parameters")
    if not isinstance(parameters, DictConfig) or not parameters:
        raise ValueError("Sweep parameters must be a nonempty mapping of dotted keys to nonempty lists")
    if any(not isinstance(key, str) or any(not part for part in key.split(".")) for key in parameters):
        raise ValueError("Parameter keys must be nonempty dotted paths")
    keys = list(parameters)
    for key, values in parameters.items():
        if key.split(".")[0] in {"run_id", "ckpt_dir"}:
            raise ValueError("Sweep controls run_id and ckpt_dir; sweep training parameters instead")
        if any(other != key and other.startswith(f"{key}.") for other in keys):
            raise ValueError(f"Overlapping sweep parameter paths: {key}")
        if not isinstance(values, ListConfig) or not values:
            raise ValueError(f"Parameter {key} must have a nonempty list of values")
    base = OmegaConf.load(root / config)
    if not isinstance(base, DictConfig):
        raise TypeError("Base config must be a mapping")
    base_fields = cast(dict[str, object], OmegaConf.to_container(base, resolve=False))
    sweep_fields = cast(dict[str, object], OmegaConf.to_container(spec, resolve=False))
    directory = sweep_fields.get("ckpt_dir", base_fields.get("ckpt_dir", "/runs/${run_id}"))
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("ckpt_dir must be a nonempty string")
    trials = []
    seen = set()
    for values in product(*(parameters[key] for key in keys)):
        trial = base.copy()
        for key, value in zip(keys, values, strict=True):
            OmegaConf.update(trial, key, value, merge=False, force_add=True)
        trial.run_id = f"{name}_{config_digest(trial)}"
        if trial.run_id in seen:
            continue
        seen.add(trial.run_id)
        trial.ckpt_dir = directory
        trials.append(trial)
    return Sweep(name=name, trials=trials)
