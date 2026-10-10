"""Validated AlphaZero settings and OmegaConf YAML loading."""

from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, NonNegativeInt, PositiveInt, field_validator
from pydantic.dataclasses import dataclass

from rl2.configuration import load_settings

type EnvId = Literal["chess", "gardner_chess", "tic_tac_toe"]
type PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ModelConfig:
    channels: Annotated[int, Field(gt=0, strict=True)] = 64
    num_blocks: Annotated[int, Field(gt=0, strict=True)] = 3


@dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class Config:
    env_id: EnvId = "chess"
    seed: Annotated[int, Field(ge=0, lt=2**32)] = 0
    iterations: PositiveInt = 100
    num_envs: PositiveInt = 8
    max_moves: PositiveInt = 512
    num_simulations: PositiveInt = 32
    model: ModelConfig = ModelConfig()
    batch_size: PositiveInt = 128
    updates_per_iteration: PositiveInt = 8
    learning_rate: PositiveFiniteFloat = 1e-3
    weight_decay: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 1e-4
    max_grad_norm: PositiveFiniteFloat = 1.0
    dirichlet_alpha: PositiveFiniteFloat = 0.3
    dirichlet_fraction: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.25
    exploration_moves: NonNegativeInt = 30
    log_dir: Annotated[str, Field(pattern=r"\S")] = "runs/alphazero/${run_id}"
    run_id: str | None = None

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or value in (".", "..") or any(c in value for c in "/\\")):
            raise ValueError("run_id must be a nonempty directory name without slashes or traversal")
        return value


def load_config(path: str | Path) -> Config:
    """Load and validate training settings from a YAML mapping."""
    return Config(**load_settings(path, asdict(Config())))
