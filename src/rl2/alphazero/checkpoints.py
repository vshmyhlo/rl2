"""Orbax snapshots at completed AlphaZero iteration boundaries."""

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, NamedTuple

# Use gcsfs for gs:// paths, matching PPO and avoiding a TensorFlow import.
os.environ.setdefault("EPATH_USE_TF", "0")

import jax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
from pydantic import BaseModel, ConfigDict, NonNegativeInt

from rl2.alphazero.config import Config


class TrainingProgress(NamedTuple):
    state: TrainState
    key: jax.Array
    iteration: int
    steps: int
    completed_games: int


class CheckpointMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1] = 1
    config: dict[str, Any]
    iteration: NonNegativeInt
    steps: NonNegativeInt
    completed_games: NonNegativeInt


def checkpoint_manager(run_dir: str) -> ocp.CheckpointManager:
    directory = f"{run_dir.rstrip('/')}/checkpoints"
    if not directory.startswith("gs://"):
        directory = str(Path(directory).resolve())
    return ocp.CheckpointManager(
        directory,
        options=ocp.CheckpointManagerOptions(max_to_keep=2, enable_async_checkpointing=False),
    )


def checkpoint_settings(config: Config) -> dict[str, Any]:
    mutable = {"run_id", "log_dir", "iterations", "log_interval_seconds", "checkpoint_interval_seconds", "evaluation"}
    return {name: value for name, value in asdict(config).items() if name not in mutable}


def save_checkpoint(manager: ocp.CheckpointManager, progress: TrainingProgress, config: Config) -> None:
    """Save weights, optimizer, RNG, and counters after a complete iteration."""
    metadata = CheckpointMetadata(
        config=checkpoint_settings(config),
        iteration=progress.iteration,
        steps=progress.steps,
        completed_games=progress.completed_games,
    )
    manager.save(
        progress.iteration,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave({"train_state": progress.state, "key": progress.key}),
            metadata=ocp.args.JsonSave(metadata.model_dump()),
        ),
        force=True,
    )


def restore_checkpoint(manager: ocp.CheckpointManager, state: TrainState, config: Config) -> TrainingProgress | None:
    iteration = manager.latest_step()
    if iteration is None:
        return None
    saved = manager.restore(iteration, args=ocp.args.Composite(metadata=ocp.args.JsonRestore())).metadata
    metadata = CheckpointMetadata.model_validate(saved)
    if metadata.config != checkpoint_settings(config):
        raise ValueError("Checkpoint training settings are incompatible with this config. Use a new run_id.")
    if metadata.iteration != iteration:
        raise ValueError("Checkpoint iteration does not match its directory")
    restored = manager.restore(
        iteration,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore({"train_state": state, "key": jax.random.PRNGKey(config.seed)})
        ),
    ).state
    return TrainingProgress(
        restored["train_state"], restored["key"], iteration, metadata.steps, metadata.completed_games
    )
