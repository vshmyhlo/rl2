"""Orbax snapshots at completed AlphaZero iteration boundaries."""

import os
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

# Use gcsfs for gs:// paths, matching PPO and avoiding a TensorFlow import.
os.environ.setdefault("EPATH_USE_TF", "0")

import jax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState

from rl2.alphazero.config import Config


class Checkpoint(NamedTuple):
    train_state: TrainState
    key: jax.Array  # (2,)
    iteration: int
    steps: int
    completed_games: int


def checkpoint_manager(run_dir: str) -> ocp.CheckpointManager:
    directory = f"{run_dir.rstrip('/')}/checkpoints"
    if not directory.startswith("gs://"):
        directory = str(Path(directory).resolve())
    return ocp.CheckpointManager(
        directory,
        options=ocp.CheckpointManagerOptions(max_to_keep=1, enable_async_checkpointing=False),
    )


def save_checkpoint(manager: ocp.CheckpointManager, checkpoint: Checkpoint, config: Config) -> None:
    """Save weights, optimizer, RNG, counters, and the full config after a complete iteration."""
    manager.save(
        checkpoint.iteration,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(checkpoint),
            config=ocp.args.JsonSave(asdict(config)),
        ),
        force=True,
    )


def restore_checkpoint(manager: ocp.CheckpointManager, checkpoint: Checkpoint) -> Checkpoint | None:
    """Restore training state using the supplied template, ignoring the saved config."""
    iteration = manager.latest_step()
    if iteration is None:
        return None
    checkpoint = manager.restore(
        iteration,
        args=ocp.args.Composite(state=ocp.args.StandardRestore(checkpoint)),
    ).state
    return checkpoint._replace(iteration=iteration)
