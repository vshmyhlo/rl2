"""Orbax rollout checkpoints shared by the Karel trainers."""

import json
import os
from pathlib import Path
from typing import Any, NamedTuple

os.environ.setdefault("EPATH_USE_TF", "0")

import chex
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import serialization
from flax.training.train_state import TrainState

from rl2.utils import read_optional


class TrainingProgress(NamedTuple):
    state: TrainState  # Parameter/optimizer pytree; step: ().
    key: jax.Array  # () typed PRNG key
    iteration: int
    steps: int
    logged_groups: int
    episodes: int


def checkpoint_manager(run_dir: str) -> ocp.CheckpointManager:
    directory = run_dir if run_dir.startswith("gs://") else str(Path(run_dir).resolve())
    return ocp.CheckpointManager(
        directory,
        options=ocp.CheckpointManagerOptions(max_to_keep=2, enable_async_checkpointing=False),
    )


def save_checkpoint(run_dir: str, progress: TrainingProgress, rng: np.random.Generator) -> None:
    """Save model, optimizer, both RNGs, and completed rollout counters."""
    with checkpoint_manager(run_dir) as manager:
        manager.save(
            progress.iteration,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(
                    {"state": serialization.to_state_dict(progress.state), "key": jax.random.key_data(progress.key)}
                ),
                metadata=ocp.args.JsonSave(
                    {
                        "version": 1,
                        "key_impl": str(jax.random.key_impl(progress.key)),
                        "numpy_rng": json.dumps(rng.bit_generator.state),
                        "iteration": progress.iteration,
                        "steps": progress.steps,
                        "logged_groups": progress.logged_groups,
                        "episodes": progress.episodes,
                    }
                ),
            ),
            force=True,
        )


def _restore_payload(payload: dict[str, Any], state: TrainState, rng: np.random.Generator) -> TrainingProgress:
    if payload["version"] != 1:
        raise ValueError("Unsupported GRPO checkpoint version")
    try:
        restored = serialization.from_state_dict(state, payload["state"])
        chex.assert_trees_all_equal_shapes_and_dtypes(
            (state.params, state.opt_state), (restored.params, restored.opt_state)
        )
    except (ValueError, AssertionError) as error:
        raise ValueError("Checkpoint model/optimizer structure is incompatible with this config.") from error
    key = jax.random.wrap_key_data(jnp.asarray(payload["key"]), impl=payload["key_impl"])  # () typed key
    for name in ("iteration", "steps", "logged_groups", "episodes"):
        if type(payload[name]) is not int or payload[name] < 0:
            raise ValueError("Invalid checkpoint counters")
    rng.bit_generator.state = json.loads(payload["numpy_rng"])
    return TrainingProgress(
        state=restored,
        key=key,
        iteration=payload["iteration"],
        steps=payload["episodes"],
        logged_groups=payload["logged_groups"],
        episodes=payload["episodes"],
    )


def restore_legacy_checkpoint(data: bytes, state: TrainState, rng: np.random.Generator) -> TrainingProgress:
    return _restore_payload(serialization.msgpack_restore(data), state, rng)


def restore_checkpoint(run_dir: str, state: TrainState, rng: np.random.Generator) -> TrainingProgress | None:
    with checkpoint_manager(run_dir) as manager:
        iteration = manager.latest_step()
        if iteration is not None:
            metadata = manager.restore(iteration, args=ocp.args.Composite(metadata=ocp.args.JsonRestore())).metadata
            key = jax.random.key(0, impl=metadata["key_impl"])  # () typed key
            saved = manager.restore(
                iteration,
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(
                        {"state": serialization.to_state_dict(state), "key": jax.random.key_data(key)}
                    )
                ),
            )
            return _restore_payload(dict(metadata) | dict(saved.state), state, rng)
    data = read_optional(f"{run_dir.rstrip('/')}/checkpoint.msgpack")
    return None if data is None else restore_legacy_checkpoint(data, state, rng)
