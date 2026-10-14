"""PPO-RND checkpoints, including curiosity models and normalization statistics."""

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

os.environ.setdefault("EPATH_USE_TF", "0")

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import serialization
from flax.training.train_state import TrainState

from rl2.configuration import BaseConfig
from rl2.utils import RunningMeanStd


class TrainingProgress(NamedTuple):
    state: TrainState  # Policy/optimizer pytree; step: ().
    predictor: TrainState  # RND predictor/optimizer pytree; step: ().
    target: TrainState  # Fixed RND target pytree; step: ().
    key: jax.Array  # () typed PRNG key
    rnd_key: jax.Array  # () typed PRNG key
    iteration: int
    episodes: int
    return_sum: float
    length_sum: int


def checkpoint_manager(directory: str) -> ocp.CheckpointManager:
    directory = directory if directory.startswith("gs://") else str(Path(directory).resolve())
    return ocp.CheckpointManager(
        directory,
        options=ocp.CheckpointManagerOptions(max_to_keep=2, enable_async_checkpointing=False),
    )


def _checkpoint_state(
    progress: TrainingProgress, obs_moments: RunningMeanStd, reward_moments: RunningMeanStd
) -> dict[str, Any]:
    return {
        "state": serialization.to_state_dict(progress.state),
        "predictor": serialization.to_state_dict(progress.predictor),
        "target": serialization.to_state_dict(progress.target),
        "key": jax.random.key_data(progress.key),  # (K,) uint32 key words
        "rnd_key": jax.random.key_data(progress.rnd_key),  # (K,) uint32 key words
        "obs_moments": vars(obs_moments),  # mean/var: (*O,), count: scalar
        "reward_moments": vars(reward_moments),  # mean/var/count: scalars
    }


def save_checkpoint(
    manager: ocp.CheckpointManager,
    progress: TrainingProgress,
    obs_moments: RunningMeanStd,
    reward_moments: RunningMeanStd,
    rng: np.random.Generator,
    config: BaseConfig,
) -> None:
    manager.save(
        progress.iteration,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(_checkpoint_state(progress, obs_moments, reward_moments)),
            metadata=ocp.args.JsonSave(
                {
                    "config": asdict(config),
                    "key_impl": str(jax.random.key_impl(progress.key)),
                    "rng": json.dumps(rng.bit_generator.state),
                    "iteration": progress.iteration,
                    "episodes": progress.episodes,
                    "return_sum": progress.return_sum,
                    "length_sum": progress.length_sum,
                }
            ),
        ),
        force=True,
    )


def restore_checkpoint(
    manager: ocp.CheckpointManager,
    progress: TrainingProgress,
    obs_moments: RunningMeanStd,
    reward_moments: RunningMeanStd,
    rng: np.random.Generator,
) -> TrainingProgress:
    iteration = manager.latest_step()
    if iteration is None:
        return progress
    saved = manager.restore(
        iteration,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(_checkpoint_state(progress, obs_moments, reward_moments)),
            metadata=ocp.args.JsonRestore(),
        ),
    )
    for name, moments in (("obs_moments", obs_moments), ("reward_moments", reward_moments)):
        moments.mean = np.asarray(saved.state[name]["mean"], dtype=np.float64)  # (*S,)
        moments.var = np.asarray(saved.state[name]["var"], dtype=np.float64)  # (*S,)
        moments.count = float(saved.state[name]["count"])
    rng.bit_generator.state = json.loads(saved.metadata["rng"])
    return TrainingProgress(
        state=serialization.from_state_dict(progress.state, saved.state["state"]),
        predictor=serialization.from_state_dict(progress.predictor, saved.state["predictor"]),
        target=serialization.from_state_dict(progress.target, saved.state["target"]),
        key=jax.random.wrap_key_data(jnp.asarray(saved.state["key"]), impl=saved.metadata["key_impl"]),
        rnd_key=jax.random.wrap_key_data(jnp.asarray(saved.state["rnd_key"]), impl=saved.metadata["key_impl"]),
        iteration=iteration,
        episodes=saved.metadata["episodes"],
        return_sum=saved.metadata["return_sum"],
        length_sum=saved.metadata["length_sum"],
    )
