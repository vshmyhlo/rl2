"""Compare exact training results from fresh Python processes."""

import contextlib
import hashlib
import io
import json
import subprocess
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import gymnasium as gym
import jax
import numpy as np
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rl2 import ppo
from rl2.observation_encoder import ConvStage


class ShortGame(gym.wrappers.TimeLimit):
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ppo.Array, dict[str, Any]]:
        if seed is not None:
            self._max_episode_steps = 7 if self.render_mode else 31 + 16 * (seed % 2)
        return super().reset(seed=seed, options=options)


make_atari = ppo.make_env


def short_game(
    env_id: str,
    render_mode: str | None = None,
    frame_stack: bool = False,
    atari_preprocessing: bool = False,
    observation_size: int | None = None,
) -> gym.Env:
    return ShortGame(
        make_atari(env_id, render_mode, frame_stack, atari_preprocessing, observation_size), max_episode_steps=31
    )


def digest(tree: Any) -> str:
    result = hashlib.sha256()
    for leaf in jax.tree.leaves(jax.device_get(tree)):
        array = np.asarray(leaf)
        if not np.isfinite(array).all():
            raise AssertionError("Non-finite training state")
        result.update(str((array.dtype, array.shape)).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def snapshot(path: str, mode: str, seed: int, videos: bool) -> None:
    config = replace(
        ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        lstm_hidden_size=16,
        encoder_stages=(ConvStage(8), ConvStage(16), ConvStage(16), ConvStage(16)),
        seed=seed,
        vector_env=mode,
        total_steps=128,
        num_envs=2,
        num_steps=16,
        num_minibatches=2,
        update_epochs=2,
        video_every_episodes=2 if videos else 0,
    )
    trajectory = []
    original_act = ppo.act

    def record_act(
        state: ppo.TrainState,
        obs: ppo.Array,
        carry: ppo.LSTMCarry,
        episode_starts: ppo.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, ppo.LSTMCarry]:
        result = original_act(state, obs, carry, episode_starts, key)
        if obs.shape[0] == config.num_envs:  # Exclude the separate video game.
            trajectory.append(digest((obs, carry, episode_starts, result)))
        return result

    with TemporaryDirectory() as log_dir:
        with (
            patch(
                "rl2.ppo.ActorCritic",
                new=partial(
                    ppo.ActorCritic,
                    encoder_stages=(ConvStage(8), ConvStage(16), ConvStage(16), ConvStage(16)),
                    embedding_size=32,
                ),
            ),
            patch("rl2.ppo.make_env", new=short_game),
            patch("rl2.ppo.act", new=record_act),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            state = ppo.train(replace(config, log_dir=log_dir))
        (run_dir,) = Path(log_dir).iterdir()
        events = EventAccumulator(str(run_dir)).Reload()
        metrics = {
            tag: [(event.step, event.value) for event in events.Scalars(tag)]
            for tag in events.Tags()["scalars"]
            if tag != "charts/steps_per_second" and not tag.startswith("time/")
        }
        result = {
            "state": digest((state.step, state.params, state.opt_state)),
            "trajectory": trajectory,
            "metrics": metrics,
        }
        if videos:
            assert events.Images("gameplay"), "Expected recorded games"
        Path(path).write_text(json.dumps(result))


def run_snapshot(path: Path, mode: str, seed: int = 11, videos: bool = False) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--snapshot", str(path), mode, str(seed), str(int(videos))],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def baseline(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return run_snapshot(tmp_path_factory.mktemp("baseline") / "snapshot.json", "sync")


@pytest.mark.parametrize(
    ("mode", "videos"),
    [("sync", False), ("async", False), ("async", False), ("async", True)],  # noqa: PT014 - Check repeated async runs.
    ids=["sync", "async-first", "async-repeat", "async-video"],
)
def test_fresh_process_runs(baseline: dict[str, Any], tmp_path: Path, mode: str, videos: bool) -> None:
    assert baseline == run_snapshot(tmp_path / "snapshot.json", mode, videos=videos)


def test_different_seed_changes_state(baseline: dict[str, Any], tmp_path: Path) -> None:
    assert baseline["state"] != run_snapshot(tmp_path / "snapshot.json", "async", seed=12)["state"]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--snapshot":
        snapshot(sys.argv[2], sys.argv[3], int(sys.argv[4]), bool(int(sys.argv[5])))
