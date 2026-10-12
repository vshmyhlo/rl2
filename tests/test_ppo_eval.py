import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from rl2 import ppo
from rl2.gymnax_env import GymnaxEnv
from rl2.training_scores import TrainingScores


def policy(
    variables: dict[str, jax.Array],
    observations: jax.Array,  # (B,C,H,W)
    carry: ppo.RecurrentCarry,  # leaves: (B,H)
    starts: jax.Array,  # (B,)
    *,
    method: str,
) -> tuple[ppo.RecurrentCarry, jax.Array, jax.Array]:
    return carry, jnp.zeros((observations.shape[0], 3)), jnp.zeros(observations.shape[0])


@pytest.mark.parametrize(
    "frame_limit,stop,expected_length,expected_frames",
    [
        (1, True, 0, 1),
        (2, True, 1, 2),
        (1, False, 2, 1),
        (None, True, 2, 3),
    ],
    ids=["initial-frame-only", "capped-video", "evaluation-finishes", "uncapped-video"],
)
def test_episode_frame_limits(frame_limit: int | None, stop: bool, expected_length: int, expected_frames: int) -> None:
    config = replace(
        ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"), video_max_frames=frame_limit
    )
    env = GymnaxEnv(config.env_id, max_steps_in_episode=2)
    state = TrainState.create(apply_fn=policy, params={}, tx=optax.sgd(0.1))
    carry = ppo.ActorCritic(3, ppo.LSTMConfig(hidden_size=1)).initial_carry(1)  # leaves: (1,1)
    with patch.object(ppo, "GymnaxEnv", return_value=env):
        score, length, frames = ppo.play_episode(
            state, config, jax.random.key(0), carry, record=True, stop_at_frame_limit=stop, env=env
        )
    assert score == 0 and length == expected_length and len(frames) == expected_frames
    assert all(frame.shape == (160, 160, 3) and frame.dtype == np.uint8 for frame in frames)


def test_evaluation_logs_raw_scores_and_independent_seeds() -> None:
    config = replace(ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"), eval_episodes=2)
    writer = Mock()
    state = Mock(spec=TrainState)
    carry = ppo.ActorCritic(3, ppo.LSTMConfig(hidden_size=1)).initial_carry(1)  # leaves: (1,1)
    frame = np.zeros((2, 2, 3), np.uint8)  # (H,W,3)
    scores = TrainingScores(1, 4.0, (4.0,), 2, (2,))
    with patch.object(ppo, "play_episode", side_effect=[(2.0, 3, []), (4.0, 5, [frame])]) as play:
        ppo.log_evaluation(state, config, writer, 1, 128, initial_carry=carry, training_scores=scores)
    for index, call in enumerate(play.call_args_list):
        np.testing.assert_array_equal(
            jax.random.key_data(call.args[2]),
            jax.random.key_data(jax.random.fold_in(jax.random.key(config.eval_seed), index)),
        )
        assert call.args[3] is carry
        assert call.kwargs["record"] == (index == 1)
    scalars = {call.args[0]: call.args[1] for call in writer.add_scalar.call_args_list}
    assert scalars["eval/return_mean"] == 3
    assert scalars["eval/return_std"] == pytest.approx(2**0.5)
    assert scalars["eval/return_sem"] == pytest.approx(1.0)
    assert scalars["training/total_return_mean"] == 4
    assert "eval/human_normalized_score_percent" not in scalars
    report = json.loads(writer.add_text.call_args.args[1].removeprefix("```json\n").removesuffix("\n```"))
    assert report["env_id"] == "Breakout-MinAtar"
    assert report["episodes"] == [{"return": 2, "length": 3}, {"return": 4, "length": 5}]
    assert report["training_scores"] == scores.as_report()
    assert writer.add_video.call_args.args[1].shape == (1, 1, 3, 2, 2)
    assert writer.add_video.call_args.kwargs["fps"] == 10 * config.video_speed


def test_log_video_wiring() -> None:
    config = ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml")
    writer = Mock()
    carry = ppo.ActorCritic(3, ppo.LSTMConfig(hidden_size=1)).initial_carry(1)  # leaves: (1,1)
    with patch.object(ppo, "play_episode", return_value=(0.0, 1, [np.zeros((2, 2, 3), np.uint8)])) as play:
        ppo.log_video(Mock(), config, writer, 2, 128, initial_carry=carry)
    assert play.call_args.kwargs["record"]
    assert play.call_args.kwargs["stop_at_frame_limit"]
    assert writer.add_video.call_args.args[0] == "gameplay"


def test_single_evaluation_omits_unavailable_uncertainty() -> None:
    config = replace(ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"), eval_episodes=1)
    writer = Mock()
    carry = ppo.ActorCritic(3, ppo.LSTMConfig(hidden_size=1)).initial_carry(1)  # leaves: (1,1)
    with patch.object(ppo, "play_episode", return_value=(2.0, 3, [])):
        ppo.log_evaluation(Mock(), config, writer, 0, 0, initial_carry=carry)
    tags = {call.args[0] for call in writer.add_scalar.call_args_list}
    assert "eval/return_mean" in tags
    assert not {"eval/return_std", "eval/return_sem"} & tags
