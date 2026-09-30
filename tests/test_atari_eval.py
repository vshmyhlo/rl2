import contextlib
import io
import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import Mock, patch

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from rl2 import ppo
from rl2.atari_eval import EvaluationConfig, ScoreBaselines, _action, evaluate, make_evaluation_env
from rl2.atari_scores import ATARI_REFERENCE_SCORES, REFERENCE_SOURCE, get_reference_scores
from rl2.ppo import Array, Config, LSTMCarry, initial_carry, load_config


def training_config() -> Config:
    return replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        env_id="ALE/Pong-v5",
        atari_preprocessing=True,
        lstm_hidden_size=2,
    )


def policy(
    variables: dict[str, Any], obs: Array, carry: LSTMCarry, starts: Array
) -> tuple[LSTMCarry, jax.Array, jax.Array]:
    del obs
    carry = tuple(jnp.where(starts[0, :, None], 0, c) + 1 for c in carry)
    return carry, variables["params"]["logits"][None, None], jnp.zeros((1, 1))


def policy_state() -> TrainState:
    return TrainState.create(apply_fn=policy, params={"logits": jnp.arange(6.0)}, tx=optax.sgd(0.1))


class ScoringEnv(gym.Env):
    """Lose a life, then end by game over or timeout, with non-unit rewards."""

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(6)
        self.episode = -1
        self.steps = 0
        self.closed = False

    def get_action_meanings(self) -> list[str]:
        return ["NOOP"]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Array, dict[str, Any]]:
        super().reset(seed=seed)
        self.episode += 1
        self.steps = 0
        return np.zeros((1, 84, 84), dtype=np.uint8), {"episode_frame_number": 3}

    def step(self, action: int) -> tuple[Array, float, bool, bool, dict[str, Any]]:
        self.steps += 1
        done = self.steps == 2
        return (
            np.zeros((1, 84, 84), dtype=np.uint8),
            5.0 if self.steps == 1 else -2.0 + self.episode * 4,
            done and self.episode == 0,
            done and self.episode != 0,
            {"lives": 1, "episode_frame_number": 3 + self.steps * 4},
        )

    def close(self) -> None:
        self.closed = True


def test_full_games_raw_returns_memory_and_json(tmp_path: Path) -> None:
    env = ScoringEnv()
    starts: list[bool] = []
    memories: list[float] = []

    def action(
        state: TrainState, obs: Array, carry: LSTMCarry, episode_start: bool, key: jax.Array, *, greedy: bool
    ) -> tuple[jax.Array, LSTMCarry]:
        starts.append(episode_start)
        memories.append(float(carry[0][0, 0]))
        return jnp.asarray(0), (carry[0] + 1, carry[1] + 1)

    output = tmp_path / "evaluation.json"
    with (
        patch("rl2.atari_eval.make_evaluation_env", return_value=env),
        patch("rl2.atari_eval._action", side_effect=action),
    ):
        result = evaluate(
            policy_state(),
            training_config(),
            EvaluationConfig(episodes=2),
            baselines=ScoreBaselines("ALE/Pong-v5", -5, 15, "test reference"),
            output_path=output,
        )
    assert json.loads(output.read_text()) == result
    assert [episode["return"] for episode in result["episodes"]] == [3, 7]
    assert result["return_mean"] == 5
    assert result["return_std"] == pytest.approx(np.sqrt(8), rel=0, abs=5e-08)
    assert result["return_sem"] == pytest.approx(2, rel=0, abs=5e-08)
    assert result["human_normalized_score_percent"] == 50
    assert result["baselines"]["source"] == "test reference"
    assert starts == [True, False, True, False]
    assert memories == [0, 1, 0, 1]
    assert result["episodes"][0]["terminated"]
    assert result["episodes"][1]["truncated"]
    assert env.closed


@pytest.mark.parametrize("stacked", (False, True))
@pytest.mark.parametrize("protocol", ("sticky", "legacy_noop"))
@pytest.mark.parametrize("cap", (101, 102, 103, 104))
def test_real_ale_frame_cap_stacking_and_protocol(stacked: bool, protocol: str, cap: int) -> None:
    config = replace(training_config(), frame_stack=stacked)
    env = make_evaluation_env(config, EvaluationConfig(episodes=1, protocol=protocol, max_episode_frames=cap))
    try:
        obs, info = env.reset(seed=7)
        assert obs.shape == (4 if stacked else 1, 84, 84)
        assert info["episode_frame_number"] >= 1
        assert info["episode_frame_number"] <= 30
        assert env.unwrapped.ale.getFloat("repeat_action_probability") == pytest.approx(
            0.25 if protocol == "sticky" else 0, rel=0, abs=5e-08
        )
        for _ in range(cap):
            obs, _, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break
        assert truncated
        assert not terminated
        assert info["episode_frame_number"] == cap
    finally:
        env.close()


def test_real_evaluation_reproducible_and_state_unchanged() -> None:
    state = policy_state()
    before = [np.asarray(leaf).copy() for leaf in jax.tree.leaves(state)]
    options = EvaluationConfig(episodes=2, max_episode_frames=104)
    first = evaluate(state, training_config(), options)
    second = evaluate(state, training_config(), options)
    assert first == second
    assert [ep["seed"] for ep in first["episodes"]] == [10000, 10001]
    for old, new in zip(before, jax.tree.leaves(state), strict=True):
        np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize("greedy", (False, True))
def test_action_selection_and_recurrent_reset(greedy: bool) -> None:
    state = policy_state()
    carry = tuple(c + 5 for c in initial_carry(1, 2))
    obs = np.zeros((1, 84, 84), dtype=np.uint8)
    key = jax.random.key(1)
    action, memory = _action(state, obs, carry, True, key, greedy=greedy)
    expected = 5 if greedy else int(jax.random.categorical(key, state.params["logits"]))
    assert int(action) == expected
    np.testing.assert_array_equal(memory[0], np.ones((1, 2)))


def test_cleanup_on_policy_failure() -> None:
    env = ScoringEnv()
    with (
        patch("rl2.atari_eval.make_evaluation_env", return_value=env),
        patch("rl2.atari_eval._action", side_effect=RuntimeError("policy failure")),
        pytest.raises(RuntimeError, match="policy failure"),
    ):
        evaluate(policy_state(), training_config(), EvaluationConfig(episodes=1))
    assert env.closed


def test_training_schedules_evaluation_by_time_without_changing_state() -> None:
    config = replace(
        training_config(),
        frame_stack=False,
        bf16=False,
        num_envs=1,
        num_steps=4,
        num_minibatches=1,
        update_epochs=1,
        total_steps=12,
        vector_env="sync",
        video_every_episodes=0,
        target_kl=None,
        eval_episodes=2,
        eval_seed=321,
    )

    def training_env(*args: Any, **kwargs: Any) -> ScoringEnv:
        return ScoringEnv()

    def evaluation_env(*args: Any) -> ScoringEnv:
        return ScoringEnv()

    original_update = ppo.update

    def timed_update(state: TrainState, batch: ppo.PPOBatch, config: Config) -> tuple[TrainState, ppo.PPOMetrics]:
        result = original_update(state, batch, config)
        clock.return_value += next(rollout_durations)
        return result

    def timed_evaluate(state: TrainState, training: Config, evaluation: EvaluationConfig) -> dict[str, Any]:
        result = evaluate(state, training, evaluation)
        clock.return_value += 1200.0  # Evaluation itself takes longer than the scheduling interval.
        return result

    states: list[TrainState] = []
    for interval, durations, expected_steps in (
        (0, [599.0, 1.0, 600.0], []),
        (10, [599.0, 1.0, 600.0], [8, 12]),
        (5, [599.0, 1.0, 600.0], [4, 12]),
        (10, [1800.0, 1.0, 600.0], [4, 12]),  # One long rollout crosses several intervals.
    ):
        clock = Mock(return_value=0.0)
        rollout_durations = iter(durations)
        with TemporaryDirectory() as directory:
            with (
                patch("rl2.ppo.make_env", side_effect=training_env),
                patch(
                    "rl2.ppo.ActorCritic",
                    new=partial(ppo.ActorCritic, encoder_channels=(2,), embedding_size=4),
                ),
                patch("rl2.atari_eval.make_evaluation_env", side_effect=evaluation_env),
                patch("rl2.atari_eval.evaluate", side_effect=timed_evaluate) as evaluation,
                patch("rl2.ppo.monotonic", new=clock),
                patch("rl2.ppo.update", side_effect=timed_update),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                states.append(ppo.train(replace(config, log_dir=directory, eval_every_minutes=interval)))
            assert evaluation.call_count == len(expected_steps)
            assert [int(call.args[0].step) for call in evaluation.call_args_list] == [
                step // 4 for step in expected_steps
            ]
            for call in evaluation.call_args_list:
                assert call.args[2] == EvaluationConfig(episodes=2, seed=321)
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            (run_dir,) = Path(directory).iterdir()
            events = EventAccumulator(str(run_dir)).Reload()
            if expected_steps:
                assert [event.step for event in events.Scalars("eval/return_mean")] == expected_steps
                normalized = events.Scalars("eval/human_normalized_score_percent")
                assert [event.step for event in normalized] == expected_steps
                assert [event.value for event in normalized] == pytest.approx(
                    [100 * (5 + 20.7) / (14.6 + 20.7)] * len(expected_steps)
                )
                for event in events.Tensors("eval/report/text_summary"):
                    text = event.tensor_proto.string_val[0].decode()
                    report = json.loads(text.removeprefix("```json\n").removesuffix("\n```"))
                    assert report["training_steps"] == event.step
                    assert report["baselines"]["source"] == REFERENCE_SOURCE
                    assert report["human_normalized_score_percent"] == pytest.approx(100 * (5 + 20.7) / 35.3)
                    assert report["training_episodes"] == event.step // 2
            else:
                assert "eval/return_mean" not in events.Tags()["scalars"]
            assert [event.value for event in events.Scalars("charts/total_episodes")] == [2, 4, 6]
    for state in states[1:]:
        for baseline, actual in zip(jax.tree.leaves(states[0]), jax.tree.leaves(state), strict=True):
            np.testing.assert_array_equal(baseline, actual)


@pytest.mark.parametrize(
    "overrides",
    (
        {"eval_every_minutes": -1},
        {"eval_every_minutes": float("nan")},
        {"eval_every_minutes": float("inf")},
        {"eval_every_minutes": "10"},
        {"eval_every_minutes": 100, "atari_preprocessing": False},
        {"eval_every_minutes": 100, "eval_episodes": 0},
        {"eval_every_minutes": 100, "eval_seed": -1},
    ),
)
def test_training_rejects_invalid_evaluation_before_creating_environments(overrides: dict[str, Any]) -> None:
    with patch("rl2.ppo.gym.vector.AsyncVectorEnv") as envs:
        with pytest.raises(ValueError):
            ppo.train(replace(training_config(), **overrides))
        envs.assert_not_called()


@pytest.mark.parametrize(
    "kwargs", [{"episodes": 0}, {"seed": -1}, {"protocol": "human_starts"}, {"max_episode_frames": 30}]
)
def test_invalid_evaluation_config(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        EvaluationConfig(**kwargs)


@pytest.mark.parametrize("overrides", [{"atari_preprocessing": False}, {"observation_size": 42}])
def test_evaluation_rejects_policy_input_mismatch(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="policy trained"):
        make_evaluation_env(replace(training_config(), **overrides), EvaluationConfig())


def test_evaluation_rejects_baseline_env_mismatch() -> None:
    with pytest.raises(ValueError, match="env_id"):
        evaluate(policy_state(), training_config(), baselines=ScoreBaselines("ALE/Breakout-v5", 0, 1, "test"))


def test_baselines_reject_equal_scores() -> None:
    with pytest.raises(ValueError):
        ScoreBaselines("ALE/Pong-v5", 1, 1, "test")


def test_reference_table_contains_all_57_games() -> None:
    assert len(ATARI_REFERENCE_SCORES) == 57
    assert get_reference_scores("ALE/SpaceInvaders-v5") == (148.0, 1668.7)
    assert get_reference_scores("ALE/Pong-v5") == (-20.7, 14.6)
    assert get_reference_scores("ALE/VideoPinball-v5") == (16256.9, 17667.9)


@pytest.mark.parametrize("game,scores", ATARI_REFERENCE_SCORES.items())
def test_reference_names_match_registered_ale_games(game: str, scores: tuple[float, float]) -> None:
    env_id = "ALE/" + "".join(part.title() for part in game.split("_")) + "-v5"
    assert gym.spec(env_id).kwargs["game"] == game
    assert get_reference_scores(env_id) == scores
    assert np.isfinite(scores).all()
    assert scores[0] != scores[1]


@pytest.mark.parametrize("env_id", ["ALE/Adventure-v5", "ALE/Pong-v4", "PongNoFrameskip-v4", "other/Pong-v5"])
def test_reference_lookup_does_not_guess_unknown_games(env_id: str) -> None:
    assert get_reference_scores(env_id) is None


@pytest.mark.parametrize(
    "env_id,score,expected",
    [
        ("ALE/SpaceInvaders-v5", 148.0, 0.0),
        ("ALE/SpaceInvaders-v5", 1668.7, 100.0),
        ("ALE/SpaceInvaders-v5", 3189.4, 200.0),
        ("ALE/SpaceInvaders-v5", -1372.7, -100.0),
        ("ALE/Skiing-v5", -17098.1, 0.0),
        ("ALE/Skiing-v5", -4336.9, 100.0),
    ],
)
def test_evaluation_automatically_normalizes_from_published_scores(
    env_id: str, score: float, expected: float, tmp_path: Path
) -> None:
    env = ScoringEnv()
    transition = (np.zeros((1, 84, 84), dtype=np.uint8), score, True, False, {"episode_frame_number": 7})
    output = tmp_path / "report.json"
    with (
        patch("rl2.atari_eval.make_evaluation_env", return_value=env),
        patch.object(env, "step", return_value=transition),
    ):
        result = evaluate(
            policy_state(), replace(training_config(), env_id=env_id), EvaluationConfig(episodes=1), output_path=output
        )
    assert result["return_mean"] == score
    assert result["human_normalized_score_percent"] == pytest.approx(expected)
    assert result["baselines"]["env_id"] == env_id
    assert result["baselines"]["source"] == REFERENCE_SOURCE
    assert json.loads(output.read_text()) == result


def test_evaluation_without_reference_scores_keeps_raw_returns() -> None:
    with patch("rl2.atari_eval.make_evaluation_env", return_value=ScoringEnv()):
        result = evaluate(
            policy_state(), replace(training_config(), env_id="ALE/Adventure-v5"), EvaluationConfig(episodes=2)
        )
    assert result["return_mean"] == 5
    assert result["baselines"] is None
    assert result["human_normalized_score_percent"] is None
    json.dumps(result, allow_nan=False)


def test_training_omits_unavailable_normalized_scalar() -> None:
    config = replace(training_config(), env_id="ALE/Adventure-v5", eval_episodes=2)
    with (
        patch("rl2.atari_eval.make_evaluation_env", return_value=ScoringEnv()),
        patch("rl2.ppo.SummaryWriter") as writer,
    ):
        ppo.log_evaluation(policy_state(), config, writer, 100, 5000)
    tags = [call.args[0] for call in writer.add_scalar.call_args_list]
    assert "eval/return_mean" in tags
    assert "eval/human_normalized_score_percent" not in tags


def test_progress_tracks_completed_games_and_long_running_games(capsys: pytest.CaptureFixture[str]) -> None:
    state = policy_state()
    options = EvaluationConfig(episodes=2)
    with (
        patch("rl2.atari_eval.make_evaluation_env", return_value=ScoringEnv()),
        patch("rl2.atari_eval.monotonic", side_effect=[100.0, 101.0, 104.0, 115.0, 120.0]),
    ):
        result = evaluate(state, training_config(), options)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    assert "completed=0/2 (0%)" in lines[0]
    assert "mean_return=n/a elapsed=0s eta=n/a" in lines[0]
    assert "completed=1/2 (50%) episode=1 return=3.0 frames=11" in lines[1]
    assert "mean_return=3.0 elapsed=4s eta=4s" in lines[1]
    # Partial returns must not enter the completed-game mean or advance the count.
    assert "completed=1/2 (50%) episode=2 return=5.0 frames=7" in lines[2]
    assert "mean_return=3.0 elapsed=15s eta=15s" in lines[2]
    assert "completed=2/2 (100%) episode=2 return=7.0 frames=11" in lines[3]
    assert "mean_return=5.0 elapsed=20s eta=0s" in lines[3]
    with (
        patch("rl2.atari_eval.make_evaluation_env", return_value=ScoringEnv()),
        patch("rl2.atari_eval.monotonic", side_effect=AssertionError("quiet evaluation should not track time")),
    ):
        quiet_result = evaluate(state, training_config(), options, show_progress=False)
    assert quiet_result == result
    assert capsys.readouterr().out == ""


def test_failed_evaluation_does_not_report_completion(capsys: pytest.CaptureFixture[str]) -> None:
    env = ScoringEnv()
    with (
        patch("rl2.atari_eval.make_evaluation_env", return_value=env),
        patch("rl2.atari_eval._action", side_effect=RuntimeError("policy failure")),
        pytest.raises(RuntimeError, match="policy failure"),
    ):
        evaluate(policy_state(), training_config(), EvaluationConfig(episodes=1))
    output = capsys.readouterr().out
    assert "completed=0/1" in output
    assert "completed=1/1" not in output
    assert env.closed
