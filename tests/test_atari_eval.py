import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState
from numpy.typing import NDArray
from pydantic import ValidationError

from rl2 import ppo, ppo_rnd
from rl2.atari_eval import (
    EvaluationAgent,
    EvaluationConfig,
    ScoreBaselines,
    TrainingScores,
    TrainStateAgent,
    evaluate,
    make_evaluation_env,
)
from rl2.atari_scores import ATARI_REFERENCE_SCORES, REFERENCE_SOURCE, get_reference_scores
from rl2.ppo import Config, LSTMStackCarry, load_config


def training_config() -> Config:
    return replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        env_id="ALE/Pong-v5",
        atari_preprocessing=True,
        model=ppo.LSTMConfig(hidden_size=2),
    )


def policy(
    variables: dict[str, Any], obs: jax.Array, carry: LSTMStackCarry, starts: jax.Array, *, method: str
) -> tuple[LSTMStackCarry, jax.Array, jax.Array]:
    assert method == "step"
    chex.assert_shape(obs, (None, None, None, None))
    chex.assert_type(obs, jnp.uint8)
    chex.assert_shape(starts, (obs.shape[0],))
    chex.assert_type(starts, jnp.bool_)
    for layer in carry:
        chex.assert_shape(layer, (obs.shape[0], None))
        chex.assert_type(layer, jnp.float32)
    carry = tuple(tuple(jnp.where(starts[:, None], 0, c) + 1 for c in layer) for layer in carry)
    return carry, variables["params"]["logits"][None], jnp.zeros(1)


def policy_state() -> TrainState:
    return TrainState.create(apply_fn=policy, params={"logits": jnp.arange(6.0)}, tx=optax.sgd(0.1))


@pytest.mark.parametrize(
    "scores,mean,recent_mean,recent_count",
    [
        pytest.param(TrainingScores(0, 0.0, ()), None, None, 0, id="no-completed-episodes"),
        pytest.param(TrainingScores(2, -4.0, (-1.0, -3.0)), -2.0, -2.0, 2, id="short-history"),
        pytest.param(TrainingScores(101, 300.0, (2.0,) * 100), 300 / 101, 2.0, 100, id="full-window"),
        pytest.param(TrainingScores(101, None, (2.0,) * 100), None, 2.0, 100, id="unknown-historical-total"),
        pytest.param(TrainingScores(101, 300.0), 300 / 101, None, None, id="unknown-recent-history"),
    ],
)
def test_training_score_means(
    scores: TrainingScores, mean: float | None, recent_mean: float | None, recent_count: int | None
) -> None:
    report = scores.as_report()
    assert report == {
        "total_episodes": scores.episode_count,
        "total_return_mean": mean,
        "return_mean_last_100": recent_mean,
        "episodes_last_100": recent_count,
        "total_episode_length_mean": None,
        "episode_length_mean_last_100": None,
    }
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "count,total,recent,mean,recent_mean",
    [
        pytest.param(0, 0, (), None, None, id="no-completed-episodes"),
        pytest.param(2, 10, (3, 7), 5.0, 5.0, id="short-history"),
        pytest.param(101, 305, (3,) * 100, 305 / 101, 3.0, id="full-window"),
        pytest.param(101, None, (3,) * 100, None, 3.0, id="unknown-historical-total"),
        pytest.param(2, 10, None, 5.0, None, id="unknown-recent-history"),
    ],
)
def test_training_episode_length_means(
    count: int,
    total: int | None,
    recent: tuple[int, ...] | None,
    mean: float | None,
    recent_mean: float | None,
) -> None:
    report = TrainingScores(count, None, length_sum=total, recent_lengths=recent).as_report()
    assert report["total_episode_length_mean"] == mean
    assert report["episode_length_mean_last_100"] == recent_mean
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "count,total,recent",
    [
        pytest.param(1, -1, None, id="negative-total"),
        pytest.param(101, None, (1,) * 101, id="window-too-long"),
        pytest.param(1, None, (1, 2), id="more-than-completed"),
        pytest.param(1, None, (0,), id="nonpositive-length"),
    ],
)
def test_training_scores_reject_invalid_lengths(count: int, total: int | None, recent: tuple[int, ...] | None) -> None:
    with pytest.raises(ValueError):
        TrainingScores(count, None, length_sum=total, recent_lengths=recent)


@pytest.mark.parametrize(
    "count,total",
    [
        pytest.param(-1, 0.0, id="negative-count"),
        pytest.param(1, float("inf"), id="nonfinite-total"),
    ],
)
def test_training_scores_reject_invalid_history(count: int, total: float | None) -> None:
    with pytest.raises(ValueError):
        TrainingScores(count, total)


@pytest.mark.parametrize(
    "count,recent_returns",
    [
        pytest.param(101, (1.0,) * 101, id="window-too-long"),
        pytest.param(1, (1.0, 2.0), id="more-than-completed"),
        pytest.param(1, (float("nan"),), id="nonfinite-return"),
    ],
)
def test_training_scores_reject_invalid_recent_history(count: int, recent_returns: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        TrainingScores(count, None, recent_returns)


class ScoringEnv(gym.Env):
    """Lose a life, then end by game over or timeout, with non-unit rewards."""

    def __init__(self) -> None:
        self.metadata = {"render_fps": 60}
        self.observation_space = gym.spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(6)
        self.ale = Mock(spec=["cloneState"])
        self.ale.cloneState.return_value.getCurrentMode.return_value = 22
        self.episode = -1
        self.steps = 0
        self.closed = False
        self.rgb = np.empty((2, 3, 3), dtype=np.uint8)  # (H, W, C)

    def get_action_meanings(self) -> list[str]:
        return ["NOOP"]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.uint8], dict[str, Any]]:  # obs: (1, 84, 84)
        super().reset(seed=seed)
        self.episode += 1
        self.steps = 0
        return np.zeros((1, 84, 84), dtype=np.uint8), {"episode_frame_number": 3}

    def step(self, action: int) -> tuple[NDArray[np.uint8], float, bool, bool, dict[str, Any]]:  # obs: (1, 84, 84)
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

    def render(self) -> NDArray[np.uint8]:  # (H, W, C)
        self.rgb.fill(self.episode * 10 + self.steps)
        return self.rgb  # (H, W, C); reuse the buffer to catch missing frame copies


@pytest.mark.parametrize(
    "episodes,frame_cap,record,expected_frames",
    [
        pytest.param(2, None, False, 0, id="disabled"),
        pytest.param(2, 1, True, 1, id="reset-frame-only"),
        pytest.param(2, 2, True, 2, id="cap-before-episode-end"),
        pytest.param(2, None, True, 3, id="uncapped-truncation"),
        pytest.param(1, 900, True, 3, id="termination-before-cap"),
    ],
)
def test_evaluation_records_only_last_episode(
    episodes: int, frame_cap: int | None, record: bool, expected_frames: int
) -> None:
    config = replace(training_config(), video_max_frames=frame_cap, video_speed=2.0)
    env = ScoringEnv()
    on_video = Mock()
    with (
        patch("rl2.atari_eval.make_evaluation_env", return_value=env) as make_env,
        patch.object(env, "render", wraps=env.render) as render,
        patch("rl2.atari_eval._action", return_value=(jnp.asarray(0), ())),
    ):
        result = evaluate(
            TrainStateAgent(policy_state(), ()),
            config,
            EvaluationConfig(episodes=episodes),
            on_video=on_video if record else None,
            show_progress=False,
        )
    assert make_env.call_args.kwargs["render_mode"] == ("rgb_array" if record else None)
    assert render.call_count == expected_frames
    assert env.closed
    assert env.episode == episodes - 1
    assert [episode["agent_steps"] for episode in result["episodes"]] == [2] * episodes
    assert [episode["return"] for episode in result["episodes"]] == [3.0 + 4 * i for i in range(episodes)]
    if record:
        on_video.assert_called_once()
        video, fps = on_video.call_args.args  # video: (1, T, C, H, W)
        assert video.shape == (1, expected_frames, 3, 2, 3)
        assert video.dtype == np.uint8
        for index in range(expected_frames):
            np.testing.assert_array_equal(video[0, index], (episodes - 1) * 10 + index)
        assert fps == 30
    else:
        on_video.assert_not_called()


def test_evaluation_closes_environment_on_video_failure() -> None:
    env = ScoringEnv()
    with (
        patch("rl2.atari_eval.make_evaluation_env", return_value=env),
        patch("rl2.atari_eval._action", return_value=(jnp.asarray(0), ())),
        pytest.raises(RuntimeError, match="video failure"),
    ):
        evaluate(
            TrainStateAgent(policy_state(), ()),
            training_config(),
            EvaluationConfig(episodes=1),
            on_video=Mock(side_effect=RuntimeError("video failure")),
            show_progress=False,
        )
    assert env.closed


def test_evaluation_accepts_custom_recurrent_agent() -> None:
    starts: list[bool] = []
    resets: list[int] = []

    class Agent:
        def initial_carry(self) -> dict[str, jax.Array]:  # memory: (2,)
            resets.append(len(starts))
            return {"memory": jnp.zeros(2)}  # memory: (2,)

        def act(
            self,
            obs: jax.Array,  # (1, 84, 84)
            carry: dict[str, jax.Array],  # memory: (2,)
            episode_start: bool,
            key: jax.Array,  # () typed PRNG key
            *,
            greedy: bool,
        ) -> tuple[jax.Array, dict[str, jax.Array]]:  # () action; memory: (2,)
            chex.assert_shape(obs, (1, 84, 84))
            chex.assert_shape(carry["memory"], (2,))
            chex.assert_shape(key, ())
            assert greedy
            starts.append(episode_start)
            return carry["memory"][0].astype(jnp.int32), jax.tree.map(lambda leaf: leaf + 1, carry)  # (); memory: (2,)

    agent: EvaluationAgent[dict[str, jax.Array]] = Agent()
    env = ScoringEnv()
    with (
        patch("rl2.atari_eval.make_evaluation_env", return_value=env),
        patch.object(env, "step", wraps=env.step) as step,
    ):
        result = evaluate(agent, training_config(), EvaluationConfig(episodes=2, action_selection="greedy"))
    assert resets == [0, 2]
    assert starts == [True, False, True, False]
    assert [call.args[0] for call in step.call_args_list] == [0, 1, 0, 1]
    assert result["return_mean"] == 5
    assert env.closed


@pytest.mark.parametrize("model_type", ("lstm", "gdn2", "mamba3"))
def test_full_games_raw_returns_memory_and_json(tmp_path: Path, model_type: ppo.ModelType) -> None:
    config = replace(
        training_config(),
        model=ppo.LSTMConfig(hidden_size=2)
        if model_type == "lstm"
        else ppo.GDN2Config(hidden_size=8, num_heads=1, head_dim=4, intermediate_size=8)
        if model_type == "gdn2"
        else ppo.Mamba3Config(hidden_size=8, num_layers=1, intermediate_size=8, state_size=4, head_dim=4),
    )
    initial = jax.tree.map(lambda leaf: leaf + 3, ppo.ActorCritic(1, config.model).initial_carry(1))  # leaves: (1, ...)
    env = ScoringEnv()
    starts: list[bool] = []
    memories: list[float] = []

    def action(
        state: TrainState,
        obs: jax.Array,  # (1, 84, 84)
        carry: ppo.RecurrentCarry,  # leaves: (1, ...)
        episode_start: bool,
        key: jax.Array,  # () typed PRNG key
        *,
        greedy: bool,
        sequence_policy: bool,
    ) -> tuple[jax.Array, ppo.RecurrentCarry]:  # () action; carry leaves: (1, ...)
        assert not sequence_policy
        starts.append(episode_start)
        chex.assert_trees_all_equal_shapes_and_dtypes(carry, initial)
        memories.append(float(jax.tree.leaves(carry)[0].reshape(-1)[0]))
        return jnp.asarray(0), jax.tree.map(lambda leaf: leaf + 1, carry)  # (); leaves: (1, ...)

    output = tmp_path / "evaluation.json"
    with (
        patch("rl2.atari_eval.make_evaluation_env", return_value=env),
        patch("rl2.atari_eval._action", side_effect=action),
    ):
        result = evaluate(
            TrainStateAgent(policy_state(), initial),
            config,
            EvaluationConfig(episodes=2),
            baselines=ScoreBaselines("ALE/Pong-v5", -5, 15, "test reference"),
            output_path=output,
        )
    assert json.loads(output.read_text()) == result
    assert [episode["return"] for episode in result["episodes"]] == [3, 7]
    assert result["return_mean"] == 5
    assert result["training_scores"] is None
    assert result["return_std"] == pytest.approx(np.sqrt(8), rel=0, abs=5e-08)
    assert result["return_sem"] == pytest.approx(2, rel=0, abs=5e-08)
    assert result["human_normalized_score_percent"] == 50
    assert result["baselines"]["source"] == "test reference"
    assert result["environment"]["mode"] == 22
    assert starts == [True, False, True, False]
    assert memories == [3, 4, 3, 4]
    for leaf in jax.tree.leaves(initial):
        np.testing.assert_array_equal(leaf, 3)
    assert result["episodes"][0]["terminated"]
    assert result["episodes"][1]["truncated"]
    assert env.closed


def test_rnd_evaluation_carry_policy_interface_and_report(tmp_path: Path) -> None:
    config = replace(
        ppo_rnd.load_config(Path(__file__).resolve().parents[1] / "configs/ppo_rnd_montezuma.yaml"),
        env_id="ALE/Pong-v5",
        atari_preprocessing=True,
        frame_stack=False,
        lstm_hidden_size=2,
    )

    def rnd_policy(
        variables: dict[str, Any], obs: jax.Array, carry: ppo_rnd.LSTMCarry, starts: jax.Array
    ) -> tuple[ppo_rnd.LSTMCarry, jax.Array, jax.Array]:
        chex.assert_shape(obs, (1, 1, 1, 84, 84))
        chex.assert_type(obs, jnp.uint8)
        chex.assert_shape(starts, (1, 1))
        chex.assert_type(starts, jnp.bool_)
        chex.assert_shape(carry, (1, 2))
        chex.assert_type(carry, jnp.float32)
        memory = tuple(jnp.where(starts[0, :, None], 0, c) + 1 for c in carry)
        # The greedy action exposes whether memory advances and resets correctly.
        logits = jax.nn.one_hot(memory[0][:, 0].astype(jnp.int32), 6)[None]
        return memory, logits, jnp.zeros((1, 1, 2))

    state = TrainState.create(apply_fn=rnd_policy, params={}, tx=optax.sgd(0.1))
    env = ScoringEnv()
    output = tmp_path / "rnd-evaluation.json"
    with (
        patch("rl2.atari_eval.make_evaluation_env", return_value=env),
        patch.object(env, "step", wraps=env.step) as step,
    ):
        result = evaluate(
            TrainStateAgent(state, ppo_rnd.initial_carry(1, config.lstm_hidden_size), sequence_policy=True),
            config,
            EvaluationConfig(episodes=2, action_selection="greedy"),
            output_path=output,
        )
    assert [call.args[0] for call in step.call_args_list] == [1, 2, 1, 2]
    assert result["return_mean"] == 5
    assert result["training_config"]["encoder_channels"] == list(config.encoder_channels)
    assert json.loads(output.read_text()) == result
    assert env.closed


@pytest.mark.parametrize(
    "game,expected_mode",
    [("Centipede", 22), ("NameThisGame", 8)],
)
def test_real_ale_nonzero_default_mode(game: str, expected_mode: int) -> None:
    config = replace(training_config(), env_id=f"ALE/{game}-v5", frame_stack=False)
    with make_evaluation_env(config, EvaluationConfig(episodes=1, max_episode_frames=104)) as env:
        obs, _ = env.reset(seed=7)
        assert obs.shape == (1, 84, 84)
        assert env.unwrapped.ale.cloneState().getCurrentMode() == expected_mode


# Cover every frame-skip remainder and both protocols/stack modes without crossing them.
@pytest.mark.parametrize(
    "stacked,protocol,cap",
    [(False, "sticky", 101), (True, "sticky", 102), (True, "legacy_noop", 103), (False, "legacy_noop", 104)],
)
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
    first = evaluate(
        TrainStateAgent(state, ppo.ActorCritic(1, training_config().model).initial_carry(1)),
        training_config(),
        options,
    )
    on_video = Mock()
    second = evaluate(
        TrainStateAgent(state, ppo.ActorCritic(1, training_config().model).initial_carry(1)),
        training_config(),
        options,
        on_video=on_video,
    )
    assert first == second
    on_video.assert_called_once()
    video, fps = on_video.call_args.args  # video: (1, T, C, H, W)
    assert video.shape == (1, second["episodes"][-1]["agent_steps"] + 1, 3, 210, 160)
    assert video.dtype == np.uint8
    assert fps > 0
    assert [ep["seed"] for ep in first["episodes"]] == [10000, 10001]
    for old, new in zip(before, jax.tree.leaves(state), strict=True):
        np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize("greedy", (False, True))
def test_action_selection_and_recurrent_reset(greedy: bool) -> None:
    state = policy_state()
    carry = jax.tree.map(
        lambda leaf: leaf + 5, ppo.ActorCritic(1, training_config().model).initial_carry(1)
    )  # leaves: (1, 2)
    obs = np.zeros((1, 84, 84), dtype=np.uint8)  # (1, 84, 84)
    key = jax.random.key(1)  # () typed PRNG key
    agent = TrainStateAgent(state, carry)
    action, memory = agent.act(obs, agent.initial_carry(), True, key, greedy=greedy)  # (); leaves: (1, 2)
    chex.assert_shape(action, ())
    chex.assert_trees_all_equal_shapes_and_dtypes(memory, carry)
    expected = 5 if greedy else int(jax.random.categorical(key, state.params["logits"]))
    assert int(action) == expected
    for leaf in jax.tree.leaves(memory):
        np.testing.assert_array_equal(leaf, np.ones((1, 2)))


def test_cleanup_on_policy_failure() -> None:
    env = ScoringEnv()
    with (
        patch("rl2.atari_eval.make_evaluation_env", return_value=env),
        patch("rl2.atari_eval._action", side_effect=RuntimeError("policy failure")),
        pytest.raises(RuntimeError, match="policy failure"),
    ):
        evaluate(
            TrainStateAgent(policy_state(), ppo.ActorCritic(1, training_config().model).initial_carry(1)),
            training_config(),
            EvaluationConfig(episodes=1),
        )
    assert env.closed


@pytest.mark.parametrize(
    "overrides",
    (
        {"eval_interval_seconds": -1},
        {"eval_interval_seconds": float("nan")},
        {"eval_interval_seconds": float("inf")},
        {"eval_interval_seconds": "10"},
        {"eval_interval_seconds": 100, "atari_preprocessing": False},
        {"eval_interval_seconds": 100, "observation_size": 42},
        {"eval_interval_seconds": 100, "env_id": "rl2/MultiAtari-v0"},
        {"eval_interval_seconds": 100, "eval_episodes": 0},
        {"eval_interval_seconds": 100, "eval_seed": -1},
        {"eval_interval_seconds": 100, "eval_seed": 2**32 - 1, "eval_episodes": 2},
    ),
)
def test_training_config_rejects_invalid_evaluation(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        replace(training_config(), **overrides)


def test_training_config_accepts_evaluation_seed_boundary() -> None:
    config = replace(training_config(), eval_interval_seconds=100, eval_episodes=2, eval_seed=2**32 - 2)
    assert config.eval_seed + config.eval_episodes == 2**32


def test_training_config_skips_disabled_evaluation_validation() -> None:
    config = replace(
        training_config(),
        eval_interval_seconds=0,
        atari_preprocessing=False,
        observation_size=42,
        env_id="rl2/MultiAtari-v0",
        eval_episodes=0,
        eval_seed=-1,
    )
    assert config.eval_interval_seconds == 0


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
        evaluate(
            TrainStateAgent(policy_state(), ppo.ActorCritic(1, training_config().model).initial_carry(1)),
            training_config(),
            baselines=ScoreBaselines("ALE/Breakout-v5", 0, 1, "test"),
        )


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
            TrainStateAgent(policy_state(), ppo.ActorCritic(1, training_config().model).initial_carry(1)),
            replace(training_config(), env_id=env_id),
            EvaluationConfig(episodes=1),
            output_path=output,
        )
    assert result["return_mean"] == score
    assert result["human_normalized_score_percent"] == pytest.approx(expected)
    assert result["baselines"]["env_id"] == env_id
    assert result["baselines"]["source"] == REFERENCE_SOURCE
    assert json.loads(output.read_text()) == result


def test_evaluation_without_reference_scores_keeps_raw_returns() -> None:
    with patch("rl2.atari_eval.make_evaluation_env", return_value=ScoringEnv()):
        result = evaluate(
            TrainStateAgent(policy_state(), ppo.ActorCritic(1, training_config().model).initial_carry(1)),
            replace(training_config(), env_id="ALE/Adventure-v5"),
            EvaluationConfig(episodes=2),
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
        ppo.log_evaluation(
            policy_state(),
            config,
            writer,
            100,
            5000,
            initial_carry=ppo.ActorCritic(1, training_config().model).initial_carry(1),
        )
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
        result = evaluate(
            TrainStateAgent(state, ppo.ActorCritic(1, training_config().model).initial_carry(1)),
            training_config(),
            options,
        )
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
        quiet_result = evaluate(
            TrainStateAgent(state, ppo.ActorCritic(1, training_config().model).initial_carry(1)),
            training_config(),
            options,
            show_progress=False,
        )
    assert quiet_result == result
    assert capsys.readouterr().out == ""


def test_failed_evaluation_does_not_report_completion(capsys: pytest.CaptureFixture[str]) -> None:
    env = ScoringEnv()
    with (
        patch("rl2.atari_eval.make_evaluation_env", return_value=env),
        patch("rl2.atari_eval._action", side_effect=RuntimeError("policy failure")),
        pytest.raises(RuntimeError, match="policy failure"),
    ):
        evaluate(
            TrainStateAgent(policy_state(), ppo.ActorCritic(1, training_config().model).initial_carry(1)),
            training_config(),
            EvaluationConfig(episodes=1),
        )
    output = capsys.readouterr().out
    assert "completed=0/1" in output
    assert "completed=1/1" not in output
    assert env.closed


@pytest.mark.parametrize("trainer", ["ppo", "ppo_rnd"])
def test_evaluation_logging_records_scores_and_report_without_mutating_state(trainer: str) -> None:
    module = ppo if trainer == "ppo" else ppo_rnd
    config = replace(
        training_config()
        if trainer == "ppo"
        else ppo_rnd.load_config(Path(__file__).resolve().parents[1] / "configs/ppo_rnd_montezuma.yaml"),
        env_id="ALE/Pong-v5",
        eval_episodes=2,
        eval_seed=321,
        video_speed=2.0,
    )
    state = policy_state()
    before = jax.tree.map(np.array, state)
    initial = (  # leaves: (1, ...), backend-specific memory dimensions
        ppo.ActorCritic(1, config.model).initial_carry(1)
        if trainer == "ppo"
        else ppo_rnd.initial_carry(1, config.lstm_hidden_size)
    )
    scores = TrainingScores(101, 300.0, (2.0,) * 100, length_sum=305, recent_lengths=(3,) * 100)
    with (
        patch("rl2.atari_eval.make_evaluation_env", return_value=ScoringEnv()),
        patch("rl2.atari_eval._action", return_value=(jnp.asarray(0), ())) as action,
        patch.object(module, "SummaryWriter") as writer,
    ):
        module.log_evaluation(state, config, writer, 101, 100, training_scores=scores, initial_carry=initial)
    assert action.call_args_list[0].args[2] is initial
    assert action.call_args_list[2].args[2] is initial
    scalars = {call.args[0]: call.args[1:] for call in writer.add_scalar.call_args_list}
    assert scalars["eval/return_mean"] == (5, 100)
    assert "eval/return_median" not in scalars
    assert scalars["eval/total_training_episodes"] == (101, 100)
    assert "eval/training_episodes" not in scalars
    assert scalars["training/total_return_mean"] == (300 / 101, 100)
    assert scalars["training/total_episodes"] == (101, 100)
    assert scalars["training/return_mean_last_100"] == (2.0, 100)
    assert scalars["training/total_episode_length_mean"] == (305 / 101, 100)
    assert scalars["training/episode_length_mean_last_100"] == (3.0, 100)
    assert scalars["training/episodes_last_100"] == (100, 100)
    assert "training/return_mean_100" not in scalars
    assert "training/last_100_episode_count" not in scalars
    assert scalars["eval/human_normalized_score_percent"] == pytest.approx((100 * (5 + 20.7) / (14.6 + 20.7), 100))
    tag, text, step = writer.add_text.call_args.args
    assert tag == "eval/report" and step == 100
    report = json.loads(text.removeprefix("```json\n").removesuffix("\n```"))
    assert "return_median" not in report
    assert report["training_steps"] == 100 and report["training_episodes"] == 101
    assert report["training_scores"] == scores.as_report()
    assert report["baselines"]["source"] == REFERENCE_SOURCE
    assert [episode["seed"] for episode in report["episodes"]] == [321, 322]
    writer.add_video.assert_called_once()
    tag, video, step = writer.add_video.call_args.args  # video: (1, T, C, H, W)
    assert tag == "eval/gameplay" and step == 100
    assert video.shape == (1, 3, 3, 2, 3)
    assert video.dtype == np.uint8
    np.testing.assert_array_equal(video[0, :, 0, 0, 0], [10, 11, 12])
    assert writer.add_video.call_args.kwargs["fps"] == 30
    for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(state), strict=True):
        np.testing.assert_array_equal(old, new)
