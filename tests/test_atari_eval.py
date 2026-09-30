import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from rl2.atari_eval import EvaluationConfig, ScoreBaselines, _action, evaluate, make_evaluation_env
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


class AtariEvaluationTests(unittest.TestCase):
    def test_full_games_raw_returns_memory_and_json(self) -> None:
        env = ScoringEnv()
        starts: list[bool] = []
        memories: list[float] = []

        def action(
            state: TrainState, obs: Array, carry: LSTMCarry, episode_start: bool, key: jax.Array, *, greedy: bool
        ) -> tuple[jax.Array, LSTMCarry]:
            starts.append(episode_start)
            memories.append(float(carry[0][0, 0]))
            return jnp.asarray(0), (carry[0] + 1, carry[1] + 1)

        with TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation.json"
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
            self.assertEqual(json.loads(output.read_text()), result)
        self.assertEqual([episode["return"] for episode in result["episodes"]], [3, 7])
        self.assertEqual(result["return_mean"], 5)
        self.assertAlmostEqual(result["return_std"], np.sqrt(8))
        self.assertAlmostEqual(result["return_sem"], 2)
        self.assertEqual(result["human_normalized_score_percent"], 50)
        self.assertEqual(starts, [True, False, True, False])
        self.assertEqual(memories, [0, 1, 0, 1])
        self.assertTrue(result["episodes"][0]["terminated"])
        self.assertTrue(result["episodes"][1]["truncated"])
        self.assertTrue(env.closed)

    def test_real_ale_frame_cap_stacking_and_protocol(self) -> None:
        for stacked in (False, True):
            for protocol in ("sticky", "legacy_noop"):
                for cap in (101, 102, 103, 104):
                    with self.subTest(stacked=stacked, protocol=protocol, cap=cap):
                        config = replace(training_config(), frame_stack=stacked)
                        env = make_evaluation_env(
                            config, EvaluationConfig(episodes=1, protocol=protocol, max_episode_frames=cap)
                        )
                        try:
                            obs, info = env.reset(seed=7)
                            self.assertEqual(obs.shape, (4 if stacked else 1, 84, 84))
                            self.assertGreaterEqual(info["episode_frame_number"], 1)
                            self.assertLessEqual(info["episode_frame_number"], 30)
                            self.assertAlmostEqual(
                                env.unwrapped.ale.getFloat("repeat_action_probability"),
                                0.25 if protocol == "sticky" else 0,
                            )
                            for _ in range(cap):
                                obs, _, terminated, truncated, info = env.step(0)
                                if terminated or truncated:
                                    break
                            self.assertTrue(truncated)
                            self.assertFalse(terminated)
                            self.assertEqual(info["episode_frame_number"], cap)
                        finally:
                            env.close()

    def test_real_evaluation_reproducible_and_state_unchanged(self) -> None:
        state = policy_state()
        before = [np.asarray(leaf).copy() for leaf in jax.tree.leaves(state)]
        options = EvaluationConfig(episodes=2, max_episode_frames=104)
        first = evaluate(state, training_config(), options)
        second = evaluate(state, training_config(), options)
        self.assertEqual(first, second)
        self.assertEqual([ep["seed"] for ep in first["episodes"]], [10_000, 10_001])
        for old, new in zip(before, jax.tree.leaves(state), strict=True):
            np.testing.assert_array_equal(old, new)

    def test_action_selection_and_recurrent_reset(self) -> None:
        state = policy_state()
        carry = tuple(c + 5 for c in initial_carry(1, 2))
        obs = np.zeros((1, 84, 84), dtype=np.uint8)
        key = jax.random.key(1)
        for greedy in (False, True):
            action, memory = _action(state, obs, carry, True, key, greedy=greedy)
            expected = 5 if greedy else int(jax.random.categorical(key, state.params["logits"]))
            self.assertEqual(int(action), expected)
            np.testing.assert_array_equal(memory[0], np.ones((1, 2)))

    def test_cleanup_on_policy_failure(self) -> None:
        env = ScoringEnv()
        with (
            patch("rl2.atari_eval.make_evaluation_env", return_value=env),
            patch("rl2.atari_eval._action", side_effect=RuntimeError("policy failure")),
            self.assertRaisesRegex(RuntimeError, "policy failure"),
        ):
            evaluate(policy_state(), training_config(), EvaluationConfig(episodes=1))
        self.assertTrue(env.closed)

    def test_invalid_protocol_and_input_mismatches(self) -> None:
        for kwargs in ({"episodes": 0}, {"seed": -1}, {"protocol": "human_starts"}, {"max_episode_frames": 30}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EvaluationConfig(**kwargs)
        for config in (
            replace(training_config(), atari_preprocessing=False),
            replace(training_config(), observation_size=42),
        ):
            with self.assertRaisesRegex(ValueError, "policy trained"):
                make_evaluation_env(config, EvaluationConfig())
        with self.assertRaisesRegex(ValueError, "env_id"):
            evaluate(policy_state(), training_config(), baselines=ScoreBaselines("ALE/Breakout-v5", 0, 1, "test"))
        with self.assertRaises(ValueError):
            ScoreBaselines("ALE/Pong-v5", 1, 1, "test")


if __name__ == "__main__":
    unittest.main()
