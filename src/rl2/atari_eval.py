"""Full-game Atari evaluation for recurrent agents, including PPO and PPO-RND.

Example::

    from rl2.configuration import load_config
    from rl2.ppo_rnd import Config, train
    from rl2.atari_eval import EvaluationConfig, TrainStateAgent, evaluate
    from rl2.ppo_rnd import initial_carry

    config = load_config("configs/ppo_rnd_montezuma.yaml", Config)
    state = train(config)
    result = evaluate(
        TrainStateAgent(state, initial_carry(1, config.lstm_hidden_size)),
        config, EvaluationConfig(), output_path="evaluation.json",
    )
    print(result["return_mean"])

Default: 100 episodes, sticky actions 0.25, 1..30 reset no-ops, action
repeat 4, last-two-frame max pooling, grayscale 84x84, minimal actions,
108,000 emulator frames (30 minutes) per game, no life-loss termination,
and undiscounted, unclipped rewards. Reset no-ops consume the frame budget;
as in Gymnasium's preprocessing, their rewards are excluded from the return.
No automatic FIRE actions: the policy must start/resume games itself.
Frame stacking follows training (one or four frames).
Game mode follows each ROM's default, matching training.
PPO actions are sampled by default; greedy evaluation is an explicit option.

There is no single protocol shared by all Atari papers. ``legacy_noop``
disables sticky actions for comparisons to non-sticky no-op-start results;
it is NOT human-start evaluation. Neither preset reproduces every paper's
horizon, action selection, observation history, or evaluation sample count.
Match those settings, training frame budget, game set, and baseline sources
before comparing. See Machado et al. (2018), https://arxiv.org/abs/1709.06009,
https://ale.farama.org/environments/, and
https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.AtariPreprocessing.

Report per-game means over all requested episodes, including timeouts. For
benchmark aggregates, normalize each game's mean before averaging across
games: 100 * (agent - random) / (human - random). The vendored DQN Zoo
Atari-57 table supplies reference scores by default; explicit baselines override
it for comparisons to papers using other tables. Games outside Atari-57 report
null baselines and a null normalized score. Normalization does not establish
that evaluation protocols match. Report multiple independent training seeds and their spread;
episode SEM below only measures evaluation variability of this fixed policy.
Do not select the best evaluation episode/checkpoint or average raw scores
across different games. This evaluator does not train or update the policy.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import monotonic
from typing import Any, Literal, Protocol

import ale_py
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from numpy.typing import NDArray

from rl2 import ppo_rnd
from rl2.atari_env import AtariPreprocessing
from rl2.atari_scores import REFERENCE_SOURCE, get_reference_scores
from rl2.training_scores import TrainingScores

type EvaluationResult = dict[str, Any]
type TrainingConfig = ppo_rnd.Config


class EvaluationAgent[Carry](Protocol):
    """Recurrent policy for one environment; Carry is an agent-defined pytree."""

    def initial_carry(self) -> Carry:  # agent-defined memory leaf shapes
        """Return fresh recurrent memory for one episode."""
        ...

    def act(
        self,
        obs: jax.Array,  # (*O,) observation dimensions O
        carry: Carry,  # agent-defined memory leaf shapes
        episode_start: bool,
        key: jax.Array,  # () typed PRNG key
        *,
        greedy: bool,
    ) -> tuple[jax.Array, Carry]:  # () action; memory leaf shapes unchanged
        """Return a scalar action ID and advanced recurrent memory."""
        ...


@dataclass(frozen=True)
class TrainStateAgent[Carry]:
    """Adapt a frozen PPO step policy or PPO-RND sequence policy for evaluation."""

    state: TrainState
    carry: Carry  # leaves: (1, ...), backend-specific memory dimensions
    sequence_policy: bool = False

    def initial_carry(self) -> Carry:  # leaves: (1, ...)
        return self.carry  # leaves: (1, ...); JAX arrays are immutable

    def act(
        self,
        obs: jax.Array,  # (*O,)
        carry: Carry,  # leaves: (1, ...)
        episode_start: bool,
        key: jax.Array,  # () typed PRNG key
        *,
        greedy: bool,
    ) -> tuple[jax.Array, Carry]:  # () action; carry leaves: (1, ...)
        return _action(
            self.state, obs, carry, episode_start, key, greedy=greedy, sequence_policy=self.sequence_policy
        )  # () action; carry leaves: (1, ...)


@dataclass(frozen=True)
class EvaluationConfig:
    episodes: int = 100
    seed: int = 10_000
    protocol: Literal["sticky", "legacy_noop"] = "sticky"
    action_selection: Literal["sample", "greedy"] = "sample"
    max_episode_frames: int = 108_000

    def __post_init__(self) -> None:
        if type(self.episodes) is not int or self.episodes < 1:
            raise ValueError("episodes must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - self.episodes:
            raise ValueError("episode seeds must fit in uint32")
        if self.protocol not in ("sticky", "legacy_noop"):
            raise ValueError("protocol must be sticky or legacy_noop")
        if self.action_selection not in ("sample", "greedy"):
            raise ValueError("action_selection must be sample or greedy")
        if type(self.max_episode_frames) is not int or self.max_episode_frames <= 30:
            raise ValueError("max_episode_frames must be an integer greater than the 30 reset no-ops")


@dataclass(frozen=True)
class ScoreBaselines:
    """Reference scores for this game; source should identify table/protocol."""

    env_id: str
    random: float
    human: float
    source: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.random) or not np.isfinite(self.human) or self.human == self.random:
            raise ValueError("baselines must be finite with distinct human and random scores")
        if not self.source.strip():
            raise ValueError("baseline source is required")


def validate_training_config(training: TrainingConfig) -> None:
    """Reject incompatible policy inputs before starting training or evaluation."""
    if not training.atari_preprocessing or training.observation_size not in (None, 84):
        raise ValueError(
            "Standard evaluation requires a policy trained with atari_preprocessing=True "
            "and observation_size=84 (or None). Retrain with those settings; changing "
            "preprocessing only at evaluation changes the policy's inputs/action timing."
        )
    if not training.env_id.startswith("ALE/") or not training.env_id.endswith("-v5"):
        raise ValueError("Use an explicit ALE/<Game>-v5 environment ID")


def make_evaluation_env(
    training: TrainingConfig, evaluation: EvaluationConfig, *, render_mode: Literal["rgb_array"] | None = None
) -> gym.Env:
    """Build an explicitly configured ALE environment without training wrappers."""
    validate_training_config(training)
    gym.register_envs(ale_py)
    # Keep the ROM's default mode, as in training; zero is not valid for every game.
    env = gym.make(
        training.env_id,
        render_mode=render_mode,
        frameskip=1,  # The preprocessing wrapper alone performs action repeat.
        repeat_action_probability=0.25 if evaluation.protocol == "sticky" else 0.0,
        full_action_space=False,
        obs_type="rgb",
        difficulty=0,
        max_num_frames_per_episode=evaluation.max_episode_frames,
        max_episode_steps=-1,  # Disable Gym TimeLimit; ALE counts actual emulator frames.
    )
    try:
        env = AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=False,
        )
        return gym.wrappers.FrameStackObservation(env, stack_size=4 if training.frame_stack else 1)
    except BaseException:
        env.close()
        raise


@partial(jax.jit, static_argnames=("greedy", "sequence_policy"))
def _action[Carry](
    state: TrainState,
    obs: jax.Array,  # (*O,) observation dimensions O
    carry: Carry,  # leaves: (1, ...), backend-specific memory dimensions
    episode_start: bool,
    key: jax.Array,  # () typed PRNG key
    *,
    greedy: bool,
    sequence_policy: bool = False,
) -> tuple[jax.Array, Carry]:  # () action; carry leaves: (1, ...)
    starts = jnp.asarray([episode_start])  # (1,)
    if sequence_policy:
        # PPO-RND exposes only a time-major sequence call, with two value heads.
        carry, logits, _ = state.apply_fn(
            {"params": state.params}, obs[None, None], carry, starts[None]
        )  # carry leaves: (1, ...); logits: (1, 1, A); values: (1, 1, 2)
        logits = logits[0]  # (1, A)
    else:
        carry, policy, _ = state.apply_fn(
            {"params": state.params}, obs[None], carry, starts, method="step"
        )  # carry leaves: (1, ...); policy.logits: (1, A); values: (1,)
        logits = policy.logits  # (1, A), Atari uses a categorical policy
    logits = logits[0]  # (A,)
    action = jnp.argmax(logits) if greedy else jax.random.categorical(key, logits)  # ()
    return action, carry  # () action; carry leaves: (1, ...)


def evaluate[Carry](
    agent: EvaluationAgent[Carry],
    training: TrainingConfig,
    evaluation: EvaluationConfig | None = None,
    *,
    baselines: ScoreBaselines | None = None,
    training_scores: TrainingScores | None = None,
    output_path: str | Path | None = None,
    show_progress: bool = True,
    on_video: Callable[[NDArray[np.uint8], float], None] | None = None,  # video: (1, T, C, H, W)
) -> EvaluationResult:
    """Evaluate frozen parameters with fresh environment, RNG, and recurrent state.

    ``agent.initial_carry()`` resets recurrent memory for each episode.
    ``training`` must describe the agent's actual training setup.
    Returns a JSON-compatible report; optionally writes it to ``output_path``.
    Baselines default to the bundled DQN Zoo Atari-57 table when available.
    Optional ``training_scores`` adds all-training raw episode means and the
    completed episode count; absent history is reported as
    null. Evaluation episodes never contribute to these training metrics.
    Progress prints at startup, after each game, and every 10 seconds during
    a game (after an environment step). Set show_progress=False to silence it.
    ``on_video`` receives (1, T, C, H, W) uint8 RGB frames from the last episode
    and playback FPS, using training.video_max_frames and training.video_speed.
    The frame cap limits recording only; evaluation always finishes the game.
    The caller owns checkpoint selection and independent-training-seed repeats.
    """
    evaluation = evaluation or EvaluationConfig()
    if baselines is not None and baselines.env_id != training.env_id:
        raise ValueError("baseline env_id must match the evaluated game")
    if baselines is None:
        scores = get_reference_scores(training.env_id)
        if scores is not None:
            baselines = ScoreBaselines(training.env_id, scores[0], scores[1], REFERENCE_SOURCE)
    env = make_evaluation_env(training, evaluation, render_mode="rgb_array" if on_video is not None else None)
    episodes: list[dict[str, Any]] = []
    total_return = 0.0
    started = last_report = monotonic() if show_progress else 0.0

    def report_progress(now: float, episode: int, score: float, frames: int) -> None:
        completed = len(episodes)
        elapsed = now - started
        mean = f"{total_return / completed:.1f}" if completed else "n/a"
        eta = f"{elapsed / completed * (evaluation.episodes - completed):.0f}s" if completed else "n/a"
        print(
            f"Evaluation progress: completed={completed}/{evaluation.episodes} "
            f"({100 * completed / evaluation.episodes:.0f}%) "
            f"episode={episode} return={score:.1f} frames={frames} "
            f"mean_return={mean} elapsed={elapsed:.0f}s eta={eta}",
            flush=True,
        )

    try:
        if show_progress:
            report_progress(started, 1, 0.0, 0)
        action_meanings = env.unwrapped.get_action_meanings()
        game_mode = int(env.unwrapped.ale.cloneState().getCurrentMode())
        for index in range(evaluation.episodes):
            seed = evaluation.seed + index
            obs, info = env.reset(seed=seed)  # obs: (*O,)
            record_video = on_video is not None and index == evaluation.episodes - 1
            frames: list[NDArray[np.uint8]] = [env.render().copy()] if record_video else []  # each (H, W, C)
            reset_frames = int(info["episode_frame_number"])
            key = jax.random.key(seed)  # () typed PRNG key
            carry = agent.initial_carry()  # agent-defined memory leaf shapes
            episode_return, agent_steps = 0.0, 0
            terminated, truncated = False, False
            while not (terminated or truncated):
                key, action_key = jax.random.split(key)  # () typed PRNG keys
                action, carry = agent.act(
                    jnp.asarray(obs),  # (*O,)
                    carry,
                    agent_steps == 0,
                    action_key,
                    greedy=evaluation.action_selection == "greedy",
                )  # () action; agent-defined memory leaf shapes
                obs, reward, terminated, truncated, info = env.step(int(action))  # obs: (*O,)
                if record_video and (training.video_max_frames is None or len(frames) < training.video_max_frames):
                    frames.append(env.render().copy())  # (H, W, C)
                episode_return += float(reward)
                agent_steps += 1
                if show_progress and not (terminated or truncated):
                    now = monotonic()
                    if now - last_report >= 10.0:
                        report_progress(now, index + 1, episode_return, int(info["episode_frame_number"]))
                        last_report = now
            total_return += episode_return
            episodes.append(
                {
                    "seed": seed,
                    "return": episode_return,
                    "agent_steps": agent_steps,
                    "emulator_frames": int(info["episode_frame_number"]),
                    "reset_frames": reset_frames,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            if show_progress:
                last_report = monotonic()
                report_progress(last_report, index + 1, episode_return, int(info["episode_frame_number"]))
        if on_video is not None:
            video = np.stack(frames).transpose(0, 3, 1, 2)[None]  # (1, T, C, H, W)
            on_video(video, training.video_speed * env.metadata["render_fps"] / 4)
    finally:
        env.close()

    returns = np.asarray([episode["return"] for episode in episodes], dtype=np.float64)  # (E,) evaluation episodes
    std = float(returns.std(ddof=1)) if len(returns) > 1 else None
    # Normalize tuples in either config so the in-memory report matches saved JSON.
    training_metadata = json.loads(json.dumps(asdict(training), allow_nan=False))
    result: EvaluationResult = {
        "env_id": training.env_id,
        "evaluation": asdict(evaluation),
        "environment": {
            "sticky_action_probability": 0.25 if evaluation.protocol == "sticky" else 0.0,
            "action_repeat": 4,
            "reset_noops": "uniform integers 1..30",
            "max_pool_frames": 2,
            "grayscale": True,
            "screen_size": 84,
            "frame_stack": 4 if training.frame_stack else 1,
            "terminal_on_life_loss": False,
            "reward_clipping": False,
            "discount": 1.0,
            "automatic_fire": False,
            "full_action_space": False,
            "action_meanings": action_meanings,
            "mode": game_mode,
            "difficulty": 0,
        },
        "training_config": training_metadata,  # Configured budget, not proof of frames actually trained.
        "training_scores": training_scores.as_report() if training_scores is not None else None,
        "versions": {name: version(name) for name in ("ale-py", "gymnasium", "jax", "flax", "numpy")},
        "return_mean": float(returns.mean()),
        "return_std": std,
        "return_sem": std / np.sqrt(len(returns)) if std is not None else None,
        "episodes": episodes,
        "baselines": None,
        "human_normalized_score_percent": None,
    }
    if baselines is not None:
        result["baselines"] = asdict(baselines)
        result["human_normalized_score_percent"] = (
            100 * (result["return_mean"] - baselines.random) / (baselines.human - baselines.random)
        )
    if output_path is not None:
        Path(output_path).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
