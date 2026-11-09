import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from rl2 import ppo, train_mappo
from rl2.actions import ContinuousSpace
from rl2.envs.base import EnvStep
from rl2.envs.gymnax_env import GymnaxEnv


def policy(
    variables: dict[str, jax.Array],
    observations: jax.Array,  # (B,C,H,W)
    carry: ppo.RecurrentCarry,  # leaves: (B,H)
    starts: jax.Array,  # (B,)
    *,
    method: str,
) -> ppo.ActorCriticOutput:
    return ppo.ActorCriticOutput(
        carry, ppo.CategoricalPolicy(jnp.zeros((observations.shape[0], 3))), jnp.zeros(observations.shape[0])
    )


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
        ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml", ppo.Config),
        video_max_frames=frame_limit,
    )
    env = GymnaxEnv(ppo.GymnaxConfig(id=config.env.id, max_steps=2))
    state = TrainState.create(apply_fn=policy, params={}, tx=optax.sgd(0.1))
    carry = ppo.ActorCritic(
        ppo.DiscreteSpace(3),
        ppo.ModelConfig(encoder=ppo.Conv2DEncoderConfig(), recurrent=ppo.LSTMConfig(hidden_size=1)),
    ).initial_carry(1)  # leaves: (1,1)
    with patch.object(ppo, "GymnaxEnv", return_value=env):
        result = ppo.play_episode(
            state, config, jax.random.key(0), carry, record=True, stop_at_frame_limit=stop, env=env
        )
    assert result.episode_return == 0 and result.length == expected_length and len(result.frames) == expected_frames
    assert all(frame.shape == (160, 160, 3) and frame.dtype == np.uint8 for frame in result.frames)


@pytest.mark.parametrize("record", [True, False])
def test_episode_renders_state_before_autoreset(record: bool) -> None:
    config = ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml", ppo.Config)
    observation = jnp.zeros((4, 2, 2), jnp.uint8)  # (C, H, W)
    initial, final, reset = {"frame": 10}, {"frame": 20}, {"frame": 30}
    env = Mock()
    env.params = {}
    env.reset.return_value = observation, initial
    env.step.return_value = observation, reset, jnp.asarray(1.0), jnp.asarray(True), {"final_state": final}
    env.render.side_effect = [np.full((3, 5, 3), pixel, np.uint8) for pixel in (10, 20)]  # each (P, Q, 3)
    state = TrainState.create(apply_fn=policy, params={}, tx=optax.sgd(0.1))
    carry = ppo.ActorCritic(
        ppo.DiscreteSpace(3),
        ppo.ModelConfig(encoder=ppo.Conv2DEncoderConfig(), recurrent=ppo.LSTMConfig(hidden_size=1)),
    ).initial_carry(1)  # leaves: (1,1)
    result = ppo.play_episode(state, config, jax.random.key(0), carry, record=record, env=env)
    assert (result.episode_return, result.length) == (1.0, 1)
    if record:
        assert [call.args[0] for call in env.render.call_args_list] == [initial, final]
        assert len(result.frames) == 2 and all(frame.shape == (3, 5, 3) for frame in result.frames)
        np.testing.assert_array_equal(result.frames[-1], 20)
    else:
        assert result.frames == []
        env.render.assert_not_called()


def test_beta_playback_reuses_compiled_sampling_across_steps_and_episodes() -> None:
    config = ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo/kinetix.yaml", ppo.Config)
    space = ContinuousSpace((2,), (-1.0, -1.0), (1.0, 1.0))
    observation = jnp.zeros(1)  # (F=1,)
    carry = ((jnp.zeros((1, 1)), jnp.zeros((1, 1))),)  # leaves: (B=1,H=1)

    def beta_policy(
        variables: dict[str, dict[str, jax.Array]],
        observations: jax.Array,  # (B,F)
        carry: ppo.RecurrentCarry,  # leaves: (B,H)
        starts: jax.Array,  # (B,)
        *,
        method: str,
    ) -> ppo.ActorCriticOutput:
        alpha = variables["params"]["alpha"]  # (B,D=2)
        return ppo.ActorCriticOutput(carry, ppo.BetaPolicy(alpha, jnp.full_like(alpha, 2.0), space), jnp.zeros(1))

    state = TrainState.create(apply_fn=beta_policy, params={"alpha": jnp.full((1, 2), 2.0)}, tx=optax.sgd(0.1))

    class Env:
        def __init__(self) -> None:
            self.params = {}

        def reset(self, key: jax.Array, params: dict) -> tuple[jax.Array, dict[str, jax.Array]]:  # key: ()
            return observation, {"step": jnp.asarray(0), "action": jnp.zeros(2)}  # (F,); (); (D,)

        def step(
            self,
            key: jax.Array,  # ()
            state: dict[str, jax.Array],  # step: (); action: (D,)
            action: jax.Array,  # (D,)
            params: dict,
        ) -> EnvStep:
            final = {"step": state["step"] + 1, "action": action}  # (); (D,)
            return EnvStep(
                observation=observation,  # (F,)
                state=final,  # (); (D,)
                reward=jnp.asarray(1.0),  # ()
                done=final["step"] == 2,  # ()
                info={
                    "final_state": final,  # (); (D,)
                    "final_observation": observation,  # (F,)
                    "terminated": final["step"] == 2,  # ()
                    "truncated": jnp.asarray(False),  # ()
                },
            )

        def render(self, state: dict[str, jax.Array]) -> np.ndarray:  # state: (); (D,)
            return np.zeros((2, 2, 3), np.uint8)  # (H, W, 3)

    env = Env()
    with (
        patch.object(ppo.BetaPolicy, "sample", autospec=True, side_effect=ppo.BetaPolicy.sample) as sample,
        patch.object(env, "render", wraps=env.render) as render,
        patch.object(env, "step", wraps=env.step) as step,
    ):
        for _ in range(2):
            score, length, frames = ppo.play_episode(state, config, jax.random.key(0), carry, record=True, env=env)
            assert (score, length, len(frames)) == (2.0, 2, 3)
            state = state.replace(params={"alpha": jnp.full((1, 2), 10.0)})
    # Sampling should execute in the cached graph, including after parameter changes.
    assert sample.call_count == 1
    assert step.call_count == 1
    actions = np.stack([render.call_args_list[i].args[0]["action"] for i in (1, 2, 4, 5)])  # (T=4, D=2)
    assert actions.shape == (4, 2)
    assert np.all((actions > -1) & (actions < 1))
    assert not np.array_equal(actions[0], actions[1])  # Advancing RNG keys changes samples.
    assert not np.array_equal(actions[:2], actions[2:])  # Updated parameters reach the cached graph.


@pytest.mark.parametrize(
    "trainer,config_path,actor_count",
    [(ppo, "configs/ppo.yaml", 1), (train_mappo, "configs/mappo/smax.yaml", 2)],
    ids=["ppo", "mappo"],
)
def test_evaluation_logs_raw_scores_and_independent_seeds(
    trainer: ModuleType, config_path: str, actor_count: int
) -> None:
    config = replace(
        trainer.load_config(Path(__file__).resolve().parents[1] / config_path, trainer.Config),
        eval_episodes=2,
        video_fps=12.5,
        video_interval=0,
    )
    writer = Mock()
    state = Mock(spec=TrainState)
    carry = ((jnp.zeros((actor_count, 1)), jnp.zeros((actor_count, 1))),)  # leaves: (B, H=1)
    frame = np.zeros((2, 2, 3), np.uint8)  # (H,W,3)
    episodes = [
        ppo.EpisodeResult(episode_return=2.0, length=3, frames=[]),
        ppo.EpisodeResult(episode_return=4.0, length=5, frames=[] if trainer is ppo else [frame]),
    ]
    if trainer is ppo:
        episodes.append(ppo.EpisodeResult(episode_return=4.0, length=5, frames=[frame]))
    with (
        patch.object(trainer, "make_env") as make_env,
        patch.object(trainer, "play_episode", side_effect=episodes) as play,
    ):
        trainer.log_evaluation(state, config, writer, 1, 128, initial_carry=carry)
    make_env.assert_called_once_with(config)
    for index, call in enumerate(play.call_args_list):
        np.testing.assert_array_equal(
            jax.random.key_data(call.args[2]),
            jax.random.key_data(jax.random.fold_in(jax.random.key(config.eval_seed), min(index, 1))),
        )
        assert call.args[3] is carry
        assert call.kwargs["env"] is make_env.return_value
        assert call.kwargs["record"] == (index == len(episodes) - 1)
    scalars = writer.log_metrics.call_args.args[0]
    assert scalars["eval/return_mean"] == 3
    assert scalars["eval/return_std"] == pytest.approx(2**0.5)
    assert scalars["eval/return_sem"] == pytest.approx(1.0)
    assert scalars["eval/training_episodes_total"] == 1
    assert not any(tag.startswith("training/") for tag in scalars)
    assert not any("last_100" in tag for tag in scalars)
    assert "eval/human_normalized_score_percent" not in scalars
    report = json.loads(writer.log_text.call_args.args[0])
    assert report["env_id"] == (config.env.id)
    assert report["episodes"] == [{"return": 2, "length": 3}, {"return": 4, "length": 5}]
    assert "training_scores" not in report
    assert len(writer.log_video.call_args.args[0]) == 1
    assert writer.log_video.call_args.args[0][0].shape == (2, 2, 3)
    assert writer.log_video.call_args.args[1] == "eval/gameplay/128.gif"
    assert writer.log_text.call_args.args[1] == "eval/report/128.json"
    assert writer.log_metrics.call_args.args[1] == 128
    assert writer.log_video.call_args.kwargs["fps"] == 12.5


@pytest.mark.parametrize(
    "returns,best_index",
    [([-5.0, -1.0, -3.0], 1), ([0.0, 0.0], 0), ([2.0], 0)],
    ids=["negative-returns", "first-tied-episode", "single-episode"],
)
def test_evaluation_records_highest_return_episode(returns: list[float], best_index: int) -> None:
    config = replace(
        ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml", ppo.Config),
        eval_episodes=len(returns),
        video_interval=0,
        video_max_frames=1,
    )
    writer = Mock()
    state = Mock(spec=TrainState)
    env = Mock()
    carry = ((jnp.zeros((1, 1)), jnp.zeros((1, 1))),)  # leaves: (B=1, H=1)
    frame = np.zeros((2, 2, 3), np.uint8)  # (H, W, 3)
    episodes = [ppo.EpisodeResult(episode_return=score, length=3, frames=[]) for score in returns]
    # The capped replay's partial return must not enter evaluation statistics.
    replay = ppo.EpisodeResult(episode_return=100.0, length=0, frames=[frame])
    with patch.object(ppo, "play_episode", side_effect=[*episodes, replay]) as play:
        ppo.log_evaluation(state, config, writer, 1, 128, initial_carry=carry, env=env)
    assert play.call_count == len(returns) + 1
    assert all(not call.kwargs["record"] for call in play.call_args_list[:-1])
    recorded = play.call_args
    assert recorded.args[0] is state
    assert recorded.args[1] is config
    assert recorded.args[3] is carry
    assert recorded.kwargs == {"record": True, "stop_at_frame_limit": True, "env": env}
    np.testing.assert_array_equal(
        jax.random.key_data(recorded.args[2]),
        jax.random.key_data(play.call_args_list[best_index].args[2]),
    )
    writer.log_video.assert_called_once()
    assert writer.log_video.call_args.args[0] is replay.frames
    report = json.loads(writer.log_text.call_args.args[0])
    assert report["episodes"] == [{"return": score, "length": 3} for score in returns]
    assert writer.log_metrics.call_args.args[0]["eval/return_mean"] == pytest.approx(sum(returns) / len(returns))


def test_log_video_wiring() -> None:
    config = ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo/xminigrid.yaml", ppo.Config)
    writer = Mock()
    carry = ppo.ActorCritic(
        ppo.DiscreteSpace(3),
        ppo.ModelConfig(encoder=ppo.Conv2DEncoderConfig(), recurrent=ppo.LSTMConfig(hidden_size=1)),
    ).initial_carry(1)  # leaves: (1,1)
    with (
        patch.object(ppo, "make_env"),
        patch.object(ppo, "play_episode", return_value=(0.0, 1, [np.zeros((2, 2, 3), np.uint8)])) as play,
    ):
        ppo.log_video(Mock(), config, writer, 2, 128, initial_carry=carry)
    assert play.call_args.kwargs["record"]
    assert play.call_args.kwargs["stop_at_frame_limit"]
    assert writer.log_video.call_args.args[1] == "gameplay/128.gif"
    assert writer.log_video.call_args.kwargs["fps"] == 10.0


@pytest.mark.parametrize(
    "trainer,config_path,actor_count",
    [(ppo, "configs/ppo.yaml", 1), (train_mappo, "configs/mappo/smax.yaml", 2)],
    ids=["ppo", "mappo"],
)
def test_playback_logging_reports_progress(
    trainer: ModuleType,
    config_path: str,
    actor_count: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = replace(
        trainer.load_config(Path(__file__).resolve().parents[1] / config_path, trainer.Config), eval_episodes=1
    )
    writer = Mock()
    carry = ((jnp.zeros((actor_count, 1)), jnp.zeros((actor_count, 1))),)  # leaves: (B, H=1)
    frame = np.zeros((2, 2, 3), np.uint8)  # (H, W, C)
    clock = 10.0

    def advance(seconds: float, *args: object, **kwargs: object) -> None:
        nonlocal clock
        clock += seconds

    def play_episode(*args: object, **kwargs: object) -> ppo.EpisodeResult:
        advance(3.0)
        return ppo.EpisodeResult(episode_return=2.0, length=1, frames=[frame])

    writer.log_video.side_effect = partial(advance, 2.0)
    writer.log_text.side_effect = partial(advance, 1.0)
    with (
        patch.object(trainer, "make_env"),
        patch.object(trainer, "play_episode", side_effect=play_episode),
        patch.object(trainer, "monotonic", side_effect=lambda: clock),
    ):
        trainer.log_video(Mock(), config, writer, 2, 128, initial_carry=carry)
        trainer.log_evaluation(Mock(), config, writer, 2, 128, initial_carry=carry)
    evaluation_seconds = 9.0 if trainer is ppo else 6.0
    assert capsys.readouterr().out.splitlines() == [
        "Recording gameplay at step 128 after 2 training episodes...",
        "Recorded 1 gameplay frames after 2 training episodes in 5.0s",
        "Evaluating 1 games at step 128...",
        f"Evaluation: return=2.0 over 1 games in {evaluation_seconds:.1f}s",
    ]
    writer.log_metrics.assert_any_call({"actor/video_seconds": 5.0}, 128)
    assert writer.log_metrics.call_args.args[0]["eval/evaluation_seconds"] == evaluation_seconds
    assert writer.log_metrics.call_args.args[1] == 128
    assert [call.args[1] for call in writer.log_video.call_args_list] == ["gameplay/128.gif", "eval/gameplay/128.gif"]
    assert all(
        len(call.args[0]) == 1 and call.args[0][0].shape == (2, 2, 3) for call in writer.log_video.call_args_list
    )


@pytest.mark.parametrize(
    "trainer,config_path,actor_count",
    [(ppo, "configs/ppo.yaml", 1), (train_mappo, "configs/mappo/smax.yaml", 2)],
    ids=["ppo", "mappo"],
)
def test_single_evaluation_omits_unavailable_uncertainty(
    trainer: ModuleType, config_path: str, actor_count: int
) -> None:
    config = replace(
        trainer.load_config(Path(__file__).resolve().parents[1] / config_path, trainer.Config), eval_episodes=1
    )
    writer = Mock()
    carry = ((jnp.zeros((actor_count, 1)), jnp.zeros((actor_count, 1))),)  # leaves: (B, H=1)
    with (
        patch.object(trainer, "make_env"),
        patch.object(trainer, "play_episode", return_value=(2.0, 3, [])),
    ):
        trainer.log_evaluation(Mock(), config, writer, 0, 0, initial_carry=carry)
    tags = set(writer.log_metrics.call_args.args[0])
    assert "eval/return_mean" in tags
    assert not {"eval/return_std", "eval/return_sem"} & tags
    report = json.loads(writer.log_text.call_args.args[0])
    assert report["return_mean"] == 2.0
    assert report["return_std"] is None
    assert report["return_sem"] is None
    assert "training_scores" not in report
    writer.log_video.assert_not_called()


def test_kinetix_video_and_evaluation_use_configured_environment() -> None:
    config = replace(
        ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo/kinetix.yaml", ppo.Config), eval_episodes=1
    )
    carry = ppo.ActorCritic(
        ppo.DiscreteSpace(11),
        ppo.ModelConfig(encoder=ppo.Conv2DEncoderConfig(), recurrent=ppo.LSTMConfig(hidden_size=1)),
    ).initial_carry(1)  # leaves: (1,1)
    writer = Mock()
    frame = np.zeros((2, 2, 3), np.uint8)  # (H,W,3)
    with (
        patch.object(ppo, "KinetixEnv") as make_env,
        patch.object(ppo, "play_episode", return_value=(1.0, 2, [frame])) as play,
    ):
        ppo.log_video(Mock(), config, writer, 0, 0, initial_carry=carry)
        ppo.log_evaluation(Mock(), config, writer, 0, 0, initial_carry=carry)
    assert make_env.call_count == 2
    assert all(call.args == (config.env,) for call in make_env.call_args_list)
    assert all(call.kwargs["env"] is make_env.return_value for call in play.call_args_list)
    report = json.loads(writer.log_text.call_args.args[0])
    assert report["env_id"] == "Kinetix" and report["return_mean"] == 1.0
