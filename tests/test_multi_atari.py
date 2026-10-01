import gymnasium as gym
import numpy as np
import pytest

from rl2.multi_atari import MultiAtariEnv


def test_shared_spaces_and_action_meanings() -> None:
    meanings: list[list[str]] = []
    for env_id in ("ALE/Pong-v5", "ALE/Breakout-v5"):
        with MultiAtariEnv([env_id]) as env:
            observation, info = env.reset(seed=42)
            assert observation.shape == (210, 160, 3)
            assert env.observation_space.contains(observation)
            assert env.action_space == gym.spaces.Discrete(18)
            assert info["env_id"] == env_id
            assert env.env is not None
            meanings.append(env.env.unwrapped.get_action_meanings())
            observation, _, _, _, info = env.step(17)
            assert env.observation_space.contains(observation)
            assert info["env_id"] == env_id
    assert meanings[0] == meanings[1]


def test_seed_reproduces_game_selection_and_observations() -> None:
    games = ["ALE/Pong-v5", "ALE/Breakout-v5"]
    with MultiAtariEnv(games) as first, MultiAtariEnv(games) as second:
        selected: set[str] = set()
        for episode in range(8):
            seed = 42 if episode == 0 else None
            obs_a, info_a = first.reset(seed=seed)
            obs_b, info_b = second.reset(seed=seed)
            assert info_a["env_id"] == info_b["env_id"]
            selected.add(info_a["env_id"])
            np.testing.assert_array_equal(obs_a, obs_b)
            obs_a, *transition_a = first.step(1)
            obs_b, *transition_b = second.step(1)
            np.testing.assert_array_equal(obs_a, obs_b)
            assert transition_a == transition_b
        assert selected == set(games)


def test_requires_reset_and_closes_emulator() -> None:
    env = MultiAtariEnv(["ALE/Pong-v5"])
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)
    try:
        env.reset(seed=42)
        assert env.env is not None
        env.env = gym.wrappers.TimeLimit(env.env, max_episode_steps=1)
        env.reset(seed=42)
        _, _, _, truncated, _ = env.step(0)
        assert truncated
        with pytest.raises(gym.error.ResetNeeded):
            env.step(0)
    finally:
        env.close()
    assert env.env is None


@pytest.mark.parametrize(
    ("preprocessing", "stacked", "size", "shape"),
    [
        (True, True, None, (4, 84, 84)),
        (True, False, 64, (1, 64, 64)),
        (False, True, None, (4, 210, 160, 3)),
        (False, False, 64, (1, 64, 64, 3)),
    ],
)
def test_ppo_wrappers_follow_the_active_game(
    preprocessing: bool, stacked: bool, size: int | None, shape: tuple[int, ...]
) -> None:
    from rl2.multi_atari import ENV_ID
    from rl2.ppo import AtariPreprocessing, make_env

    with make_env(
        ENV_ID,
        atari_preprocessing=preprocessing,
        frame_stack=stacked,
        observation_size=size,
        render_mode="rgb_array",
    ) as env:
        raw = env.unwrapped
        assert isinstance(raw, MultiAtariEnv)
        assert len(raw.env_ids) == 57
        assert raw.observation_space.shape == (210, 160, 3)
        assert env.action_space == gym.spaces.Discrete(18)
        selected: set[str] = set()
        for episode in range(4):
            obs, info = env.reset(seed=42 if episode == 0 else None)
            assert obs.shape == shape
            assert env.observation_space.contains(obs)
            assert raw.env is not None
            assert raw.ale is raw.env.unwrapped.ale
            assert info["env_id"] == raw.env_id
            selected.add(info["env_id"])
            for frame in obs:
                np.testing.assert_array_equal(frame, obs[0])
            if preprocessing:
                wrapper = env
                while not isinstance(wrapper, AtariPreprocessing):
                    wrapper = wrapper.env
                assert wrapper.ale is raw.ale
            assert env.render().shape == (210, 160, 3)
            frame_number = raw.ale.getFrameNumber()
            obs, _, terminated, truncated, step_info = env.step(17)
            assert env.observation_space.contains(obs)
            assert step_info["env_id"] == info["env_id"]
            if not (terminated or truncated):
                assert raw.ale.getFrameNumber() - frame_number == (4 if preprocessing else 1)
        assert len(selected) > 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_vector_env_can_reset_individual_games(asynchronous: bool) -> None:
    from functools import partial

    from rl2.multi_atari import ENV_ID
    from rl2.ppo import make_env

    vector_cls = gym.vector.AsyncVectorEnv if asynchronous else gym.vector.SyncVectorEnv
    options = {"context": "spawn"} if asynchronous else {}
    envs = vector_cls(
        [partial(make_env, ENV_ID, atari_preprocessing=True, frame_stack=True)] * 2,
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
        **options,
    )
    try:
        obs, info = envs.reset(seed=42)
        assert obs.shape == (2, 4, 84, 84)
        second_game = info["env_id"][1]
        second_obs = obs[1].copy()
        obs, _ = envs.reset(options={"reset_mask": np.array([True, False])})
        np.testing.assert_array_equal(obs[1], second_obs)
        obs, _, _, _, info = envs.step(np.array([17, 17]))
        assert obs.shape == (2, 4, 84, 84)
        assert info["env_id"][1] == second_game
    finally:
        envs.close()


def test_ppo_config_selects_the_registered_mixture() -> None:
    from pathlib import Path

    from rl2.multi_atari import ENV_ID
    from rl2.ppo import load_config, make_env

    config = load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml")
    assert config.env_id == ENV_ID
    assert config.eval_every_minutes == 0
    with make_env(
        config.env_id,
        frame_stack=config.frame_stack,
        atari_preprocessing=config.atari_preprocessing,
        observation_size=config.observation_size,
    ) as env:
        assert isinstance(env.unwrapped, MultiAtariEnv)
        assert env.observation_space.shape == (4, 84, 84)


def test_skiing_switch_keeps_shared_actions_and_preprocessing() -> None:
    from rl2.ppo import AtariPreprocessing

    raw = MultiAtariEnv(["ALE/Pong-v5", "ALE/Skiing-v5"])
    with gym.wrappers.FrameStackObservation(AtariPreprocessing(raw), stack_size=4) as env:
        # These seeds select Skiing, Pong, then Skiing again.
        for seed, game in ((0, "Skiing"), (1, "Pong"), (0, "Skiing")):
            obs, info = env.reset(seed=seed)
            assert info["env_id"] == f"ALE/{game}-v5"
            assert env.action_space == gym.spaces.Discrete(18)
            assert obs.shape == (4, 84, 84)
            obs, _, _, _, _ = env.step(17)
            assert env.observation_space.contains(obs)


@pytest.mark.parametrize(
    ("game", "local_actions"),
    [
        ("Skiing", (0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8)),
        ("Pong", tuple(range(18))),
    ],
)
def test_shared_actions_match_native_transitions(game: str, local_actions: tuple[int, ...]) -> None:
    from rl2.multi_atari import FullAtariActionSpace

    kwargs = {"frameskip": 1, "full_action_space": True, "repeat_action_probability": 0.0}
    with (
        FullAtariActionSpace(gym.make(f"ALE/{game}-v5", **kwargs)) as shared,
        gym.make(f"ALE/{game}-v5", **kwargs) as native,
    ):
        shared.reset(seed=123)
        native.reset(seed=123)
        for action, local_action in enumerate(local_actions):
            assert shared.action(np.int64(action)) == local_action
            actual_obs, *actual_transition = shared.step(action)
            expected_obs, *expected_transition = native.step(local_action)
            np.testing.assert_array_equal(actual_obs, expected_obs)
            assert actual_transition == expected_transition
        for invalid_action in (-1, 18):
            with pytest.raises(gym.error.InvalidAction):
                shared.step(invalid_action)


def test_every_default_game_accepts_the_shared_spaces() -> None:
    with MultiAtariEnv() as env:
        # Find a deterministic reset seed selecting each game, covering every
        # switch without depending on a short random sequence hitting Skiing.
        seeds: dict[int, int] = {}
        seed = 0
        while len(seeds) < len(env.env_ids):
            index = int(np.random.default_rng(seed).integers(len(env.env_ids)))
            seeds.setdefault(index, seed)
            seed += 1
        for index, seed in sorted(seeds.items()):
            obs, info = env.reset(seed=seed)
            assert info["env_id"] == env.env_ids[index]
            assert env.observation_space.contains(obs)
            assert env.action_space == gym.spaces.Discrete(18)
            for action in range(18):
                obs, _, terminated, truncated, _ = env.step(action)
                assert env.observation_space.contains(obs)
                if terminated or truncated:
                    break
