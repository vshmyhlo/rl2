import asyncio
import random
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import crafter
import pytest
from crafter import constants, engine, objects
from pydantic import ValidationError

from rl2.agent_task import StepResult, ToolCall, ToolCallError
from rl2.lm_envs import crafter as env


def map_rows(observation: str) -> list[str]:
    return observation.split("rows):\n", 1)[1].split("\nVitals:", 1)[0].splitlines()


@pytest.mark.parametrize("position", [(6, 5), (0, 0)], ids=["interior", "world_edge"])
def test_text_view_orientation_bounds_and_state(position: tuple[int, int]) -> None:
    world = engine.World(area=(12, 10), materials=constants.materials, chunk_size=(12, 12))
    for x in range(12):
        for y in range(10):
            world[x, y] = "grass"
    player = objects.Player(world=world, pos=position)
    world.add(player)
    x, y = position
    world[x + 1, y] = "tree"
    world[x, y + 1] = "water"
    world[11, 9] = "diamond"  # Outside both local viewports.
    plant = objects.Plant(world=world, pos=(x + 2, y))
    plant.grown = 301
    world.add(plant)
    world.add(objects.Plant(world=world, pos=(x + 3, y)))
    world.add(objects.Zombie(world=world, pos=(x, y + 2), player=player))
    player.inventory["wood"] = 2
    player.achievements["collect_wood"] = 2
    player.sleeping = True
    game = Mock(spec=crafter.Env)
    game._world = world
    game._player = player
    game._local_view = SimpleNamespace(_grid=(9, 7))
    game._step = 3
    game._length = 10
    observation = env.text_observation(game)
    rows = map_rows(observation)
    assert len(rows) == 7 and all(len(row) == 9 for row in rows)
    assert rows[3][4:8] == "@TPp"
    assert rows[4][4] == "~" and rows[5][4] == "Z"
    assert "d" not in "".join(rows)
    if position == (0, 0):
        assert rows[:3] == ["?????????"] * 3
        assert all(row.startswith("????") for row in rows)
    else:
        assert "?" not in "".join(rows)
    assert "health=9/9" in observation and "wood=2" in observation and "iron=0" in observation
    assert "collect_wood=2" in observation and "collect_diamond" not in observation
    assert "facing: down; sleeping: yes; daylight: 0.00" in observation


def test_upstream_episode_rewards_determinism_and_step_limit() -> None:
    config = env.CrafterConfig(max_episode_steps=2)
    episode = env.CrafterEpisode(config=config, seed=7)
    action = ToolCall(name="do", arguments={})
    with pytest.raises(RuntimeError, match="Reset"):
        episode.step(action)
    initial = episode.reset()
    assert initial.single_tool_call
    assert [tool["function"]["name"] for tool in initial.tools] == episode.env.action_names
    assert episode.reset() == initial
    native = crafter.Env(seed=7, length=2, reward=True)
    native.reset()
    assert env.text_observation(native) == initial.prompt
    for expected_reward in (1.0, 0.0):
        # Collect wood twice: only the first achievement earns a bonus.
        for game in (episode.env, native):
            x, y = game._player.pos  # Scalars from (2,) position.
            target = (x, y + 1)
            _, obj = game._world[target]
            if obj is not None:
                game._world.remove(obj)
            game._world[target] = "tree"
        result = episode.step(action)
        _, reward, done, _ = native.step(native.action_names.index("do"))
        assert result.reward == reward == expected_reward
        assert result.done == bool(done)
        assert result.observation.startswith(env.text_observation(native))
    assert result.done and "step limit reached" in result.observation
    assert "collect_wood=2" in result.observation
    with pytest.raises(RuntimeError, match="Reset"):
        episode.step(action)
    assert episode.reset() == initial


def test_death_preserves_negative_reward() -> None:
    episode = env.CrafterEpisode(config=env.CrafterConfig(), seed=7)
    episode.reset()
    x, y = episode.env._player.pos  # Scalars from (2,) position.
    _, obj = episode.env._world[x, y + 1]
    if obj is not None:
        episode.env._world.remove(obj)
    episode.env._world[x, y + 1] = "lava"
    result = episode.step(ToolCall(name="move_down", arguments={}))
    assert result.done and result.reward == -0.9
    assert "Episode ended: dead" in result.observation
    assert "health=0/9" in result.observation


@pytest.mark.parametrize(
    "action",
    [ToolCall(name="fly", arguments={}), ToolCall(name="noop", arguments={"steps": 2})],
    ids=["unknown_action", "unexpected_arguments"],
)
def test_invalid_actions_do_not_advance_game(action: ToolCall) -> None:
    episode = env.CrafterEpisode(config=env.CrafterConfig(), seed=7)
    episode.done = False
    episode.env = Mock()
    with pytest.raises(ToolCallError):
        episode.step(action)
    episode.env.step.assert_not_called()
    assert not episode.done


def test_seed_stream_restores_from_rng_state() -> None:
    config = env.CrafterConfig()
    rng = random.Random(4)
    stream = env.stream_crafter_tasks(config, rng)
    first = next(stream)
    state = rng.getstate()
    second = next(stream)
    rng.setstate(state)
    assert next(env.stream_crafter_tasks(config, rng)) == second
    assert first.id != second.id and first.config == config


def test_invalid_episode_length() -> None:
    with pytest.raises(ValidationError):
        env.CrafterConfig(max_episode_steps=0)


@pytest.mark.parametrize("failure", [None, RuntimeError, asyncio.CancelledError, TimeoutError])
def test_session_requests_and_cleanup(monkeypatch: pytest.MonkeyPatch, failure: type[BaseException] | None) -> None:
    actor = Mock()
    actor.reset.remote = AsyncMock(return_value="initial")
    actor.step.remote = AsyncMock(return_value="result")
    worker = Mock()
    worker.remote.return_value = actor
    kill = Mock()
    monkeypatch.setattr(env, "CrafterWorker", worker)
    monkeypatch.setattr(env.ray, "kill", kill)
    task = env.CrafterTask(config=env.CrafterConfig(), seed=5)

    async def pending_step(action: ToolCall) -> StepResult:
        return await asyncio.Future()

    async def run() -> None:
        async with task.init_env() as session:
            assert await session.reset(timeout=1) == "initial"
            action = ToolCall(name="noop", arguments={})
            assert await session.step(action, timeout=1) == "result"
            actor.step.remote.assert_awaited_once_with(action)
            with pytest.raises(ToolCallError):
                await session.step(ToolCall(name="noop", arguments={"extra": True}))
            assert actor.step.remote.await_count == 1
            if failure is TimeoutError:
                # Exercise the session's deadline on a pending remote request.
                actor.step.remote.side_effect = pending_step
                await session.step(action, timeout=0)
            elif failure is not None:
                raise failure()

    if failure is None:
        asyncio.run(run())
    else:
        with pytest.raises(failure):
            asyncio.run(run())
    worker.remote.assert_called_once_with(config=task.config, seed=task.seed)
    kill.assert_called_once_with(actor)
