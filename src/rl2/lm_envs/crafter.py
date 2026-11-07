"""Crafter's local tiles and player state as text for tool-calling rollouts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from random import Random
from typing import Annotated, ClassVar, Literal

import crafter
import ray
from crafter import constants, objects
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass as config_dataclass
from ray.actor import ActorHandle

from rl2.agent_task import Environment, ResetResult, StepResult, Task, ToolCall, ToolCallError, ToolSchema

MATERIAL_SYMBOLS = {
    None: "?",
    "water": "~",
    "grass": ".",
    "stone": "#",
    "path": ":",
    "sand": "_",
    "tree": "T",
    "lava": "L",
    "coal": "c",
    "iron": "i",
    "diamond": "d",
    "table": "t",
    "furnace": "f",
}
OBJECT_SYMBOLS = {objects.Player: "@", objects.Cow: "C", objects.Zombie: "Z", objects.Skeleton: "S", objects.Arrow: "a"}
DIRECTIONS = {(-1, 0): "left", (1, 0): "right", (0, -1): "up", (0, 1): "down"}
VITALS = ("health", "food", "drink", "energy")
SYSTEM_PROMPT = """Survive in Crafter and unlock as many achievements as possible.
Use exactly one action tool per turn, then wait for the observation.
Close </think> before writing a complete <tool_call> JSON block outside reasoning.
Every response must include an action until the episode ends.

The map shows only the local view. Up is north; right is east. @ is you.
Legend: ?=outside world, ~=water, .=grass, #=stone, :=path, _=sand, T=tree,
L=lava, c=coal, i=iron, d=diamond, t=table, f=furnace,
C=cow, Z=zombie, S=skeleton, a=arrow, p=growing plant, P=ripe plant.
Objects cover terrain. Tiles remain readable at night and while sleeping.
Move to face a tile, then use do to gather, drink, eat, or attack it.
Movement changes facing even when blocked. Lava is lethal. Sleep restores energy;
while sleeping, actions are ignored until you wake. Each action advances one game step.
Gather wood from trees. Mining stone/coal needs a wood pickaxe, iron needs a stone
pickaxe, and diamond needs an iron pickaxe. Craft tools near the required stations.
Rewards are Crafter's achievement and health-change rewards; repeated achievements
do not award the achievement bonus again. Invalid tool calls do not advance time.
"""


@config_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class CrafterConfig:
    id: Literal["crafter"] = "crafter"
    max_episode_steps: Annotated[int, Field(gt=0)] = 10_000


def action_tools() -> list[ToolSchema]:
    descriptions = {
        "noop": "Wait for one game step.",
        "do": "Interact with the tile you face: gather, drink, eat a ripe plant, or attack a creature.",
        "sleep": "Start sleeping to restore energy; time advances one step per call.",
    }
    descriptions.update(
        {f"move_{direction}": f"Face and attempt to move one tile {direction}." for direction in DIRECTIONS.values()}
    )
    for name, recipe in constants.place.items():
        descriptions[f"place_{name}"] = (
            f"Place {name} on the tile you face. Costs: {recipe['uses']}. Terrain: {', '.join(recipe['where'])}."
        )
    for name, recipe in constants.make.items():
        descriptions[f"make_{name}"] = (
            f"Craft {name}. Costs: {recipe['uses']}. Required nearby stations: {', '.join(recipe['nearby'])}."
        )
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": descriptions[name],
                "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            },
        }
        for name in constants.actions
    ]


def validate_tool_call(action: ToolCall) -> None:
    if action.name not in constants.actions:
        raise ToolCallError(f"Unknown Crafter action {action.name!r}. Choose one of: {', '.join(constants.actions)}.")
    if action.arguments:
        raise ToolCallError(f"{action.name} takes no arguments; use an empty arguments object.")


def text_observation(env: crafter.Env) -> str:
    """Render only the upstream local viewport, without revealing the rest of the world."""
    # Crafter 1.8.3 has no public local-state API at reset; keep private access here.
    player = env._player
    width, height = env._local_view._grid  # Scalars from the (2,) viewport dimensions.
    px, py = player.pos  # Scalars from the (2,) world position, in (x, y) order.
    rows: list[str] = []
    for y in range(py - height // 2, py - height // 2 + height):
        row: list[str] = []
        for x in range(px - width // 2, px - width // 2 + width):
            material, obj = env._world[x, y]
            if isinstance(obj, objects.Plant):
                symbol = "P" if obj.ripe else "p"
            elif obj is not None:
                symbol = OBJECT_SYMBOLS[type(obj)]
            else:
                symbol = MATERIAL_SYMBOLS[material]
            row.append(symbol)
        rows.append("".join(row))
    vitals = ", ".join(f"{name}={player.inventory[name]}/{constants.items[name]['max']}" for name in VITALS)
    inventory = ", ".join(f"{name}={amount}" for name, amount in player.inventory.items() if name not in VITALS)
    achievements = ", ".join(f"{name}={count}" for name, count in player.achievements.items() if count) or "none"
    return (
        f"Step: {env._step}/{env._length}\n"
        f"Position: ({px}, {py}); facing: {DIRECTIONS[tuple(player.facing)]}; "
        f"sleeping: {'yes' if player.sleeping else 'no'}; daylight: {env._world.daylight:.2f}\n"
        f"Local map ({width} columns x {height} rows):\n" + "\n".join(rows) + "\n"
        f"Vitals: {vitals}\nInventory: {inventory}\nAchievements: {achievements}"
    )


class CrafterEpisode:
    """Synchronous game execution, confined to one Ray worker per rollout."""

    def __init__(self, config: CrafterConfig, seed: int) -> None:
        self.config = config
        self.seed = seed
        self.done = True

    def reset(self) -> ResetResult:
        # Recreate to reset upstream's episode counter as well as its world seed.
        self.env = crafter.Env(seed=self.seed, length=self.config.max_episode_steps, reward=True)
        self.env.reset()
        self.done = False
        return ResetResult(
            tools=action_tools(), system_prompt=SYSTEM_PROMPT, prompt=text_observation(self.env), single_tool_call=True
        )

    def step(self, action: ToolCall) -> StepResult:
        if self.done:
            raise RuntimeError("Reset the episode before calling a tool")
        validate_tool_call(action)
        # Keep upstream RGB rendering: its night noise consumes the game's RNG.
        _, reward, done, _ = self.env.step(self.env.action_names.index(action.name))
        self.done = bool(done)
        observation = text_observation(self.env)
        if self.done:
            observation += "\nEpisode ended: " + ("dead" if self.env._player.health <= 0 else "step limit reached")
        return StepResult(observation=observation, reward=float(reward), done=self.done)


CrafterWorker = ray.remote(CrafterEpisode)


@dataclass
class CrafterSession(Environment):
    actor: ActorHandle

    async def reset(self, *, timeout: float | None = None) -> ResetResult:
        async with asyncio.timeout(timeout):
            return await self.actor.reset.remote()

    async def step(self, action: ToolCall, *, timeout: float | None = None) -> StepResult:
        validate_tool_call(action)
        async with asyncio.timeout(timeout):
            return await self.actor.step.remote(action)


@dataclass(frozen=True, slots=True)
class CrafterTask(Task):
    config: CrafterConfig
    seed: int
    name: ClassVar[str] = "crafter"

    @property
    def id(self) -> str:
        return f"crafter:{self.seed}"

    @asynccontextmanager
    async def init_env(self) -> AsyncGenerator[CrafterSession]:
        async with AsyncExitStack() as resources:
            actor = CrafterWorker.remote(config=self.config, seed=self.seed)
            resources.push_async_callback(asyncio.to_thread, ray.kill, actor)
            yield CrafterSession(actor=actor)


def stream_crafter_tasks(config: CrafterConfig, rng: Random) -> Generator[CrafterTask]:
    while True:
        yield CrafterTask(config=config, seed=rng.getrandbits(32))
