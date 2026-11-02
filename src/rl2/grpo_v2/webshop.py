"""Gymnasium adapter loaded in WebShop Ray processes."""

import json
import random
from pathlib import Path
from typing import Any, NamedTuple

import gymnasium as gym
from web_agent_site.engine import engine
from web_agent_site.envs import WebAgentTextEnv


class WebShopStep(NamedTuple):
    observation: str
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class WebShop(gym.Env[str, str]):
    def __init__(self, root: str) -> None:
        data = Path(root) / "data"
        engine.DEFAULT_ATTR_PATH = str(data / "items_ins_v2.json")
        engine.HUMAN_ATTR_PATH = str(data / "items_human_ins.json")
        # Product prices are sampled before upstream fixes the goal-shuffle seed.
        random.seed(233)
        self.env = WebAgentTextEnv(
            observation_mode="text",
            file_path=str(data / "items_shuffle.json"),
            num_products=None,
            human_goals=True,
            limit_goals=-1,
        )
        self.done = True

    def task_ids(self) -> list[str]:
        # Upstream reserves 500 test and 1,000 development goals after shuffling.
        return [str(index) for index in range(1500, len(self.env.server.goals))]

    def observe(self, observation: str) -> str:
        return observation + "\nAvailable actions: " + json.dumps(self.env.get_available_actions())

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
        super().reset(seed=seed)
        if options is None or "task_id" not in options:
            raise ValueError("WebShop reset requires options['task_id']")
        index = int(options["task_id"])
        if not 1500 <= index < len(self.env.server.goals):
            raise ValueError("WebShop task is outside the training split")
        observation, info = self.env.reset(session=index)
        self.done = False
        return self.observe(observation), info or {}

    def step(self, action: str) -> WebShopStep:
        if self.done:
            raise RuntimeError("Reset the episode before taking an action")
        observation, reward, done, _ = self.env.step(action)
        self.done = bool(done)
        return WebShopStep(
            observation=self.observe(observation), reward=float(reward), terminated=self.done, truncated=False, info={}
        )

    def close(self) -> None:
        self.env.close()
