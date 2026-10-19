from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from common import main
from web_agent_site.engine import engine
from web_agent_site.envs import WebAgentTextEnv


class WebShop:
    def __init__(self, config: dict[str, Any]) -> None:
        data = Path(config["root"]) / "data"
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
        self.reward = 0.0

    def task_ids(self) -> list[str]:
        # Upstream's fixed goal shuffle reserves 500 test and 1,000 development goals.
        return [str(index) for index in range(1500, len(self.env.server.goals))]

    def observe(self, observation: str, done: bool = False) -> dict[str, Any]:
        return {
            "observation": observation + "\nAvailable actions: " + json.dumps(self.env.get_available_actions()),
            "reward": self.reward,
            "done": done,
        }

    def reset(self, task_id: str) -> dict[str, Any]:
        index = int(task_id)
        if not 1500 <= index < len(self.env.server.goals):
            raise ValueError("WebShop task is outside the training split")
        observation, _ = self.env.reset(session=index)
        self.reward = 0.0
        return self.observe("Return one search[keywords] or click[value] action per turn.\n" + observation)

    def step(self, action: str) -> dict[str, Any]:
        observation, reward, done, _ = self.env.step(action.strip())
        self.reward = float(reward)
        return self.observe(observation, bool(done))

    def score(self) -> float:
        return self.reward

    def close(self) -> None:
        self.env.close()


if __name__ == "__main__":
    main(WebShop)
