from __future__ import annotations

import json
from contextlib import ExitStack
from typing import Any

from common import main
from tau2.gym.gym_agent import AgentGymEnv
from tau2.run import load_task_splits


class Tau2:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.resources = ExitStack()
        self.reward = 0.0
        self.done = False

    def task_ids(self) -> list[str]:
        ids = []
        for domain in self.config["domains"]:
            splits = load_task_splits(domain)
            if not splits or not splits.get("train"):
                raise ValueError(f"tau2 {domain} has no official nonempty train split")
            ids.extend(f"{domain}:{task_id}" for task_id in splits["train"])
        return ids

    def reset(self, task_id: str) -> dict[str, Any]:
        if task_id not in self.task_ids():
            raise ValueError("tau2 task is outside the training split")
        self.resources.close()
        domain, identifier = task_id.split(":", 1)
        self.env = AgentGymEnv(
            domain=domain,
            task_id=identifier,
            user_llm=self.config["user_llm"],
            max_steps=self.config["max_steps"],
            solo_mode=False,
        )
        self.resources.callback(self.env.close)
        observation, info = self.env.reset()
        if not json.loads(info["simulation_run"]) and self.env._simulation_done.is_set():
            raise RuntimeError("tau2 simulator failed during reset; see worker logs")
        self.reward, self.done = 0.0, False
        tools = [tool.openai_schema for tool in info["tools"]]
        return {
            "observation": (
                'Return a plain-text message to the user or one JSON {"name": ..., "arguments": {...}} tool call.\n'
                f"{info['policy']}\nTools: {json.dumps(tools)}\n{observation}"
            )
        }

    def step(self, action: str) -> dict[str, Any]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.done = bool(terminated or truncated)
        if self.done and not json.loads(info["simulation_run"]):
            raise RuntimeError("tau2 simulator failed; see worker logs")
        self.reward = float(reward)
        return {"observation": observation, "reward": self.reward, "done": self.done}

    def score(self) -> float:
        if not self.done:
            # Finalize at the trainer's turn limit so upstream evaluates partial progress.
            self.step('{"name": "done", "arguments": {}}')
        return self.reward

    def close(self) -> None:
        self.resources.close()


if __name__ == "__main__":
    main(Tau2)
