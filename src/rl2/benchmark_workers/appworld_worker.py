from __future__ import annotations

import json
import re
from contextlib import ExitStack
from tempfile import TemporaryDirectory
from typing import Any

from appworld import AppWorld, load_task_ids
from common import main


class AppWorldEnvironment:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.resources = ExitStack()
        self.done = False

    def task_ids(self) -> list[str]:
        return load_task_ids("train")

    def reset(self, task_id: str) -> dict[str, Any]:
        if task_id not in self.task_ids():
            raise ValueError("AppWorld task is outside the training split")
        self.resources.close()
        output = self.resources.enter_context(TemporaryDirectory(prefix="rl2-appworld-"))
        self.world = self.resources.enter_context(
            AppWorld(
                task_id=task_id,
                experiment_name=output,
                max_interactions=self.config["max_steps"],
            )
        )
        task = self.world.task
        self.done = self.world.task_completed() if self.config.get("tool_calls") else False
        action_prompt = (
            "Execute Python using the execute tool and the preloaded apis object. Print results to see them.\n"
            if self.config.get("tool_calls")
            else "Execute Python using the preloaded apis object. Return one ```python code block per turn.\n"
        )
        prompt = (
            action_prompt + "Discover APIs with apis.api_docs.show_app_descriptions(), "
            "apis.api_docs.show_api_descriptions(app_name=...), and "
            "apis.api_docs.show_api_doc(app_name=..., api_name=...).\n"
            "Finish with apis.supervisor.complete_task().\n"
            f"Supervisor: {json.dumps(dict(task.supervisor))}\nCurrent time: {task.datetime}\n{task.instruction}"
        )
        return {"observation": prompt, "done": self.done}

    def step(self, action: str) -> dict[str, Any]:
        if self.config.get("tool_calls"):
            if self.done:
                raise RuntimeError("AppWorld episode finished; reset before stepping again")
            observation = self.world.execute(action)
            self.done = self.world.task_completed()
            return {"observation": observation, "reward": self.score() if self.done else 0.0, "done": self.done}
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", action, re.DOTALL)
        if len(blocks) != 1:
            return {"observation": "Return exactly one fenced Python code block."}
        observation = self.world.execute(blocks[0])
        return {"observation": observation, "done": self.world.task_completed()}

    def score(self) -> float:
        return float(self.world.evaluate().success)

    def close(self) -> None:
        self.resources.close()


if __name__ == "__main__":
    main(AppWorldEnvironment)
