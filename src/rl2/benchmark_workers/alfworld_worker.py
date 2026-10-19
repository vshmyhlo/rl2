from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any

import yaml
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
from common import main


class ALFWorld:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = yaml.safe_load(Path(config["config_file"]).read_text())
        self.config["dataset"]["num_train_games"] = -1
        self.config["env"].update(task_types=[1, 2, 3, 4, 5, 6], domain_randomization=False)
        # DQN mode omits expert demonstrations from observations.
        self.config["general"]["training_method"] = "dqn"
        self.config["rl"]["training"]["max_nb_steps_per_episode"] = config["max_steps"]
        self.loader = AlfredTWEnv(self.config, train_eval="train")
        self.games = sorted(self.loader.game_files)
        self.resources = ExitStack()
        self.reward = 0.0

    def task_ids(self) -> list[str]:
        return self.games

    def observe(self, observation: str, info: dict[str, Any], done: bool = False) -> dict[str, Any]:
        return {
            "observation": observation + "\nAdmissible commands: " + ", ".join(info["admissible_commands"][0]),
            "reward": self.reward,
            "done": done,
        }

    def reset(self, task_id: str) -> dict[str, Any]:
        if task_id not in self.games:
            raise ValueError("ALFWorld task is outside the training split")
        self.resources.close()
        self.loader.game_files = [task_id]
        self.loader.num_games = 1
        self.env = self.loader.init_env(batch_size=1)
        self.resources.callback(self.env.close)
        observations, info = self.env.reset()
        self.reward = 0.0
        return self.observe("Return one admissible command per turn.\n" + observations[0], info)

    def step(self, action: str) -> dict[str, Any]:
        observations, _, dones, info = self.env.step([action.strip()])
        self.reward = float(info["won"][0])
        return self.observe(observations[0], info, bool(dones[0]))

    def score(self) -> float:
        return self.reward

    def close(self) -> None:
        self.resources.close()


if __name__ == "__main__":
    main(ALFWorld)
