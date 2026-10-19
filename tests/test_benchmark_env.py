import asyncio
import json
import os
import random
import runpy
import sys
from contextlib import closing
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from rl2 import benchmark_env as env

WORKERS = Path(env.__file__).with_name("benchmark_workers")


def load_worker(name: str, modules: dict[str, dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.syspath_prepend(str(WORKERS))
    for module_name, attributes in modules.items():
        module = ModuleType(module_name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, module_name, module)
    return runpy.run_path(str(WORKERS / f"{name}_worker.py"))


def test_webshop_full_split(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = Mock()
    upstream.server.goals = range(1503)
    upstream.reset.return_value = ("Buy a hat", None)
    upstream.get_available_actions.return_value = {"clickables": ["Buy Now"]}
    upstream.step.return_value = ("Purchased", 0.75, True, None)
    prices: list[float] = []

    def create_environment(**kwargs: Any) -> Mock:
        prices.append(random.random())
        return upstream

    constructor = Mock(side_effect=create_environment)
    engine = SimpleNamespace()
    module = load_worker(
        "webshop",
        {
            "web_agent_site.engine": {"engine": engine},
            "web_agent_site.envs": {"WebAgentTextEnv": constructor},
        },
        monkeypatch,
    )
    with closing(module["WebShop"]({"root": "/data/webshop"})) as worker:
        assert worker.task_ids() == ["1500", "1501", "1502"]
        assert constructor.call_args.kwargs["num_products"] is None
        assert constructor.call_args.kwargs["human_goals"] is True
        assert constructor.call_args.kwargs["file_path"] == "/data/webshop/data/items_shuffle.json"
        assert engine.DEFAULT_ATTR_PATH == "/data/webshop/data/items_ins_v2.json"
        with pytest.raises(ValueError, match="training split"):
            worker.reset("1499")
        assert "Buy a hat" in worker.reset("1502")["observation"]
        upstream.reset.assert_called_once_with(session=1502)
        assert worker.step("click[Buy Now]")["done"]
        assert worker.score() == 0.75
    upstream.close.assert_called_once()
    with closing(module["WebShop"]({"root": "/data/webshop"})):
        assert prices[0] == prices[1]


def test_alfworld_all_training_games(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "alfworld.json"
    config.write_text(json.dumps({"dataset": {"num_train_games": 1}, "env": {}, "general": {}, "rl": {"training": {}}}))
    upstream = Mock()
    upstream.reset.return_value = (["Put apple on table"], {"admissible_commands": [["look"]]})
    upstream.step.return_value = (["Done"], [0], [True], {"admissible_commands": [[]], "won": [True]})
    loader = Mock(game_files=["train/b", "train/a"])
    loader.init_env.return_value = upstream
    constructor = Mock(return_value=loader)
    module = load_worker(
        "alfworld", {"alfworld.agents.environment.alfred_tw_env": {"AlfredTWEnv": constructor}}, monkeypatch
    )
    with closing(module["ALFWorld"]({"config_file": str(config), "max_steps": 3})) as worker:
        assert worker.task_ids() == ["train/a", "train/b"]
        settings = constructor.call_args.args[0]
        assert constructor.call_args.kwargs == {"train_eval": "train"}
        assert settings["dataset"]["num_train_games"] == -1
        assert settings["env"]["task_types"] == [1, 2, 3, 4, 5, 6]
        assert settings["general"]["training_method"] == "dqn"
        assert settings["rl"]["training"]["max_nb_steps_per_episode"] == 3
        with pytest.raises(ValueError, match="training split"):
            worker.reset("dev/a")
        assert "look" in worker.reset("train/b")["observation"]
        assert loader.game_files == ["train/b"]
        assert worker.step("look")["done"]
        assert worker.score() == 1.0
    upstream.close.assert_called_once()


def test_appworld_train_execution_and_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    world = MagicMock()
    world.__enter__.return_value = world
    world.task = SimpleNamespace(supervisor={"email": "a@example.com"}, datetime="2026-01-01", instruction="Buy milk")
    world.execute.return_value = "ok"
    world.task_completed.return_value = True
    world.evaluate.return_value.success = True
    constructor = Mock(return_value=world)
    tasks = Mock(return_value=["train-a", "train-b"])
    module = load_worker("appworld", {"appworld": {"AppWorld": constructor, "load_task_ids": tasks}}, monkeypatch)
    with closing(module["AppWorldEnvironment"]({"max_steps": 3})) as worker:
        assert worker.task_ids() == ["train-a", "train-b"]
        tasks.assert_called_with("train")
        with pytest.raises(ValueError, match="training split"):
            worker.reset("dev-a")
        prompt = worker.reset("train-b")["observation"]
        assert "Buy milk" in prompt and "show_api_doc" in prompt
        output = Path(constructor.call_args.kwargs["experiment_name"])
        assert output.is_dir()
        assert "exactly one" in worker.step("no code")["observation"]
        world.execute.assert_not_called()
        assert worker.step("```python\nprint(1)\n```")["done"]
        world.execute.assert_called_once_with("print(1)\n")
        assert worker.score() == 1.0
    assert not output.exists()
    world.__exit__.assert_called_once()


def test_tau2_train_domains_reward_and_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = Mock()
    upstream._simulation_done.is_set.return_value = False
    upstream.reset.return_value = (
        "Hello",
        {"simulation_run": "{}", "policy": "Policy", "tools": [SimpleNamespace(openai_schema={"name": "lookup"})]},
    )
    upstream.step.return_value = ("Finished", 0.6, True, False, {"simulation_run": '{"id": "run"}'})
    constructor = Mock(return_value=upstream)
    splits = Mock(return_value={"train": ["0", "1"], "test": ["2"], "full": ["0", "1", "2", "3"]})
    module = load_worker(
        "tau2",
        {
            "tau2.gym.gym_agent": {"AgentGymEnv": constructor},
            "tau2.run": {"load_task_splits": splits},
        },
        monkeypatch,
    )
    config = {"domains": ["retail", "airline", "telecom"], "max_steps": 3, "user_llm": "test-user"}
    with closing(module["Tau2"](config)) as worker:
        assert worker.task_ids() == [f"{domain}:{i}" for domain in config["domains"] for i in [0, 1]]
        with pytest.raises(ValueError, match="training split"):
            worker.reset("retail:2")
        prompt = worker.reset("telecom:0")["observation"]
        assert "Policy" in prompt and "lookup" in prompt and "Hello" in prompt
        assert constructor.call_args.kwargs == {
            "domain": "telecom",
            "task_id": "0",
            "user_llm": "test-user",
            "max_steps": 3,
            "solo_mode": False,
        }
        assert worker.score() == 0.6
        upstream.step.assert_called_once_with('{"name": "done", "arguments": {}}')
        assert worker.score() == 0.6
        upstream.step.return_value = ("", 0, True, False, {"simulation_run": "{}"})
        with pytest.raises(RuntimeError, match="simulator failed"):
            worker.step("hello")
        splits.return_value = {"test": ["0"]}
        with pytest.raises(ValueError, match="no official"):
            worker.task_ids()
    upstream.close.assert_called_once()


def test_task_manifest_and_worker_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    config = env.WebShopConfig(python=sys.executable, root="/data/webshop")
    run = Mock(return_value=SimpleNamespace(stdout='startup noise\nRL2_BENCHMARK ["1500", "1501"]\n'))
    monkeypatch.setattr(env.subprocess, "run", run)
    tasks = env.load_benchmark_tasks(config, max_steps=2, timeout=30)
    assert [task.data.task_id for task in tasks] == ["1500", "1501"]
    assert [task.data.id for task in tasks] == ["webshop:1500", "webshop:1501"]
    assert run.call_args.args[0][-1] == "--list"
    assert run.call_args.kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0] == "/data/webshop"
    assert json.loads(run.call_args.args[0][-2])["max_steps"] == 2
    assert env.worker_environment(env.AppWorldConfig(python=sys.executable, root="/apps"))["APPWORLD_ROOT"] == "/apps"


@pytest.mark.parametrize("manifest", [[], ["duplicate", "duplicate"], [""]])
def test_invalid_task_manifest(manifest: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        env.subprocess, "run", Mock(return_value=SimpleNamespace(stdout=env.PREFIX + json.dumps(manifest)))
    )
    with pytest.raises(ValueError):
        env.load_benchmark_tasks(env.AppWorldConfig(python=sys.executable, root="/apps"), max_steps=2, timeout=30)


@pytest.mark.parametrize("mode", ["normal", "crash", "timeout", "cancel", "cleanup-error"])
def test_worker_process_lifecycle(mode: str, tmp_path: Path) -> None:
    script = tmp_path / "worker.py"
    script.write_text("""import time
from pathlib import Path
from typing import Any
from common import main

class Environment:
    def __init__(self, config: dict[str, Any]) -> None:
        self.action = ""
    def task_ids(self) -> list[str]:
        return ["a", "b"]
    def reset(self, task_id: str) -> dict[str, str]:
        return {"observation": task_id}
    def step(self, action: str) -> dict[str, str]:
        self.action = action
        if action == "crash":
            raise RuntimeError("worker crash")
        if action in ("timeout", "cancel"):
            time.sleep(60)
        return {"observation": "ok"}
    def score(self) -> float:
        return 0.5
    def close(self) -> None:
        if self.action == "cleanup-error":
            raise RuntimeError("cleanup failed")
main(Environment)
""")

    async def run() -> None:
        session = env.BenchmarkSession(
            [sys.executable, str(script), "{}"],
            timeout=0.2 if mode == "timeout" else 10,
            env=dict(os.environ, PYTHONPATH=str(WORKERS)),
        )

        async def interact() -> None:
            async with session:
                assert await session.request(op="reset", task_id="a") == {"observation": "a"}
                assert await session.request(op="score") == 0.5
                if mode == "cancel":
                    asyncio.get_running_loop().call_later(0.05, asyncio.current_task().cancel)
                await session.request(op="step", action=mode)

        if mode == "normal":
            await interact()
        else:
            with pytest.raises(
                {
                    "crash": RuntimeError,
                    "timeout": TimeoutError,
                    "cancel": asyncio.CancelledError,
                    "cleanup-error": RuntimeError,
                }[mode]
            ):
                await interact()
        assert session.process.returncode is not None

    asyncio.run(run())
