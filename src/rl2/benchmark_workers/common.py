"""JSON-lines transport; imported by workers using their own Python installations."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from contextlib import closing, redirect_stdout
from typing import Any, Protocol, TextIO

PREFIX = "RL2_BENCHMARK "


class Environment(Protocol):
    def task_ids(self) -> list[str]: ...
    def reset(self, task_id: str) -> dict[str, Any]: ...
    def step(self, action: str) -> dict[str, Any]: ...
    def score(self) -> float: ...
    def close(self) -> None: ...


def emit(value: Any, output: TextIO) -> None:
    print(PREFIX + json.dumps(value), file=output, flush=True)


def main(factory: Callable[[dict[str, Any]], Environment]) -> None:
    output = sys.stdout
    with redirect_stdout(sys.stderr), closing(factory(json.loads(sys.argv[1]))) as env:
        if "--list" in sys.argv[2:]:
            emit(env.task_ids(), output)
            return
        for line in sys.stdin:
            request = json.loads(line)
            if request["op"] == "reset":
                value = env.reset(request["task_id"])
            elif request["op"] == "step":
                value = env.step(request["action"])
            elif request["op"] == "score":
                value = env.score()
            else:
                raise ValueError(f"Unknown operation: {request['op']}")
            emit(value, output)
