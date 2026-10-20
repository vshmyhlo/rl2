"""Periodic Ray timeline snapshots that survive local cluster shutdown."""

import json
import logging
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from threading import Event, Thread

import ray

from rl2.utils import write_bytes

logger = logging.getLogger(__name__)


@contextmanager
def record_timeline(path: str, interval: float = 60) -> Iterator[None]:
    """Export while Ray is running; keep the previous snapshot if export fails."""
    stopped = Event()

    def save() -> None:
        try:
            write_bytes(path, json.dumps(ray.timeline()).encode())
        except Exception:
            logger.exception("Could not save Ray timeline to %s", path)

    def periodically_save() -> None:
        while not stopped.wait(interval):
            save()

    with ExitStack() as resources:
        resources.callback(save)
        thread = Thread(target=periodically_save, name="ray-timeline", daemon=True)
        thread.start()
        resources.callback(thread.join)
        resources.callback(stopped.set)
        logger.info("Ray timeline: %s (updated every %ss and on exit)", path, interval)
        if url := os.environ.get("RL2_RAY_TIMELINE_URL"):
            logger.info("Open Ray timeline: %s", url)
        yield
