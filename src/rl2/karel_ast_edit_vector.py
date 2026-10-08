"""Masked batched AST editing with persistent, spawned environment workers."""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from types import TracebackType
from typing import NamedTuple, Self

import chex
import numpy as np
from numpy.typing import NDArray

from rl2.karel import KarelProgramEnv
from rl2.karel_ast import KarelAST
from rl2.karel_ast_edit import EditConfig, EditStep, EditingObservation, Evaluation, KarelASTEditEnv


class EditSummary(NamedTuple):
    """Final worker state needed for rewards, diagnostics, and program logging."""

    tree: KarelAST
    result: Evaluation
    completed_edits: int
    remaining: int


def _summaries(envs: list[KarelASTEditEnv]) -> list[EditSummary]:
    """Read episode results without transferring the complete environment objects."""
    if any(env.result is None for env in envs):
        raise RuntimeError("Reset environments before reading their summaries")
    return [EditSummary(env.tree, env.result, env.completed_edits, env.remaining) for env in envs]


_worker_envs: list[KarelASTEditEnv] = []


def _initialize_worker(config: EditConfig, count: int) -> None:
    """Construct one persistent shard in a worker, without initializing JAX devices."""
    global _worker_envs
    _worker_envs = [KarelASTEditEnv(config) for _ in range(count)]


def _reset_worker(tasks: list[KarelProgramEnv]) -> list[EditingObservation]:
    """Reset the worker's shard from independently copied, already sampled tasks."""
    return [env.reset(task=task) for env, task in zip(_worker_envs, tasks, strict=True)]


def _step_worker(actions: list[int | None]) -> list[EditStep | None]:
    """Step selected members; None leaves paused or completed members untouched."""
    return [env.step(action) if action is not None else None for env, action in zip(_worker_envs, actions, strict=True)]


def _summarize_worker() -> list[EditSummary]:
    """Return the shard's final programs and scores after a rollout."""
    return _summaries(_worker_envs)


class KarelASTEditVectorEnv:
    """Step a fixed batch with optional process parallelism and no automatic resets.

    workers=0 runs locally. Otherwise each spawned process owns a contiguous
    shard for the lifetime of the vector environment. step submits all shards
    in parallel, then waits for their results in episode order. No environment or AST
    is copied back on ordinary steps; full programs are read via summaries().
    """

    def __init__(self, config: EditConfig, num_envs: int, *, workers: int = 0) -> None:
        """Allocate a batch and optional persistent worker pools; reset before stepping."""
        if type(num_envs) is not int or type(workers) is not int:
            raise TypeError("Environment and worker counts must be integers")
        chex.assert_scalar_positive(num_envs)
        chex.assert_scalar_non_negative(workers)
        self.config = config
        self.num_envs = num_envs
        self.closed = False
        self._pools: list[ProcessPoolExecutor] = []
        self._slices: list[slice] = []
        self._envs = [] if workers else [KarelASTEditEnv(config) for _ in range(num_envs)]
        try:
            count = min(workers, num_envs)
            for index in range(count):
                start, end = index * num_envs // count, (index + 1) * num_envs // count
                self._slices.append(slice(start, end))
                self._pools.append(
                    ProcessPoolExecutor(
                        max_workers=1,
                        mp_context=get_context("spawn"),
                        initializer=_initialize_worker,
                        initargs=(config, end - start),
                    )
                )
        except BaseException:
            self.close()
            raise

    def _check_open(self) -> None:
        """Reject reuse after closing the workers."""
        if self.closed:
            raise RuntimeError("Vector environment is closed")

    def reset(self, tasks: list[KarelProgramEnv]) -> list[EditingObservation]:
        """Reset every member, preserving task order and same-task rollout groups."""
        self._check_open()
        if len(tasks) != self.num_envs or any(task.config != self.config.env for task in tasks):
            raise ValueError("Expected one matching task per environment")
        try:
            if not self._pools:
                return [env.reset(task=task) for env, task in zip(self._envs, tasks, strict=True)]
            futures = [pool.submit(_reset_worker, tasks[part]) for pool, part in zip(self._pools, self._slices)]
            return [observation for future in futures for observation in future.result()]
        except BaseException:
            self.close()
            raise

    def step(self, actions: NDArray[np.int32], mask: NDArray[np.bool_]) -> list[EditStep | None]:
        """Step masked members in parallel and wait for ordered results; paused members stay untouched."""
        self._check_open()
        chex.assert_shape((actions, mask), (self.num_envs,))
        chex.assert_type(actions, np.int32)
        chex.assert_type(mask, np.bool_)
        selected = [int(action) if active else None for action, active in zip(actions, mask)]
        try:
            if self._pools:
                futures = [pool.submit(_step_worker, selected[part]) for pool, part in zip(self._pools, self._slices)]
                return [transition for future in futures for transition in future.result()]
            return [
                env.step(action) if action is not None else None
                for env, action in zip(self._envs, selected, strict=True)
            ]
        except BaseException:
            self.close()
            raise

    def summaries(self) -> list[EditSummary]:
        """Fetch final programs and diagnostics once, after stepping has completed."""
        self._check_open()
        if not self._pools:
            return _summaries(self._envs)
        futures = [pool.submit(_summarize_worker) for pool in self._pools]
        return [summary for future in futures for summary in future.result()]

    def close(self) -> None:
        """Join workers, including pending work; safe to call more than once."""
        self.closed = True
        for pool in self._pools:
            pool.shutdown(wait=True, cancel_futures=True)
        self._pools.clear()

    def __enter__(self) -> Self:
        """Use the batch as a context manager for reliable worker cleanup."""
        self._check_open()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        """Close workers on normal completion or an exception."""
        self.close()
