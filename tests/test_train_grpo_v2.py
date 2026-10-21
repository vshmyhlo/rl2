import asyncio
import io
import json
import os
import pickle
import runpy
import subprocess
import sys
from collections.abc import Iterator
from contextlib import ExitStack, aclosing
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock, Mock
from urllib.error import URLError

import httpx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from openai import APIStatusError, AsyncOpenAI
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError
from ray import cloudpickle

from rl2 import train_grpo_v2 as training


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel_trace, "get_tracer", Mock(return_value=provider.get_tracer("test")))
    with ExitStack() as resources:
        resources.callback(provider.shutdown)
        yield exporter


def test_config_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    defaults = training.Config()
    assert defaults.minibatches_per_epoch == 4
    assert defaults.grad_accum_steps == 1
    assert defaults.update_epochs == 1
    assert defaults.checkpoint_interval == 100
    assert defaults.padding_multiple == 256
    assert defaults.gradient_checkpointing
    assert not defaults.enable_ray_dashboard
    assert not defaults.enable_ray_timeline
    assert defaults.enable_tracing
    assert defaults.weight_update_pause_mode == "wait"
    preset = Path(__file__).resolve().parents[1] / "configs/grpo_v2/alfworld.yaml"
    config = training.load_config(preset, training.Config)
    assert config.run_id == "grpo_v2_alfworld_Qwen_Qwen3-0.6B_seed3"
    assert config.ckpt_dir == f"/runs/grpo_v2/{config.run_id}"
    assert config.total_iterations is None
    assert config.minibatches_per_epoch == 4
    assert config.grad_accum_steps == 1
    assert config.gradient_checkpointing
    assert config.weight_update_pause_mode == "wait"
    assert asdict(config.taskset) == {"id": "alfworld", "config_file": "data/alfworld/base_config.yaml"}
    assert "dataset_file" not in asdict(config)
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            asdict(config)
            | {
                "max_tokens": "${group_size}",
                "total_iterations": 1,
                "gradient_checkpointing": False,
                "enable_ray_dashboard": True,
                "enable_ray_timeline": True,
                "enable_tracing": False,
                "padding_multiple": 64,
            }
        )
    )
    train = Mock()
    monkeypatch.setattr(training, "train", train)
    monkeypatch.setattr(sys, "argv", ["train_grpo_v2", "--config", str(path)])
    training.main()
    train.assert_called_once_with(
        replace(
            config,
            max_tokens=4,
            total_iterations=1,
            gradient_checkpointing=False,
            enable_ray_dashboard=True,
            enable_ray_timeline=True,
            enable_tracing=False,
            padding_multiple=64,
        )
    )
    assert pickle.loads(pickle.dumps(config)) == config


def test_presets_use_telemetry_defaults() -> None:
    presets = Path(__file__).resolve().parents[1] / "configs/grpo_v2"
    for preset in sorted(presets.glob("alfworld.yaml")):
        config = training.load_config(preset, training.Config)
        assert config.enable_tracing, preset.name
        assert not config.enable_ray_dashboard, preset.name
        assert not config.enable_ray_timeline, preset.name


@pytest.mark.parametrize(
    "settings",
    [
        {"num_groups": 0},
        {"group_size": 1},
        {"total_iterations": 0},
        {"checkpoint_interval": 0},
        {"checkpoint_interval": 1.5},
        {"max_tokens": 0},
        {"padding_multiple": 0},
        {"padding_multiple": 1.5},
        {"max_turns": 0},
        {"temperature": 0.001},
        {"temperature": 2.1},
        {"seed": -1},
        {"group_size": 2.5},
        {"unknown_option": 1},
        {"learning_rate": 0.0},
        {"clip_coef": 1.0},
        {"max_grad_norm": 0.0},
        {"minibatches_per_epoch": 0},
        {"minibatches_per_epoch": 3},
        {"minibatches_per_epoch": 8},
        {"grad_accum_steps": 0},
        {"grad_accum_steps": 3},
        {"minibatches_per_epoch": 2, "grad_accum_steps": 4},
        {"update_epochs": 0},
        {"weight_update_pause_mode": "unknown"},
        {"taskset": "unknown"},
    ],
)
def test_config_rejects_invalid_settings(settings: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        training.Config(**settings)


@pytest.mark.parametrize(
    ("settings", "error"),
    [
        ({"taskset": {"id": "addition", "config_file": "config.yaml"}}, "literal_error"),
        ({"taskset": {"id": "alfworld", "config_file": None}}, "string_type"),
        ({"taskset": {"id": "alfworld", "config_file": "config.yaml", "unknown": True}}, "unexpected_keyword_argument"),
        ({"dataset_file": "train.parquet"}, "unexpected_keyword_argument"),
    ],
)
def test_taskset_config_rejects_invalid_yaml(settings: dict[str, object], error: str, tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValidationError, match=error):
        training.load_config(path, training.Config)


@pytest.mark.parametrize("temperature", [0.01, 2.0])
def test_sampling_temperature_bounds(temperature: float) -> None:
    assert training.Config(temperature=temperature).temperature == temperature


@pytest.mark.parametrize("failure", [None, "launch", "rollout", "shutdown"])
def test_serve_vllm(failure: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "Qwen/Qwen3-0.6B",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--dtype",
        "bfloat16",
        "--generation-config",
        "vllm",
        "--logprobs-mode",
        "processed_logprobs",
        "--weight-transfer-config",
        '{"backend":"nccl"}',
    ]
    server = MagicMock()
    server.__enter__.return_value = server
    launch = Mock(return_value=server, side_effect=OSError("launch failed") if failure == "launch" else None)
    monkeypatch.setattr(training.subprocess, "Popen", launch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    if failure == "shutdown":
        server.wait.side_effect = subprocess.TimeoutExpired(cmd=command, timeout=10)

    if failure in {"launch", "rollout"}:
        with pytest.raises(OSError if failure == "launch" else RuntimeError), training.serve_vllm():
            raise RuntimeError("rollout failed")
    else:
        with training.serve_vllm() as process:
            assert process is server
            server.terminate.assert_not_called()

    launch.assert_called_once()
    assert launch.call_args.args == (command,)
    assert launch.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert launch.call_args.kwargs["env"]["VLLM_SERVER_DEV_MODE"] == "1"
    if failure == "launch":
        server.terminate.assert_not_called()
    else:
        server.terminate.assert_called_once_with()
        server.wait.assert_called_once_with(timeout=10)
        server.__exit__.assert_called_once()
        assert server.kill.call_count == (1 if failure == "shutdown" else 0)


def test_actor_and_learner_each_reserve_one_gpu() -> None:
    assert training.Actor._default_options["num_gpus"] == 1
    assert training.Learner._default_options["num_gpus"] == 1
    assert training.Batcher._default_options.get("num_gpus", 0) == 0


def test_rollout_workers_restrict_accidental_jax_gpu_use() -> None:
    actor_env = training.Actor._default_options.get("runtime_env", {}).get("env_vars", {})
    batcher_env = training.Batcher._default_options.get("runtime_env", {}).get("env_vars", {})
    assert actor_env == batcher_env == {"JAX_PLATFORMS": "cpu"}


def test_script_actor_serialization() -> None:
    # A script's definitions must travel by value to workers with a different entrypoint.
    namespace = runpy.run_path(training.__file__)
    actors = {name: namespace[name].__ray_metadata__.modified_class for name in ("Learner", "Actor", "Batcher")}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "from ray import cloudpickle\n"
                "actors = cloudpickle.loads(sys.stdin.buffer.read())\n"
                "assert all(actor.__name__ == name for name, actor in actors.items())\n"
            ),
        ],
        input=cloudpickle.dumps(actors),
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr.decode()


def test_wait_for_server_retries_until_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    response = io.BytesIO()
    request = Mock(side_effect=[URLError("connection refused"), response])
    monkeypatch.setattr(training, "urlopen", request)
    monkeypatch.setattr(training.time, "sleep", Mock())

    training.wait_for_server(Mock(poll=Mock(return_value=None)))

    assert request.call_count == 2
    request.assert_called_with(f"{training.SERVER_URL}/health", timeout=5)
    assert response.closed


def test_wait_for_server_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training, "urlopen", Mock(side_effect=URLError("connection refused")))
    monkeypatch.setattr(training.time, "monotonic", Mock(side_effect=[0, 0, 1, 2]))
    monkeypatch.setattr(training.time, "sleep", Mock())

    with pytest.raises(TimeoutError, match="vLLM was not ready"):
        training.wait_for_server(Mock(poll=Mock(return_value=None)), timeout=2)


@pytest.mark.parametrize("returncode", [1, 0, -9], ids=["error", "early-exit", "killed"])
def test_wait_for_server_detects_exit(returncode: int, monkeypatch: pytest.MonkeyPatch) -> None:
    server = Mock(poll=Mock(side_effect=[None, returncode]))
    request = Mock(side_effect=URLError("connection refused"))
    monkeypatch.setattr(training, "urlopen", request)
    sleep = Mock()
    monkeypatch.setattr(training.time, "sleep", sleep)

    with pytest.raises(RuntimeError, match=f"vLLM exited with code {returncode}"):
        training.wait_for_server(server)

    request.assert_called_once()
    sleep.assert_called_once()


@pytest.fixture(autouse=True)
def task_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        training,
        "load_alfworld_tasks",
        Mock(
            return_value=[training.ALFWorldTask(id=f"train/{index}", config_file="config.yaml") for index in range(2)]
        ),
    )


@pytest.fixture
def actor(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(training, "serve_vllm", Mock(return_value=MagicMock()))
    monkeypatch.setattr(training, "wait_for_server", Mock())
    actor = training.Actor.__ray_metadata__.modified_class()
    monkeypatch.setattr(actor.rng, "choice", Mock(return_value=actor.tasks[0]))
    return actor


async def collect_and_close(actor: Any, task: training.ALFWorldTask) -> training.RolloutGroup:
    async with aclosing(actor):
        return await actor.collect_group(task, group_size=2)


def test_actor_reuses_agent(actor: Any, rollout: training.Rollout, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = AsyncMock()
    entering = asyncio.Event()
    release = asyncio.Event()

    async def enter(context: training.Agent) -> Any:
        entering.set()
        await release.wait()
        return agent

    agent.__aenter__.side_effect = enter
    agent.run.return_value = rollout
    create = Mock(return_value=agent)
    monkeypatch.setattr(training, "create_agent", create)

    async def collect() -> None:
        async with aclosing(actor), asyncio.timeout(5):
            first = asyncio.create_task(actor.collect_group(actor.tasks[0], group_size=2))
            await entering.wait()
            second = asyncio.create_task(actor.collect_group(actor.tasks[1], group_size=2))
            await asyncio.sleep(0)
            create.assert_called_once_with(actor.config)
            release.set()
            groups = await asyncio.gather(first, second)
            assert [group.task_id for group in groups] == [f"alfworld:{task.id}" for task in actor.tasks[:2]]
            await actor.collect_group(actor.tasks[0], group_size=2)
            agent.__aexit__.assert_not_awaited()
        await actor.aclose()
        with pytest.raises(RuntimeError, match="Actor is closed"):
            await actor.collect_group(actor.tasks[0], group_size=2)

    asyncio.run(collect())
    create.assert_called_once_with(actor.config)
    agent.__aenter__.assert_awaited_once()
    agent.__aexit__.assert_awaited_once()
    assert agent.run.await_count == 6


def test_actor_retries_agent_startup(actor: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    failed, agent = AsyncMock(), AsyncMock()
    failed.__aenter__.side_effect = RuntimeError("startup failed")
    agent.__aenter__.return_value = agent
    create = Mock(side_effect=[failed, agent])
    monkeypatch.setattr(training, "create_agent", create)

    async def start() -> None:
        async with aclosing(actor):
            with pytest.raises(RuntimeError, match="startup failed"):
                await actor._get_agent()
            assert await actor._get_agent() is agent

    asyncio.run(start())
    assert create.call_count == 2
    agent.__aexit__.assert_awaited_once()


def test_actor_close_cancels_rollouts(actor: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = AsyncMock()
    agent.__aenter__.return_value = agent
    started = asyncio.Event()
    active = 0

    async def run(task: training.ALFWorldTask) -> training.Rollout:
        nonlocal active
        active += 1
        if active == 2:
            started.set()
        try:
            return await asyncio.Future[training.Rollout]()
        finally:
            active -= 1
            agent.__aexit__.assert_not_awaited()

    agent.run.side_effect = run
    monkeypatch.setattr(training, "create_agent", Mock(return_value=agent))

    async def collect() -> None:
        async with aclosing(actor), asyncio.timeout(5):
            group = asyncio.create_task(actor.collect_group(actor.tasks[0], group_size=2))
            await started.wait()
            await actor.aclose()
            with pytest.raises(asyncio.CancelledError):
                await group

    asyncio.run(collect())
    assert active == 0
    agent.__aexit__.assert_awaited_once()


@pytest.mark.parametrize("startup_fails", [False, True])
def test_actor_server_lifetime(startup_fails: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    config = training.Config(model="Qwen/custom")
    server = MagicMock()
    serve = Mock(return_value=server)
    monkeypatch.setattr(training, "serve_vllm", serve)
    ready = Mock(side_effect=TimeoutError("not ready") if startup_fails else None)
    monkeypatch.setattr(training, "wait_for_server", ready)
    group = training.RolloutGroup(task_id="0", prompt="test", rollouts=[])

    if startup_fails:
        actor_class = training.Actor.__ray_metadata__.modified_class
        actor = actor_class.__new__(actor_class)
        with pytest.raises(TimeoutError, match="not ready"):
            actor.__init__(config=config)
        actor.__ray_shutdown__()
    else:
        actor = training.Actor.__ray_metadata__.modified_class(config=config)
        actor.collect_group = AsyncMock(return_value=group)
        monkeypatch.setattr(actor.rng, "choice", Mock(return_value=actor.tasks[0]))
        for _ in range(2):
            assert asyncio.run(actor.fetch_next_group(group_size=2)).value is group
        actor.collect_group.assert_awaited_with(actor.tasks[0], group_size=2)
        assert actor.collect_group.await_count == 2
        server.__exit__.assert_not_called()
        actor.__ray_shutdown__()
    serve.assert_called_once_with(model=config.model)
    ready.assert_called_once_with(server.__enter__.return_value)
    server.__enter__.assert_called_once()
    server.__exit__.assert_called_once()


@pytest.fixture
def rollout() -> training.Rollout:
    return training.Rollout(
        sequence=training.TrainingSequence(
            token_ids=np.array([1, 2], dtype=np.int32),  # (T=2,)
            loss_mask=np.array([False, True]),  # (T=2,)
            old_log_probs=np.array([0.0, -0.2], dtype=np.float32),  # (T=2,)
        ),
        reward=1.0,
        advantage=0.5,
        completion="2",
        interaction="2",
        num_turns=1,
        num_input_tokens=1,
        num_output_tokens=1,
        rollout_seconds=0.1,
        stop_condition="agent_completed",
    )


@pytest.mark.parametrize(("num_groups", "fails"), [(1, False), (2, False), (2, True)])
def test_learner_pulls_groups_through_batcher(
    num_groups: int, fails: bool, learner: Any, rollout: training.Rollout, monkeypatch: pytest.MonkeyPatch
) -> None:
    actor = Mock()
    groups = [training.RolloutGroup(task_id=str(i), prompt="test", rollouts=[rollout] * 3) for i in range(num_groups)]
    results = [training.Transfer(value=group, ready_at=training.time.monotonic()) for group in groups]
    actor.fetch_next_group.remote = AsyncMock(
        side_effect=[results[0], RuntimeError("rollout failed")] if fails else results
    )
    batcher = training.Batcher.__ray_metadata__.modified_class(actor=actor)
    batcher_handle = Mock()
    batcher_handle.fetch_next_batch.remote.side_effect = batcher.fetch_next_batch
    learner.batcher = batcher_handle

    def get(result: Any) -> Any:
        return asyncio.run(result)

    monkeypatch.setattr(training.ray, "get", get)

    if fails:
        with pytest.raises(ExceptionGroup) as error:
            learner.fetch_next_batch(num_groups=num_groups, group_size=3)
        assert len(error.value.exceptions) == 1
        assert isinstance(error.value.exceptions[0], RuntimeError)
        assert str(error.value.exceptions[0]) == "rollout failed"
    else:
        batch = learner.fetch_next_batch(num_groups=num_groups, group_size=3)
        assert not hasattr(batch, "groups")
        assert batch.stop_counts == {"agent_completed": num_groups * 3}
        assert batch.num_output_tokens == num_groups * 3
        assert batch.examples.count("Full interaction (special tokens preserved):") == 3
        assert batch.training_batch.token_ids.shape == (num_groups * 3, 8)
        np.testing.assert_array_equal(batch.training_batch.advantages, 0.5)
    batcher_handle.fetch_next_batch.remote.assert_called_once_with(
        num_groups=num_groups, group_size=3, context_length=8
    )
    assert actor.fetch_next_group.remote.call_count == num_groups
    assert all(call.kwargs == {"group_size": 3} for call in actor.fetch_next_group.remote.call_args_list)


def test_batch_requires_groups() -> None:
    actor = Mock()
    batcher = training.Batcher.__ray_metadata__.modified_class(actor=actor)
    with pytest.raises(ValueError, match="num_groups"):
        asyncio.run(batcher.fetch_next_batch(num_groups=0, group_size=2, context_length=8))
    actor.fetch_next_group.remote.assert_not_called()


def test_batch_failure_cancels_pending_fetches() -> None:
    started = asyncio.Event()
    cleaned_up = asyncio.Event()
    requested = 0

    async def fetch_group(group_size: int) -> training.Transfer[training.RolloutGroup]:
        nonlocal requested
        requested += 1
        if requested == 1:
            await started.wait()
            raise RuntimeError("rollout failed")
        started.set()
        try:
            return await asyncio.Future[training.Transfer[training.RolloutGroup]]()
        finally:
            cleaned_up.set()

    actor = Mock()
    actor.fetch_next_group.remote.side_effect = fetch_group
    batcher = training.Batcher.__ray_metadata__.modified_class(actor=actor)
    batcher.prepare_batch = Mock()

    async def collect() -> None:
        async with asyncio.timeout(5):
            with pytest.raises(ExceptionGroup) as error:
                await batcher.fetch_next_batch(num_groups=2, group_size=2, context_length=8)
            assert len(error.value.exceptions) == 1
            assert isinstance(error.value.exceptions[0], RuntimeError)
            assert str(error.value.exceptions[0]) == "rollout failed"
            # Cleanup must finish before control returns to the caller.
            assert cleaned_up.is_set()

    asyncio.run(collect())
    assert requested == 2
    batcher.prepare_batch.assert_not_called()


def test_ray_transfer_timings_exclude_generation_and_slow_group_wait(
    learner: Any, rollout: training.Rollout, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = Mock(return_value=0.0)
    monkeypatch.setattr(training, "time", Mock(monotonic=clock))
    caplog.set_level("INFO", logger=training.__name__)
    groups = [
        training.RolloutGroup(task_id=str(i), prompt="test", rollouts=[replace(rollout, advantage=float(i))] * 2)
        for i in range(2)
    ]
    fast_ready = asyncio.Event()
    requested = 0

    async def fetch_group(group_size: int) -> training.Transfer[training.RolloutGroup]:
        nonlocal requested
        assert group_size == 2
        index = requested
        requested += 1
        if index == 0:
            await fast_ready.wait()
            ready_at, received_at = 50.0, 50.5
        else:
            ready_at, received_at = 10.0, 10.25
            fast_ready.set()
        result = training.Transfer(value=groups[index], ready_at=ready_at)
        # Account for the full return path, including serialization/deserialization.
        result = pickle.loads(pickle.dumps(result))
        clock.return_value = received_at
        return result

    actor = Mock()
    actor.fetch_next_group.remote.side_effect = fetch_group
    batcher = training.Batcher.__ray_metadata__.modified_class(actor=actor)
    learner.batcher.fetch_next_batch.remote.side_effect = batcher.fetch_next_batch
    prepare_batch = batcher.prepare_batch

    def prepare(groups: list[training.RolloutGroup], context_length: int) -> training.GRPOBatch[np.ndarray]:
        batch = prepare_batch(groups, context_length=context_length)  # (B, T), (B,) leaves.
        clock.return_value += 2.0
        return batch  # (B, T), (B,) leaves.

    monkeypatch.setattr(batcher, "prepare_batch", prepare)

    def get(pending: Any) -> training.Transfer[training.RolloutBatch]:
        result = asyncio.run(pending)
        assert result.ready_at == 52.5
        result = pickle.loads(pickle.dumps(result))
        clock.return_value = 53.25
        return result

    monkeypatch.setattr(training.ray, "get", get)
    batch = learner.fetch_next_batch(num_groups=2, group_size=2)
    # Preserve request order despite out-of-order completion and serialization.
    np.testing.assert_array_equal(batch.training_batch.advantages, [0.0, 0.0, 1.0, 1.0])
    assert "Example 1 | task=0" in batch.examples
    assert "Example 3 | task=1" in batch.examples
    assert batch.training_batch.token_ids.shape == (4, 8)
    assert batch.transfer_metrics == {
        "time/batcher/fetch_rollout_seconds": 50.5,
        "time/batcher/prepare_batch_seconds": 2.0,
        "time/batcher/actor_to_batcher_seconds": 0.375,
        "time/learner/batcher_to_learner_seconds": 0.75,
    }
    assert "Actor -> Batcher: task=1 seconds=0.250000" in caplog.text
    assert "Actor -> Batcher: task=0 seconds=0.500000" in caplog.text
    assert "Batcher -> Learner: groups=2 seconds=0.750000" in caplog.text
    assert "array_bytes" not in caplog.text


@pytest.fixture
def learner(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(training, "checkpoint_manager", Mock(return_value=Mock()))
    monkeypatch.setattr(training, "restore_checkpoint", Mock(return_value=None))
    monkeypatch.setattr(training, "save_checkpoint", Mock(return_value=0.5))
    monkeypatch.setattr(training.ray, "get", Mock())
    device = jax.devices("cpu")[0]
    params = {
        "model.norm.weight": jnp.ones(3, dtype=jnp.bfloat16),  # (D,)
        "model.embed_tokens.weight": jnp.arange(6, dtype=jnp.bfloat16).reshape(2, 3),  # (V, D)
    }
    model = Mock()
    model.config.max_position_embeddings = 8
    loader = Mock(return_value=(model, {"params": params}))
    monkeypatch.setattr(training, "load_qwen3_checkpoint", loader)
    monkeypatch.setattr(training.jax, "devices", Mock(return_value=[device]))
    config = training.Config(model="Qwen/custom", gradient_checkpointing=False)
    learner = training.Learner.__ray_metadata__.modified_class(batcher=Mock(), actor=Mock(), config=config)
    learner.model = model
    loader.assert_not_called()
    assert not hasattr(learner, "state")
    return learner


@pytest.mark.parametrize(
    ("fails", "group_rewards", "degenerate_fraction"),
    [
        pytest.param(False, ((0.0, 0.0), (1.0, 1.0)), 1.0, id="all-degenerate"),
        pytest.param(False, ((0.0, 0.0), (0.0, 1.0)), 0.5, id="mixed"),
        pytest.param(False, ((0.0, 1.0), (1.0, 0.0)), 0.0, id="none-degenerate"),
        pytest.param(True, ((0.0, 0.0), (1.0, 1.0)), 1.0, id="failure"),
    ],
)
def test_learner_logs_rollouts(
    fails: bool,
    group_rewards: tuple[tuple[float, float], tuple[float, float]],
    degenerate_fraction: float,
    learner: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    spans: InMemorySpanExporter,
) -> None:
    caplog.set_level("INFO", logger=training.__name__)
    sequence = training.TrainingSequence(
        token_ids=np.array([1], dtype=np.int32),  # (T=1,)
        loss_mask=np.array([True]),  # (T=1,)
        old_log_probs=np.array([-0.2], dtype=np.float32),  # (T=1,)
    )
    rollout = training.Rollout(
        sequence=sequence,
        reward=0.0,
        advantage=-1.0,
        completion="wrong",
        interaction="wrong",
        num_turns=1,
        num_input_tokens=4,
        num_output_tokens=2,
        rollout_seconds=0.5,
        stop_condition="agent_completed",
    )
    batch = [
        training.RolloutGroup(
            task_id="0", prompt="first", rollouts=[replace(rollout, reward=reward) for reward in group_rewards[0]]
        ),
        training.RolloutGroup(
            task_id="1",
            prompt="second",
            rollouts=[
                replace(
                    rollout,
                    reward=reward,
                    num_turns=3,
                    num_output_tokens=6,
                    rollout_seconds=1.5,
                    stop_condition=condition,
                )
                for reward, condition in zip(group_rewards[1], ("max_turns", None), strict=True)
            ],
        ),
    ]
    rewards = [reward for group in group_rewards for reward in group]
    learner.config = replace(learner.config, num_groups=2, group_size=2, total_iterations=None if fails else 2)
    writer = MagicMock()
    writer.run_id = "mlflow-run"
    writer.__enter__.return_value = writer
    factory = Mock(return_value=writer)
    monkeypatch.setattr(training, "MLflowLogger", factory)
    # Isolate this clock from logging and other users of the time module.
    monkeypatch.setattr(training, "time", Mock(monotonic=Mock(side_effect=range(20))))
    group = Mock()
    monkeypatch.setattr(training, "get_open_port", Mock(return_value=12345))
    monkeypatch.setattr(training.NCCLWeightTransferEngine, "trainer_init", Mock(return_value=group))
    monkeypatch.setattr(training.ray, "get", Mock())
    steps = Mock()
    steps.initialize = Mock(wraps=learner.initialize_weight_transfer)
    learner.initialize_weight_transfer = steps.initialize
    learner.fetch_next_batch = steps.fetch
    transfer_metrics = {
        "time/batcher/fetch_rollout_seconds": 0.5,
        "time/batcher/prepare_batch_seconds": 0.25,
        "time/batcher/actor_to_batcher_seconds": 0.25,
        "time/learner/batcher_to_learner_seconds": 0.125,
    }
    prepared = Mock(spec=training.GRPOBatch)
    prepared.token_ids = Mock(shape=(4, 8))  # (B=4, T=8)
    received = training.build_rollout_batch(groups=batch, training_batch=prepared, transfer_metrics=transfer_metrics)
    next_batch = training.build_rollout_batch(
        groups=[
            replace(group, rollouts=[replace(rollout, stop_condition="codegym_done") for rollout in group.rollouts])
            for group in batch
        ],
        training_batch=prepared,
        transfer_metrics=transfer_metrics,
    )
    learner.fetch_next_batch.side_effect = [received, RuntimeError("done") if fails else next_batch]
    learner.publish_weights = steps.publish

    def optimize(batch: training.GRPOBatch[np.ndarray]) -> dict[str, float]:
        assert batch is prepared
        learner.state = learner.state.replace(step=learner.state.step + 1)
        return {"losses/policy": -0.25}

    steps.optimize.side_effect = optimize
    learner.optimize = steps.optimize

    def exit_writer(*args: object) -> None:
        group.destroy.assert_called_once_with()

    writer.__exit__.side_effect = exit_writer
    if fails:
        with pytest.raises(RuntimeError, match="done"):
            learner.start()
        training.save_checkpoint.assert_not_called()
    else:
        learner.start()
        assert training.save_checkpoint.call_args.args[1].iteration == 2
    training.checkpoint_manager.return_value.close.assert_called_once_with()
    training.load_qwen3_checkpoint.assert_called_once_with(
        learner.config.model,
        dtype="bfloat16",
        attention_backend="tokamax",
        gradient_checkpointing=learner.config.gradient_checkpointing,
    )
    assert learner.model is training.load_qwen3_checkpoint.return_value[0]
    assert learner.mesh.axis_names == ("batch",)
    for value in learner.state.params.values():  # Model parameter shapes.
        assert value.sharding == training.NamedSharding(learner.mesh, training.PartitionSpec())
        assert value.devices() == {jax.devices("cpu")[0]}
    assert all(value.dtype == jnp.float32 for value in learner.state.params.values())
    learner.fetch_next_batch.assert_called_with(num_groups=2, group_size=2)
    expected = ["initialize", "publish", "fetch", "optimize", "publish", "fetch"]
    assert [call[0] for call in steps.mock_calls] == expected + ([] if fails else ["optimize", "publish"])
    factory.assert_called_once_with(experiment="grpo_v2", run_name=None)
    writer.log_params.assert_called_once_with(asdict(learner.config))
    assert yaml.safe_load(writer.log_text.call_args.args[0]) == asdict(learner.config)
    calls = writer.log_metrics.call_args_list
    assert len(calls) == (2 if fails else 3)
    assert calls[0].kwargs == {"step": 0}
    assert calls[0].args[0] == {"model/params_millions": 9 / 1_000_000, "time/learner/setup_seconds": 1}
    assert calls[1].kwargs == {"step": 1}
    assert calls[1].args[0] == {
        **transfer_metrics,
        "losses/policy": -0.25,
        "charts/reward_mean": mean(rewards),
        "charts/reward_std": pstdev(rewards),
        "charts/reward_min": 0.0,
        "charts/reward_max": 1.0,
        "charts/degenerate_group_fraction": degenerate_fraction,
        "charts/stop_condition_fraction/agent_completed": 0.5,
        "charts/stop_condition_fraction/max_turns": 0.25,
        "charts/stop_condition_fraction/unspecified": 0.25,
        "charts/turns_mean": 2.0,
        "charts/input_tokens_mean": 4.0,
        "charts/output_tokens_mean": 4.0,
        "charts/total_iterations": 1,
        "charts/total_updates": 1,
        "charts/learning_rate": learner.config.learning_rate,
        "charts/total_rollouts": 4,
        "time/learner/iterations_per_second": 1 / 3,
        "time/learner/rollouts_per_second": 4 / 3,
        "time/learner/output_tokens_per_second": 16 / 3,
        "time/actor/rollout_seconds": 1.0,
        "time/learner/batch_fetch_seconds": 1.0,
        "time/learner/update_seconds": 1.0,
        "time/learner/weight_transfer_seconds": 1.0,
        "time/learner/iteration_seconds": 3.0,
    }
    if not fails:
        assert calls[2].kwargs == {"step": 2}
        assert calls[2].args[0]["charts/total_rollouts"] == 8
        assert calls[2].args[0]["charts/degenerate_group_fraction"] == degenerate_fraction
        assert {key: value for key, value in calls[2].args[0].items() if "stop_condition_fraction/" in key} == {
            "charts/stop_condition_fraction/agent_completed": 0.0,
            "charts/stop_condition_fraction/max_turns": 0.0,
            "charts/stop_condition_fraction/unspecified": 0.0,
            "charts/stop_condition_fraction/codegym_done": 1.0,
        }
    assert "Run ID: None; MLflow run: mlflow-run" in caplog.messages
    assert (
        f"iteration=1 reward={mean(rewards):.3f} degenerate_group_fraction={degenerate_fraction:.3f} "
        "batch=1.00s group_transfer=0.250000s batch_transfer=0.125000s "
        "loss=-0.2500 update=1.00s weights=1.00s"
    ) in caplog.messages
    writer.__exit__.assert_called_once()
    assert writer.__exit__.call_args.args[0] is (RuntimeError if fails else None)
    iterations = [item for item in spans.get_finished_spans() if item.name == "train.iteration"]
    assert len(iterations) == 2
    assert iterations[0].context.trace_id != iterations[1].context.trace_id
    assert all(item.parent is None for item in iterations)
    assert [item.attributes["iteration"] for item in iterations] == [1, 2]
    assert iterations[-1].status.status_code == (otel_trace.StatusCode.ERROR if fails else otel_trace.StatusCode.UNSET)
    for item in spans.get_finished_spans():
        if item.name.startswith("learner."):
            assert item.parent.span_id in {iteration.context.span_id for iteration in iterations}


def test_learner_checkpoints_and_resumes(learner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    learner.config = replace(learner.config, total_iterations=3, checkpoint_interval=2)
    learner.initialize_weight_transfer = Mock()
    published_steps: list[int] = []
    saved: list[training.TrainingProgress] = []
    writer = MagicMock()
    writer.__enter__.return_value = writer
    monkeypatch.setattr(training, "MLflowLogger", Mock(return_value=writer))
    monkeypatch.setattr(training.ray, "get", lambda value: value)
    learner.actor.get_sampling_state.remote.return_value = {"task_rng": "saved"}

    def publish() -> None:
        published_steps.append(int(learner.state.step))

    def fetch(num_groups: int, group_size: int) -> training.RolloutBatch:
        prepared = Mock(spec=training.GRPOBatch)
        prepared.token_ids = Mock(shape=(num_groups * group_size, 8))  # (B, T=8)
        return training.RolloutBatch(
            training_batch=prepared,
            rollout_metrics={"charts/reward_mean": 1.0, "charts/degenerate_group_fraction": 0.0},
            stop_counts={"max_turns" if not saved else "agent_completed": num_groups * group_size},
            num_output_tokens=4,
            examples="",
            transfer_metrics={
                "time/batcher/actor_to_batcher_seconds": 0.0,
                "time/learner/batcher_to_learner_seconds": 0.0,
            },
        )

    def optimize(batch: training.GRPOBatch[np.ndarray]) -> dict[str, float]:
        learner.state = learner.state.replace(
            step=learner.state.step + learner.config.update_epochs * learner.config.minibatches_per_epoch
        )
        return {"losses/policy": 0.0}

    def save(manager: Any, progress: training.TrainingProgress, config: training.Config) -> float:
        assert published_steps[-1] == int(progress.train_state.step)
        saved.append(progress)
        return 0.5

    learner.publish_weights = publish
    learner.fetch_next_batch = Mock(side_effect=fetch)
    learner.optimize = optimize
    monkeypatch.setattr(training, "save_checkpoint", Mock(side_effect=save))
    training.restore_checkpoint.side_effect = lambda *args: saved[-1] if saved else None
    learner.start()
    assert [progress.iteration for progress in saved] == [2, 3]
    assert [progress.total_rollouts for progress in saved] == [8, 12]
    assert published_steps == [0, 4, 8, 12]
    learner.actor.set_sampling_state.remote.assert_not_called()
    saved_key = jax.random.key_data(saved[-1].shuffle_key)  # (K,)

    # New settings govern future work; restored totals are not recomputed from batch size.
    learner.config = replace(learner.config, total_iterations=5, num_groups=2, update_epochs=2, seed=99)
    learner.shuffle_key = jax.random.key(99)  # () simulate a restarted learner.
    published_steps.clear()
    writer.reset_mock()
    learner.start()
    assert [progress.iteration for progress in saved] == [2, 3, 4, 5]
    assert [progress.total_rollouts for progress in saved] == [8, 12, 20, 28]
    assert published_steps == [12, 20, 28]
    learner.actor.set_sampling_state.remote.assert_called_once_with({"task_rng": "saved"})
    np.testing.assert_array_equal(jax.random.key_data(learner.shuffle_key), saved_key)
    writer.log_text.assert_called_once_with(yaml.safe_dump(asdict(learner.config)), "config/3.yaml")
    assert [call.kwargs["step"] for call in writer.log_metrics.call_args_list] == [3, 4, 5]
    metrics = writer.log_metrics.call_args.args[0]
    assert metrics["charts/total_rollouts"] == 28
    assert metrics["charts/total_updates"] == 28
    assert metrics["charts/stop_condition_fraction/max_turns"] == 0
    assert metrics["time/checkpoint_seconds"] == 0.5

    # A run already at its cumulative target does not train or overwrite its checkpoint.
    learner.fetch_next_batch.reset_mock()
    learner.start()
    learner.fetch_next_batch.assert_not_called()
    assert len(saved) == 4


def test_learner_logs_examples_every_ten_steps(
    learner: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger=training.__name__)
    rollout = training.Rollout(
        sequence=training.TrainingSequence(
            token_ids=np.array([1, 2], dtype=np.int32),  # (T=2,)
            loss_mask=np.array([False, True]),  # (T=2,)
            old_log_probs=np.array([0.0, -0.2], dtype=np.float32),  # (T=2,)
        ),
        reward=1.0,
        advantage=0.5,
        completion='<think>Use add.</think>\n<tool_call>{"name": "add"}</tool_call>',
        interaction='<|im_start|>assistant\n<think>Use add.</think>\n<tool_call>{"name": "add"}</tool_call><|im_end|>',
        num_turns=1,
        num_input_tokens=1,
        num_output_tokens=1,
        rollout_seconds=0.1,
        stop_condition="max_turns",
    )
    groups = [
        training.RolloutGroup(task_id="task-1", prompt="Add:\n1 + 2", rollouts=[rollout, rollout]),
        training.RolloutGroup(
            task_id=None,
            prompt="Another prompt",
            rollouts=[replace(rollout, interaction="", stop_condition=None), replace(rollout, interaction="omitted")],
        ),
    ]
    learner.config = replace(learner.config, num_groups=2, group_size=2, total_iterations=21)
    learner.initialize_weight_transfer = Mock()
    learner.publish_weights = Mock()
    learner.optimize = Mock(return_value={"losses/policy": 0.0})
    prepared = Mock(spec=training.GRPOBatch)
    prepared.token_ids = Mock(shape=(4, 8))  # (B=4, T=8)
    learner.fetch_next_batch = Mock(
        return_value=training.build_rollout_batch(
            groups=groups,
            training_batch=prepared,
            transfer_metrics={
                "time/batcher/actor_to_batcher_seconds": 0.0,
                "time/learner/batcher_to_learner_seconds": 0.0,
            },
        )
    )
    writer = MagicMock()
    writer.__enter__.return_value = writer
    monkeypatch.setattr(training, "MLflowLogger", Mock(return_value=writer))

    learner.start()

    artifacts = writer.log_text.call_args_list[1:]  # Skip the config artifact.
    assert [call.args[1] for call in artifacts] == ["examples/10.txt", "examples/20.txt"]
    messages = [message for message in caplog.messages if "Rollout examples | step" in message]
    assert len(messages) == 2
    for step, call, message in zip((10, 20), artifacts, messages, strict=True):
        text = call.args[0]
        assert message == f"\n{text}"
        assert text.startswith(f"Rollout examples | step {step}\n\n{'=' * 72}\n")
        assert "Example 1 | task=task-1 | reward=1.000 | advantage=0.500" in text
        assert "Turns: 1 | Output tokens: 1 | Stop: max_turns" in text
        assert "Prompt:\n    Add:\n    1 + 2" in text
        assert (
            "Full interaction (special tokens preserved):\n<|im_start|>assistant\n"
            '<think>Use add.</think>\n<tool_call>{"name": "add"}</tool_call><|im_end|>'
        ) in text
        assert "Example 3 | task=None" in text
        assert "Full interaction (special tokens preserved):\n(empty)" in text
        assert text.count("reward=1.000") == 3
        assert text.count("Token IDs:\n[1, 2]") == 3
        assert "Example 4" not in text and "omitted" not in text


@pytest.mark.parametrize("failure", ["receiver", "publish"])
def test_learner_cleans_up_failed_start(failure: str, learner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    group = Mock()
    monkeypatch.setattr(training, "get_open_port", Mock(return_value=12345))

    def initialize(info: dict[str, Any]) -> Mock:
        learner.actor.initialize_weight_transfer.remote.assert_called_once_with(info)
        assert info == {"master_address": "127.0.0.1", "master_port": 12345, "rank_offset": 1, "world_size": 2}
        return group

    monkeypatch.setattr(training.NCCLWeightTransferEngine, "trainer_init", initialize)
    get = Mock(side_effect=RuntimeError("receiver failed") if failure == "receiver" else None)
    monkeypatch.setattr(training.ray, "get", get)
    writer = MagicMock()
    writer.__enter__.return_value = writer
    monkeypatch.setattr(training, "MLflowLogger", Mock(return_value=writer))
    learner.publish_weights = Mock(side_effect=RuntimeError("publish failed"))
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        learner.start()
    get.assert_called_once_with(learner.actor.initialize_weight_transfer.remote.return_value)
    group.destroy.assert_called_once_with()
    assert learner.publish_weights.call_count == (failure == "publish")
    writer.__exit__.assert_called_once()
    assert writer.__exit__.call_args.args[0] is RuntimeError


@pytest.mark.parametrize("buffer_bytes", [64 * 1024**2, 8], ids=["default-buffer", "oversized-embedding"])
def test_learner_republishes_unchanged_parameters(
    buffer_bytes: int, learner: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    learner.state = training.create_state(
        training.load_qwen3_checkpoint.return_value[0],
        training.load_qwen3_checkpoint.return_value[1]["params"],
        learner.config,
    )
    monkeypatch.setattr(training, "WEIGHT_BUFFER_BYTES", buffer_bytes)
    learner._transfer_group = Mock()
    initial = jax.tree.map(np.asarray, learner.state.params)  # HF parameter shapes: (D,) and (V, D).
    sends: list[dict[str, Any]] = []

    def send(iterator: Any, trainer_args: training.NCCLTrainerSendWeightsArgs) -> None:
        assert learner.actor.update_weights.remote.call_count == len(sends) + 1
        weights = dict(iterator)  # HF parameter shapes.
        info = learner.actor.update_weights.remote.call_args.args[0]
        assert info["names"] == list(weights)
        assert info["shapes"] == [list(weight.shape) for weight in weights.values()]
        assert info["dtype_names"] == ["bfloat16", "bfloat16"]
        assert info["packed"] and trainer_args.packed
        assert info["packed_buffer_size_bytes"] == trainer_args.packed_buffer_size_bytes
        assert trainer_args.packed_buffer_size_bytes == max(buffer_bytes, 12)
        assert trainer_args.group is learner._transfer_group
        for name, weight in weights.items():
            assert tuple(weight.shape) == initial[name].shape
            np.testing.assert_array_equal(weight.float().numpy(), initial[name].astype(np.float32))
        sends.append(info)

    monkeypatch.setattr(training.NCCLWeightTransferEngine, "trainer_send_weights", send)
    synchronize = Mock()
    monkeypatch.setattr(training.torch.cuda, "synchronize", synchronize)

    def acknowledge(pending: Any) -> None:
        assert pending is learner.actor.update_weights.remote.return_value
        assert synchronize.call_count == len(sends)

    monkeypatch.setattr(training.ray, "get", acknowledge)
    learner.publish_weights()
    learner.publish_weights()
    assert len(sends) == 2
    assert sends[0] == sends[1]
    for name, value in learner.state.params.items():  # (D,) or (V, D)
        np.testing.assert_array_equal(value, initial[name])


@pytest.mark.parametrize(
    ("pause_mode", "failure_path"),
    [("wait", None), ("abort", None), ("keep", None), ("wait", "/update_weights"), ("wait", "/finish_weight_update")],
)
def test_actor_weight_update_lifecycle(
    pause_mode: Literal["wait", "abort", "keep"],
    failure_path: str | None,
    actor: Any,
    monkeypatch: pytest.MonkeyPatch,
    spans: InMemorySpanExporter,
) -> None:
    actor.config = replace(actor.config, weight_update_pause_mode=pause_mode)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code=500 if request.url.path == failure_path else 200)

    client = httpx.AsyncClient(base_url=training.SERVER_URL, transport=httpx.MockTransport(respond))
    monkeypatch.setattr(training.httpx, "AsyncClient", Mock(return_value=client))
    update_info = {"names": ["model.norm.weight"]}
    if failure_path:
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(actor.update_weights(update_info))
        with pytest.raises(RuntimeError, match="failed weight update"):
            asyncio.run(actor.fetch_next_group(group_size=2))
    else:
        asyncio.run(actor.update_weights(update_info))
        actor.collect_group = AsyncMock(return_value=Mock(spec=training.RolloutGroup))
        result = asyncio.run(actor.fetch_next_group(group_size=2))
        assert result.value is actor.collect_group.return_value
    paths = ["/pause", "/start_weight_update", "/update_weights", "/finish_weight_update", "/resume"]
    expected = paths if failure_path is None else paths[: paths.index(failure_path) + 1]
    assert [request.url.path for request in requests] == expected
    assert all(request.method == "POST" for request in requests)
    assert dict(requests[0].url.params) == {"mode": pause_mode, "clear_cache": "true"}
    assert json.loads(requests[2].content) == {"update_info": update_info}
    assert client.is_closed
    finished = spans.get_finished_spans()
    parent = next(item for item in finished if item.name == "actor.update_weights")
    phases = [item for item in finished if item.name.startswith("actor.update_weights.")]
    expected_phases = ["pause", "start", "transfer", "finish", "resume"][: len(expected)]
    assert [item.name for item in phases] == [f"actor.update_weights.{name}" for name in expected_phases]
    assert all(item.parent.span_id == parent.context.span_id for item in phases)
    assert phases[0].attributes["mode"] == pause_mode
    assert phases[2].attributes["tensor_count"] == 1
    for index, item in enumerate(phases):
        assert item.status.status_code == (
            otel_trace.StatusCode.ERROR if expected[index] == failure_path else otel_trace.StatusCode.UNSET
        )
    assert parent.status.status_code == (otel_trace.StatusCode.ERROR if failure_path else otel_trace.StatusCode.UNSET)


def test_actor_initializes_weight_receiver(
    actor: Any, monkeypatch: pytest.MonkeyPatch, spans: InMemorySpanExporter
) -> None:
    init_info = {"master_address": "127.0.0.1", "master_port": 12345, "rank_offset": 1, "world_size": 2}

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/init_weight_transfer_engine"
        assert json.loads(request.content) == {"init_info": init_info}
        return httpx.Response(status_code=200)

    client = httpx.AsyncClient(base_url=training.SERVER_URL, transport=httpx.MockTransport(respond))
    monkeypatch.setattr(training.httpx, "AsyncClient", Mock(return_value=client))
    asyncio.run(actor.initialize_weight_transfer(init_info))
    assert client.is_closed
    assert [item.name for item in spans.get_finished_spans()] == ["actor.initialize_weight_transfer"]


@pytest.mark.parametrize(
    ("enable_dashboard", "enable_timeline", "enable_tracing", "fails", "close_fails"),
    [
        (False, False, False, False, False),
        (True, False, False, False, False),
        (False, True, False, False, False),
        (False, False, True, False, False),
        (True, True, True, True, False),
        (False, False, False, False, True),
    ],
    ids=["disabled", "dashboard-only", "timeline-only", "tracing-only", "all-on-failure", "close-failure"],
)
def test_train_gracefully_terminates_actor(
    enable_dashboard: bool,
    enable_timeline: bool,
    enable_tracing: bool,
    fails: bool,
    close_fails: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = training.Config(
        ckpt_dir=str(tmp_path),
        padding_multiple=64,
        enable_ray_dashboard=enable_dashboard,
        enable_ray_timeline=enable_timeline,
        enable_tracing=enable_tracing,
    )
    monkeypatch.delenv("RL2_RAY_TIMELINE_PATH", raising=False)
    monkeypatch.delenv("RL2_RAY_DASHBOARD_HOST", raising=False)
    monkeypatch.setenv("RAY_AUTH_MODE", "token")
    monkeypatch.setenv("RAY_DASHBOARD_ENABLE_PROFILING", "0" if enable_dashboard else "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://tempo:4317")
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    flush = Mock()
    monkeypatch.setattr(training, "flush_tracing", flush)
    if fails or not enable_timeline:
        monkeypatch.setenv("RL2_RAY_TIMELINE_PATH", str(tmp_path / "submitted" / "train-test.json"))
    if fails:
        monkeypatch.setenv("RL2_RAY_DASHBOARD_HOST", "0.0.0.0")
    actor, batcher, learner = Mock(), Mock(), Mock()
    actor.__ray_terminate__ = Mock()
    learner.__ray_terminate__ = Mock()
    batcher.__ray_terminate__ = Mock()
    monkeypatch.setattr(training, "Actor", Mock(remote=Mock(return_value=actor)))
    monkeypatch.setattr(training, "Batcher", Mock(remote=Mock(return_value=batcher)))
    monkeypatch.setattr(training, "Learner", Mock(remote=Mock(return_value=learner)))

    def initialize_ray(**kwargs: Any) -> MagicMock:
        assert os.environ["RAY_DASHBOARD_ENABLE_PROFILING"] == ("1" if enable_dashboard else "0")
        return training.ray.init.return_value

    monkeypatch.setattr(training.ray, "init", MagicMock(side_effect=initialize_ray))
    timeline = Mock(return_value=[{"name": "training"}])
    monkeypatch.setattr(training.ray, "timeline", timeline)

    def get_result(result: Any) -> None:
        if result is actor.aclose.remote.return_value:
            actor.__ray_terminate__.remote.assert_not_called()
            if close_fails:
                raise RuntimeError("agent cleanup failed")

    get = Mock(side_effect=get_result)
    monkeypatch.setattr(training.ray, "get", get)
    wait = Mock()
    monkeypatch.setattr(training.ray, "wait", wait)

    def start_learner() -> Any:
        assert os.environ["RAY_AUTH_MODE"] == "disabled"
        training.Actor.remote.assert_called_once_with(config=config)
        if fails:
            raise RuntimeError("training failed")
        return learner.start.remote.return_value

    learner.start.remote.side_effect = start_learner

    def export() -> list[dict[str, str]]:
        assert training.ray.init.return_value.__exit__.call_count == 0
        return [{"name": "training"}]

    timeline.side_effect = export
    if fails or close_fails:
        with pytest.raises(RuntimeError, match="agent cleanup failed" if close_fails else "training failed"):
            training.train(config)
    else:
        training.train(config)
    training.Actor.remote.assert_called_once_with(config=config)
    training.Batcher.remote.assert_called_once_with(actor=actor, config=config)
    training.Learner.remote.assert_called_once_with(batcher=batcher, actor=actor, config=config)
    options = training.ray.init.call_args.kwargs
    assert options["include_dashboard"] is enable_dashboard
    assert options["dashboard_host"] == ("0.0.0.0" if fails else "127.0.0.1")
    assert options["dashboard_port"] == 8265
    assert options["runtime_env"]["env_vars"] == ({"RAY_PROFILING": "1"} if enable_timeline else {})
    assert options["_system_config"] == {
        "enable_timeline": enable_timeline,
        "task_events_report_interval_ms": 1000,
        "kill_child_processes_on_worker_exit_with_raylet_subreaper": True,
    }
    assert options["_tracing_startup_hook"] == ("rl2.tracing:setup_tracing" if enable_tracing else None)
    traces = list(tmp_path.rglob("*.json"))
    if enable_timeline:
        assert len(traces) == 1
        assert traces[0].parent == tmp_path / ("submitted" if fails else "profiles")
        assert json.loads(traces[0].read_text()) == [{"name": "training"}]
        timeline.assert_called_once_with()
    else:
        assert not traces
        timeline.assert_not_called()
    actor.start.remote.assert_not_called()
    actor.aclose.remote.assert_called_once_with()
    actor.__ray_terminate__.remote.assert_called_once_with()
    learner.__ray_terminate__.remote.assert_called_once_with()
    batcher.__ray_terminate__.remote.assert_called_once_with()
    flush.assert_called_once_with()
    wait.assert_called_once_with(
        [
            learner.__ray_terminate__.remote.return_value,
            actor.__ray_terminate__.remote.return_value,
            batcher.__ray_terminate__.remote.return_value,
        ],
        num_returns=3,
    )
    if fails:
        get.assert_called_once_with(actor.aclose.remote.return_value)
    else:
        assert [call.args[0] for call in get.call_args_list] == [
            learner.start.remote.return_value,
            actor.aclose.remote.return_value,
        ]


@pytest.mark.parametrize("size", [0, 1])
def test_group_requires_multiple_rollouts(size: int, actor: Any) -> None:
    with pytest.raises(ValueError, match="group_size"):
        asyncio.run(actor.fetch_next_group(group_size=size))


@pytest.mark.parametrize(
    "rewards",
    [[-1.0, 0.0, 2.0], [0.7] * 8, [1.0, 1.0 + 1e-8]],
    ids=["population-std", "fractional-ties", "differences-below-float32-precision"],
)
def test_score_group_normalizes_rewards(
    rewards: list[float], rollout: training.Rollout, actor: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = actor.tasks[0]
    rollouts = [replace(rollout, reward=reward) for reward in rewards]
    group = training.score_group(task, rollouts)
    assert group.task_id == f"{task.name}:{task.id}"
    assert group.prompt == task.prompt
    expected = [(reward - mean(rewards)) / (pstdev(rewards) + 1e-4) for reward in rewards]
    assert len(group.rollouts) == len(rewards)
    assert [rollout.reward for rollout in group.rollouts] == rewards
    assert [rollout.advantage for rollout in group.rollouts] == pytest.approx(expected, rel=1e-6, abs=1e-7)
    if len(set(rewards)) == 1:
        assert all(rollout.advantage == 0.0 for rollout in group.rollouts)


def test_batcher_collects_groups_concurrently(
    actor: Any, rollout: training.Rollout, monkeypatch: pytest.MonkeyPatch, spans: InMemorySpanExporter
) -> None:
    # Actor scoring, transcript construction, and batching must work without JAX.
    monkeypatch.setattr(training, "jax", Mock(spec=[]))
    monkeypatch.setattr(training, "jnp", Mock(spec=[]))
    all_started = asyncio.Event()
    started = 0
    env = AsyncMock()
    env.__aenter__.return_value = env

    async def run(task: training.ALFWorldTask) -> training.Rollout:
        nonlocal started
        before = otel_trace.get_current_span().get_span_context()
        started += 1
        if started == 4:
            all_started.set()
        # A serialized actor would wait forever for the second group's rollouts.
        await all_started.wait()
        assert otel_trace.get_current_span().get_span_context() == before
        return rollout

    env.run.side_effect = run
    monkeypatch.setattr(training, "create_agent", Mock(return_value=env))
    monkeypatch.setattr(actor.rng, "choice", Mock(side_effect=actor.tasks[:2]))
    handle = Mock()
    handle.fetch_next_group.remote.side_effect = actor.fetch_next_group
    batcher = training.Batcher.__ray_metadata__.modified_class(actor=handle)

    async def collect() -> training.Transfer[training.RolloutBatch]:
        async with aclosing(actor):
            with training.span("test.batch") as parent:
                async with asyncio.timeout(5):
                    result = await batcher.fetch_next_batch(num_groups=2, group_size=2, context_length=16)
                assert otel_trace.get_current_span() is parent
                return result

    batch = asyncio.run(collect()).value
    assert started == 4
    assert f"Example 1 | task=alfworld:{actor.tasks[0].id}" in batch.examples
    assert f"Example 3 | task=alfworld:{actor.tasks[1].id}" in batch.examples
    assert batch.training_batch.token_ids.shape == (4, 16)
    assert all(isinstance(value, np.ndarray) for value in batch.training_batch)
    env.__aenter__.assert_awaited_once()
    env.__aexit__.assert_awaited_once()
    finished = spans.get_finished_spans()
    groups = {item.attributes["group_id"]: item for item in finished if item.name == "actor.collect_group"}
    rollouts = [item for item in finished if item.name == "actor.rollout"]
    root = next(item for item in finished if item.name == "test.batch")
    assert len(groups) == 2 and len(rollouts) == 4
    assert all(item.parent.span_id == root.context.span_id for item in groups.values())
    assert len({item.context.span_id for item in rollouts}) == 4
    for item in rollouts:
        parent = groups[item.attributes["group_id"]]
        assert item.parent.span_id == parent.context.span_id
        assert item.context.trace_id == parent.context.trace_id
    assert not otel_trace.get_current_span().get_span_context().is_valid


def test_group_cancels_pending_rollouts(
    actor: Any, monkeypatch: pytest.MonkeyPatch, spans: InMemorySpanExporter
) -> None:
    env = AsyncMock()
    env.__aenter__.return_value = env
    started = asyncio.Event()
    cancelled: list[bool] = []
    count = 0

    async def run(task: training.ALFWorldTask) -> training.Rollout:
        nonlocal count
        count += 1
        if count == 1:
            await started.wait()
            raise ValueError("generation failed")
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.append(True)

    env.run.side_effect = run
    monkeypatch.setattr(training, "create_agent", Mock(return_value=env))

    async def collect() -> None:
        async with aclosing(actor):
            with training.span("test.parent") as parent:
                try:
                    await actor.collect_group(actor.tasks[0], group_size=2)
                except ExceptionGroup:
                    assert otel_trace.get_current_span() is parent
                    raise

    with pytest.raises(ExceptionGroup, match="TaskGroup"):
        asyncio.run(collect())
    assert cancelled == [True]
    env.__aexit__.assert_awaited_once()
    finished = spans.get_finished_spans()
    assert len(finished) == 4  # Both rollouts, their group, and the caller end.
    assert all(item.status.status_code == otel_trace.StatusCode.ERROR for item in finished)
    assert sum(item.attributes.get("cancelled", False) for item in finished) == 1
    assert not otel_trace.get_current_span().get_span_context().is_valid


@pytest.mark.parametrize("fails", [False, True])
def test_create_agent_config_and_cleanup(fails: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    config = training.Config(model="Qwen/custom", max_tokens=16, max_turns=2, temperature=0.7, rollout_timeout=300)
    tokenizer = Mock()
    load = Mock(return_value=tokenizer)
    monkeypatch.setattr(training.AutoTokenizer, "from_pretrained", load)

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            500 if fails else 200,
            json={
                "object": "list",
                "data": [
                    {"id": config.model, "object": "model", "created": 0, "owned_by": "vllm", "max_model_len": 4096}
                ],
            },
        )

    client = AsyncOpenAI(
        base_url=f"{training.SERVER_URL}/v1",
        api_key="EMPTY",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    factory = Mock(return_value=client)
    monkeypatch.setattr(training, "AsyncOpenAI", factory)

    async def run() -> None:
        async with training.create_agent(config) as agent:
            assert agent.tokenizer is tokenizer and agent.client is client
            assert agent.model == config.model
            assert agent.context_length == 4096
            assert agent.max_tokens == 16 and agent.max_turns == 2
            assert agent.temperature == 0.7 and agent.rollout_timeout == 300
            assert agent.environment_timeout == config.environment_timeout

    if fails:
        with pytest.raises(APIStatusError):
            asyncio.run(run())
    else:
        asyncio.run(run())
    load.assert_called_once_with(config.model)
    factory.assert_called_once_with(
        base_url=f"{training.SERVER_URL}/v1", api_key="EMPTY", timeout=config.rollout_timeout, max_retries=0
    )
    assert client.is_closed()
