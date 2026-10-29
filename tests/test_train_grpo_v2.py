import asyncio
import io
import json
import logging
import os
import pickle
import random
import runpy
import subprocess
import sys
from collections.abc import Generator, Iterator
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
from rl2.agent_task import Environment
from rl2.alfworld_env import ALFWorldTask
from rl2.qwen3 import ATTENTION_BACKENDS, AttentionBackend


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel_trace, "get_tracer", Mock(return_value=provider.get_tracer("test")))
    with ExitStack() as resources:
        resources.callback(provider.shutdown)
        yield exporter


@pytest.mark.parametrize("backend", ATTENTION_BACKENDS)
def test_attention_backend_config(backend: AttentionBackend) -> None:
    assert training.Config(attention_backend=backend).attention_backend == backend


def test_config_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    defaults = training.Config()
    assert defaults.finetuning_mode == "lora"
    assert defaults.minibatches_per_epoch == 4
    assert defaults.grad_accum_steps == 1
    assert defaults.prefetch_batches == 0
    assert not defaults.drop_degenerate_groups
    assert defaults.length_penalty_coef == 0.0
    assert defaults.update_epochs == 1
    assert defaults.checkpoint_interval == 100
    assert defaults.padding_multiple == 256
    assert defaults.attention_backend == "cudnn"
    assert training.Qwen3LM().dtype == "bfloat16"
    assert defaults.scan_layers
    assert defaults.gradient_checkpointing
    assert not defaults.enable_ray_dashboard
    assert not defaults.enable_ray_timeline
    assert defaults.enable_tracing
    assert defaults.weight_update_pause_mode == "wait"
    config = training.Config(run_id="test_full", ckpt_dir="/runs/test_full")
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            asdict(config)
            | {
                "run_id": "test_${finetuning_mode}",
                "ckpt_dir": "/runs/${run_id}",
                "max_tokens": "${group_size}",
                "total_iterations": 1,
                "attention_backend": "xla",
                "scan_layers": False,
                "gradient_checkpointing": False,
                "enable_ray_dashboard": True,
                "enable_ray_timeline": True,
                "enable_tracing": False,
                "padding_multiple": 64,
                "drop_degenerate_groups": False,
                "length_penalty_coef": 0.01,
                "rollout_is_threshold": 3.0,
                "prefetch_batches": 3,
                "finetuning_mode": "full",
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
            max_tokens=config.group_size,
            total_iterations=1,
            attention_backend="xla",
            scan_layers=False,
            gradient_checkpointing=False,
            enable_ray_dashboard=True,
            enable_ray_timeline=True,
            enable_tracing=False,
            padding_multiple=64,
            drop_degenerate_groups=False,
            length_penalty_coef=0.01,
            rollout_is_threshold=3.0,
            prefetch_batches=3,
            finetuning_mode="full",
        )
    )
    assert pickle.loads(pickle.dumps(config)) == config


def test_example_log_interval_config(tmp_path: Path) -> None:
    assert training.Config().example_log_interval == 10
    path = tmp_path / "config.yaml"
    path.write_text("example_log_interval: 1\n")
    assert training.load_config(path, training.Config).example_log_interval == 1


def test_appworld_actor_task_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    config = training.Config(taskset=training.AppWorldConfig(python="/test/python", root="/test/appworld"))
    assert pickle.loads(pickle.dumps(config)) == config
    task = Mock(spec=training.Task)
    stream = iter([task])
    stream_tasks = Mock(return_value=stream)
    monkeypatch.setattr(training, "stream_appworld_tasks", stream_tasks)
    monkeypatch.setattr(training, "serve_vllm", Mock(return_value=MagicMock()))
    monkeypatch.setattr(training, "wait_for_server", Mock())
    actor = training.Actor.__ray_metadata__.modified_class(config)

    with actor._server:
        assert actor.tasks is stream
        assert next(actor.tasks) is task
    stream_tasks.assert_called_once_with(config.taskset, actor.rng, config.environment_timeout)


def test_reasoning_gym_cli_and_actor_task_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = training.Config(
        taskset=training.ReasoningGymConfig(task="leg_counting", task_config={"min_animals": 1, "max_animals": 5}),
        run_id="test_reasoning_gym",
        ckpt_dir=str(tmp_path),
    )
    assert pickle.loads(pickle.dumps(config)) == config
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(asdict(config)))
    train = Mock()
    monkeypatch.setattr(training, "train", train)
    monkeypatch.setattr(sys, "argv", ["train_grpo_v2", "--config", str(path)])
    training.main()
    train.assert_called_once_with(config)

    monkeypatch.setattr(training, "serve_vllm", Mock(return_value=MagicMock()))
    monkeypatch.setattr(training, "wait_for_server", Mock())
    actor_class = training.Actor.__ray_metadata__.modified_class
    actor = actor_class(config)
    restored = actor_class(config)
    actor.collect_group = AsyncMock(return_value=Mock(spec=training.RolloutGroup))
    restored.collect_group = AsyncMock(return_value=Mock(spec=training.RolloutGroup))

    async def fetch_groups(instance: Any) -> None:
        async with asyncio.TaskGroup() as pending:
            for _ in range(3):
                pending.create_task(instance.fetch_next_group(group_size=2))

    with actor._server, restored._server:
        assert iter(actor.tasks) is actor.tasks
        asyncio.run(actor.fetch_next_group(group_size=2))
        first = actor.collect_group.call_args.args[0]
        state = json.loads(json.dumps(actor.get_sampling_state()))
        actor.collect_group.reset_mock()
        asyncio.run(fetch_groups(actor))
        expected = actor.collect_group.call_args_list
        tasks = [call.args[0] for call in expected]
        assert len({first.id, *(task.id for task in tasks)}) == 4
        assert all(task.config == config.taskset for task in tasks)
        assert all(call.kwargs == {"group_size": 2} for call in expected)
        restored.set_sampling_state(state)
        asyncio.run(fetch_groups(restored))
        assert restored.collect_group.call_args_list == expected
        assert restored.get_sampling_state() == actor.get_sampling_state()


@pytest.mark.parametrize(
    "settings",
    [
        {"num_groups": 0},
        {"group_size": 1},
        {"prefetch_batches": -1},
        {"prefetch_batches": 1.5},
        {"total_iterations": 0},
        {"checkpoint_interval": 0},
        {"checkpoint_interval": 1.5},
        {"example_log_interval": 0},
        {"example_log_interval": 1.5},
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
        {"length_penalty_coef": -0.01},
        {"clip_coef": 1.0},
        {"rollout_is_threshold": 0.0},
        {"max_grad_norm": 0.0},
        {"minibatches_per_epoch": 0},
        {"minibatches_per_epoch": 3},
        {"minibatches_per_epoch": 8},
        {"grad_accum_steps": 0},
        {"grad_accum_steps": 3},
        {"minibatches_per_epoch": 2, "grad_accum_steps": 4},
        {"update_epochs": 0},
        {"attention_backend": "unknown"},
        {"weight_update_pause_mode": "unknown"},
        pytest.param({"weight_update_pause_mode": "abort"}, id="removed-abort-mode"),
        {"taskset": "unknown"},
    ],
)
def test_config_rejects_invalid_settings(settings: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        training.Config(**settings)


@pytest.mark.parametrize(
    ("settings", "error"),
    [
        ({"taskset": {"id": "addition", "config_file": "config.yaml"}}, "union_tag_invalid"),
        ({"taskset": {"id": "alfworld", "config_file": None}}, "string_type"),
        ({"taskset": {"id": "alfworld", "config_file": "config.yaml", "unknown": True}}, "unexpected_keyword_argument"),
        ({"dataset_file": "train.parquet"}, "unexpected_keyword_argument"),
    ],
)
def test_taskset_config_rejects_invalid_yaml(settings: dict[str, object], error: str, tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValidationError, match=error) as raised:
        training.load_config(path, training.Config)
    assert len(raised.value.errors()) == 1


def test_explicit_taskset_requires_discriminator() -> None:
    with pytest.raises(ValidationError) as raised:
        training.Config(taskset={"config_file": "config.yaml"})
    assert [(error["loc"], error["type"]) for error in raised.value.errors()] == [(("taskset",), "union_tag_not_found")]


@pytest.mark.parametrize("temperature", [0.01, 2.0])
def test_sampling_temperature_bounds(temperature: float) -> None:
    assert training.Config(temperature=temperature).temperature == temperature


@pytest.mark.parametrize("failure", [None, "launch", "rollout", "pod-network", "full"])
def test_serve_vllm(failure: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RL2_VLLM_HOST", raising=False)
    if failure == "pod-network":
        monkeypatch.setenv("RL2_VLLM_HOST", "0.0.0.0")
    command = [
        sys.executable,
        "-m",
        "rl2.vllm_support.server",
        "--model",
        "Qwen/Qwen3-0.6B",
        "--host",
        "0.0.0.0" if failure == "pod-network" else "127.0.0.1",
        "--port",
        "8000",
        "--disable-uvicorn-access-log",
        "--dtype",
        "bfloat16",
        "--hf-overrides",
        '{"head_dtype":"float32"}',
        "--generation-config",
        "vllm",
        "--logprobs-mode",
        "processed_logprobs",
        "--enable-lora",
        "--max-lora-rank",
        "16",
        "--lora-dtype",
        "bfloat16",
    ]
    if failure == "full":
        command = command[: command.index("--enable-lora")] + ["--weight-transfer-config", '{"backend":"nccl"}']
    server = MagicMock()
    server.__enter__.return_value = server

    def exit_server(*args: Any) -> None:
        server.terminate.assert_called_once_with()

    server.__exit__.side_effect = exit_server

    def start_server(command: list[str], *, env: dict[str, str]) -> MagicMock:
        if failure == "launch":
            raise OSError("launch failed")
        return server

    launch = Mock(side_effect=start_server)
    monkeypatch.setattr(training.subprocess, "Popen", launch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    if failure in {"launch", "rollout"}:
        with pytest.raises(OSError if failure == "launch" else RuntimeError), training.serve_vllm():
            raise RuntimeError("rollout failed")
    else:
        with training.serve_vllm(lora_rank=None if failure == "full" else 16) as process:
            assert process is server
            server.terminate.assert_not_called()

    launch.assert_called_once()
    assert launch.call_args.args == (command,)
    assert launch.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert launch.call_args.kwargs["env"]["VLLM_SERVER_DEV_MODE"] == "1"
    assert launch.call_args.kwargs["env"]["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "True"
    assert launch.call_args.kwargs["env"] == {
        **os.environ,
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
    }
    if failure == "launch":
        server.terminate.assert_not_called()
    else:
        server.terminate.assert_called_once_with()
        server.__exit__.assert_called_once()
        server.kill.assert_not_called()


@pytest.mark.parametrize(("rank", "maximum"), [(17, 32), (512, 512), (513, None)])
def test_serve_vllm_lora_rank(rank: int, maximum: int | None, monkeypatch: pytest.MonkeyPatch) -> None:
    launch = Mock(return_value=MagicMock())
    monkeypatch.setattr(training.subprocess, "Popen", launch)
    if maximum is None:
        with pytest.raises(ValueError, match="LoRA ranks up to 512"), training.serve_vllm(lora_rank=rank):
            pytest.fail("Unsupported rank should fail before server startup")
        launch.assert_not_called()
    else:
        with training.serve_vllm(lora_rank=rank):
            pass
        command = launch.call_args.args[0]
        assert command[command.index("--max-lora-rank") + 1] == str(maximum)


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
                "from types import SimpleNamespace\n"
                "import jax\n"
                "import jax.numpy as jnp\n"
                "import numpy as np\n"
                "from ray import cloudpickle\n"
                "actors = cloudpickle.loads(sys.stdin.buffer.read())\n"
                "assert all(actor.__name__ == name for name, actor in actors.items())\n"
                "create_state = actors['Learner'].start.__wrapped__.__globals__['create_state']\n"
                "config = create_state.__annotations__['config'](lora_rank=1, lora_alpha=3)\n"
                "name = 'model.layers.0.self_attn.q_proj.weight'\n"
                "params = {name: jnp.ones((2, 3), dtype=jnp.bfloat16)}  # (O=2, I=3)\n"
                "state = create_state(SimpleNamespace(apply=None), params, config)\n"
                "mesh = jax.sharding.Mesh(jax.devices()[:1], ('batch',))\n"
                "sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())\n"
                "placed = jax.device_put(state, sharding)\n"
                "restored = jax.jit(lambda value: value)(placed)\n"
                "assert restored.lora_scale == 3\n"
                "assert restored.base_params[name].shape == (2, 3)\n"
                "assert restored.params[name].a.shape == (1, 3)\n"
                "assert restored.params[name].b.shape == (2, 1)\n"
                "for actual, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(state), strict=True):\n"
                "    assert actual.shape == expected.shape\n"
                "    np.testing.assert_array_equal(actual, expected)\n"
            ),
        ],
        input=cloudpickle.dumps(actors),
        env=os.environ | {"JAX_PLATFORMS": "cpu"},
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
def task_discovery(monkeypatch: pytest.MonkeyPatch) -> list[ALFWorldTask]:
    tasks = [ALFWorldTask(id=f"train/{index}", config_file="config.yaml") for index in range(2)]

    def stream(config: training.ALFWorldConfig, rng: random.Random, timeout: float) -> Generator[ALFWorldTask]:
        while True:
            yield rng.choice(tasks)

    monkeypatch.setattr(training, "stream_alfworld_tasks", Mock(side_effect=stream))
    return tasks


@pytest.fixture(autouse=True)
def httpx_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(logging.getLogger("httpx"), "level", logging.INFO)


def test_actor_suppresses_http_request_logs(actor: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200)

    async def request() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            response = await client.post(f"{training.SERVER_URL}/v1/completions")
            assert response.status_code == 200

    asyncio.run(request())
    logging.getLogger("httpx").warning("HTTP warning")
    logging.getLogger("httpx").error("HTTP error")
    training.logger.info("Training progress")
    assert "HTTP Request:" not in caplog.text
    assert "HTTP warning" in caplog.text
    assert "HTTP error" in caplog.text
    assert "Training progress" in caplog.text


@pytest.fixture
def actor(monkeypatch: pytest.MonkeyPatch, task_discovery: list[ALFWorldTask]) -> Any:
    def init_env() -> AsyncMock:
        environment = AsyncMock()
        environment.__aenter__.return_value = environment
        return environment

    monkeypatch.setattr(ALFWorldTask, "init_env", Mock(side_effect=init_env))
    monkeypatch.setattr(training, "serve_vllm", Mock(return_value=MagicMock()))
    monkeypatch.setattr(training, "wait_for_server", Mock())
    actor = training.Actor.__ray_metadata__.modified_class()
    monkeypatch.setattr(actor.rng, "choice", Mock(return_value=task_discovery[0]))
    return actor


async def collect_and_close(actor: Any, task: ALFWorldTask) -> training.RolloutGroup:
    async with aclosing(actor):
        return await actor.collect_group(task, group_size=2)


def test_actor_fetch_next_group(
    actor: Any, monkeypatch: pytest.MonkeyPatch, task_discovery: list[ALFWorldTask]
) -> None:
    collect = AsyncMock(return_value=Mock(spec=training.RolloutGroup))
    monkeypatch.setattr(actor, "collect_group", collect)

    result = asyncio.run(actor.fetch_next_group(group_size=2))

    training.stream_alfworld_tasks.assert_called_once_with(
        actor.config.taskset, actor.rng, actor.config.environment_timeout
    )
    actor.rng.choice.assert_called_once_with(task_discovery)
    collect.assert_awaited_once_with(task_discovery[0], group_size=2)
    assert result.value is collect.return_value


def test_actor_reuses_agent(
    actor: Any, rollout: training.Rollout, monkeypatch: pytest.MonkeyPatch, task_discovery: list[ALFWorldTask]
) -> None:
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
            first = asyncio.create_task(actor.collect_group(task_discovery[0], group_size=2))
            await entering.wait()
            second = asyncio.create_task(actor.collect_group(task_discovery[1], group_size=2))
            await asyncio.sleep(0)
            create.assert_called_once_with(actor.config)
            release.set()
            groups = await asyncio.gather(first, second)
            assert [group.task_id for group in groups] == [f"alfworld:{task.id}" for task in task_discovery[:2]]
            await actor.collect_group(task_discovery[0], group_size=2)
            agent.__aexit__.assert_not_awaited()
        await actor.aclose()
        with pytest.raises(RuntimeError, match="Actor is closed"):
            await actor.collect_group(task_discovery[0], group_size=2)

    asyncio.run(collect())
    create.assert_called_once_with(actor.config)
    agent.__aenter__.assert_awaited_once()
    agent.__aexit__.assert_awaited_once()
    assert agent.run.await_count == 6
    assert ALFWorldTask.init_env.call_count == 6
    environments = [call.kwargs["env"] for call in agent.run.await_args_list]
    assert len({id(env) for env in environments}) == 6
    for call, env in zip(agent.run.await_args_list, environments, strict=True):
        assert call.kwargs == {"env": env}
        env.__aenter__.assert_awaited_once()
        env.__aexit__.assert_awaited_once()


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


def test_actor_close_cancels_rollouts(
    actor: Any, monkeypatch: pytest.MonkeyPatch, task_discovery: list[ALFWorldTask]
) -> None:
    agent = AsyncMock()
    agent.__aenter__.return_value = agent
    started = asyncio.Event()
    active = 0

    async def run(env: Environment) -> training.Rollout:
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
            group = asyncio.create_task(actor.collect_group(task_discovery[0], group_size=2))
            await started.wait()
            await actor.aclose()
            with pytest.raises(asyncio.CancelledError):
                await group

    asyncio.run(collect())
    assert active == 0
    agent.__aexit__.assert_awaited_once()


@pytest.mark.parametrize("startup_fails,mode", [(False, "lora"), (True, "lora"), (False, "full")])
def test_actor_server_lifetime(
    startup_fails: bool,
    mode: Literal["lora", "full"],
    monkeypatch: pytest.MonkeyPatch,
    task_discovery: list[ALFWorldTask],
) -> None:
    config = training.Config(model="Qwen/custom", finetuning_mode=mode)
    server = MagicMock()
    serve = Mock(return_value=server)
    monkeypatch.setattr(training, "serve_vllm", serve)
    ready = Mock(side_effect=TimeoutError("not ready") if startup_fails else None)
    monkeypatch.setattr(training, "wait_for_server", ready)
    group = training.RolloutGroup(task_id="0", rollouts=[])

    if startup_fails:
        actor_class = training.Actor.__ray_metadata__.modified_class
        actor = actor_class.__new__(actor_class)
        with pytest.raises(TimeoutError, match="not ready"):
            actor.__init__(config=config)
        asyncio.run(actor.aclose())
    else:
        actor = training.Actor.__ray_metadata__.modified_class(config=config)
        assert actor.ready() is None
        actor.collect_group = AsyncMock(return_value=group)
        monkeypatch.setattr(actor.rng, "choice", Mock(return_value=task_discovery[0]))
        for _ in range(2):
            assert asyncio.run(actor.fetch_next_group(group_size=2)).value is group
        actor.collect_group.assert_awaited_with(task_discovery[0], group_size=2)
        assert actor.collect_group.await_count == 2
        server.__exit__.assert_not_called()
        asyncio.run(actor.aclose())
        asyncio.run(actor.aclose())
    serve.assert_called_once_with(model=config.model, lora_rank=config.lora_rank if mode == "lora" else None)
    ready.assert_called_once_with(server.__enter__.return_value)
    server.__enter__.assert_called_once()
    server.__exit__.assert_called_once()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_actor_close_releases_server_after_agent(
    cleanup_fails: bool, actor: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = AsyncMock()
    agent.__aenter__.return_value = agent
    if cleanup_fails:
        agent.__aexit__.side_effect = RuntimeError("agent cleanup failed")
    monkeypatch.setattr(training, "create_agent", Mock(return_value=agent))

    def close_server() -> None:
        agent.__aexit__.assert_awaited_once()

    close = Mock(side_effect=close_server)
    actor._server.callback(close)

    async def run() -> None:
        await actor._get_agent()
        if cleanup_fails:
            with pytest.raises(RuntimeError, match="agent cleanup failed"):
                await actor.aclose()
        else:
            await actor.aclose()
        close.assert_called_once_with()
        await actor.aclose()

    asyncio.run(run())
    close.assert_called_once_with()


@pytest.fixture
def rollout() -> training.Rollout:
    return training.Rollout(
        prompt="Environment prompt",
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


@pytest.mark.parametrize(("num_groups", "fails", "iteration"), [(1, False, 1), (2, False, 10), (2, True, 1)])
def test_learner_pulls_groups_through_batcher(
    num_groups: int,
    fails: bool,
    iteration: int,
    learner: Any,
    rollout: training.Rollout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = Mock()
    learner.config = replace(learner.config, num_groups=num_groups, group_size=3, minibatches_per_epoch=1)
    groups = [training.RolloutGroup(task_id=str(i), rollouts=[rollout] * 3) for i in range(num_groups)]
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
    format_examples = Mock(wraps=training.format_rollout_examples)
    monkeypatch.setattr(training, "format_rollout_examples", format_examples)

    if fails:
        with pytest.raises(ExceptionGroup) as error:
            learner.fetch_next_batch(iteration=iteration)
        assert len(error.value.exceptions) == 1
        assert isinstance(error.value.exceptions[0], RuntimeError)
        assert str(error.value.exceptions[0]) == "rollout failed"
    else:
        batch = learner.fetch_next_batch(iteration=iteration)
        assert not hasattr(batch, "groups")
        assert batch.stop_counts == {"agent_completed": num_groups * 3}
        assert batch.num_output_tokens == num_groups * 3
        assert batch.examples.count("Full token sequence (special tokens preserved):") == num_groups * 3
        format_examples.assert_called_once()
        assert batch.training_batch.token_ids.shape == (num_groups * 3, 8)
        np.testing.assert_array_equal(batch.training_batch.advantages, 0.5)
    batcher_handle.fetch_next_batch.remote.assert_called_once_with(
        num_groups=num_groups, group_size=3, context_length=8, include_examples=True
    )
    assert actor.fetch_next_group.remote.call_count == num_groups
    assert all(call.kwargs == {"group_size": 3} for call in actor.fetch_next_group.remote.call_args_list)


@pytest.mark.parametrize(
    ("prefetch_batches", "checkpoint_interval", "total_iterations"),
    [(0, 100, 9), (3, 100, 12), (3, 12, None)],
    ids=["disabled", "final-iteration", "checkpoint"],
)
def test_learner_prefetches_bounded_batches_in_order(
    learner: Any,
    prefetch_batches: int,
    checkpoint_interval: int,
    total_iterations: int | None,
    rollout: training.Rollout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner.config = replace(
        learner.config,
        num_groups=1,
        group_size=2,
        minibatches_per_epoch=1,
        prefetch_batches=prefetch_batches,
        checkpoint_interval=checkpoint_interval,
        total_iterations=total_iterations,
    )
    batch = training.build_rollout_batch(
        groups=[training.RolloutGroup(task_id="0", rollouts=[rollout] * 2)],
        training_batch=Mock(spec=training.GRPOBatch, attention_mask=np.ones((2, 2), dtype=np.bool_)),  # (B, T)
        transfer_metrics={},
    )
    results = [
        training.Transfer(value=replace(batch, examples=str(i)), ready_at=0.0) for i in range(prefetch_batches + 2)
    ]
    submit = learner.batcher.fetch_next_batch.remote
    submit.side_effect = results

    def wait_next(refs: list[Any], *, num_returns: int) -> tuple[list[Any], list[Any]]:
        assert num_returns == 1
        # Later submissions are ready first, but must not overtake the oldest.
        return refs[-1:], refs[:-1]

    wait = Mock(side_effect=wait_next)
    monkeypatch.setattr(training.ray, "wait", wait)
    get = Mock(side_effect=lambda value: value)
    monkeypatch.setattr(training.ray, "get", get)
    cancel = Mock()
    monkeypatch.setattr(training.ray, "cancel", cancel)

    first = learner.fetch_next_batch(iteration=8)
    assert first.examples == "0"
    assert submit.call_count == prefetch_batches + 1
    get.assert_called_once_with(results[0])
    second = learner.fetch_next_batch(iteration=9)
    assert second.examples == "1"
    assert submit.call_count == prefetch_batches + 2
    for index in range(2, len(results)):
        assert learner.fetch_next_batch(iteration=8 + index).examples == str(index)
    assert submit.call_count == len(results)
    assert all(call.kwargs["include_examples"] for call in submit.call_args_list)
    assert [call.args[0] for call in get.call_args_list] == results
    wait.assert_not_called()
    learner.cancel_pending_batches()
    cancel.assert_not_called()


def test_learner_cancels_prefetch_on_update_failure(
    learner: Any, rollout: training.Rollout, monkeypatch: pytest.MonkeyPatch
) -> None:
    learner.config = replace(learner.config, total_iterations=4, prefetch_batches=3)
    learner.publish_weights = Mock()
    batch = training.build_rollout_batch(
        groups=[training.RolloutGroup(task_id="0", rollouts=[rollout] * 4)],
        training_batch=Mock(spec=training.GRPOBatch, attention_mask=np.ones((4, 2), dtype=np.bool_)),  # (B, T)
        transfer_metrics={},
    )
    first = training.Transfer(value=batch, ready_at=0.0)
    pending = [Mock() for _ in range(3)]
    learner.batcher.fetch_next_batch.remote.side_effect = [first, *pending]
    monkeypatch.setattr(training.ray, "get", lambda value: value)
    cancel = Mock()
    monkeypatch.setattr(training.ray, "cancel", cancel)
    monkeypatch.setattr(training, "MLflowLogger", MagicMock())

    def optimize(batch: training.GRPOBatch[np.ndarray]) -> dict[str, float]:
        assert learner.batcher.fetch_next_batch.remote.call_count == 4
        raise RuntimeError("update failed")

    learner.optimize = optimize
    with pytest.raises(RuntimeError, match="update failed"):
        learner.start()
    assert [call.args[0] for call in cancel.call_args_list] == pending
    assert all(call.kwargs == {"recursive": True} for call in cancel.call_args_list)
    learner.cancel_pending_batches()
    assert cancel.call_count == len(pending)


@pytest.mark.parametrize("drop_degenerate_groups", [False, True])
def test_batch_refills_degenerate_groups(drop_degenerate_groups: bool, rollout: training.Rollout) -> None:
    groups = [
        training.score_group(Mock(id=str(index)), [replace(rollout, reward=reward) for reward in rewards])
        for index, rewards in enumerate(((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)))
    ]
    actor = Mock()
    actor.fetch_next_group.remote = AsyncMock(
        side_effect=[training.Transfer(value=group, ready_at=training.time.monotonic()) for group in groups]
    )
    batcher = training.Batcher.__ray_metadata__.modified_class(
        actor=actor, config=training.Config(drop_degenerate_groups=drop_degenerate_groups)
    )

    result = asyncio.run(batcher.fetch_next_batch(num_groups=2, group_size=2, context_length=8, include_examples=True))

    batch = result.value
    expected_groups = groups[2:] if drop_degenerate_groups else groups[:2]
    assert actor.fetch_next_group.remote.await_count == (4 if drop_degenerate_groups else 2)
    assert all(call.kwargs == {"group_size": 2} for call in actor.fetch_next_group.remote.call_args_list)
    assert batch.training_batch.token_ids.shape == (4, 8)
    assert batch.training_batch.advantages.shape == (4,)
    np.testing.assert_allclose(
        batch.training_batch.advantages,
        [rollout.advantage for group in expected_groups for rollout in group.rollouts],
    )
    assert batch.examples == training.format_rollout_examples(expected_groups)
    assert batch.rollout_metrics["rollout/degenerate_group_fraction"] == (0.0 if drop_degenerate_groups else 1.0)
    assert batch.num_output_tokens == 4
    assert batch.stop_counts == {"agent_completed": 4}


def test_batch_requires_groups() -> None:
    actor = Mock()
    batcher = training.Batcher.__ray_metadata__.modified_class(actor=actor)
    with pytest.raises(ValueError, match="num_groups"):
        asyncio.run(batcher.fetch_next_batch(num_groups=0, group_size=2, context_length=8))
    actor.fetch_next_group.remote.assert_not_called()


@pytest.mark.parametrize("drop_degenerate_groups", [False, True])
def test_batch_failure_cancels_pending_fetches(drop_degenerate_groups: bool, rollout: training.Rollout) -> None:
    started = asyncio.Event()
    cleaned_up = asyncio.Event()
    requested = 0

    async def fetch_group(group_size: int) -> training.Transfer[training.RolloutGroup]:
        nonlocal requested
        requested += 1
        if drop_degenerate_groups and requested == 1:
            return training.Transfer(
                value=training.RolloutGroup(task_id="degenerate", rollouts=[rollout] * 2),
                ready_at=training.time.monotonic(),
            )
        if requested == 1 + drop_degenerate_groups:
            await started.wait()
            raise RuntimeError("rollout failed")
        started.set()
        try:
            return await asyncio.Future[training.Transfer[training.RolloutGroup]]()
        finally:
            cleaned_up.set()

    actor = Mock()
    actor.fetch_next_group.remote.side_effect = fetch_group
    batcher = training.Batcher.__ray_metadata__.modified_class(
        actor=actor, config=training.Config(drop_degenerate_groups=drop_degenerate_groups)
    )
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
    assert requested == 2 + drop_degenerate_groups
    batcher.prepare_batch.assert_not_called()


def test_ray_transfer_timings_exclude_generation_and_slow_group_wait(
    learner: Any, rollout: training.Rollout, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    learner.config = replace(learner.config, num_groups=2, group_size=2)
    clock = Mock(return_value=0.0)
    monkeypatch.setattr(training, "time", Mock(monotonic=clock))
    caplog.set_level("INFO", logger=training.__name__)
    groups = [
        training.RolloutGroup(task_id=str(i), rollouts=[replace(rollout, advantage=float(i))] * 2) for i in range(2)
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
    batch = learner.fetch_next_batch(iteration=10)
    # Preserve request order despite out-of-order completion and serialization.
    np.testing.assert_array_equal(batch.training_batch.advantages, [0.0, 0.0, 1.0, 1.0])
    assert "Example 1 | task=0" in batch.examples
    assert "Example 3 | task=1" in batch.examples
    assert batch.training_batch.token_ids.shape == (4, 8)
    assert batch.transfer_metrics == {
        "batcher/fetch_rollout_seconds": 50.5,
        "batcher/prepare_batch_seconds": 2.0,
        "batcher/actor_transfer_seconds_mean": 0.375,
        "learner/batcher_transfer_seconds": 0.75,
    }
    assert not [record for record in caplog.records if record.name == training.__name__]


@pytest.fixture
def learner(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(training, "full_weight_publisher", Mock(return_value=MagicMock()))
    monkeypatch.setattr(training, "lora_weight_publisher", Mock(return_value=MagicMock()))
    monkeypatch.setattr(training, "checkpoint_manager", Mock(return_value=Mock()))
    monkeypatch.setattr(training, "restore_checkpoint", Mock(return_value=None))
    monkeypatch.setattr(training, "save_checkpoint", Mock(return_value=0.5))
    monkeypatch.setattr(training.ray, "get", Mock())

    device = jax.devices("cpu")[0]
    params = {
        "model.layers.0.self_attn.q_proj.weight": jnp.ones((2, 3), dtype=jnp.bfloat16),  # (O, I)
        "model.norm.weight": jnp.ones(3, dtype=jnp.bfloat16),  # (D,)
        "model.embed_tokens.weight": jnp.arange(6, dtype=jnp.bfloat16).reshape(2, 3),  # (V, D)
    }
    model = Mock()
    model.config.max_position_embeddings = 8
    loader = Mock(return_value=(model, {"params": params}))
    monkeypatch.setattr(training, "load_qwen3_checkpoint", loader)
    monkeypatch.setattr(training.jax, "devices", Mock(return_value=[device]))
    config = training.Config(
        model="Qwen/custom",
        attention_backend="xla",
        scan_layers=False,
        gradient_checkpointing=False,
        prefetch_batches=0,
    )
    learner = training.Learner.__ray_metadata__.modified_class(batcher=Mock(), actor=Mock(), config=config)
    learner.model = model
    loader.assert_not_called()
    assert not hasattr(learner, "state")
    return learner


def test_learner_initializes_replicated_state(learner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    devices = jax.local_devices(backend="cpu")[:2]
    if len(devices) < 2:
        pytest.skip("Requires two CPU devices; use JAX_NUM_CPU_DEVICES=2")
    monkeypatch.setattr(training.jax, "devices", Mock(return_value=devices))
    monkeypatch.setattr(training, "MLflowLogger", MagicMock())
    learner.config = replace(learner.config, learner_gpus=2, minibatches_per_epoch=2)
    previous_mesh = jax.sharding.get_mesh()

    def load(*args: Any, **kwargs: Any) -> Any:
        assert jax.sharding.get_mesh() == learner.mesh
        return training.load_qwen3_checkpoint.return_value

    def publish() -> None:
        assert jax.sharding.get_mesh() == learner.mesh
        raise RuntimeError("initialized")

    training.load_qwen3_checkpoint.side_effect = load
    learner.publish_weights = publish
    with pytest.raises(RuntimeError, match="initialized"):
        learner.start()
    assert jax.sharding.get_mesh() == previous_mesh
    assert learner.mesh.size == 2
    for value in jax.tree.leaves(learner.state):  # HF/LoRA parameter shapes and scalar optimizer counters.
        assert value.sharding.is_fully_replicated
        assert value.devices() == set(devices)


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_learner_waits_for_actor_before_publisher_setup(
    learner: Any, monkeypatch: pytest.MonkeyPatch, mode: Literal["lora", "full"]
) -> None:
    learner.config = replace(learner.config, finetuning_mode=mode)
    monkeypatch.setattr(training, "MLflowLogger", MagicMock())

    def enter(_context: MagicMock) -> None:
        learner.actor.ready.remote.assert_called_once_with()
        training.ray.get.assert_called_once_with(learner.actor.ready.remote.return_value)
        raise RuntimeError("publisher reached")

    publisher = getattr(training, f"{mode}_weight_publisher")
    publisher.return_value.__enter__.side_effect = enter
    with pytest.raises(RuntimeError, match="publisher reached"):
        learner.start()


def test_learner_propagates_actor_startup_failure(learner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training, "MLflowLogger", MagicMock())
    monkeypatch.setattr(training.ray, "cancel", Mock())
    training.ray.get.side_effect = RuntimeError("actor startup failed")
    with pytest.raises(RuntimeError, match="actor startup failed"):
        learner.start()
    training.lora_weight_publisher.assert_not_called()
    training.full_weight_publisher.assert_not_called()


@pytest.mark.parametrize(
    ("fails", "group_rewards", "degenerate_fraction", "mode"),
    [
        pytest.param(False, ((0.0, 0.0), (1.0, 1.0)), 1.0, "lora", id="all-degenerate"),
        pytest.param(False, ((0.0, 0.0), (0.0, 1.0)), 0.5, "lora", id="mixed"),
        pytest.param(False, ((0.0, 1.0), (1.0, 0.0)), 0.0, "lora", id="none-degenerate"),
        pytest.param(True, ((0.0, 0.0), (1.0, 1.0)), 1.0, "lora", id="failure"),
        pytest.param(False, ((0.0, 1.0), (1.0, 0.0)), 0.0, "full", id="full-weights"),
    ],
)
def test_learner_logs_rollouts(
    fails: bool,
    group_rewards: tuple[tuple[float, float], tuple[float, float]],
    degenerate_fraction: float,
    mode: Literal["lora", "full"],
    learner: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    spans: InMemorySpanExporter,
) -> None:
    learner.config = replace(learner.config, finetuning_mode=mode)
    caplog.set_level("INFO", logger=training.__name__)
    sequence = training.TrainingSequence(
        token_ids=np.array([1], dtype=np.int32),  # (T=1,)
        loss_mask=np.array([True]),  # (T=1,)
        old_log_probs=np.array([-0.2], dtype=np.float32),  # (T=1,)
    )
    rollout = training.Rollout(
        prompt="Environment prompt",
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
        training.RolloutGroup(task_id="0", rollouts=[replace(rollout, reward=reward) for reward in group_rewards[0]]),
        training.RolloutGroup(
            task_id="1",
            rollouts=[
                replace(
                    rollout,
                    reward=reward,
                    num_turns=3,
                    num_output_tokens=6,
                    rollout_seconds=1.5,
                    stop_condition=condition,
                )
                for reward, condition in zip(group_rewards[1], ("max_turns", "unspecified"), strict=True)
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
    monkeypatch.setattr(training.ray, "get", Mock())
    steps = Mock()
    learner.fetch_next_batch = steps.fetch
    transfer_metrics = {
        "batcher/fetch_rollout_seconds": 0.5,
        "batcher/prepare_batch_seconds": 0.25,
        "batcher/actor_transfer_seconds_mean": 0.25,
        "learner/batcher_transfer_seconds": 0.125,
    }
    prepared = Mock(spec=training.GRPOBatch)
    prepared.token_ids = Mock(shape=(4, 8))  # (B=4, T=8)
    prepared.attention_mask = np.tile([True, True, False, False, False, False, False, False], (4, 1))  # (B=4, T=8)
    received = training.build_rollout_batch(groups=batch, training_batch=prepared, transfer_metrics=transfer_metrics)
    next_batch = training.build_rollout_batch(
        groups=[
            replace(group, rollouts=[replace(rollout, stop_condition="environment_done") for rollout in group.rollouts])
            for group in batch
        ],
        training_batch=prepared,
        transfer_metrics=transfer_metrics,
    )
    learner.fetch_next_batch.side_effect = [received, RuntimeError("done") if fails else next_batch]
    learner.publish_weights = steps.publish

    def optimize(batch: training.GRPOBatch[np.ndarray]) -> dict[str, float]:
        assert batch is prepared
        assert jax.sharding.get_mesh() == learner.mesh
        learner.state = learner.state.replace(step=learner.state.step + 1)
        return {"train/policy_loss": -0.25}

    steps.optimize.side_effect = optimize
    learner.optimize = steps.optimize

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
        attention_backend=learner.config.attention_backend,
        scan_layers=learner.config.scan_layers,
        gradient_checkpointing=learner.config.gradient_checkpointing,
    )
    assert learner.model is training.load_qwen3_checkpoint.return_value[0]
    if mode == "full":
        training.full_weight_publisher.assert_called_once()
        factory_call = training.full_weight_publisher.call_args
        assert factory_call.args[0]() is learner.state.params
        assert factory_call.kwargs == {
            "server_url": training.SERVER_URL,
            "device_index": learner.mesh.devices.flat[0].local_hardware_id,
            "pause_mode": learner.config.weight_update_pause_mode,
        }
        training.full_weight_publisher.return_value.__exit__.assert_called_once()
        training.lora_weight_publisher.assert_not_called()
    else:
        training.full_weight_publisher.assert_not_called()
        training.lora_weight_publisher.assert_called_once()
        factory_call = training.lora_weight_publisher.call_args
        assert factory_call.args[0]() is learner.state.params
        assert factory_call.kwargs == {
            "server_url": training.SERVER_URL,
            "model": learner.config.model,
            "adapter_name": training.LORA_NAME,
            "lora_rank": learner.config.lora_rank,
            "lora_alpha": learner.config.lora_alpha,
            "pause_mode": learner.config.weight_update_pause_mode,
        }
        training.lora_weight_publisher.return_value.__exit__.assert_called_once()
    assert learner.mesh.axis_names == ("batch",)
    for value in jax.tree.leaves(learner.state.params):  # Model parameter shapes.
        assert value.sharding == training.NamedSharding(learner.mesh, training.PartitionSpec())
        assert value.devices() == {jax.devices("cpu")[0]}
    assert all(value.dtype == jnp.float32 for value in jax.tree.leaves(learner.state.params))
    learner.fetch_next_batch.assert_called_with(iteration=2)
    expected = ["publish", "fetch", "optimize", "publish", "fetch"]
    assert [call[0] for call in steps.mock_calls] == expected + ([] if fails else ["optimize", "publish"])
    factory.assert_called_once_with(experiment="grpo_v2", run_name=None)
    writer.log_params.assert_called_once_with(asdict(learner.config))
    assert yaml.safe_load(writer.log_text.call_args.args[0]) == asdict(learner.config)
    calls = writer.log_metrics.call_args_list
    assert len(calls) == (2 if fails else 3)
    assert calls[0].kwargs == {"step": 0}
    assert calls[0].args[0] == {
        "model/parameters_billions": 15 / 1_000_000_000,
        "model/trainable_parameters_billions": (80 if mode == "lora" else 15) / 1_000_000_000,
        "learner/setup_seconds": 1,
    }
    assert calls[1].kwargs == {"step": 1}
    assert calls[1].args[0] == {
        **transfer_metrics,
        "train/policy_loss": -0.25,
        "rollout/reward_mean": mean(rewards),
        "rollout/reward_std": pstdev(rewards),
        "rollout/reward_min": 0.0,
        "rollout/reward_max": 1.0,
        "rollout/adjusted_reward_mean": mean(rewards),
        "rollout/adjusted_reward_std": pstdev(rewards),
        "rollout/adjusted_reward_min": 0.0,
        "rollout/adjusted_reward_max": 1.0,
        "rollout/degenerate_group_fraction": degenerate_fraction,
        "rollout/stop_fraction/agent_completed": 0.5,
        "rollout/stop_fraction/environment_done": 0.0,
        "rollout/stop_fraction/max_tokens": 0.0,
        "rollout/stop_fraction/context_length": 0.0,
        "rollout/stop_fraction/max_turns": 0.25,
        "rollout/stop_fraction/unspecified": 0.25,
        "rollout/turns_mean": 2.0,
        "rollout/input_tokens_mean": 4.0,
        "rollout/output_tokens_mean": 4.0,
        "rollout/total_tokens_mean": 2.0,
        "batcher/token_utilization_fraction": 0.25,
        "batcher/padded_sequence_tokens": 8.0,
        "train/iterations_total": 1,
        "train/updates_total": 1,
        "train/learning_rate": learner.config.learning_rate,
        "train/rollouts_total": 4,
        "learner/iterations_per_second": 1 / 3,
        "learner/rollouts_per_second": 4 / 3,
        "learner/output_tokens_per_second": 16 / 3,
        "actor/rollout_seconds_mean": 1.0,
        "learner/batch_fetch_seconds": 1.0,
        "learner/update_seconds": 1.0,
        "learner/weight_transfer_seconds": 1.0,
        "learner/iteration_seconds": 3.0,
    }
    if not fails:
        assert calls[2].kwargs == {"step": 2}
        assert calls[2].args[0]["train/rollouts_total"] == 8
        assert calls[2].args[0]["rollout/degenerate_group_fraction"] == degenerate_fraction
        assert {key: value for key, value in calls[2].args[0].items() if "stop_fraction/" in key} == {
            "rollout/stop_fraction/agent_completed": 0.0,
            "rollout/stop_fraction/max_tokens": 0.0,
            "rollout/stop_fraction/context_length": 0.0,
            "rollout/stop_fraction/max_turns": 0.0,
            "rollout/stop_fraction/unspecified": 0.0,
            "rollout/stop_fraction/environment_done": 1.0,
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


def test_learner_skips_publication_without_updates(
    learner: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger=training.__name__)
    learner.config = replace(learner.config, total_iterations=3)
    writer = MagicMock()
    writer.__enter__.return_value = writer
    monkeypatch.setattr(training, "MLflowLogger", Mock(return_value=writer))
    monkeypatch.setattr(training, "time", Mock(monotonic=Mock(side_effect=range(20))))
    prepared = Mock(spec=training.GRPOBatch)
    prepared.token_ids = Mock(shape=(4, 2))  # (B=4, T=2)
    learner.fetch_next_batch = Mock(
        return_value=training.RolloutBatch(
            training_batch=prepared,
            rollout_metrics={"rollout/reward_mean": 0.0, "rollout/degenerate_group_fraction": 1.0},
            stop_counts={},
            num_output_tokens=4,
            examples="",
            transfer_metrics={"batcher/actor_transfer_seconds_mean": 0.0, "learner/batcher_transfer_seconds": 0.0},
        )
    )
    increments = iter((0, 1, 0))  # All skipped, partially skipped, then all skipped at a nonzero step.
    published_steps: list[int] = []

    def optimize(batch: training.GRPOBatch[np.ndarray]) -> dict[str, float]:  # (B, T), (B,) leaves.
        learner.state = learner.state.replace(step=learner.state.step + next(increments))
        return {"train/policy_loss": 0.0}

    def publish() -> None:
        published_steps.append(int(learner.state.step))

    learner.optimize = optimize
    learner.publish_weights = publish
    learner.start()

    assert published_steps == [0, 1]
    metrics = [call.args[0] for call in writer.log_metrics.call_args_list[1:]]
    assert [values["train/updates_total"] for values in metrics] == [0, 1, 1]
    assert [values["train/iterations_total"] for values in metrics] == [1, 2, 3]
    assert [values["train/rollouts_total"] for values in metrics] == [4, 8, 12]
    assert "learner/weight_transfer_seconds" not in metrics[0]
    assert metrics[1]["learner/weight_transfer_seconds"] == 1.0
    assert "learner/weight_transfer_seconds" not in metrics[2]
    messages = [message for message in caplog.messages if message.startswith("iteration=")]
    assert ["weights=" in message for message in messages] == [False, True, False]
    progress = training.save_checkpoint.call_args.args[1]
    assert progress.iteration == 3
    assert int(progress.train_state.step) == 1


def test_learner_checkpoints_and_resumes(learner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    learner.config = replace(learner.config, total_iterations=3, checkpoint_interval=2, prefetch_batches=4)
    published_steps: list[int] = []
    saved: list[training.TrainingProgress] = []
    writer = MagicMock()
    writer.__enter__.return_value = writer
    monkeypatch.setattr(training, "MLflowLogger", Mock(return_value=writer))
    monkeypatch.setattr(training.ray, "get", lambda value: value)
    learner.actor.get_sampling_state.remote.return_value = {"task_rng": "saved"}

    def publish() -> None:
        published_steps.append(int(learner.state.step))

    def fetch(*, iteration: int) -> training.RolloutBatch:
        num_rollouts = learner.config.num_groups * learner.config.group_size
        prepared = Mock(spec=training.GRPOBatch)
        prepared.token_ids = Mock(shape=(num_rollouts, 8))  # (B, T=8)
        return training.RolloutBatch(
            training_batch=prepared,
            rollout_metrics={"rollout/reward_mean": 1.0, "rollout/degenerate_group_fraction": 0.0},
            stop_counts={"max_turns" if not saved else "agent_completed": num_rollouts},
            num_output_tokens=4,
            examples="",
            transfer_metrics={
                "batcher/actor_transfer_seconds_mean": 0.0,
                "learner/batcher_transfer_seconds": 0.0,
            },
        )

    def optimize(batch: training.GRPOBatch[np.ndarray]) -> dict[str, float]:
        learner.state = learner.state.replace(
            step=learner.state.step + learner.config.update_epochs * learner.config.minibatches_per_epoch
        )
        return {"train/policy_loss": 0.0}

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
    assert [call.kwargs["iteration"] for call in learner.fetch_next_batch.call_args_list] == [1, 2, 3]
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
    assert metrics["train/rollouts_total"] == 28
    assert metrics["train/updates_total"] == 28
    assert metrics["rollout/stop_fraction/max_turns"] == 0
    assert metrics["checkpoint/save_seconds"] == 0.5

    # A run already at its cumulative target does not train or overwrite its checkpoint.
    learner.fetch_next_batch.reset_mock()
    learner.start()
    learner.fetch_next_batch.assert_not_called()
    assert len(saved) == 4


def test_learner_logs_examples_at_configured_interval(
    learner: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger=training.__name__)
    rollout = training.Rollout(
        prompt="Environment prompt",
        sequence=training.TrainingSequence(
            token_ids=np.array([1, 2], dtype=np.int32),  # (T=2,)
            loss_mask=np.array([False, True]),  # (T=2,)
            old_log_probs=np.array([0.0, -0.2], dtype=np.float32),  # (T=2,)
        ),
        reward=1.0,
        advantage=0.5,
        completion='<think>Use add.</think>\n<tool_call>{"name": "add"}</tool_call>',
        interaction=(
            "<|im_start|>system\nUse tools.<|im_end|>\n"
            "<|im_start|>user\nAdd:\n1 + 2<|im_end|>\n"
            '<|im_start|>assistant\n<think>Use add.</think>\n<tool_call>{"name": "add"}</tool_call><|im_end|>'
        ),
        num_turns=1,
        num_input_tokens=1,
        num_output_tokens=1,
        rollout_seconds=0.1,
        stop_condition="max_turns",
    )
    groups = [
        training.RolloutGroup(task_id="task-1", rollouts=[rollout] * 5),
        training.RolloutGroup(
            task_id=None,
            rollouts=[
                replace(rollout, interaction="", stop_condition="unspecified"),
                rollout,
                rollout,
                replace(rollout, interaction="omitted"),
                replace(rollout, interaction="omitted"),
            ],
        ),
    ]
    learner.config = replace(
        learner.config,
        num_groups=2,
        group_size=5,
        minibatches_per_epoch=1,
        total_iterations=7,
        prefetch_batches=3,
        example_log_interval=3,
    )
    learner.publish_weights = Mock()
    learner.optimize = Mock(return_value={"train/policy_loss": 0.0})
    prepared = Mock(spec=training.GRPOBatch)
    prepared.token_ids = Mock(shape=(10, 8))  # (B=10, T=8)
    prepared.attention_mask = np.ones((10, 8), dtype=np.bool_)  # (B=10, T=8)
    submitted: list[training.Transfer[training.RolloutBatch]] = []

    def fetch(
        num_groups: int, group_size: int, context_length: int, *, include_examples: bool
    ) -> training.Transfer[training.RolloutBatch]:
        batch = training.build_rollout_batch(
            groups=[replace(groups[0], task_id=f"batch-{len(submitted) + 1}"), groups[1]],
            training_batch=prepared,
            transfer_metrics={"batcher/actor_transfer_seconds_mean": 0.0},
            include_examples=include_examples,
        )
        result = training.Transfer(value=batch, ready_at=training.time.monotonic())
        submitted.append(result)
        return result

    learner.batcher.fetch_next_batch.remote.side_effect = fetch
    monkeypatch.setattr(training.ray, "get", lambda value: value)
    writer = MagicMock()
    writer.__enter__.return_value = writer
    monkeypatch.setattr(training, "MLflowLogger", Mock(return_value=writer))

    learner.start()

    artifacts = writer.log_text.call_args_list[1:]  # Skip the config artifact.
    assert [call.args[1] for call in artifacts] == ["examples/3.txt", "examples/6.txt"]
    messages = [message for message in caplog.messages if "Rollout examples | step" in message]
    assert not messages
    assert len([message for message in caplog.messages if message.startswith("iteration=")]) == 7
    for step, call in zip((3, 6), artifacts, strict=True):
        text = call.args[0]
        assert text.startswith(f"Rollout examples | step {step}\n\n{'=' * 72}\n")
        assert f"Example 1 | task=batch-{step} | reward=1.000 | advantage=0.500" in text
        assert "Turns: 1 | Output tokens: 1 | Stop: max_turns" in text
        assert "Prompt:" not in text
        assert (
            "Full token sequence (special tokens preserved):\n"
            "<|im_start|>system\nUse tools.<|im_end|>\n"
            "<|im_start|>user\nAdd:\n1 + 2<|im_end|>\n"
            "<|im_start|>assistant\n"
            '<think>Use add.</think>\n<tool_call>{"name": "add"}</tool_call><|im_end|>'
        ) in text
        assert "Example 6 | task=None" in text
        assert "Example 8 | task=None" in text
        assert "Full token sequence (special tokens preserved):\n(empty)" in text
        assert text.count("reward=1.000") == 8
        assert "Token IDs:" not in text
        assert "Example 9" not in text and "omitted" not in text


def test_learner_cleans_up_failed_start(learner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    writer = MagicMock()
    writer.__enter__.return_value = writer
    monkeypatch.setattr(training, "MLflowLogger", Mock(return_value=writer))

    def fail() -> None:
        raise RuntimeError("publish failed")

    learner.publish_weights = fail
    with pytest.raises(RuntimeError, match="publish failed"):
        learner.start()
    writer.__exit__.assert_called_once()
    assert writer.__exit__.call_args.args[0] is RuntimeError
    training.lora_weight_publisher.return_value.__exit__.assert_called_once()


def test_learner_delegates_publication(learner: Any) -> None:
    learner._publish_weights = Mock()
    learner.publish_weights()
    learner._publish_weights.assert_called_once_with()
    learner._publish_weights.side_effect = RuntimeError("transfer failed")
    with pytest.raises(RuntimeError, match="transfer failed"):
        learner.publish_weights()


def test_console_logging_omits_ray_metadata(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("", "ray"):
        log = logging.getLogger(name)
        monkeypatch.setattr(log, "handlers", [logging.StreamHandler()] if name else [])
        monkeypatch.setattr(log, "level", logging.INFO)
        monkeypatch.setattr(log, "propagate", False)
    training.configure_console_logging()
    metadata = {
        "job_id": "job",
        "worker_id": "worker",
        "node_id": "node",
        "actor_id": "actor",
        "task_id": "task",
        "task_name": "Learner.start",
        "task_func_name": "train_grpo_v2.Learner.start",
        "timestamp_ns": 123,
    }
    for name in ("rl2.train_grpo_v2", "ray"):
        log = logging.getLogger(name)
        log.info("iteration=%s reward=0.188", 107, extra=metadata)
        try:
            raise RuntimeError("training failed")
        except RuntimeError:
            log.exception("Learner failed", extra=metadata)
    output = capsys.readouterr().err
    assert output.count(" -- iteration=107 reward=0.188\n") == 2
    assert "INFO test_train_grpo_v2.py:" in output
    assert output.count(" -- Learner failed\nTraceback (most recent call last):") == 2
    assert output.count("RuntimeError: training failed") == 2
    for key in metadata:
        assert f"{key}=" not in output


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
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in ("", "ray", "openai._base_client", "vllm", "httpx", "httpcore"):
        log = logging.getLogger(name)
        monkeypatch.setattr(log, "handlers", list(log.handlers))
        monkeypatch.setattr(log, "level", log.level)
        monkeypatch.setattr(log, "propagate", log.propagate)
        for handler in log.handlers:
            monkeypatch.setattr(handler, "formatter", handler.formatter)
    config = training.Config(
        ckpt_dir=str(tmp_path),
        padding_multiple=64,
        learner_gpus=2,
        minibatches_per_epoch=2,
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
    monkeypatch.setattr(training, "Learner", Mock(options=Mock(return_value=Mock(remote=Mock(return_value=learner)))))

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
    training.Learner.options.assert_called_once_with(num_gpus=2)
    training.Learner.options.return_value.remote.assert_called_once_with(batcher=batcher, actor=actor, config=config)
    options = training.ray.init.call_args.kwargs
    assert options["include_dashboard"] is enable_dashboard
    assert options["dashboard_host"] == ("0.0.0.0" if fails else "127.0.0.1")
    assert options["dashboard_port"] == 8265
    assert "logging_config" not in options
    assert logging.getLogger().level == logging.INFO
    assert options["runtime_env"] == {
        "env_vars": {"RAY_PROFILING": "1"} if enable_timeline else {},
        "worker_process_setup_hook": training.configure_console_logging,
    }
    for name in ("vllm", "httpx", "httpcore", "openai._base_client"):
        log = logging.getLogger(name)
        monkeypatch.setattr(log, "level", logging.DEBUG)
        for source in (log, logging.getLogger(f"{name}.child")):
            source.warning("%s diagnostic", source.name)
    output = capsys.readouterr().err
    for name in ("vllm", "httpx", "httpcore", "openai._base_client"):
        assert f"{name} diagnostic" in output
        assert f"{name}.child diagnostic" in output
    assert "Creating batcher, learner, and actor" in output
    assert "timestamp_ns=" not in output
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
    rewards: list[float], rollout: training.Rollout, task_discovery: list[ALFWorldTask]
) -> None:
    task = task_discovery[0]
    rollouts = [replace(rollout, reward=reward) for reward in rewards]
    group = training.score_group(task, rollouts)
    assert group.task_id == f"{task.name}:{task.id}"
    expected = [(reward - mean(rewards)) / (pstdev(rewards) + 1e-4) for reward in rewards]
    assert len(group.rollouts) == len(rewards)
    assert [rollout.reward for rollout in group.rollouts] == rewards
    assert [rollout.advantage for rollout in group.rollouts] == pytest.approx(expected, rel=1e-6, abs=1e-7)
    if len(set(rewards)) == 1:
        assert all(rollout.advantage == 0.0 for rollout in group.rollouts)


@pytest.mark.parametrize(
    ("rewards", "coefficient", "expected"),
    [
        pytest.param([1.0, 1.0], 0.0, [1.0, 1.0], id="disabled"),
        pytest.param([1.0, 1.0], 0.25, [0.75, 0.25], id="equal-successes"),
        pytest.param([0.0, 0.0], 0.25, [0.0, 0.0], id="zero-rewards-unpenalized"),
        pytest.param([-1.0, -1.0], 0.25, [-1.25, -1.75], id="negative-rewards-penalized"),
        pytest.param([0.0, 1.0], 0.25, [0.0, 0.25], id="task-reward-outweighs-length"),
        pytest.param([0.0, 0.75], 0.25, [0.0, 0.0], id="adjusted-tie"),
        pytest.param([0.0, 1.0], 1.0, [0.0, -2.0], id="length-outweighs-task-reward"),
    ],
)
def test_score_group_length_penalty(
    rewards: list[float], coefficient: float, expected: list[float], rollout: training.Rollout
) -> None:
    samples = [
        replace(rollout, reward=reward, num_output_tokens=length, num_input_tokens=100, num_turns=2)
        for reward, length in zip(rewards, (1, 3), strict=True)
    ]
    group = training.score_group(Mock(id="task"), samples, length_penalty_coef=coefficient)
    assert group.training_rewards == expected
    assert [sample.reward for sample in group.rollouts] == rewards
    assert [sample.advantage for sample in samples] == [rollout.advantage] * 2
    assert [sample.advantage for sample in group.rollouts] == pytest.approx(
        [(reward - mean(expected)) / (pstdev(expected) + 1e-4) for reward in expected]
    )
    assert group.is_degenerate == (expected[0] == expected[1])


def test_length_penalty_actor_batcher_integration(
    actor: Any, rollout: training.Rollout, monkeypatch: pytest.MonkeyPatch
) -> None:
    actor.config = replace(actor.config, length_penalty_coef=0.25, drop_degenerate_groups=True)
    longer = replace(
        rollout,
        num_output_tokens=2,
        sequence=training.TrainingSequence(
            token_ids=np.array([1, 2, 3], dtype=np.int32),  # (T=3,)
            loss_mask=np.array([False, True, True]),  # (T=3,)
            old_log_probs=np.array([0.0, -0.2, -0.2], dtype=np.float32),  # (T=3,)
        ),
    )
    agent = AsyncMock()
    agent.__aenter__.return_value = agent
    agent.run.side_effect = [rollout, longer]
    monkeypatch.setattr(training, "create_agent", Mock(return_value=agent))
    handle = Mock()
    handle.fetch_next_group.remote.side_effect = actor.fetch_next_group
    batcher = training.Batcher.__ray_metadata__.modified_class(actor=handle, config=actor.config)

    async def collect() -> training.RolloutBatch:
        async with aclosing(actor):
            return (await batcher.fetch_next_batch(num_groups=1, group_size=2, context_length=8)).value

    batch = asyncio.run(collect())
    assert handle.fetch_next_group.remote.call_count == 1
    assert batch.training_batch.advantages.shape == (2,)
    np.testing.assert_allclose(batch.training_batch.advantages, [0.125 / 0.1251, -0.125 / 0.1251])
    assert batch.rollout_metrics["rollout/degenerate_group_fraction"] == 0.0
    for suffix, raw, adjusted in (("mean", 1.0, 0.625), ("std", 0.0, 0.125), ("min", 1.0, 0.5), ("max", 1.0, 0.75)):
        assert batch.rollout_metrics[f"rollout/reward_{suffix}"] == raw
        assert batch.rollout_metrics[f"rollout/adjusted_reward_{suffix}"] == adjusted


def test_batcher_collects_groups_concurrently(
    actor: Any,
    rollout: training.Rollout,
    monkeypatch: pytest.MonkeyPatch,
    spans: InMemorySpanExporter,
    task_discovery: list[ALFWorldTask],
) -> None:
    # Actor scoring, transcript construction, and batching must work without JAX.
    monkeypatch.setattr(training, "jax", Mock(spec=[]))
    monkeypatch.setattr(training, "jnp", Mock(spec=[]))
    all_started = asyncio.Event()
    started = 0
    env = AsyncMock()
    env.__aenter__.return_value = env

    async def run(env: Environment) -> training.Rollout:
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
    monkeypatch.setattr(actor.rng, "choice", Mock(side_effect=task_discovery[:2]))
    handle = Mock()
    handle.fetch_next_group.remote.side_effect = actor.fetch_next_group
    batcher = training.Batcher.__ray_metadata__.modified_class(actor=handle)

    async def collect() -> training.Transfer[training.RolloutBatch]:
        async with aclosing(actor):
            with training.span("test.batch") as parent:
                async with asyncio.timeout(5):
                    result = await batcher.fetch_next_batch(
                        num_groups=2, group_size=2, context_length=16, include_examples=True
                    )
                assert otel_trace.get_current_span() is parent
                return result

    batch = asyncio.run(collect()).value
    assert started == 4
    assert f"Example 1 | task=alfworld:{task_discovery[0].id}" in batch.examples
    assert f"Example 3 | task=alfworld:{task_discovery[1].id}" in batch.examples
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


def test_rollout_deadline_excludes_cleanup(
    actor: Any, rollout: training.Rollout, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = AsyncMock()
    agent.__aenter__.return_value = agent
    agent.run.return_value = rollout
    monkeypatch.setattr(training, "create_agent", Mock(return_value=agent))
    environment = AsyncMock()
    task = Mock(init_env=Mock(return_value=environment))

    async def collect() -> training.RolloutGroup:
        loop = asyncio.get_running_loop()
        now = loop.time()
        monkeypatch.setattr(loop, "time", lambda: now)
        closing = 0
        both_closing = asyncio.Event()

        async def close(*args: Any) -> None:
            nonlocal closing, now
            closing += 1
            if closing == 2:
                both_closing.set()
            await both_closing.wait()
            now += actor.config.rollout_timeout + 1
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        environment.__aexit__.side_effect = close
        async with aclosing(actor):
            return await actor.collect_group(task, group_size=2)

    group = asyncio.run(collect())
    assert len(group.rollouts) == 2
    assert environment.__aexit__.await_count == 2


@pytest.mark.parametrize("phase", ["setup", "rollout"])
def test_rollout_deadline_still_limits_work(actor: Any, monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    actor.config = replace(actor.config, rollout_timeout=0.001)
    agent = AsyncMock()
    agent.__aenter__.return_value = agent
    monkeypatch.setattr(training, "create_agent", Mock(return_value=agent))
    environment = AsyncMock()
    task = Mock(init_env=Mock(return_value=environment))

    async def block(*args: Any, **kwargs: Any) -> None:
        await asyncio.Future()

    if phase == "setup":
        environment.__aenter__.side_effect = block
    else:
        agent.run.side_effect = block

    async def collect() -> None:
        async with aclosing(actor):
            await actor.collect_group(task, group_size=2)

    with pytest.raises(ExceptionGroup) as raised:
        asyncio.run(collect())
    assert all(isinstance(error, TimeoutError) for error in raised.value.exceptions)
    assert environment.__aexit__.await_count == (0 if phase == "setup" else 2)


def test_group_cancels_pending_rollouts(
    actor: Any, monkeypatch: pytest.MonkeyPatch, spans: InMemorySpanExporter, task_discovery: list[ALFWorldTask]
) -> None:
    env = AsyncMock()
    env.__aenter__.return_value = env
    started = asyncio.Event()
    cancelled: list[bool] = []
    count = 0

    async def run(env: Environment) -> training.Rollout:
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
                    await actor.collect_group(task_discovery[0], group_size=2)
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


@pytest.mark.parametrize("fails,mode", [(False, "lora"), (True, "lora"), (False, "full")])
def test_create_agent_config_and_cleanup(
    fails: bool, mode: Literal["lora", "full"], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = training.Config(
        model="Qwen/custom", max_tokens=16, max_turns=2, temperature=0.7, rollout_timeout=300, finetuning_mode=mode
    )
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
            assert agent.model == (training.LORA_NAME if mode == "lora" else config.model)
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
        base_url=f"{training.SERVER_URL}/v1", api_key="EMPTY", timeout=config.rollout_timeout, max_retries=3
    )
    assert client.is_closed()
