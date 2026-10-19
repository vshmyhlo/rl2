import asyncio
import io
import json
import pickle
import runpy
import subprocess
import sys
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
import renderers.base as renderer_base
import verifiers.v1 as vf
import yaml
from pydantic import ValidationError
from ray import cloudpickle
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from verifiers.v1.clients import base as client_base
from verifiers.v1.clients.train import ElasticRendererPool
from verifiers.v1.runtimes import subprocess as subprocess_runtime
from verifiers.v1.trace import AgentInfo, TraceTask

from rl2 import codegym_env
from rl2 import train_grpo_v2 as training
from rl2.addition_env import AdditionTaskset, AdditionTools
from rl2.toolrl_env import ToolCall, ToolRLData, ToolRLRenderer, ToolRLTask


def test_config_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    defaults = training.Config()
    assert defaults.minibatches_per_epoch == 4
    assert defaults.grad_accum_steps == 1
    assert defaults.update_epochs == 1
    assert defaults.gradient_checkpointing
    assert defaults.weight_update_pause_mode == "wait"
    preset = Path(__file__).resolve().parents[1] / "configs/grpo_v2/addition.yaml"
    config = training.load_config(preset, training.Config)
    assert config.run_id == "grpo_v2_addition_Qwen_Qwen3-0.6B_seed1"
    assert config.ckpt_dir == f"/runs/grpo_v2/{config.run_id}"
    assert config.total_iterations is None
    assert config.minibatches_per_epoch == 4
    assert config.grad_accum_steps == 1
    assert config.gradient_checkpointing
    assert config.weight_update_pause_mode == "wait"
    assert asdict(config.taskset) == {"id": "addition"}
    assert "dataset_file" not in asdict(config)
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            asdict(config) | {"max_tokens": "${group_size}", "total_iterations": 1, "gradient_checkpointing": False}
        )
    )
    train = Mock()
    monkeypatch.setattr(training, "train", train)
    monkeypatch.setattr(sys, "argv", ["train_grpo_v2", "--config", str(path)])
    training.main()
    train.assert_called_once_with(replace(config, max_tokens=4, total_iterations=1, gradient_checkpointing=False))
    assert pickle.loads(pickle.dumps(config)) == config


@pytest.mark.parametrize(
    "settings",
    [
        {"num_groups": 0},
        {"group_size": 1},
        {"total_iterations": 0},
        {"max_tokens": 0},
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
        {"taskset": training.ToolRLConfig(), "max_turns": 2},
    ],
)
def test_config_rejects_invalid_settings(settings: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        training.Config(**settings)


@pytest.mark.parametrize("dataset", [None, "/data/train.parquet"], ids=["default", "local"])
def test_toolrl_subconfig(dataset: str | None, tmp_path: Path) -> None:
    taskset = {"id": "toolrl"}
    if dataset is not None:
        taskset["dataset_file"] = dataset
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"taskset": taskset, "max_turns": 1}))
    config = training.load_config(path, training.Config)
    assert isinstance(config.taskset, training.ToolRLConfig)
    assert config.taskset.dataset_file == (training.TOOLRL_TRAIN_FILE if dataset is None else dataset)
    assert "dataset_file" not in asdict(config)


@pytest.mark.parametrize(
    ("settings", "error"),
    [
        ({"taskset": {"id": "unknown"}}, "union_tag_invalid"),
        ({"taskset": {"id": "addition", "dataset_file": "train.parquet"}}, "unexpected_keyword_argument"),
        ({"taskset": {"id": "toolrl", "unknown_option": True}, "max_turns": 1}, "unexpected_keyword_argument"),
        ({"dataset_file": "train.parquet"}, "unexpected_keyword_argument"),
    ],
    ids=["unknown-tag", "addition-dataset", "unknown-toolrl-field", "top-level-dataset"],
)
def test_taskset_config_rejects_invalid_yaml(settings: dict[str, object], error: str, tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValidationError, match=error):
        training.load_config(path, training.Config)


def test_agent_uses_config() -> None:
    config = training.Config(model="Qwen/custom", max_tokens=16, max_turns=2, temperature=0.7)
    agent = training.create_agent(config)
    assert agent.config.model == config.model
    assert agent.config.client.renderer_model_name == config.model
    assert agent.config.sampling.max_tokens == 16
    assert agent.config.sampling.temperature == 0.7
    assert agent.config.max_turns == 2


def test_toolrl_preset_and_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    preset = Path(__file__).resolve().parents[1] / "configs/grpo_v2/toolrl.yaml"
    config = training.load_config(preset, training.Config)
    assert isinstance(config.taskset, training.ToolRLConfig)
    assert config.taskset.dataset_file == training.TOOLRL_TRAIN_FILE
    assert config.max_turns == 1
    assert config.max_tokens == 1024
    assert config.minibatches_per_epoch == training.DEFAULT_CONFIG.minibatches_per_epoch == 4
    assert config.grad_accum_steps == 1
    assert config.update_epochs == training.DEFAULT_CONFIG.update_epochs == 1
    assert config.run_id == f"grpo_v2_toolrl_Qwen_Qwen3-0.6B_seed{config.seed}"
    assert pickle.loads(pickle.dumps(config)) == config
    tasks = Mock(return_value=[Mock()])
    monkeypatch.setattr(training, "ToolRLTaskset", tasks)
    monkeypatch.setattr(training, "serve_vllm", Mock(return_value=MagicMock()))
    monkeypatch.setattr(training, "wait_for_server", Mock())
    actor = training.Actor.__ray_metadata__.modified_class(config=config)
    assert actor.tasks == tasks.return_value
    assert tasks.call_args.kwargs["config"].data_file == config.taskset.dataset_file
    actor.__ray_shutdown__()


def test_codegym_preset_and_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    preset = Path(__file__).resolve().parents[1] / "configs/grpo_v2/codegym.yaml"
    config = training.load_config(preset, training.Config)
    assert isinstance(config.taskset, training.CodeGymConfig)
    assert config.taskset.dataset_file == training.CODEGYM_TRAIN_FILE
    assert config.taskset.manager_url == "http://codegym:8000"
    assert config.max_turns == 32
    assert config.update_epochs == 1
    assert config.minibatches_per_epoch == 4
    assert config.grad_accum_steps == 1
    assert config.run_id == f"grpo_v2_codegym_Qwen_Qwen3-0.6B_seed{config.seed}"
    assert pickle.loads(pickle.dumps(config)) == config
    tasks = Mock(return_value=[Mock()])
    monkeypatch.setattr(training, "CodeGymTaskset", tasks)
    monkeypatch.setattr(training, "serve_vllm", Mock(return_value=MagicMock()))
    monkeypatch.setattr(training, "wait_for_server", Mock())
    actor = training.Actor.__ray_metadata__.modified_class(config=config)
    assert actor.tasks == tasks.return_value
    assert tasks.call_args.kwargs["config"].data_file == config.taskset.dataset_file
    actor.__ray_shutdown__()


@pytest.mark.parametrize("url", ["", "codegym:8000", "file:///tmp/codegym"])
def test_codegym_invalid_manager_url(url: str) -> None:
    with pytest.raises(ValidationError, match="HTTP"):
        training.CodeGymConfig(manager_url=url)


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


def test_script_actor_serialization() -> None:
    # A script's definitions must travel by value to workers with a different entrypoint.
    namespace = runpy.run_path(training.__file__)
    for name in ("Learner", "Actor", "Batcher"):
        actor_class = namespace[name].__ray_metadata__.modified_class
        restored = cloudpickle.loads(cloudpickle.dumps(actor_class))
        assert restored.__name__ == name


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


@pytest.fixture
def actor(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(training, "serve_vllm", Mock(return_value=MagicMock()))
    monkeypatch.setattr(training, "wait_for_server", Mock())
    actor = training.Actor.__ray_metadata__.modified_class()
    monkeypatch.setattr(training.random, "choice", Mock(return_value=actor.tasks[0]))
    return actor


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
        monkeypatch.setattr(training.random, "choice", Mock(return_value=actor.tasks[0]))
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
            token_ids=jnp.array([1, 2], dtype=jnp.int32),  # (T=2,)
            loss_mask=jnp.array([False, True]),  # (T=2,)
            old_log_probs=jnp.array([0.0, -0.2], dtype=jnp.float32),  # (T=2,)
        ),
        reward=1.0,
        advantage=0.5,
        completion="2",
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
        with pytest.raises(RuntimeError, match="rollout failed"):
            learner.fetch_next_batch(num_groups=num_groups, group_size=3)
    else:
        batch = learner.fetch_next_batch(num_groups=num_groups, group_size=3)
        assert batch.groups == groups
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


def test_ray_transfer_timings_exclude_generation_and_slow_group_wait(
    learner: Any, rollout: training.Rollout, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock = Mock(return_value=0.0)
    monkeypatch.setattr(training, "time", Mock(monotonic=clock))
    caplog.set_level("INFO", logger=training.__name__)
    groups = [training.RolloutGroup(task_id=str(i), prompt="test", rollouts=[rollout] * 2) for i in range(2)]
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

    def prepare(groups: list[training.RolloutGroup], context_length: int) -> training.GRPOBatch:
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
    jax.tree.map(
        np.testing.assert_array_equal, [asdict(group) for group in batch.groups], [asdict(group) for group in groups]
    )
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
) -> None:
    caplog.set_level("INFO", logger=training.__name__)
    sequence = training.TrainingSequence(
        token_ids=jnp.array([1], dtype=jnp.int32),  # (T=1,)
        loss_mask=jnp.array([True]),  # (T=1,)
        old_log_probs=jnp.array([-0.2], dtype=jnp.float32),  # (T=1,)
    )
    rollout = training.Rollout(
        sequence=sequence,
        reward=0.0,
        advantage=-1.0,
        completion="wrong",
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
    received = training.RolloutBatch(groups=batch, training_batch=prepared, transfer_metrics=transfer_metrics)
    next_batch = replace(
        received,
        groups=[
            replace(group, rollouts=[replace(rollout, stop_condition="codegym_done") for rollout in group.rollouts])
            for group in batch
        ],
    )
    learner.fetch_next_batch.side_effect = [received, RuntimeError("done") if fails else next_batch]
    learner.publish_weights = steps.publish

    def optimize(batch: training.GRPOBatch) -> dict[str, float]:
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
    else:
        learner.start()
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


def test_learner_logs_examples_every_ten_steps(
    learner: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger=training.__name__)
    rollout = training.Rollout(
        sequence=training.TrainingSequence(
            token_ids=jnp.array([1, 2], dtype=jnp.int32),  # (T=2,)
            loss_mask=jnp.array([False, True]),  # (T=2,)
            old_log_probs=jnp.array([0.0, -0.2], dtype=jnp.float32),  # (T=2,)
        ),
        reward=1.0,
        advantage=0.5,
        completion='<think>Use add.</think>\n<tool_call>{"name": "add"}</tool_call>',
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
            rollouts=[replace(rollout, completion="", stop_condition=None), replace(rollout, completion="omitted")],
        ),
    ]
    learner.config = replace(learner.config, num_groups=2, group_size=2, total_iterations=21)
    learner.initialize_weight_transfer = Mock()
    learner.publish_weights = Mock()
    learner.optimize = Mock(return_value={"losses/policy": 0.0})
    learner.fetch_next_batch = Mock(
        return_value=training.RolloutBatch(
            groups=groups,
            training_batch=Mock(spec=training.GRPOBatch),
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
            'Final assistant output:\n    <think>Use add.</think>\n    <tool_call>{"name": "add"}</tool_call>'
        ) in text
        assert "Example 3 | task=None" in text
        assert "Final assistant output:\n    (empty)" in text
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


def test_actor_initializes_weight_receiver(actor: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    init_info = {"master_address": "127.0.0.1", "master_port": 12345, "rank_offset": 1, "world_size": 2}

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/init_weight_transfer_engine"
        assert json.loads(request.content) == {"init_info": init_info}
        return httpx.Response(status_code=200)

    client = httpx.AsyncClient(base_url=training.SERVER_URL, transport=httpx.MockTransport(respond))
    monkeypatch.setattr(training.httpx, "AsyncClient", Mock(return_value=client))
    asyncio.run(actor.initialize_weight_transfer(init_info))
    assert client.is_closed


@pytest.mark.parametrize("fails", [False, True])
def test_train_gracefully_terminates_actor(fails: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    config = training.Config()
    actor, batcher, learner = Mock(), Mock(), Mock()
    actor.__ray_terminate__ = Mock()
    learner.__ray_terminate__ = Mock()
    monkeypatch.setattr(training, "Actor", Mock(remote=Mock(return_value=actor)))
    monkeypatch.setattr(training, "Batcher", Mock(remote=Mock(return_value=batcher)))
    monkeypatch.setattr(training, "Learner", Mock(remote=Mock(return_value=learner)))
    monkeypatch.setattr(training.ray, "init", MagicMock())
    get = Mock()
    monkeypatch.setattr(training.ray, "get", get)
    wait = Mock()
    monkeypatch.setattr(training.ray, "wait", wait)

    def start_learner() -> Any:
        training.Actor.remote.assert_called_once_with(config=config)
        if fails:
            raise RuntimeError("training failed")
        return learner.start.remote.return_value

    learner.start.remote.side_effect = start_learner
    if fails:
        with pytest.raises(RuntimeError, match="training failed"):
            training.train(config)
    else:
        training.train(config)
    training.Actor.remote.assert_called_once_with(config=config)
    training.Batcher.remote.assert_called_once_with(actor=actor)
    training.Learner.remote.assert_called_once_with(batcher=batcher, actor=actor, config=config)
    actor.start.remote.assert_not_called()
    actor.__ray_terminate__.remote.assert_called_once_with()
    learner.__ray_terminate__.remote.assert_called_once_with()
    wait.assert_called_once_with(
        [learner.__ray_terminate__.remote.return_value, actor.__ray_terminate__.remote.return_value], num_returns=2
    )
    if fails:
        get.assert_not_called()
    else:
        get.assert_called_once_with(learner.start.remote.return_value)


def test_environment_examples(actor: Any) -> None:
    assert len(actor.tasks) == 5
    assert [task.data.idx for task in actor.tasks] == list(range(5))
    assert [task.data.answer for task in actor.tasks] == [42, 21, 9, 3, 43]
    task = actor.tasks[0]
    assert task.data.id == "0"
    assert task.data.prompt == "What is 17 + 25?"
    tools = task.toolsets(task.config)
    assert isinstance(tools[0], AdditionTools)
    assert asyncio.run(tools[0].add(a=17, b=25)) == 42
    agent = training.create_agent()
    assert agent.harness.SUPPORTS_MCP
    config = agent.config
    assert config.harness.id == "null"
    assert config.runtime.type == "subprocess"
    assert config.client.base_url == f"{training.SERVER_URL}/v1"
    assert isinstance(config.client, vf.TrainClientConfig)
    assert config.client.renderer_model_name == training.MODEL
    assert config.max_turns == 3


def test_taskset_applies_config_and_prompt_override(tmp_path: Path) -> None:
    prompt = tmp_path / "system.txt"
    prompt.write_text("Use the addition tool.")
    config = vf.TasksetConfig(system_prompt=prompt)
    tasks = list(AdditionTaskset(config=config))
    assert all(task.config is config.task for task in tasks)
    assert all(task.data.system_prompt == "Use the addition tool." for task in tasks)


@pytest.mark.parametrize(("reply", "reward"), [(" 42\n", 1.0), ("41", 0.0), ("", 0.0)])
def test_correct_answer(reply: str, reward: float, actor: Any) -> None:
    task = actor.tasks[0]
    trace = vf.Trace(
        task=TraceTask(type="AdditionTask", data=task.data),
        agent=AgentInfo(config=training.create_agent().config),
        root_reply=reply,
    )
    assert asyncio.run(task.correct_answer(trace)) == reward


@pytest.mark.parametrize("size", [0, 1])
def test_group_requires_multiple_rollouts(size: int, actor: Any) -> None:
    with pytest.raises(ValueError, match="group_size"):
        asyncio.run(actor.fetch_next_group(group_size=size))


@pytest.mark.parametrize("failure", [None, "rollout", "score", "tokens"])
def test_actor_returns_only_complete_groups(failure: str | None, actor: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    task = actor.tasks[0]
    traces = [
        vf.Trace(
            task=TraceTask(type="AdditionTask", data=task.data),
            agent=AgentInfo(config=training.create_agent().config),
            root_reply=reply,
            nodes=[
                vf.MessageNode(
                    message=vf.AssistantMessage(content=reply),
                    sampled=True,
                    token_ids=[42 if reply == "42" else 41],
                    mask=[True],
                    logprobs=[-0.2],
                )
            ],
            ok=True,
            stop_condition="agent_completed",
        )
        for reply in ["42", "41"]
    ]
    if failure == "rollout":
        traces[1].ok = False
    if failure == "score":
        monkeypatch.setattr(task, "score", AsyncMock(side_effect=ValueError("scoring failed")))
    if failure == "tokens":
        traces[1].nodes = []
    env = AsyncMock()
    env.__aenter__.return_value = env
    seen: list[vf.Task] = []

    async def run(sample: vf.Task) -> vf.Trace:
        index = len(seen)
        seen.append(sample)
        # Finish out of order to check reward/trace/advantage alignment.
        if index == 0:
            await asyncio.sleep(0)
        return traces[index]

    env.run.side_effect = run
    monkeypatch.setattr(training, "create_agent", Mock(return_value=env))
    if failure:
        with pytest.raises(RuntimeError if failure == "rollout" else ValueError):
            asyncio.run(actor.fetch_next_group(group_size=2))
    else:
        result = asyncio.run(actor.fetch_next_group(group_size=2))
        group = result.value
    assert len(seen) == 2
    assert seen[0] is not seen[1] and all(sample is not task for sample in seen)
    assert all(sample.data == task.data and sample.scoring_deferred for sample in seen)
    assert not task.scoring_deferred
    env.__aexit__.assert_awaited_once()
    if not failure:
        assert group.task_id == task.data.id
        assert [rollout.reward for rollout in group.rollouts] == [1.0, 0.0]
        assert [rollout.advantage for rollout in group.rollouts] == pytest.approx([1.0, -1.0], abs=3e-4)
        assert group.prompt == task.data.prompt_text
        assert [rollout.completion for rollout in group.rollouts] == ["42", "41"]
        np.testing.assert_array_equal([rollout.sequence.token_ids for rollout in group.rollouts], [[42], [41]])
        jax.tree.map(np.testing.assert_array_equal, asdict(pickle.loads(pickle.dumps(result))), asdict(result))


@pytest.fixture
def token_trace(actor: Any) -> vf.Trace:
    return vf.Trace(
        task=TraceTask(type="AdditionTask", data=actor.tasks[0].data),
        agent=AgentInfo(config=training.create_agent().config),
        nodes=[
            vf.MessageNode(message=vf.UserMessage(content="question"), token_ids=[10], mask=[False]),
            vf.MessageNode(
                parent=0,
                message=vf.AssistantMessage(content="tool call"),
                sampled=True,
                token_ids=[20],
                mask=[True],
                logprobs=[-0.2],
            ),
            vf.MessageNode(
                parent=1,
                message=vf.ToolMessage(content="42", tool_call_id="call_add"),
                token_ids=[30],
                mask=[False],
            ),
            vf.MessageNode(
                parent=2,
                message=vf.AssistantMessage(content="42"),
                sampled=True,
                token_ids=[40, 41],
                mask=[True, True],
                logprobs=[-0.4, -0.5],
            ),
        ],
    )


@pytest.mark.parametrize("turns", [1, 2], ids=["single-turn", "multi-turn"])
def test_training_sequence_preserves_tokens_and_masks_context(token_trace: vf.Trace, turns: int) -> None:
    token_trace.nodes = token_trace.nodes[: 2 * turns]
    length = 2 if turns == 1 else 5
    sequence = training.training_sequence(token_trace)
    np.testing.assert_array_equal(sequence.token_ids, [10, 20, 30, 40, 41][:length])
    np.testing.assert_array_equal(sequence.loss_mask, [False, True, False, True, True][:length])
    np.testing.assert_allclose(sequence.old_log_probs, [0.0, -0.2, 0.0, -0.4, -0.5][:length])
    assert sequence.token_ids.shape == sequence.loss_mask.shape == sequence.old_log_probs.shape == (length,)
    assert all(isinstance(value, jax.Array) and value.device.platform == "cpu" for value in asdict(sequence).values())
    assert sequence.token_ids.dtype == jnp.int32
    assert sequence.loss_mask.dtype == jnp.bool_
    assert sequence.old_log_probs.dtype == jnp.float32
    jax.tree.map(np.testing.assert_array_equal, asdict(pickle.loads(pickle.dumps(sequence))), asdict(sequence))


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("empty", "exactly one"),
        ("branched", "exactly one"),
        ("no-tokens", "no trainable tokens"),
        ("mask-length", "equal lengths"),
        ("missing-logprobs", "log-probability"),
        ("missing-turn-tokens", "Sampled messages"),
    ],
)
def test_training_sequence_rejects_invalid_tokens(failure: str, message: str, token_trace: vf.Trace) -> None:
    trace = token_trace
    if failure == "empty":
        trace.nodes = []
    elif failure == "branched":
        trace.nodes.append(
            vf.MessageNode(
                parent=1,
                message=vf.AssistantMessage(content="alternative"),
                sampled=True,
                token_ids=[50],
                mask=[True],
                logprobs=[-0.6],
            )
        )

    elif failure == "no-tokens":
        for node in trace.nodes:
            node.token_ids, node.mask, node.logprobs = [], [], []
    elif failure == "mask-length":
        trace.nodes[-1].mask = [True]
    elif failure == "missing-logprobs":
        trace.nodes[-1].logprobs = []
    elif failure == "missing-turn-tokens":
        trace.nodes[-1].token_ids, trace.nodes[-1].mask, trace.nodes[-1].logprobs = [], [], []
    with pytest.raises(ValueError, match=message):
        training.training_sequence(trace)


def test_batcher_collects_groups_concurrently(
    actor: Any, token_trace: vf.Trace, monkeypatch: pytest.MonkeyPatch
) -> None:
    all_started = asyncio.Event()
    started = 0
    env = AsyncMock()
    env.__aenter__.return_value = env

    async def run(task: vf.Task) -> vf.Trace:
        nonlocal started
        started += 1
        if started == 4:
            all_started.set()
        # A serialized actor would wait forever for the second group's rollouts.
        await all_started.wait()
        return token_trace.model_copy(deep=True, update={"ok": True})

    env.run.side_effect = run
    monkeypatch.setattr(training, "create_agent", Mock(return_value=env))
    monkeypatch.setattr(training.random, "choice", Mock(side_effect=actor.tasks[:2]))
    handle = Mock()
    handle.fetch_next_group.remote.side_effect = actor.fetch_next_group
    batcher = training.Batcher.__ray_metadata__.modified_class(actor=handle)

    async def collect() -> training.Transfer[training.RolloutBatch]:
        async with asyncio.timeout(5):
            return await batcher.fetch_next_batch(num_groups=2, group_size=2, context_length=16)

    groups = asyncio.run(collect()).value.groups
    assert started == 4
    assert [group.task_id for group in groups] == [task.data.id for task in actor.tasks[:2]]
    assert [len(group.rollouts) for group in groups] == [2, 2]
    assert env.__aexit__.await_count == 2


def test_group_cancels_pending_rollouts(actor: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    env = AsyncMock()
    env.__aenter__.return_value = env
    started = asyncio.Event()
    cancelled: list[bool] = []
    count = 0

    async def run(task: vf.Task) -> vf.Trace:
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
    with pytest.raises(ExceptionGroup, match="TaskGroup"):
        asyncio.run(actor.collect_group(actor.tasks[0], group_size=2))
    assert cancelled == [True]
    env.__aexit__.assert_awaited_once()


@pytest.fixture
def local_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Use installed dependencies instead of provisioning a uv environment in tests.
    async def prepare_script(
        self: subprocess_runtime.SubprocessRuntime,
        script: str | bytes,
        env: dict[str, str] | None = None,
        *,
        activate: bool = True,
    ) -> list[str]:
        path = tmp_path / "harness.py"
        path.write_bytes(script.encode() if isinstance(script, str) else script)
        return [sys.executable, str(path)]

    monkeypatch.setattr(subprocess_runtime, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(subprocess_runtime.SubprocessRuntime, "prepare_uv_script", prepare_script)


@pytest.fixture
def tokenizer(monkeypatch: pytest.MonkeyPatch) -> PreTrainedTokenizerFast:
    # Exercise the real Qwen3 renderer without downloading a model tokenizer.
    backend = Tokenizer(
        models.BPE(vocab={c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}, merges=[])
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        name_or_path=training.MODEL,
        eos_token="<|im_end|>",
        additional_special_tokens=[
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
            "<think>",
            "</think>",
        ],
    )
    monkeypatch.setattr(renderer_base, "load_tokenizer", Mock(return_value=tokenizer))
    monkeypatch.setattr(ElasticRendererPool, "_renderers", {})
    monkeypatch.setattr(ElasticRendererPool, "_locks", {})
    return tokenizer


@pytest.mark.parametrize("mode", ["success", "tool-error", "provider-error"])
def test_verifiers_tool_rollout(
    mode: str, actor: Any, local_harness: None, tokenizer: PreTrainedTokenizerFast, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[dict[str, Any]] = []
    arguments = '{"a":17}' if mode == "tool-error" else '{"a":17,"b":25}'
    tool_text = '<tool_call>\n{"name":"add","arguments":' + arguments + "}\n</tool_call><|im_end|>"
    tool_ids = tokenizer.encode(tool_text)  # (C_tool,)
    answer_ids = tokenizer.encode("42<|im_end|>")  # (C_answer,)

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(status_code=200, json={"data": [{"id": training.MODEL, "max_model_len": 8192}]})
        assert str(request.url) == f"{training.SERVER_URL}/inference/v1/generate"
        payload = json.loads(request.content)
        requests.append(payload)
        if mode == "provider-error":
            return httpx.Response(status_code=400, json={"error": {"message": "invalid model"}})
        prompt = tokenizer.decode(payload["token_ids"])
        completion_ids = tool_ids if "<tool_response>" not in prompt or mode == "tool-error" else answer_ids  # (C,)
        return httpx.Response(
            status_code=200,
            json={
                "request_id": f"completion-{len(requests)}",
                "choices": [
                    {
                        "index": 0,
                        "token_ids": completion_ids,
                        "finish_reason": "stop",
                        "logprobs": {
                            "content": [{"token": f"token_id:{token}", "logprob": -0.2} for token in completion_ids]
                        },
                    }
                ],
            },
        )

    clients: list[httpx.AsyncClient] = []

    def build_client(config: vf.TrainClientConfig) -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        clients.append(client)
        return client

    monkeypatch.setattr(client_base, "build_async_httpx", build_client)
    if mode == "provider-error":
        with pytest.raises(RuntimeError, match="Rollout failed"):
            asyncio.run(actor.collect_group(actor.tasks[0], group_size=2))
    else:
        group = asyncio.run(actor.collect_group(actor.tasks[0], group_size=2))
        assert len(group.rollouts) == 2
        assert len(requests) == 2 * (2 if mode == "success" else 3)
        expected_samples = tool_ids + answer_ids if mode == "success" else tool_ids * 3  # (S,)
        for rollout in group.rollouts:
            assert rollout.num_turns == (2 if mode == "success" else 3)
            assert rollout.num_output_tokens == len(expected_samples)
            assert rollout.rollout_seconds > 0
            assert rollout.reward == (1.0 if mode == "success" else 0.0)
            assert rollout.advantage == 0.0
            sequence = rollout.sequence
            assert sequence.token_ids.shape == sequence.loss_mask.shape == sequence.old_log_probs.shape
            assert rollout.num_input_tokens == len(sequence.token_ids) - sequence.loss_mask.sum()
            np.testing.assert_array_equal(sequence.token_ids[sequence.loss_mask], expected_samples)
            np.testing.assert_allclose(sequence.old_log_probs, np.where(sequence.loss_mask, -0.2, 0.0))
            if mode == "success":
                assert rollout.completion == "42"
                assert rollout.stop_condition == "agent_completed"
            else:
                assert rollout.stop_condition == "max_turns"
        prompts = [tokenizer.decode(request["token_ids"]) for request in requests]
        if mode == "success":
            assert any("<tool_response>\n42" in prompt for prompt in prompts)
        else:
            observations = [
                prompt.split("<tool_response>\n", 1)[1] for prompt in prompts if "<tool_response>\n" in prompt
            ]
            assert observations and all(
                "required" in observation.lower() and "b" in observation for observation in observations
            )
        jax.tree.map(np.testing.assert_array_equal, asdict(pickle.loads(pickle.dumps(group))), asdict(group))
    assert clients and all(client.is_closed for client in clients)
    assert requests[0]["model"] == training.MODEL
    assert requests[0]["sampling_params"]["max_tokens"] == 512
    assert requests[0]["sampling_params"]["top_p"] == 1.0
    assert requests[0]["sampling_params"]["logprobs"] == 1
    prompt = tokenizer.decode(requests[0]["token_ids"])
    assert '"name": "add"' in prompt
    assert prompt.endswith("<think>\n\n</think>\n\n")


def test_toolrl_renderer_preserves_text(tokenizer: PreTrainedTokenizerFast) -> None:
    tokenizer.clean_up_tokenization_spaces = True
    renderer = ToolRLRenderer(tokenizer)
    text = '<think>Plan .</think>\n<tool_call>\n{"name":"say","parameters":{"text":"Hello ."}}\n</tool_call>'
    tokens = tokenizer.encode(text + "<|im_end|>")  # (T,)
    assert tokenizer.decode(tokens).startswith("<think>Plan.</think>")
    parsed = renderer.parse_response(tokens)
    assert parsed.content == text
    assert parsed.tool_calls == []
    assert parsed.reasoning_content is None
    unfinished = "<think>Still thinking ."
    assert renderer.parse_response(tokenizer.encode(unfinished)).content == unfinished


def test_verifiers_toolrl_rollout(
    actor: Any, local_harness: None, tokenizer: PreTrainedTokenizerFast, monkeypatch: pytest.MonkeyPatch
) -> None:
    actor.config = training.Config(taskset=training.ToolRLConfig(), max_turns=1)
    answer = '<think>Use add.</think>\n<tool_call>\n{"name":"add","parameters":{"a":17,"b":25}}\n</tool_call>'
    task = ToolRLTask(
        data=ToolRLData(
            id="toolrl-test",
            prompt="Add 17 and 25",
            system_prompt="Predict the add call as text.",
            answer=answer,
            expected_calls=[ToolCall(name="add", parameters={"a": 17, "b": 25})],
        ),
        config=vf.TaskConfig(),
    )
    # A truncated second completion must receive a low reward, not abort the batch.
    completions = [answer, '<think>Use add.</think>\n<tool_call>\n{"name":']
    sampled = [tokenizer.encode(text + "<|im_end|>") for text in completions]  # Each (C,).
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(status_code=200, json={"data": [{"id": training.MODEL, "max_model_len": 8192}]})
        ids = sampled[len(requests)]  # (C,)
        requests.append(json.loads(request.content))
        return httpx.Response(
            status_code=200,
            json={
                "request_id": f"toolrl-{len(requests)}",
                "choices": [
                    {
                        "index": 0,
                        "token_ids": ids,
                        "finish_reason": "stop",
                        "logprobs": {"content": [{"token": f"token_id:{token}", "logprob": -0.2} for token in ids]},
                    }
                ],
            },
        )

    def build_client(config: vf.TrainClientConfig) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(respond))

    monkeypatch.setattr(client_base, "build_async_httpx", build_client)
    group = asyncio.run(actor.collect_group(task, group_size=2))
    assert len(requests) == 2  # No tool execution or follow-up model request.
    by_completion = {rollout.completion: rollout for rollout in group.rollouts}
    assert by_completion[answer].reward == 4.0
    assert by_completion[completions[1]].reward == -3.0
    assert by_completion[answer].advantage == pytest.approx(1.0, abs=1e-4)
    assert by_completion[completions[1]].advantage == pytest.approx(-1.0, abs=1e-4)
    for text, ids in zip(completions, sampled, strict=True):
        rollout = by_completion[text]
        assert rollout.num_turns == 1
        sequence = rollout.sequence
        assert sequence.token_ids.shape == sequence.loss_mask.shape == sequence.old_log_probs.shape
        np.testing.assert_array_equal(sequence.token_ids[sequence.loss_mask], ids)
        np.testing.assert_allclose(sequence.old_log_probs, np.where(sequence.loss_mask, -0.2, 0.0))
    prompt = tokenizer.decode(requests[0]["token_ids"])
    assert prompt.endswith("<|im_start|>assistant\n")
    assert answer not in prompt


@pytest.mark.parametrize("mode", ["success", "invalid-action", "task-limit", "agent-limit", "step-error"])
def test_verifiers_codegym_rollout(
    mode: str, actor: Any, local_harness: None, tokenizer: PreTrainedTokenizerFast, monkeypatch: pytest.MonkeyPatch
) -> None:
    actor.config = training.Config(
        taskset=training.CodeGymConfig(manager_url="http://codegym:8000"),
        max_turns=1 if mode == "agent-limit" else 3,
    )
    task = codegym_env.CodeGymTask(
        data=codegym_env.CodeGymData(
            id="codegym-test",
            prompt=[vf.UserMessage(content="Use Observe then Done to add the values.")],
            ability='codegym_v1@Example__SumEnv@{"values":[1,2]}',
            max_steps=1 if mode == "task-limit" else 3,
        ),
        config=vf.TaskConfig(),
    )
    first_text = (
        "not a function call"
        if mode == "invalid-action"
        else '<|FunctionCallBegin|>[{"name":"Observe","parameters":{}}]<|FunctionCallEnd|>'
    )
    final_text = '<|FunctionCallBegin|>[{"name":"Done","parameters":{"answer":3}}]<|FunctionCallEnd|>'
    first_ids, final_ids = [
        tokenizer.encode(text + "<|im_end|>") for text in [first_text, final_text]
    ]  # (C_first,), (C_final,)
    sessions: dict[str, dict[str, Any]] = {}
    allocated: list[str] = []
    released: list[str] = []
    stopped: list[str] = []
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": training.MODEL, "max_model_len": 8192}]})
        if path == "/inference/v1/generate":
            payload = json.loads(request.content)
            requests.append(payload)
            prompt = tokenizer.decode(payload["token_ids"])
            later = "Observed values" in prompt or "Invalid function call" in prompt
            ids = final_ids if later else first_ids  # (C,)
            assert "Initial state" in prompt
            return httpx.Response(
                200,
                json={
                    "request_id": f"codegym-{len(requests)}",
                    "choices": [
                        {
                            "index": 0,
                            "token_ids": ids,
                            "finish_reason": "stop",
                            "logprobs": {"content": [{"token": f"token_id:{token}", "logprob": -0.2} for token in ids]},
                        }
                    ],
                },
            )
        if path == "/get_instance":
            uid = str(len(allocated))
            allocated.append(uid)
            return httpx.Response(200, json={"uid": uid, "port": 10000 + int(uid)})
        if path == "/release_instance":
            uid = request.url.params["uid"]
            assert uid in stopped
            released.append(uid)
            return httpx.Response(200, json={})
        payload = json.loads(request.content) if request.method == "POST" else dict(request.url.params)
        session_id = payload["session_id"]
        if path == "/start":
            assert session_id not in sessions
            sessions[session_id] = {"done": False, "steps": [], "uid": str(request.url.port - 10000)}
            return httpx.Response(200, json={"observation": "Initial state"})
        session = sessions[session_id]
        if path == "/step":
            action = json.loads(payload["action"])
            session["steps"].append(action["name"])
            session["done"] = action["name"] == "Done"
            return httpx.Response(
                500 if mode == "step-error" else 200, json={"status": True, "observation": "Observed values: [1, 2]"}
            )
        if path == "/is_done":
            return httpx.Response(200, json={"done": session["done"]})
        if path == "/reward":
            # Distinct rewards exercise per-rollout state and within-group normalization.
            reward = 2 if session["done"] and session["uid"] == "0" else 0
            return httpx.Response(200, json={"reward": reward})
        assert path == "/stop"
        stopped.append(session["uid"])
        return httpx.Response(200, json={})

    class MockClient(httpx.AsyncClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(transport=httpx.MockTransport(respond), **kwargs)

    def build_client(config: vf.TrainClientConfig) -> httpx.AsyncClient:
        return MockClient()

    monkeypatch.setattr(codegym_env.httpx, "AsyncClient", MockClient)
    monkeypatch.setattr(client_base, "build_async_httpx", build_client)
    if mode == "step-error":
        with pytest.raises(ExceptionGroup) as caught:
            asyncio.run(actor.collect_group(task, group_size=2))
        assert all(isinstance(error, httpx.HTTPStatusError) for error in caught.value.exceptions)
    else:
        group = asyncio.run(actor.collect_group(task, group_size=2))
        limited = mode in {"task-limit", "agent-limit"}
        expected_ids = first_ids if limited else first_ids + final_ids  # (S,)
        assert len(requests) == 2 * (1 if limited else 2)
        assert sorted(rollout.reward for rollout in group.rollouts) == ([0, 0] if limited else [0, 1])
        assert sorted(rollout.advantage for rollout in group.rollouts) == pytest.approx(
            [0, 0] if limited else [-1, 1], abs=3e-4
        )
        for rollout in group.rollouts:
            assert rollout.num_turns == (1 if limited else 2)
            assert rollout.stop_condition == (
                "codegym_max_steps"
                if mode == "task-limit"
                else "max_turns"
                if mode == "agent-limit"
                else "codegym_done"
            )
            sequence = rollout.sequence
            assert sequence.token_ids.shape == sequence.loss_mask.shape == sequence.old_log_probs.shape
            assert sequence.token_ids.ndim == 1
            np.testing.assert_array_equal(sequence.token_ids[sequence.loss_mask], expected_ids)
            np.testing.assert_allclose(sequence.old_log_probs, np.where(sequence.loss_mask, -0.2, 0))
        for session in sessions.values():
            assert session["steps"] == (
                ["Observe"] if limited else ["Done"] if mode == "invalid-action" else ["Observe", "Done"]
            )
    assert len(sessions) == 2
    assert sorted(released) == sorted(stopped) == allocated
