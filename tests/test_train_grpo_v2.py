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
import threading
from collections.abc import Callable, Generator, Iterator
from contextlib import AsyncExitStack, ExitStack, aclosing
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
from pydantic import TypeAdapter, ValidationError
from ray import cloudpickle
from ray._private import ray_constants
from ray.exceptions import RayTaskError

from rl2 import train_grpo_v2 as training
from rl2.agent_task import Environment, Task
from rl2.alfworld_env import ALFWorldTask
from rl2.grpo_v2.reasoning_gym import ReasoningGymConfig, ReasoningGymSession, ReasoningGymTask, create_dataset
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
    assert defaults.finetuning == training.LoRAConfig()
    assert defaults.learning_rate == 1e-6
    assert defaults.weight_decay == 0.0
    assert defaults.adamw_kwargs["weight_decay"] == 0.0
    assert defaults.adamw_batch_size_scale == 1.0
    assert defaults.prefetch_batches == 0
    assert defaults.parameter_sync_steps == 1
    assert defaults.grad_accum_steps == 1
    assert defaults.max_staleness == 100
    assert not defaults.drop_degenerate_groups
    assert defaults.length_penalty == training.LengthPenaltyConfig(reference_tokens=8192, penalty=0.0)
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
                "run_id": "test_${finetuning.mode}",
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
                "length_penalty": {"reference_tokens": 4096, "penalty": 0.05},
                "loss": {"kind": "grpo", "rollout_is_threshold": 3.0},
                "prefetch_batches": 3,
                "parameter_sync_steps": 4,
                "max_staleness": 2,
                "finetuning": {"mode": "full"},
                "weight_decay": 0.05,
                "adamw_batch_size_scale": 0.5,
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
            length_penalty=training.LengthPenaltyConfig(reference_tokens=4096, penalty=0.05),
            loss=training.GRPOLossConfig(rollout_is_threshold=3.0),
            prefetch_batches=3,
            parameter_sync_steps=4,
            max_staleness=2,
            finetuning=training.FullFinetuningConfig(),
            weight_decay=0.05,
            adamw_batch_size_scale=0.5,
        )
    )
    assert pickle.loads(pickle.dumps(config)) == config


@pytest.mark.parametrize(
    "mode,expected",
    [
        ({"mode": "lora"}, training.LoRAConfig()),
        ({"mode": "lora", "rank": 512, "alpha": 8}, training.LoRAConfig(rank=512, alpha=8)),
        ({"mode": "full"}, training.FullFinetuningConfig()),
    ],
    ids=["lora-defaults", "lora-overrides", "full"],
)
def test_finetuning_config(
    tmp_path: Path, mode: dict[str, Any], expected: training.LoRAConfig | training.FullFinetuningConfig
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"finetuning": mode}))
    config = training.load_config(path, training.Config)
    assert config.finetuning == expected
    assert asdict(config)["finetuning"] == asdict(expected)


@pytest.mark.parametrize(
    "settings",
    [
        {"finetuning": {"mode": "full", "rank": 16}},
        {"finetuning": {"mode": "full", "alpha": 32}},
        {"finetuning": {"mode": "other"}},
        {"finetuning": {"rank": 16}},
        {"finetuning": "lora"},
        {"lora_rank": 16},
        {"lora_alpha": 32},
    ],
    ids=["full-rank", "full-alpha", "unknown-tag", "missing-tag", "old-string", "top-level-rank", "top-level-alpha"],
)
def test_finetuning_rejects_invalid_config(tmp_path: Path, settings: dict[str, Any]) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValidationError):
        training.load_config(path, training.Config)


@pytest.mark.parametrize(
    "batch_size_scale,expected",
    [
        (1.0, {"learning_rate": 1e-6, "b1": 0.9, "b2": 0.999, "eps": 1e-8, "weight_decay": 0.01}),
        (
            0.5,
            {
                "learning_rate": 7.071067811865476e-7,
                "b1": 0.95,
                "b2": 0.9995,
                "eps": 1.414213562373095e-8,
                "weight_decay": 0.007071067811865476,
            },
        ),
        (
            2.0,
            {
                "learning_rate": 1.4142135623730952e-6,
                "b1": 0.8,
                "b2": 0.998,
                "eps": 7.071067811865475e-9,
                "weight_decay": 0.014142135623730952,
            },
        ),
        (
            10.0,
            {
                "learning_rate": 3.16227766016838e-6,
                "b1": 0.0,
                "b2": 0.99,
                "eps": 3.162277660168379e-9,
                "weight_decay": 0.0316227766016838,
            },
        ),
    ],
    ids=["identity", "smaller-batch", "larger-batch", "zero-beta1-boundary"],
)
def test_adamw_scaling(batch_size_scale: float, expected: dict[str, float]) -> None:
    config = training.Config(adamw_batch_size_scale=batch_size_scale, weight_decay=0.01)
    assert config.adamw_kwargs == pytest.approx(expected)
    assert config.learning_rate == 1e-6
    assert config.weight_decay == 0.01
    assert (config.batch_size, config.group_size) == (4, 4)


def test_example_log_interval_config(tmp_path: Path) -> None:
    assert training.Config().example_log_interval == 10
    path = tmp_path / "config.yaml"
    path.write_text("example_log_interval: 1\n")
    assert training.load_config(path, training.Config).example_log_interval == 1


@pytest.mark.parametrize(
    ("taskset", "stream_name"),
    [
        (training.AppWorldConfig(python="/test/python", root="/test/appworld"), "stream_appworld_tasks"),
        (training.WebShopConfig(root="/test/webshop"), "stream_webshop_tasks"),
    ],
    ids=["appworld", "webshop"],
)
def test_external_actor_task_stream(
    taskset: training.AppWorldConfig | training.WebShopConfig, stream_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = training.Config(tasksets={taskset.id: training.WeightedTasksetConfig(taskset=taskset)})
    assert pickle.loads(pickle.dumps(config)) == config
    task = Mock(spec=training.Task)
    stream = iter([task])
    stream_tasks = Mock(return_value=stream)
    monkeypatch.setattr(training, stream_name, stream_tasks)
    monkeypatch.setattr(training, "serve_vllm", Mock(return_value=MagicMock()))
    monkeypatch.setattr(training, "wait_for_server", Mock())
    actor = training.Actor.__ray_metadata__.modified_class(config)

    with actor._server:
        assert next(actor.tasks) == training.SampledTask(taskset_id=taskset.id, task=task)
    stream_tasks.assert_called_once_with(taskset, actor.rng, config.environment_timeout)


def test_reasoning_gym_cli_and_actor_task_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = training.Config(
        tasksets={
            "easy": training.WeightedTasksetConfig(
                taskset=training.ReasoningGymConfig(
                    task="leg_counting",
                    task_config={"min_animals": 1, "max_animals": 5},
                    python_namespace="sandbox-test",
                    python_kubectl=("sudo", "-n", "k3s", "kubectl"),
                ),
                weight=1.0,
            ),
            "hard": training.WeightedTasksetConfig(
                taskset=training.ReasoningGymConfig(
                    task="leg_counting", task_config={"min_animals": 6, "max_animals": 8}
                ),
                weight=3.0,
            ),
        },
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
    restored = actor_class(replace(config, tasksets=dict(reversed(config.tasksets.items()))))
    actor.collect_group = AsyncMock(return_value=Mock(spec=training.RolloutGroup))
    restored.collect_group = AsyncMock(return_value=Mock(spec=training.RolloutGroup))

    async def fetch_groups(instance: Any) -> None:
        async with asyncio.TaskGroup() as tg:
            for _ in range(8):
                tg.create_task(instance.fetch_next_group(group_size=2))

    with actor._server, restored._server:
        assert iter(actor.tasks) is actor.tasks
        asyncio.run(actor.fetch_next_group(group_size=2))
        first = actor.collect_group.call_args.args[0]
        state = json.loads(json.dumps(actor.get_sampling_state()))
        actor.collect_group.reset_mock()
        asyncio.run(fetch_groups(actor))
        expected = actor.collect_group.call_args_list
        tasks = [call.args[0] for call in expected]
        assert len({first.id, *(task.id for task in tasks)}) == 9
        assert all(task.config in [entry.taskset for entry in config.tasksets.values()] for task in tasks)
        assert {task.config.task_config["min_animals"] for task in tasks} == {1, 6}
        assert all(
            call.kwargs
            == {
                "group_size": 2,
                "taskset_id": "easy" if call.args[0].config.task_config["min_animals"] == 1 else "hard",
            }
            for call in expected
        )
        restored.set_sampling_state(state)
        asyncio.run(fetch_groups(restored))
        assert restored.collect_group.call_args_list == expected
        assert restored.get_sampling_state() == actor.get_sampling_state()


@pytest.mark.parametrize(
    "settings",
    [
        {"batch_size": 0},
        {"batch_size": 1.5},
        pytest.param({"num_groups": 1}, id="removed-num-groups"),
        {"group_size": 1},
        pytest.param({"continual_group_filling": True}, id="removed-continual-group-filling"),
        pytest.param({"max_resampling_attempts": 100}, id="removed-max-resampling-attempts"),
        {"prefetch_batches": -1},
        {"prefetch_batches": 1.5},
        {"prefetch_mode": "next_ready"},
        {"parameter_sync_steps": 0},
        {"parameter_sync_steps": 1.5},
        {"max_staleness": -1},
        {"max_staleness": 1.5},
        {"max_staleness": None},
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
        {"weight_decay": -0.01},
        {"adamw_batch_size_scale": 0.0},
        {"adamw_batch_size_scale": -0.5},
        {"adamw_batch_size_scale": 10.01},
        pytest.param({"length_penalty_coef": 0.0}, id="removed-length-coefficient"),
        pytest.param({"max_tokens_penalty": 0.0}, id="removed-max-tokens-penalty"),
        {"loss": {"kind": "grpo", "clip_coef": 1.0}},
        {"loss": {"kind": "grpo", "rollout_is_threshold": 0.0}},
        pytest.param({"clip_coef": 0.2}, id="removed-top-level-loss-setting"),
        {"max_grad_norm": 0.0},
        pytest.param({"minibatches_per_epoch": 1}, id="removed-minibatching"),
        {"grad_accum_steps": 0},
        {"grad_accum_steps": 1.5},
        pytest.param({"parameter_sync_step": 1}, id="renamed-parameter-sync"),
        pytest.param({"update_epochs": 1}, id="removed-update-epochs"),
        {"attention_backend": "unknown"},
        {"weight_update_pause_mode": "unknown"},
        pytest.param({"weight_update_pause_mode": "abort"}, id="removed-abort-mode"),
        {"tasksets": "unknown"},
    ],
)
def test_config_rejects_invalid_settings(settings: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        training.Config(**settings)


@pytest.mark.parametrize(
    "loss",
    [
        training.CISPOConfig(clip_low_threshold=0.5, clip_high_threshold=2.0),
        training.GRPOLossConfig(clip_coef=0.15, rollout_is_threshold=3.0),
        training.StableReinforceConfig(top_k=8, kl_coeff=0.02, kl_estimator="k3"),
    ],
    ids=["cispo", "grpo", "stable-reinforce"],
)
def test_tagged_loss_round_trip(loss: training.LossConfig, tmp_path: Path) -> None:
    path = tmp_path / "loss.yaml"
    path.write_text(yaml.safe_dump({"loss": asdict(loss)}))
    config = training.load_config(path, training.Config)
    assert type(config.loss) is type(loss)
    assert config.loss == loss
    assert pickle.loads(pickle.dumps(config)) == config
    assert TypeAdapter(training.Config).validate_json(json.dumps(asdict(config))) == config


@pytest.mark.parametrize(
    "loss,error",
    [
        ({}, "union_tag_not_found"),
        ({"kind": "unknown"}, "union_tag_invalid"),
        ({"kind": "cispo", "clip_coef": 0.2}, "unexpected_keyword_argument"),
        ({"kind": "grpo", "top_k": 32}, "unexpected_keyword_argument"),
        ({"kind": "stable_reinforce", "clip_coef": 0.2}, "unexpected_keyword_argument"),
        (None, "dict_type"),
    ],
)
def test_tagged_loss_rejects_invalid_variants(loss: dict[str, object] | None, error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        TypeAdapter(training.Config).validate_json(json.dumps({"loss": loss}))


@pytest.mark.parametrize(
    ("settings", "error"),
    [
        ({"tasksets": {"example": {"taskset": {"id": "addition", "config_file": "config.yaml"}}}}, "union_tag_invalid"),
        ({"tasksets": {"example": {"taskset": {"id": "alfworld", "config_file": None}}}}, "string_type"),
        (
            {"tasksets": {"example": {"taskset": {"id": "alfworld", "config_file": "config.yaml", "unknown": True}}}},
            "unexpected_keyword_argument",
        ),
        ({"tasksets": {}}, "too_short"),
        ({"tasksets": [{"taskset": {"id": "reasoning_gym"}}]}, "dict_type"),
        ({"tasksets": {" ": {"taskset": {"id": "reasoning_gym"}}}}, "string_pattern_mismatch"),
        ({"tasksets": {"example": {"taskset": {"id": "reasoning_gym"}, "weight": 0}}}, "positive weight"),
        ({"taskset": {"id": "alfworld", "config_file": "config.yaml"}}, "unexpected_keyword_argument"),
        ({"dataset_file": "train.parquet"}, "unexpected_keyword_argument"),
    ],
)
def test_taskset_config_rejects_invalid_yaml(settings: dict[str, object], error: str, tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValidationError, match=error):
        training.load_config(path, training.Config)


def test_duplicate_taskset_identifiers_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "tasksets:\n"
        "  easy:\n    taskset: {id: reasoning_gym, task: leg_counting}\n"
        "  easy:\n    taskset: {id: reasoning_gym, task: basic_arithmetic}\n"
    )
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key"):
        training.load_config(path, training.Config)


@pytest.mark.parametrize("name", ["Legs", "leg-counting", "legs/other", "leg counting", "1legs", "legs_", "a__b"])
def test_taskset_identifiers_require_snake_case(name: str) -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        training.Config(tasksets={name: training.WeightedTasksetConfig(taskset=training.ReasoningGymConfig())})


def test_default_taskset_mapping_is_not_shared() -> None:
    first = training.Config()
    second = training.Config()
    first.tasksets.clear()
    assert list(second.tasksets) == ["alfworld"]


def test_explicit_taskset_requires_discriminator() -> None:
    with pytest.raises(ValidationError) as raised:
        training.WeightedTasksetConfig(taskset={"config_file": "config.yaml"})
    assert [(error["loc"], error["type"]) for error in raised.value.errors()] == [(("taskset",), "union_tag_not_found")]


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")], ids=["negative", "nan", "infinite"])
def test_taskset_weight_rejects_invalid_values(weight: float) -> None:
    with pytest.raises(ValidationError):
        training.WeightedTasksetConfig(taskset=training.ReasoningGymConfig(), weight=weight)


@pytest.mark.parametrize("name", ["alfworld", "appworld", "reasoning_gym", "codegym", "webshop"])
def test_taskset_presets(name: str) -> None:
    path = Path(__file__).parents[1] / "configs" / "grpo_v2" / f"{name}.yaml"
    config = training.load_config(path, training.Config)
    assert len(config.tasksets) == 1
    key = "leg_counting" if name == "reasoning_gym" else name
    assert list(config.tasksets) == [key]
    assert config.tasksets[key].taskset.id == name
    assert config.tasksets[key].weight == 1.0
    if name == "reasoning_gym":
        assert "reasoning_gym_leg_counting_" in config.run_id


def test_codegym_task_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    config = training.CodeGymConfig()
    task = Mock(spec=training.Task)
    load = Mock(return_value=iter([task]))
    monkeypatch.setattr(training, "stream_codegym_tasks", load)
    rng = random.Random(3)
    stream = training.stream_tasks({"codegym": training.WeightedTasksetConfig(taskset=config)}, rng, timeout=120)
    assert next(stream) == training.SampledTask(taskset_id="codegym", task=task)
    load.assert_called_once_with(config, rng)


def test_mix_preset_accepts_native_reasoning_rewards() -> None:
    path = Path(__file__).parents[1] / "configs/grpo_v2/mix.yaml"
    config = training.load_config(path, training.Config)
    assert config.tasksets["alfworld"].taskset.id == "alfworld"
    assert all(entry.weight == 1.0 for entry in config.tasksets.values())
    tasks = [
        entry.taskset for entry in config.tasksets.values() if isinstance(entry.taskset, training.ReasoningGymConfig)
    ]
    assert {task.task for task in tasks} == {
        "basic_arithmetic",
        "leg_counting",
        "letter_counting",
        "number_sorting",
        "gcd",
        "lcm",
        "count_primes",
        "number_filtering",
        "word_sorting",
        "caesar_cipher",
        "rotate_matrix",
        "base_conversion",
        "shortest_path",
        "group_anagrams",
        "fraction_simplification",
        "largest_island",
        "dice",
        "bitwise_arithmetic",
        "puzzle24",
        "mini_sudoku",
        "knights_knaves",
        "course_schedule",
        "game_of_life",
    }
    rewards: list[float] = []
    for task in tasks:
        dataset = create_dataset(task, seed=config.seed)
        entry = dataset[0]
        assert entry["question"]
        assert dataset.score_answer(answer=entry["answer"], entry=entry) == 1.0
        rewards.append(dataset.score_answer(answer=f"The answer is {entry['answer']}", entry=entry))
    assert any(0 < reward < 1 for reward in rewards)
    group = training.RolloutGroup(
        task_id="test-task",
        rollouts=[Mock(spec=training.Rollout, reward=reward, num_output_tokens=4096) for reward in rewards],
        length_penalty=config.length_penalty,
        taskset_id="test",
    )
    assert group.training_rewards == pytest.approx([reward * 0.975 for reward in rewards])


def test_weighted_task_stream(monkeypatch: pytest.MonkeyPatch, task_discovery: list[ALFWorldTask]) -> None:
    config = training.Config(
        tasksets={
            "household": training.WeightedTasksetConfig(
                taskset=training.ALFWorldConfig(config_file="config.yaml"), weight=0.5
            ),
            "legs": training.WeightedTasksetConfig(taskset=training.ReasoningGymConfig(), weight=1.5),
            "apps": training.WeightedTasksetConfig(
                taskset=training.AppWorldConfig(python="/test/python", root="/test/appworld"), weight=0.0
            ),
        }
    )
    rng = random.Random(1)
    monkeypatch.setattr(rng, "choices", Mock(side_effect=AssertionError("Taskset selection must be deterministic")))
    appworld_stream = Mock(side_effect=AssertionError("Disabled taskset must not be initialized"))
    monkeypatch.setattr(training, "stream_appworld_tasks", appworld_stream)
    tasks = training.stream_tasks(config.tasksets, rng, config.environment_timeout)
    training.stream_alfworld_tasks.assert_called_once_with(
        config.tasksets["household"].taskset, rng, config.environment_timeout
    )
    for name in ["legs", "household", "legs", "legs"] * 3:
        sampled = next(tasks)
        assert sampled.taskset_id == name
        if name == "household":
            assert sampled.task in task_discovery
        else:
            assert sampled.task.config == config.tasksets["legs"].taskset
    appworld_stream.assert_not_called()


def test_weighted_task_stream_large_weights() -> None:
    config = training.Config(
        tasksets={
            "legs": training.WeightedTasksetConfig(taskset=training.ReasoningGymConfig(), weight=1e308),
            "arithmetic": training.WeightedTasksetConfig(
                taskset=training.ReasoningGymConfig(task="basic_arithmetic"), weight=1e308
            ),
        }
    )
    tasks = training.stream_tasks(config.tasksets, random.Random(1), config.environment_timeout)
    reordered = training.stream_tasks(
        dict(reversed(config.tasksets.items())), random.Random(99), config.environment_timeout
    )
    expected = ["arithmetic", "legs"] * 4
    assert [next(tasks).taskset_id for _ in expected] == expected
    assert [next(reordered).taskset_id for _ in expected] == expected


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
        "--enable-scale-out",
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
        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "30",
    }
    if failure == "launch":
        server.terminate.assert_not_called()
    else:
        server.terminate.assert_called_once_with()
        server.__exit__.assert_called_once()
        server.kill.assert_not_called()


def test_serve_vllm_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "vllm:\n"
        "  max_model_len: 8192\n"
        "  max_num_seqs: 160\n"
        "  max_num_batched_tokens: 8192\n"
        "  gpu_memory_utilization: 0.95\n"
        "  enable_prefix_caching: true\n"
        "  enable_chunked_prefill: false\n"
        "  async_scheduling: true\n"
    )
    config = training.load_config(path, training.Config)
    launch = Mock(return_value=MagicMock())
    monkeypatch.setattr(training.subprocess, "Popen", launch)
    with training.serve_vllm(config=config.vllm):
        pass
    command = launch.call_args.args[0]
    start = command.index("--max-model-len")
    assert command[start : command.index("--enable-lora")] == [
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "160",
        "--max-num-batched-tokens",
        "8192",
        "--gpu-memory-utilization",
        "0.95",
        "--enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--async-scheduling",
    ]


@pytest.mark.parametrize(
    "values",
    [
        {"max_model_len": 0},
        {"max_num_seqs": 0},
        {"max_num_batched_tokens": 0},
        {"gpu_memory_utilization": 0},
        {"gpu_memory_utilization": 1.01},
        {"unknown_option": True},
    ],
)
def test_vllm_config_rejects_invalid_settings(values: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        training.VLLMConfig(**values)


def test_serve_vllm_kills_unresponsive_server(monkeypatch: pytest.MonkeyPatch) -> None:
    server = MagicMock()
    server.__enter__.return_value = server
    server.wait.side_effect = subprocess.TimeoutExpired(cmd="vllm", timeout=30)
    monkeypatch.setattr(training.subprocess, "Popen", Mock(return_value=server))

    with training.serve_vllm():
        pass

    server.terminate.assert_called_once_with()
    server.wait.assert_called_once_with(timeout=30)
    server.kill.assert_called_once_with()
    server.__exit__.assert_called_once()


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
                "config_type = create_state.__annotations__['config']\n"
                "lora_config = type(config_type().finetuning)(rank=1, alpha=3)\n"
                "config = config_type(finetuning=lora_config)\n"
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


async def collect_and_close(actor: Any, task: Task) -> training.RolloutGroup:
    async with aclosing(actor):
        return await actor.collect_group(task, group_size=2, taskset_id="alfworld")


def test_actor_fetch_next_group(
    actor: Any, monkeypatch: pytest.MonkeyPatch, task_discovery: list[ALFWorldTask]
) -> None:
    collect = AsyncMock(return_value=Mock(spec=training.RolloutGroup))
    monkeypatch.setattr(actor, "collect_group", collect)

    result = asyncio.run(actor.fetch_next_group(group_size=2))

    training.stream_alfworld_tasks.assert_called_once_with(
        actor.config.tasksets["alfworld"].taskset, actor.rng, actor.config.environment_timeout
    )
    actor.rng.choice.assert_called_once_with(task_discovery)
    collect.assert_awaited_once_with(task_discovery[0], group_size=2, taskset_id="alfworld")
    assert result is collect.return_value


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
            first = asyncio.create_task(actor.collect_group(task_discovery[0], group_size=2, taskset_id="alfworld"))
            await entering.wait()
            second = asyncio.create_task(actor.collect_group(task_discovery[1], group_size=2, taskset_id="alfworld"))
            await asyncio.sleep(0)
            create.assert_called_once_with(actor.config)
            release.set()
            groups = await asyncio.gather(first, second)
            assert [group.task_id for group in groups] == [f"alfworld:{task.id}" for task in task_discovery[:2]]
            await actor.collect_group(task_discovery[0], group_size=2, taskset_id="alfworld")
            agent.__aexit__.assert_not_awaited()
        await actor.aclose()
        with pytest.raises(RuntimeError, match="Actor is closed"):
            await actor.collect_group(task_discovery[0], group_size=2, taskset_id="alfworld")

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
    actor: Any,
    monkeypatch: pytest.MonkeyPatch,
    task_discovery: list[ALFWorldTask],
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
            group = asyncio.create_task(actor.collect_group(task_discovery[0], group_size=2, taskset_id="alfworld"))
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
    config = training.Config(
        model="Qwen/custom",
        finetuning=training.LoRAConfig() if mode == "lora" else training.FullFinetuningConfig(),
        vllm=training.VLLMConfig(max_num_batched_tokens=4096),
        loss=training.StableReinforceConfig() if mode == "full" else training.GRPOLossConfig(),
    )
    server = MagicMock()
    serve = Mock(return_value=server)
    monkeypatch.setattr(training, "serve_vllm", serve)
    ready = Mock(side_effect=TimeoutError("not ready") if startup_fails else None)
    monkeypatch.setattr(training, "wait_for_server", ready)
    group = training.RolloutGroup(task_id="0", rollouts=[], taskset_id="test")

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
            assert asyncio.run(actor.fetch_next_group(group_size=2)) is group
        actor.collect_group.assert_awaited_with(task_discovery[0], group_size=2, taskset_id="alfworld")
        assert actor.collect_group.await_count == 2
        server.__exit__.assert_not_called()
        asyncio.run(actor.aclose())
        asyncio.run(actor.aclose())
    serve.assert_called_once_with(
        model=config.model,
        lora_rank=config.finetuning.rank if mode == "lora" else None,
        config=config.vllm,
        **({"sampler_top_k": 32} if mode == "full" else {}),
    )
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
            weight_versions=np.where(np.array([False, True]), 0, -1).astype(np.int32),  # (T,)
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
    learner.config = replace(learner.config, batch_size=num_groups * 3, group_size=3)
    groups = [
        training.RolloutGroup(task_id=str(i), rollouts=[rollout] * 3, taskset_id="test") for i in range(num_groups)
    ]
    actor.fetch_next_group.remote = AsyncMock(
        side_effect=[groups[0], RuntimeError("rollout failed")] if fails else groups
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
    include_examples = iteration % learner.config.example_log_interval == 0

    if fails:
        with pytest.raises(RuntimeError, match="rollout failed"):
            learner.fetch_next_batch(iteration=iteration)
    else:
        batch = learner.fetch_next_batch(iteration=iteration)
        assert not hasattr(batch, "groups")
        assert batch.stop_counts == {"agent_completed": num_groups * 3}
        assert batch.num_output_tokens == num_groups * 3
        if include_examples:
            assert batch.examples.count("Full token sequence (special tokens preserved):") == 1
            format_examples.assert_called_once()
        else:
            assert batch.examples == ""
            format_examples.assert_not_called()
        assert batch.training_batch.token_ids.shape == (num_groups * 3, 8)
        np.testing.assert_array_equal(batch.training_batch.advantages, 0.5)
    batcher_handle.fetch_next_batch.remote.assert_called_once_with(
        batch_size=num_groups * 3,
        group_size=3,
        context_length=8,
        iteration=iteration,
        consumption_weight_version=0,
        include_examples=include_examples,
    )
    assert actor.fetch_next_group.remote.call_count == num_groups
    assert all(call.kwargs == {"group_size": 3} for call in actor.fetch_next_group.remote.call_args_list)


@pytest.mark.parametrize(
    ("prefetch_batches", "checkpoint_interval", "total_iterations", "expected_requests"),
    [
        (0, 100, 9, [2, 4]),
        (3, 100, 12, [8, 10, 10, 10, 10]),
        (3, 100, 8, [2]),
        (3, 12, None, [8, 10, 10, 10, 10, 18]),
        (1, 100, None, [4, 6, 8]),
    ],
    ids=["disabled", "training-end", "final-batch-only", "checkpoint", "steady-pool"],
)
def test_batcher_prefetches_bounded_groups(
    prefetch_batches: int,
    checkpoint_interval: int,
    total_iterations: int | None,
    expected_requests: list[int],
    rollout: training.Rollout,
) -> None:
    actor = Mock()
    actor.fetch_next_group.remote = AsyncMock(
        return_value=training.RolloutGroup(task_id="0", rollouts=[rollout] * 2, taskset_id="test")
    )
    batcher = training.Batcher.__ray_metadata__.modified_class(
        actor=actor,
        config=training.Config(
            prefetch_batches=prefetch_batches,
            checkpoint_interval=checkpoint_interval,
            total_iterations=total_iterations,
        ),
    )

    async def collect() -> None:
        async with asyncio.timeout(5):
            for iteration, requests in enumerate(expected_requests, start=8):
                batch = await batcher.fetch_next_batch(
                    batch_size=4, group_size=2, context_length=8, iteration=iteration
                )
                assert batch.training_batch.token_ids.shape == (4, 8)
                # Let newly scheduled groups start while the learner processes the batch.
                await asyncio.sleep(0)
                assert actor.fetch_next_group.remote.call_count == requests
                assert len(batcher._pending_groups) == requests - 2 * (iteration - 7)
            await batcher.cancel_pending_groups()
            assert not batcher._pending_groups

    asyncio.run(collect())


def test_batcher_replenishes_each_pop(rollout: training.Rollout, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training.ray, "cancel", Mock())

    async def collect() -> None:
        pending = [asyncio.Future[training.RolloutGroup]() for _ in range(6)]
        started: asyncio.Queue[int] = asyncio.Queue()
        requested = 0
        group = training.RolloutGroup(task_id="0", taskset_id="test", rollouts=[rollout] * 2)

        async def fetch(group_size: int) -> training.RolloutGroup:
            nonlocal requested
            index = requested
            requested += 1
            started.put_nowait(index)
            return await pending[index]

        actor = Mock()
        actor.fetch_next_group.remote.side_effect = fetch
        batcher = training.Batcher.__ray_metadata__.modified_class(
            actor=actor, config=training.Config(prefetch_batches=2)
        )
        async with asyncio.timeout(5), asyncio.TaskGroup() as tg:
            batch = tg.create_task(batcher.fetch_next_batch(batch_size=4, group_size=2, context_length=8))
            assert [await started.get() for _ in range(4)] == [0, 1, 2, 3]
            pending[0].set_result(group)
            assert await started.get() == 4
            assert not batch.done()
            pending[1].set_result(group)
            assert await started.get() == 5
            result = await batch
            assert result.training_batch.token_ids.shape == (4, 8)
            assert len(batcher._pending_groups) == 4
            await batcher.cancel_pending_groups()
            assert all(item.cancelled() for item in pending[2:])

    asyncio.run(collect())


def test_batcher_selects_groups_across_batch_boundaries(rollout: training.Rollout) -> None:
    groups = [
        training.RolloutGroup(
            task_id=str(index), rollouts=[replace(rollout, advantage=float(index))] * 2, taskset_id="test"
        )
        for index in range(4)
    ]

    async def collect() -> None:
        pending = [asyncio.Future[training.RolloutGroup]() for _ in groups]
        started: asyncio.Queue[int] = asyncio.Queue()
        requested = 0

        async def fetch(group_size: int) -> training.RolloutGroup:
            nonlocal requested
            index = requested
            requested += 1
            started.put_nowait(index)
            return await pending[index]

        actor = Mock()
        actor.fetch_next_group.remote.side_effect = fetch
        batcher = training.Batcher.__ray_metadata__.modified_class(
            actor=actor,
            config=training.Config(prefetch_batches=2, checkpoint_interval=2),
        )
        async with asyncio.timeout(5), asyncio.TaskGroup() as tg:
            first = tg.create_task(
                batcher.fetch_next_batch(batch_size=4, group_size=2, context_length=8, include_examples=True)
            )
            assert [await started.get() for _ in groups] == list(range(4))
            # The fastest groups belong to different submission-order batches.
            pending[3].set_result(groups[3])
            await asyncio.sleep(0)
            pending[1].set_result(groups[1])
            await asyncio.sleep(0)
            first_batch = await first
            first_indices = [int(value) for value in first_batch.training_batch.advantages[::2]]
            assert sorted(first_indices) == [1, 3]
            assert f"Example 1 | task={first_indices[-1]} |" in first_batch.examples
            assert first_batch.examples.count("Full token sequence") == 1
            np.testing.assert_array_equal(first_batch.training_batch.advantages, np.repeat(first_indices, 2))
            assert first_batch.training_batch.token_ids.shape == (4, 8)

            second = tg.create_task(
                batcher.fetch_next_batch(
                    batch_size=4, group_size=2, context_length=8, iteration=2, include_examples=True
                )
            )
            pending[0].set_result(groups[0])
            await asyncio.sleep(0)
            pending[2].set_result(groups[2])
            second_batch = await second
            second_indices = [int(value) for value in second_batch.training_batch.advantages[::2]]
            assert sorted(second_indices) == [0, 2]
            assert f"Example 1 | task={second_indices[-1]} |" in second_batch.examples
            assert second_batch.examples.count("Full token sequence") == 1
            np.testing.assert_array_equal(second_batch.training_batch.advantages, np.repeat(second_indices, 2))
            assert actor.fetch_next_group.remote.call_count == 4
            assert not batcher._pending_groups
            await batcher.cancel_pending_groups()

    asyncio.run(collect())


def test_learner_cancels_prefetch_on_update_failure(
    learner: Any, rollout: training.Rollout, monkeypatch: pytest.MonkeyPatch
) -> None:
    learner.config = replace(learner.config, total_iterations=4, prefetch_batches=3)
    learner.publish_weights = Mock()
    batch = training.build_rollout_batch(
        groups=[training.RolloutGroup(task_id="0", rollouts=[rollout] * 4, taskset_id="test")],
        consumption_weight_version=0,
        max_staleness=learner.config.max_staleness,
        training_batch=Mock(
            spec=training.GRPOBatch,
            attention_mask=np.ones((4, 2), dtype=np.bool_),  # (B=4, T=2)
            loss_mask=np.tile([False, True], (4, 1)),  # (B, T)
        ),
        timing_metrics={},
    )
    learner.batcher.fetch_next_batch.remote.return_value = batch
    monkeypatch.setattr(training.ray, "get", lambda value: value)
    monkeypatch.setattr(training, "MLflowLogger", MagicMock())

    def optimize(
        batch: training.GRPOBatch[np.ndarray],  # (B, T), (B,) leaves.
        gradient_sum: training.TrainParameters,  # Parameter shapes.
    ) -> training.BatchGradientResult:
        learner.batcher.fetch_next_batch.remote.assert_called_once_with(
            batch_size=4,
            group_size=4,
            context_length=8,
            iteration=1,
            consumption_weight_version=0,
            include_examples=False,
        )
        raise RuntimeError("update failed")

    learner.optimize = optimize
    with pytest.raises(RuntimeError, match="update failed"):
        learner.start()
    learner.batcher.cancel_pending_groups.remote.assert_called_once_with()


def test_batcher_cancels_prefetched_groups(
    rollout: training.Rollout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancel = Mock()
    monkeypatch.setattr(training.ray, "cancel", cancel)

    async def collect() -> None:
        pending = [asyncio.Future[training.RolloutGroup]() for _ in range(5)]
        pending[0].set_result(training.RolloutGroup(task_id="0", rollouts=[rollout] * 2, taskset_id="test"))
        actor = Mock()
        actor.fetch_next_group.remote.side_effect = pending
        batcher = training.Batcher.__ray_metadata__.modified_class(
            actor=actor, config=training.Config(prefetch_batches=2)
        )
        async with asyncio.timeout(5):
            await batcher.fetch_next_batch(batch_size=2, group_size=2, context_length=8)
            await batcher.cancel_pending_groups()
            assert all(item.cancelled() for item in pending[1:])
            assert [call.args[0] for call in cancel.call_args_list] == pending[1:]
            assert all(call.kwargs == {"recursive": True} for call in cancel.call_args_list)
            assert not batcher._pending_groups
            await batcher.cancel_pending_groups()
            assert cancel.call_count == 4

    asyncio.run(collect())


def test_batcher_cancels_active_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    cancel = Mock()
    monkeypatch.setattr(training.ray, "cancel", cancel)

    async def collect() -> None:
        pending = [asyncio.Future[training.RolloutGroup]() for _ in range(2)]
        started: asyncio.Queue[int] = asyncio.Queue()
        requests = iter(enumerate(pending))

        def fetch(group_size: int) -> asyncio.Future[training.RolloutGroup]:
            index, result = next(requests)
            started.put_nowait(index)
            return result

        actor = Mock()
        actor.fetch_next_group.remote.side_effect = fetch
        batcher = training.Batcher.__ray_metadata__.modified_class(
            actor=actor, config=training.Config(prefetch_batches=1)
        )
        batcher.prepare_batch = Mock()
        async with asyncio.timeout(5), asyncio.TaskGroup() as tg:
            batch = tg.create_task(batcher.fetch_next_batch(batch_size=2, group_size=2, context_length=8))
            assert [await started.get() for _ in pending] == [0, 1]
            batch.cancel()
            with pytest.raises(asyncio.CancelledError):
                await batch
            assert all(item.cancelled() for item in pending)
            assert [call.args[0] for call in cancel.call_args_list] == pending
            assert all(call.kwargs == {"recursive": True} for call in cancel.call_args_list)
            assert not batcher._pending_groups
            batcher.prepare_batch.assert_not_called()

    asyncio.run(collect())


@pytest.mark.parametrize("failure_stage", ["group", "prepare"])
def test_batcher_next_ready_failure_cancels_pool(
    failure_stage: str, rollout: training.Rollout, monkeypatch: pytest.MonkeyPatch
) -> None:
    cancel = Mock()
    monkeypatch.setattr(training.ray, "cancel", cancel)

    async def collect() -> None:
        pending = [asyncio.Future[training.RolloutGroup]() for _ in range(4 + (failure_stage == "prepare"))]
        error = RuntimeError("batch failed")
        if failure_stage == "group":
            pending[3].set_exception(error)
        else:
            pending[3].set_result(training.RolloutGroup(task_id="3", rollouts=[rollout] * 2, taskset_id="test"))
        actor = Mock()
        actor.fetch_next_group.remote.side_effect = pending
        batcher = training.Batcher.__ray_metadata__.modified_class(
            actor=actor, config=training.Config(prefetch_batches=2)
        )
        if failure_stage == "prepare":
            batcher.prepare_batch = Mock(side_effect=error)
        async with asyncio.timeout(5):
            with pytest.raises(RuntimeError, match="batch failed") as caught:
                await batcher.fetch_next_batch(batch_size=2, group_size=2, context_length=8)
            assert caught.value is error
            unconsumed = [item for index, item in enumerate(pending) if index != 3]
            assert all(item.cancelled() for item in unconsumed)
            assert [call.args[0] for call in cancel.call_args_list] == unconsumed
            assert not batcher._pending_groups

    asyncio.run(collect())


@pytest.mark.parametrize("drop_degenerate_groups", [False, True])
def test_batch_refills_degenerate_groups(drop_degenerate_groups: bool, rollout: training.Rollout) -> None:
    groups = [
        training.score_group(
            Mock(id=str(index)),
            [replace(rollout, reward=reward) for reward in rewards],
            taskset_id="degenerate" if index < 2 else "retained",
        )
        for index, rewards in enumerate(((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)))
    ]
    actor = Mock()
    actor.fetch_next_group.remote = AsyncMock(side_effect=groups)
    batcher = training.Batcher.__ray_metadata__.modified_class(
        actor=actor, config=training.Config(drop_degenerate_groups=drop_degenerate_groups)
    )

    batch = asyncio.run(batcher.fetch_next_batch(batch_size=4, group_size=2, context_length=8, include_examples=True))

    expected_groups = groups[2:] if drop_degenerate_groups else groups[:2]
    assert actor.fetch_next_group.remote.await_count == (4 if drop_degenerate_groups else 2)
    assert all(call.kwargs == {"group_size": 2} for call in actor.fetch_next_group.remote.call_args_list)
    assert batch.training_batch.token_ids.shape == (4, 8)
    assert batch.training_batch.advantages.shape == (4,)
    np.testing.assert_allclose(
        batch.training_batch.advantages,
        [rollout.advantage for group in expected_groups for rollout in group.rollouts],
    )
    assert f"Example 1 | task={expected_groups[-1].task_id} |" in batch.examples
    assert batch.examples.count("Full token sequence") == 1
    assert batch.rollout_metrics["rollout/degenerate_group_fraction"] == (0.0 if drop_degenerate_groups else 1.0)
    assert batch.rollout_metrics["batcher/skipped_groups"] == (2 if drop_degenerate_groups else 0)
    assert {
        name: value for name, value in batch.rollout_metrics.items() if name.startswith("rollout/reward_mean/")
    } == {f"rollout/reward_mean/{'retained' if drop_degenerate_groups else 'degenerate'}": 0.5}
    assert batch.num_output_tokens == 4
    assert batch.stop_counts == {"agent_completed": 4}


def test_batch_skipped_groups_follow_prefetched_groups(rollout: training.Rollout) -> None:
    retained = training.score_group(Mock(id="retained"), [rollout, replace(rollout, reward=0.0)], taskset_id="alfworld")
    skipped = training.score_group(Mock(id="skipped"), [rollout] * 2, taskset_id="alfworld")
    actor = Mock()
    actor.fetch_next_group.remote = AsyncMock(side_effect=[retained, skipped, skipped, retained])
    batcher = training.Batcher.__ray_metadata__.modified_class(
        actor=actor,
        config=training.Config(drop_degenerate_groups=True, prefetch_batches=1, checkpoint_interval=2),
    )

    async def collect() -> None:
        async with asyncio.timeout(5):
            first = await batcher.fetch_next_batch(batch_size=2, group_size=2, context_length=8)
            await asyncio.sleep(0)
            assert actor.fetch_next_group.remote.await_count == 4
            assert first.rollout_metrics["batcher/skipped_groups"] == 0
            second = await batcher.fetch_next_batch(batch_size=2, group_size=2, context_length=8, iteration=2)
            assert second.rollout_metrics["batcher/skipped_groups"] == 2
            assert not batcher._pending_groups

    asyncio.run(collect())


@pytest.mark.parametrize(
    ("max_staleness", "consumption_weight_version", "sampled_version", "complete_before_fetch", "masked"),
    [
        pytest.param(8, 8, 0, True, False, id="inclusive-limit"),
        pytest.param(8, 9, 0, True, True, id="too-stale"),
        pytest.param(0, 1, 0, True, True, id="current-weights-only"),
        pytest.param(0, 1, 0, False, True, id="in-flight"),
        pytest.param(0, 1, 1, False, False, id="in-flight-fresh-weights"),
    ],
)
def test_batch_masks_stale_prefetched_tokens(
    max_staleness: int,
    consumption_weight_version: int,
    sampled_version: int,
    complete_before_fetch: bool,
    masked: bool,
    rollout: training.Rollout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_inline[T](func: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)

    # Thread offloading is covered by test_batch_timings_and_order_survive_serialization.
    monkeypatch.setattr(training.asyncio, "to_thread", run_inline)
    monkeypatch.setattr(training.ray, "cancel", Mock())

    async def collect() -> None:
        pending = asyncio.Future[training.RolloutGroup]()
        groups = [
            training.RolloutGroup(
                task_id=str(index), taskset_id=f"taskset_{index}", rollouts=[rollout, replace(rollout, reward=0.0)]
            )
            for index in range(3)
        ]
        groups[1] = replace(
            groups[1],
            rollouts=[
                replace(
                    sample,
                    sequence=replace(
                        sample.sequence,
                        weight_versions=np.where(sample.sequence.loss_mask, sampled_version, -1).astype(
                            np.int32
                        ),  # (T,)
                    ),
                )
                for sample in groups[1].rollouts
            ],
        )
        requested = 0

        async def fetch(group_size: int) -> training.RolloutGroup:
            nonlocal requested
            index = requested
            requested += 1
            return await pending if index == 1 else groups[index]

        actor = Mock()
        actor.fetch_next_group.remote.side_effect = fetch
        batcher = training.Batcher.__ray_metadata__.modified_class(
            actor=actor,
            config=training.Config(
                prefetch_batches=1,
                checkpoint_interval=2,
                max_staleness=max_staleness,
            ),
        )
        async with asyncio.timeout(5), AsyncExitStack() as resources, asyncio.TaskGroup() as tg:
            resources.push_async_callback(batcher.cancel_pending_groups)
            first = await batcher.fetch_next_batch(batch_size=2, group_size=2, context_length=8)
            assert requested == 2
            assert first.rollout_metrics["batcher/stale_tokens"] == 0
            assert first.rollout_metrics["batcher/stale_transcript_token_fraction"] == 0
            if complete_before_fetch:
                pending.set_result(groups[1])
                await batcher._pending_groups[0]
            second = tg.create_task(
                batcher.fetch_next_batch(
                    batch_size=2,
                    group_size=2,
                    context_length=8,
                    iteration=2,
                    consumption_weight_version=consumption_weight_version,
                    include_examples=True,
                )
            )
            if not complete_before_fetch:
                await asyncio.sleep(0)
                pending.set_result(groups[1])
            batch = await second
            assert requested == 2
            assert batch.training_batch.token_ids.shape == (2, 8)
            assert batch.training_batch.advantages.shape == (2,)
            assert batch.rollout_metrics["rollout/reward_mean/taskset_1"] == 0.5
            assert "task=1 |" in batch.examples
            assert batch.rollout_metrics["batcher/stale_tokens"] == 2 * int(masked)
            assert batch.rollout_metrics["batcher/stale_transcript_token_fraction"] == 0.5 * masked
            age = "older" if masked else str(consumption_weight_version - sampled_version)
            assert batch.rollout_metrics[f"batcher/weight_age_transcript_token_fraction/{age}"] == 0.5
            assert batch.rollout_metrics[f"batcher/weight_age_masked_transcript_token_fraction/{age}"] == 0.5 * masked
            assert int(batch.training_batch.loss_mask.sum()) == (0 if masked else 2)
            np.testing.assert_array_equal(batch.training_batch.attention_mask[:, :2], True)
            np.testing.assert_array_equal(batch.training_batch.advantages, 0.5)
            assert not batcher._pending_groups

    asyncio.run(collect())


def test_batch_requires_groups() -> None:
    actor = Mock()
    batcher = training.Batcher.__ray_metadata__.modified_class(actor=actor)
    with pytest.raises(ValueError, match="batch_size"):
        asyncio.run(batcher.fetch_next_batch(batch_size=0, group_size=2, context_length=8))
    actor.fetch_next_group.remote.assert_not_called()


def test_degenerate_retry_masks_actual_token_versions(rollout: training.Rollout) -> None:
    async def collect() -> None:
        pending = asyncio.Future[training.RolloutGroup]()
        retained = training.score_group(Mock(id="retained"), [rollout, replace(rollout, reward=0.0)], taskset_id="test")
        degenerate = replace(retained, rollouts=[rollout] * 2)
        requested = 0

        async def fetch(group_size: int) -> training.RolloutGroup:
            nonlocal requested
            requested += 1
            return await pending if requested == 2 else retained

        actor = Mock()
        actor.fetch_next_group.remote.side_effect = fetch
        batcher = training.Batcher.__ray_metadata__.modified_class(
            actor=actor,
            config=training.Config(
                prefetch_batches=1, checkpoint_interval=2, drop_degenerate_groups=True, max_staleness=0
            ),
        )
        async with asyncio.timeout(5), asyncio.TaskGroup() as tg:
            await batcher.fetch_next_batch(batch_size=2, group_size=2, context_length=8)
            second = tg.create_task(
                batcher.fetch_next_batch(
                    batch_size=2, group_size=2, context_length=8, iteration=2, consumption_weight_version=1
                )
            )
            await asyncio.sleep(0)
            pending.set_result(degenerate)
            batch = await second
            assert requested == 3
            assert batch.rollout_metrics["batcher/skipped_groups"] == 1
            assert batch.rollout_metrics["batcher/stale_tokens"] == 2
            assert batch.rollout_metrics["batcher/stale_transcript_token_fraction"] == 0.5
            assert not batcher._pending_groups

    asyncio.run(collect())


@pytest.mark.parametrize("drop_degenerate_groups", [False, True])
def test_batch_failure_cancels_pending_fetches(
    drop_degenerate_groups: bool, rollout: training.Rollout, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training.ray, "cancel", Mock())
    started = asyncio.Event()
    cleaned_up = asyncio.Event()
    requested = 0

    async def fetch_group(group_size: int) -> training.RolloutGroup:
        nonlocal requested
        requested += 1
        if drop_degenerate_groups and requested == 1:
            return training.RolloutGroup(task_id="degenerate", rollouts=[rollout] * 2, taskset_id="test")
        if requested == 1 + drop_degenerate_groups:
            await started.wait()
            raise RuntimeError("rollout failed")
        started.set()
        try:
            return await asyncio.Future[training.RolloutGroup]()
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
            with pytest.raises(RuntimeError, match="rollout failed"):
                await batcher.fetch_next_batch(batch_size=4, group_size=2, context_length=8)
            # Cleanup must finish before control returns to the caller.
            assert cleaned_up.is_set()

    asyncio.run(collect())
    assert requested == 2 + drop_degenerate_groups
    batcher.prepare_batch.assert_not_called()


@pytest.mark.parametrize(
    ("consumption_weight_version", "max_staleness", "version_offset", "age_fractions"),
    [
        pytest.param(4, 3, 0, {"0": 0.1, "1": 0.3, "3": 0.2}, id="all-sampled-unmasked"),
        pytest.param(4, 1, 0, {"0": 0.1, "1": 0.3, "older": 0.2}, id="mixed-at-staleness-boundary"),
        pytest.param(6, 1, 0, {"older": 0.6}, id="all-sampled-masked"),
        pytest.param(4, 0, 0, {"0": 0.1, "older": 0.5}, id="current-weights-only"),
        pytest.param(10004, 1, 10000, {"0": 0.1, "1": 0.3, "older": 0.2}, id="absolute-version-independent"),
    ],
)
def test_batch_weight_age_token_fractions(
    rollout: training.Rollout,
    consumption_weight_version: int,
    max_staleness: int,
    version_offset: int,
    age_fractions: dict[str, float],
) -> None:
    rollouts = []
    for versions in ((-1, 1, 1, -1, 3, -1, 4), (-1, 3, 3)):
        weight_versions = np.asarray(versions, dtype=np.int32)  # (L,) includes interleaved context.
        mask = weight_versions >= 0  # (L,)
        weight_versions[mask] += version_offset  # (S,) sampled tokens only.
        rollouts.append(
            replace(
                rollout,
                sequence=training.TrainingSequence(
                    token_ids=np.arange(len(versions), dtype=np.int32),  # (L,)
                    loss_mask=mask,  # (L,)
                    old_log_probs=np.zeros(len(versions), dtype=np.float32),  # (L,)
                    weight_versions=weight_versions,  # (L,)
                ),
                num_output_tokens=int(mask.sum()),
                advantage=0.0,  # Zero advantages do not change the loss mask or token fractions.
            )
        )
    groups = [training.RolloutGroup(task_id="task", taskset_id="test", rollouts=rollouts)]
    batcher = training.Batcher.__ray_metadata__.modified_class(
        actor=Mock(), config=training.Config(max_staleness=max_staleness, padding_multiple=4)
    )
    prepared = batcher.prepare_batch(
        groups, context_length=8, consumption_weight_version=consumption_weight_version
    )  # (B=2, T=8) leaves.
    assert prepared.loss_mask.shape == (2, 8)
    batch = training.build_rollout_batch(
        groups=groups,
        training_batch=prepared,
        timing_metrics={},
        consumption_weight_version=consumption_weight_version,
        max_staleness=max_staleness,
    )
    metrics = batch.rollout_metrics
    stale_fraction = age_fractions.get("older", 0.0)
    assert metrics["batcher/context_transcript_token_fraction"] == 0.4
    assert metrics["batcher/stale_transcript_token_fraction"] == stale_fraction
    assert metrics["batcher/masked_transcript_token_fraction"] == pytest.approx(0.4 + stale_fraction)
    assert metrics["batcher/unmasked_transcript_token_fraction"] == pytest.approx(0.6 - stale_fraction)
    assert "batcher/stale_token_fraction" not in metrics
    expected = {}
    for age, fraction in age_fractions.items():
        expected[f"batcher/weight_age_transcript_token_fraction/{age}"] = fraction
        expected[f"batcher/weight_age_masked_transcript_token_fraction/{age}"] = fraction if age == "older" else 0.0
        expected[f"batcher/weight_age_unmasked_transcript_token_fraction/{age}"] = 0.0 if age == "older" else fraction
    assert not any(name.startswith("batcher/weight_version") for name in metrics)
    actual = {name: value for name, value in batch.rollout_metrics.items() if name.startswith("batcher/weight_age")}
    assert actual == pytest.approx(expected)
    assert sum(value for name, value in actual.items() if "_masked_" in name) == pytest.approx(
        metrics["batcher/stale_transcript_token_fraction"]
    )
    assert sum(value for name, value in actual.items() if "_unmasked_" in name) == pytest.approx(
        metrics["batcher/unmasked_transcript_token_fraction"]
    )


def test_batch_taskset_reward_means(rollout: training.Rollout) -> None:
    penalty = training.LengthPenaltyConfig(reference_tokens=1, penalty=0.5)
    groups = [
        training.RolloutGroup(
            task_id="same_environment:same_task",
            taskset_id=taskset_id,
            rollouts=[replace(rollout, reward=reward)] * count,
            length_penalty=penalty,
        )
        for taskset_id, reward, count in (("easy", 1.0, 2), ("easy", 0.0, 4), ("hard", 0.0, 2))
    ]
    groups = pickle.loads(pickle.dumps(groups))
    batcher = training.Batcher.__ray_metadata__.modified_class(actor=Mock())
    prepared = batcher.prepare_batch(groups, context_length=8)  # (B=8, T=8), (B=8,) leaves.
    assert prepared.token_ids.shape == (8, 8)
    batch = training.build_rollout_batch(
        groups=groups,
        training_batch=prepared,
        timing_metrics={},
        consumption_weight_version=0,
        max_staleness=batcher.config.max_staleness,
    )
    assert {
        name: value for name, value in batch.rollout_metrics.items() if name.startswith("rollout/reward_mean/")
    } == {"rollout/reward_mean/easy": pytest.approx(1 / 3), "rollout/reward_mean/hard": 0.0}
    assert batch.rollout_metrics["rollout/reward_mean"] == 0.25
    assert batch.rollout_metrics["rollout/adjusted_reward_mean"] == 0.125


def test_batch_timings_and_order_survive_serialization(
    learner: Any, rollout: training.Rollout, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    learner.config = replace(learner.config, batch_size=4, group_size=2)
    clock = Mock(return_value=0.0)
    monkeypatch.setattr(training, "time", Mock(monotonic=clock))
    caplog.set_level("INFO", logger=training.__name__)
    groups = [
        training.RolloutGroup(
            task_id=str(i),
            taskset_id=f"source_{i}",
            rollouts=[replace(rollout, advantage=float(i), reward=float(i))] * 2,
        )
        for i in range(2)
    ]
    fast_ready = asyncio.Event()
    requested = 0

    async def fetch_group(group_size: int) -> training.RolloutGroup:
        nonlocal requested
        assert group_size == 2
        index = requested
        requested += 1
        if index == 0:
            await fast_ready.wait()
            received_at = 50.5
        else:
            received_at = 10.25
            fast_ready.set()
        result = pickle.loads(pickle.dumps(groups[index]))
        clock.return_value = received_at
        return result

    actor = Mock()
    actor.fetch_next_group.remote.side_effect = fetch_group
    batcher = training.Batcher.__ray_metadata__.modified_class(actor=actor)
    learner.batcher.fetch_next_batch.remote.side_effect = batcher.fetch_next_batch
    prepare_batch = batcher.prepare_batch
    event_loop_thread = threading.get_ident()

    def prepare(
        groups: list[training.RolloutGroup], context_length: int, *, consumption_weight_version: int
    ) -> training.GRPOBatch[np.ndarray]:
        assert threading.get_ident() != event_loop_thread
        batch = prepare_batch(
            groups, context_length=context_length, consumption_weight_version=consumption_weight_version
        )  # (B, T), (B,) leaves.
        clock.return_value += 2.0
        return batch  # (B, T), (B,) leaves.

    monkeypatch.setattr(batcher, "prepare_batch", prepare)

    def get(pending: Any) -> training.RolloutBatch:
        result = asyncio.run(pending)
        result = pickle.loads(pickle.dumps(result))
        clock.return_value = 53.25
        return result

    monkeypatch.setattr(training.ray, "get", get)
    batch = learner.fetch_next_batch(iteration=10)
    # Preserve request order despite out-of-order completion and serialization.
    np.testing.assert_array_equal(batch.training_batch.advantages, [0.0, 0.0, 1.0, 1.0])
    assert "Example 1 | task=0" in batch.examples
    assert "Example 2 | task=1" in batch.examples
    assert batch.training_batch.token_ids.shape == (4, 8)
    assert batch.rollout_metrics["rollout/reward_mean/source_0"] == 0.0
    assert batch.rollout_metrics["rollout/reward_mean/source_1"] == 1.0
    assert batch.timing_metrics == {
        "batcher/fetch_rollout_seconds": 50.5,
        "batcher/prepare_batch_seconds": 2.0,
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
    learner.batcher.get_pool_state.remote.return_value = []
    learner.model = model
    learner.apply_update = Mock(return_value={})
    loader.assert_not_called()
    assert not hasattr(learner, "state")
    return learner


def test_learner_initializes_replicated_state(learner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    devices = jax.local_devices(backend="cpu")[:2]
    if len(devices) < 2:
        pytest.skip("Requires two CPU devices; use JAX_NUM_CPU_DEVICES=2")
    monkeypatch.setattr(training.jax, "devices", Mock(return_value=devices))
    monkeypatch.setattr(training, "MLflowLogger", MagicMock())
    learner.config = replace(learner.config, learner_gpus=2)
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
    learner.config = replace(
        learner.config, finetuning=training.LoRAConfig() if mode == "lora" else training.FullFinetuningConfig()
    )
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
    learner.config = replace(
        learner.config,
        finetuning=training.LoRAConfig() if mode == "lora" else training.FullFinetuningConfig(),
        adamw_batch_size_scale=4.0,
    )
    caplog.set_level("INFO", logger=training.__name__)
    sequence = training.TrainingSequence(
        token_ids=np.array([0, 1], dtype=np.int32),  # (T=2,)
        loss_mask=np.array([False, True]),  # (T=2,)
        old_log_probs=np.array([0.0, -0.2], dtype=np.float32),  # (T=2,)
        weight_versions=np.array([-1, 0], dtype=np.int32),  # (T=2,)
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
        training.RolloutGroup(
            task_id="0", taskset_id="easy", rollouts=[replace(rollout, reward=reward) for reward in group_rewards[0]]
        ),
        training.RolloutGroup(
            task_id="1",
            taskset_id="hard",
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
    learner.config = replace(learner.config, batch_size=4, group_size=2, total_iterations=None if fails else 2)
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
    timing_metrics = {
        "batcher/fetch_rollout_seconds": 0.5,
        "batcher/prepare_batch_seconds": 0.25,
    }
    prepared = Mock(spec=training.GRPOBatch)
    prepared.token_ids = Mock(shape=(4, 8))  # (B=4, T=8)
    prepared.attention_mask = np.tile([True, True, False, False, False, False, False, False], (4, 1))  # (B=4, T=8)
    prepared.loss_mask = np.zeros((4, 8), dtype=np.bool_)  # (B=4, T=8) all stale.
    received = training.build_rollout_batch(
        groups=batch,
        training_batch=prepared,
        timing_metrics=timing_metrics,
        consumption_weight_version=1,
        max_staleness=0,
        skipped_groups=3,
    )
    prepared.loss_mask[:, 1] = True  # (B=4,) fresh sampled tokens.
    next_batch = training.build_rollout_batch(
        groups=[
            replace(
                group,
                rollouts=[
                    replace(
                        rollout,
                        stop_condition="environment_done",
                        sequence=replace(
                            rollout.sequence,
                            weight_versions=np.array([-1, 1], dtype=np.int32),  # (T=2,)
                        ),
                    )
                    for rollout in group.rollouts
                ],
            )
            for group in batch
        ],
        training_batch=prepared,
        timing_metrics=timing_metrics,
        consumption_weight_version=1,
        max_staleness=0,
    )
    learner.fetch_next_batch.side_effect = [received, RuntimeError("done") if fails else next_batch]
    learner.publish_weights = steps.publish

    def optimize(
        batch: training.GRPOBatch[np.ndarray],  # (B, T), (B,) leaves.
        gradient_sum: training.TrainParameters,  # Parameter shapes.
    ) -> training.BatchGradientResult:
        assert batch is prepared
        assert jax.sharding.get_mesh() == learner.mesh
        learner.state = learner.state.replace(step=learner.state.step + 1)
        return training.BatchGradientResult(
            gradient_sum=gradient_sum, metrics={"train/policy_loss": -0.25}, has_signal=True
        )

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
            "lora_rank": learner.config.finetuning.rank,
            "lora_alpha": learner.config.finetuning.alpha,
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
    expected_config = asdict(learner.config)
    assert yaml.safe_load(writer.log_text.call_args.args[0]) == expected_config
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
        **timing_metrics,
        "train/policy_loss": -0.25,
        "rollout/reward_mean": mean(rewards),
        "rollout/reward_mean/easy": mean(group_rewards[0]),
        "rollout/reward_mean/hard": mean(group_rewards[1]),
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
        "batcher/skipped_groups": 3,
        "batcher/stale_tokens": 4,
        "batcher/context_transcript_token_fraction": 0.5,
        "batcher/stale_transcript_token_fraction": 0.5,
        "batcher/masked_transcript_token_fraction": 1.0,
        "batcher/unmasked_transcript_token_fraction": 0.0,
        "batcher/weight_age_transcript_token_fraction/older": 0.5,
        "batcher/weight_age_masked_transcript_token_fraction/older": 0.5,
        "batcher/weight_age_unmasked_transcript_token_fraction/older": 0.0,
        "train/iterations_total": 1,
        "train/updates_total": 1,
        "train/learning_rate": learner.config.learning_rate * 2,
        "train/rollouts_total": 4,
        "learner/iterations_per_second": 1 / 4,
        "learner/rollouts_per_second": 1.0,
        "learner/output_tokens_per_second": 4.0,
        "actor/rollout_seconds_mean": 1.0,
        "learner/batch_fetch_seconds": 1.0,
        "learner/update_seconds": 2.0,
        "learner/weight_transfer_seconds": 1.0,
        "learner/iteration_seconds": 4.0,
    }
    if not fails:
        assert calls[2].kwargs == {"step": 2}
        assert calls[2].args[0]["train/rollouts_total"] == 8
        assert calls[2].args[0]["rollout/degenerate_group_fraction"] == degenerate_fraction
        assert calls[2].args[0]["batcher/skipped_groups"] == 0
        assert calls[2].args[0]["batcher/stale_tokens"] == 0
        assert calls[2].args[0]["batcher/context_transcript_token_fraction"] == 0.5
        assert calls[2].args[0]["batcher/stale_transcript_token_fraction"] == 0.0
        assert calls[2].args[0]["batcher/masked_transcript_token_fraction"] == 0.5
        assert calls[2].args[0]["batcher/unmasked_transcript_token_fraction"] == 0.5
        assert calls[2].args[0]["batcher/weight_age_transcript_token_fraction/0"] == 0.5
        assert calls[2].args[0]["batcher/weight_age_masked_transcript_token_fraction/0"] == 0.0
        assert calls[2].args[0]["batcher/weight_age_unmasked_transcript_token_fraction/0"] == 0.5
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
        "skipped_groups=3 stale_tokens=4 batch=1.00s "
        "loss=-0.2500 update=2.00s weights=1.00s"
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
            rollout_metrics={
                "rollout/reward_mean": 0.0,
                "rollout/degenerate_group_fraction": 1.0,
                "batcher/skipped_groups": 0,
                "batcher/stale_tokens": 0,
            },
            stop_counts={},
            num_output_tokens=4,
            examples="",
            timing_metrics={},
        )
    )
    increments = iter((0, 1, 0))  # All skipped, partially skipped, then all skipped at a nonzero step.
    published_steps: list[int] = []

    def optimize(
        batch: training.GRPOBatch[np.ndarray],  # (B, T), (B,) leaves.
        gradient_sum: training.TrainParameters,  # Parameter shapes.
    ) -> training.BatchGradientResult:
        learner.state = learner.state.replace(step=learner.state.step + next(increments))
        return training.BatchGradientResult(
            gradient_sum=gradient_sum, metrics={"train/policy_loss": 0.0}, has_signal=True
        )

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


def test_learner_parameter_sync_cadence(learner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    learner.config = replace(learner.config, parameter_sync_steps=3, total_iterations=7, checkpoint_interval=2)
    writer = MagicMock()
    writer.__enter__.return_value = writer
    monkeypatch.setattr(training, "MLflowLogger", Mock(return_value=writer))
    monkeypatch.setattr(training, "time", Mock(monotonic=Mock(side_effect=range(100))))
    prepared = Mock(spec=training.GRPOBatch)
    prepared.token_ids = Mock(shape=(4, 2))  # (B=4, T=2)
    learner.fetch_next_batch = Mock(
        return_value=training.RolloutBatch(
            training_batch=prepared,
            rollout_metrics={
                "rollout/reward_mean": 1.0,
                "rollout/degenerate_group_fraction": 0.0,
                "batcher/skipped_groups": 0,
                "batcher/stale_tokens": 0,
            },
            stop_counts={},
            num_output_tokens=4,
            examples="",
            timing_metrics={},
        )
    )
    increments = iter([1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    published_steps: list[int] = []
    consumed_versions: list[int] = []

    def fetch(*, iteration: int) -> training.RolloutBatch:
        consumed_versions.append(learner.weight_version)
        return learner.fetch_next_batch.return_value

    learner.fetch_next_batch.side_effect = fetch

    def optimize(
        batch: training.GRPOBatch[np.ndarray],  # (B, T), (B,) leaves.
        gradient_sum: training.TrainParameters,  # Parameter shapes.
    ) -> training.BatchGradientResult:
        learner.state = learner.state.replace(step=learner.state.step + next(increments))
        return training.BatchGradientResult(
            gradient_sum=gradient_sum, metrics={"train/policy_loss": 0.0}, has_signal=True
        )

    def publish() -> None:
        published_steps.append(int(learner.state.step))

    learner.optimize = optimize
    learner.publish_weights = publish
    learner.start()
    # Skipped windows do not advance the publication schedule; iteration 7 completes update 3.
    assert published_steps == [0, 3]
    assert consumed_versions == [0, 0, 0, 0, 0, 0, 0]
    metrics = [call.args[0] for call in writer.log_metrics.call_args_list[1:]]
    assert [values["train/updates_total"] for values in metrics] == [1, 2, 2, 2, 2, 2, 3]
    assert [values.get("learner/weight_transfer_seconds") for values in metrics] == [None] * 6 + [1.0]
    progress = training.save_checkpoint.call_args.args[1]
    assert int(progress.train_state.step) == 3
    assert [call.args[1].iteration for call in training.save_checkpoint.call_args_list] == [2, 4, 6, 7]

    # Resume publishes immediately; scheduled publication follows cumulative optimizer update 6.
    training.restore_checkpoint.return_value = progress
    learner.config = replace(learner.config, total_iterations=10)
    published_steps.clear()
    consumed_versions.clear()
    writer.reset_mock()
    learner.start()
    assert published_steps == [3, 6]
    assert consumed_versions == [2, 2, 2]
    calls = writer.log_metrics.call_args_list[1:]
    assert [call.kwargs["step"] for call in calls] == [8, 9, 10]
    assert [call.args[0].get("learner/weight_transfer_seconds") for call in calls] == [None, None, 1.0]


def test_learner_checkpoints_and_resumes(learner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    learner.config = replace(learner.config, total_iterations=3, checkpoint_interval=2, prefetch_batches=4)
    published_steps: list[int] = []
    saved: list[training.TrainingProgress] = []
    writer = MagicMock()
    writer.__enter__.return_value = writer
    monkeypatch.setattr(training, "MLflowLogger", Mock(return_value=writer))
    monkeypatch.setattr(training.ray, "get", lambda value: value)
    learner.actor.get_sampling_state.remote.return_value = {"task_rng": "saved"}
    learner.batcher.get_pool_state.remote.return_value = [{"task_id": "pending"}]

    def publish() -> None:
        published_steps.append(int(learner.state.step))

    def fetch(*, iteration: int) -> training.RolloutBatch:
        num_rollouts = learner.config.batch_size
        prepared = Mock(spec=training.GRPOBatch)
        prepared.token_ids = Mock(shape=(num_rollouts, 8))  # (B, T=8)
        return training.RolloutBatch(
            training_batch=prepared,
            rollout_metrics={
                "rollout/reward_mean": 1.0,
                "rollout/degenerate_group_fraction": 0.0,
                "batcher/skipped_groups": 0,
                "batcher/stale_tokens": 0,
            },
            stop_counts={"max_turns" if not saved else "agent_completed": num_rollouts},
            num_output_tokens=4,
            examples="",
            timing_metrics={},
        )

    def optimize(
        batch: training.GRPOBatch[np.ndarray],  # (B, T), (B,) leaves.
        gradient_sum: training.TrainParameters,  # Parameter shapes.
    ) -> training.BatchGradientResult:
        learner.state = learner.state.replace(step=learner.state.step + 1)
        return training.BatchGradientResult(
            gradient_sum=gradient_sum, metrics={"train/policy_loss": 0.0}, has_signal=True
        )

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
    assert published_steps == [0, 1, 2, 3]
    assert saved[-1].rollout_pool == [{"task_id": "pending"}]
    assert saved[-1].weight_version == 3
    learner.actor.set_sampling_state.remote.assert_not_called()
    saved_key = jax.random.key_data(saved[-1].shuffle_key)  # (K,)

    # New settings govern future work; restored totals are not recomputed from batch size.
    learner.config = replace(learner.config, total_iterations=5, batch_size=8, seed=99)
    learner.shuffle_key = jax.random.key(99)  # () simulate a restarted learner.
    published_steps.clear()
    writer.reset_mock()
    learner.start()
    assert [progress.iteration for progress in saved] == [2, 3, 4, 5]
    assert [progress.total_rollouts for progress in saved] == [8, 12, 20, 28]
    assert published_steps == [3, 4, 5]
    learner.actor.set_sampling_state.remote.assert_called_once_with({"task_rng": "saved"})
    learner.batcher.set_pool_state.remote.assert_called_once_with(saved[1].rollout_pool)
    np.testing.assert_array_equal(jax.random.key_data(learner.shuffle_key), saved_key)
    writer.log_text.assert_called_once_with(yaml.safe_dump(asdict(learner.config)), "config/3.yaml")
    assert [call.kwargs["step"] for call in writer.log_metrics.call_args_list] == [3, 4, 5]
    metrics = writer.log_metrics.call_args.args[0]
    assert metrics["train/rollouts_total"] == 28
    assert metrics["train/updates_total"] == 5
    assert metrics["rollout/stop_fraction/max_turns"] == 0
    assert metrics["checkpoint/save_blocking_seconds"] == 0.5

    # A run already at its cumulative target does not train or overwrite its checkpoint.
    learner.fetch_next_batch.reset_mock()
    learner.start()
    learner.fetch_next_batch.assert_not_called()
    assert len(saved) == 4


def test_batcher_logs_every_taskset(rollout: training.Rollout) -> None:
    groups = [
        training.RolloutGroup(task_id="shared-id", taskset_id=f"source_{index}", rollouts=[rollout] * 2)
        for index in range(9)
    ]
    actor = Mock()
    actor.fetch_next_group.remote = AsyncMock(side_effect=groups)
    batcher = training.Batcher.__ray_metadata__.modified_class(actor=actor)

    batch = asyncio.run(batcher.fetch_next_batch(batch_size=18, group_size=2, context_length=8, include_examples=True))

    assert batch.examples.count("Full token sequence") == 9
    for index in range(9):
        assert batch.examples.count(f"taskset=source_{index}\n") == 1


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
            weight_versions=np.where(np.array([False, True]), 0, -1).astype(np.int32),  # (T,)
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
        training.RolloutGroup(
            task_id="task-1", rollouts=[replace(rollout, interaction="omitted"), rollout], taskset_id="common"
        ),
        training.RolloutGroup(
            task_id="task-2",
            rollouts=[rollout, replace(rollout, interaction="", stop_condition="unspecified")],
            taskset_id="rare",
        ),
    ]
    learner.config = replace(
        learner.config,
        batch_size=4,
        group_size=2,
        total_iterations=7,
        prefetch_batches=0,
        example_log_interval=3,
    )
    learner.publish_weights = Mock()
    learner.optimize = Mock(
        side_effect=lambda batch, gradient_sum: training.BatchGradientResult(
            gradient_sum=gradient_sum, metrics={"train/policy_loss": 0.0}, has_signal=False
        )
    )
    actor = Mock()
    actor.fetch_next_group.remote = AsyncMock(
        side_effect=[
            group
            for step in range(1, 8)
            for group in (
                replace(groups[0], task_id=f"old-{step}"),
                replace(groups[1] if step == 1 else groups[0], task_id=f"batch-{step}"),
            )
        ]
    )
    batcher = training.Batcher.__ray_metadata__.modified_class(actor=actor, config=learner.config)
    submitted: list[training.RolloutBatch] = []

    def fetch(
        batch_size: int,
        group_size: int,
        context_length: int,
        *,
        iteration: int,
        consumption_weight_version: int,
        include_examples: bool,
    ) -> training.RolloutBatch:
        batch = asyncio.run(
            batcher.fetch_next_batch(
                batch_size=batch_size,
                group_size=group_size,
                context_length=context_length,
                iteration=iteration,
                consumption_weight_version=consumption_weight_version,
                include_examples=include_examples,
            )
        )
        submitted.append(batch)
        return batch

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
    assert [bool(batch.examples) for batch in submitted] == [False, False, True, False, False, True, False]
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
        assert "Example 2 | task=batch-1" in text
        assert text.count("taskset=common") == 1
        assert text.count("taskset=rare") == 1
        assert "Full token sequence (special tokens preserved):\n(empty)" in text
        assert text.count("reward=1.000") == 2
        assert "Token IDs:" not in text
        assert "Example 3" not in text and "omitted" not in text


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
    learner._publish_weights.assert_called_once_with(learner.weight_version + 1)
    learner._publish_weights.side_effect = RuntimeError("transfer failed")
    with pytest.raises(RuntimeError, match="transfer failed"):
        learner.publish_weights()


def test_console_logging_omits_ray_metadata(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sdk_logger = logging.getLogger("openai._base_client")
    monkeypatch.setattr(sdk_logger, "filters", [])
    for name in ("", "ray"):
        log = logging.getLogger(name)
        monkeypatch.setattr(log, "handlers", [logging.StreamHandler()] if name else [])
        monkeypatch.setattr(log, "level", logging.INFO)
        monkeypatch.setattr(log, "propagate", False)
    training.configure_console_logging()
    assert training.add_retry_cause in sdk_logger.filters
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
        monkeypatch.setattr(log, "filters", list(log.filters))
        monkeypatch.setattr(log, "handlers", list(log.handlers))
        monkeypatch.setattr(log, "level", log.level)
        monkeypatch.setattr(log, "propagate", log.propagate)
        for handler in log.handlers:
            monkeypatch.setattr(handler, "formatter", handler.formatter)
    config = training.Config(
        ckpt_dir=str(tmp_path),
        padding_multiple=64,
        learner_gpus=2,
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


@pytest.mark.parametrize("rewards", [[0.0, 0.0], [0.0, 1.0]], ids=["degenerate", "nondegenerate"])
def test_group_collects_fixed_size(
    rewards: list[float],
    actor: Any,
    rollout: training.Rollout,
    monkeypatch: pytest.MonkeyPatch,
    task_discovery: list[ALFWorldTask],
    spans: InMemorySpanExporter,
) -> None:
    samples = [replace(rollout, reward=reward, completion=str(index)) for index, reward in enumerate(rewards)]
    agent = AsyncMock()
    agent.__aenter__.return_value = agent
    agent.run.side_effect = samples
    monkeypatch.setattr(training, "create_agent", Mock(return_value=agent))

    async def collect() -> training.RolloutGroup:
        async with aclosing(actor), asyncio.timeout(5):
            return await actor.fetch_next_group(group_size=2)

    group = asyncio.run(collect())
    assert group == training.score_group(task_discovery[0], samples, taskset_id="alfworld")
    assert len(group.rollouts) == 2
    assert agent.run.await_count == len(samples)
    actor.rng.choice.assert_called_once_with(task_discovery)
    environments = [call.kwargs["env"] for call in agent.run.call_args_list]
    assert len({id(env) for env in environments}) == len(samples)
    for env in environments:
        env.__aexit__.assert_awaited_once()
    assert not actor._runs
    rollout_spans = [item for item in spans.get_finished_spans() if item.name == "actor.rollout"]
    assert sorted(item.attributes["rollout_index"] for item in rollout_spans) == list(range(len(samples)))
    assert len({item.attributes["group_id"] for item in rollout_spans}) == 1


@pytest.mark.parametrize(
    "rewards",
    [[-1.0, 0.0, 2.0], [0.7] * 8, [1.0, 1.0 + 1e-8]],
    ids=["mixed-rewards", "fractional-ties", "differences-below-float32-precision"],
)
def test_score_group_centers_rewards(
    rewards: list[float], rollout: training.Rollout, task_discovery: list[ALFWorldTask]
) -> None:
    task = task_discovery[0]
    rollouts = [replace(rollout, reward=reward) for reward in rewards]
    group = training.score_group(task, rollouts, taskset_id="alfworld")
    assert group.task_id == f"{task.name}:{task.id}"
    expected = [reward - mean(rewards) for reward in rewards]
    assert len(group.rollouts) == len(rewards)
    assert [rollout.reward for rollout in group.rollouts] == rewards
    assert [rollout.advantage for rollout in group.rollouts] == pytest.approx(expected, rel=1e-6, abs=1e-12)
    if len(set(rewards)) == 1:
        assert all(rollout.advantage == 0.0 for rollout in group.rollouts)


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param({"reference_tokens": 0}, id="nonpositive-reference"),
        pytest.param({"penalty": -0.1}, id="negative-penalty"),
        pytest.param({"penalty": 1.0}, id="success-must-remain-positive"),
        pytest.param({"start_fraction": 0.9}, id="removed-start-fraction"),
        pytest.param({"end_fraction": 1.0}, id="removed-end-fraction"),
    ],
)
def test_length_penalty_rejects_invalid_settings(settings: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        training.LengthPenaltyConfig(**settings)


def test_score_group_length_penalty_generated_tokens(rollout: training.Rollout) -> None:
    lengths = [0, 1024, 4096, 8192, 9000]
    expected = [1.0, 0.9875, 0.95, 0.9, 0.9]
    samples = [replace(rollout, num_input_tokens=100, num_output_tokens=length) for length in lengths]
    penalty = training.LengthPenaltyConfig(reference_tokens=8192, penalty=0.1)
    group = training.score_group(Mock(id="task"), samples, length_penalty=penalty, taskset_id="alfworld")
    assert group.training_rewards == pytest.approx(expected)
    assert [sample.reward for sample in group.rollouts] == [1.0] * len(lengths)
    assert [sample.advantage for sample in samples] == [rollout.advantage] * len(lengths)
    assert [sample.advantage for sample in group.rollouts] == pytest.approx(
        [reward - mean(expected) for reward in expected]
    )
    assert not group.is_degenerate
    more_context = training.score_group(
        Mock(id="task"),
        [replace(sample, num_input_tokens=10000) for sample in samples],
        length_penalty=penalty,
        taskset_id="alfworld",
    )
    assert more_context.training_rewards == group.training_rewards
    assert [sample.advantage for sample in more_context.rollouts] == [sample.advantage for sample in group.rollouts]


def test_score_group_length_penalty_preserves_coefficient(rollout: training.Rollout) -> None:
    samples = [replace(rollout, num_output_tokens=length) for length in (1, 2)]
    group = training.score_group(
        Mock(id="task"),
        samples,
        length_penalty=training.LengthPenaltyConfig(reference_tokens=2, penalty=0.1),
        taskset_id="alfworld",
    )
    weaker = training.score_group(
        Mock(id="task"),
        samples,
        length_penalty=training.LengthPenaltyConfig(reference_tokens=2, penalty=0.01),
        taskset_id="alfworld",
    )
    assert [sample.advantage for sample in group.rollouts] == pytest.approx([0.025, -0.025])
    assert [sample.advantage for sample in weaker.rollouts] == pytest.approx([0.0025, -0.0025])


def test_score_group_length_penalty_mixed_outcomes(rollout: training.Rollout) -> None:
    samples = [replace(rollout, num_output_tokens=8192), replace(rollout, reward=0.0, num_output_tokens=1)]
    group = training.score_group(
        Mock(id="task"), samples, length_penalty=training.LengthPenaltyConfig(penalty=0.05), taskset_id="alfworld"
    )
    assert group.training_rewards == pytest.approx([0.95, 0.0])
    assert [sample.advantage for sample in group.rollouts] == pytest.approx([0.475, -0.475])


@pytest.mark.parametrize(
    ("reward", "penalty"),
    [pytest.param(0.0, 0.1, id="failures"), pytest.param(1.0, 0.0, id="disabled")],
)
def test_score_group_length_penalty_neutral_groups(reward: float, penalty: float, rollout: training.Rollout) -> None:
    samples = [
        replace(rollout, reward=reward, num_output_tokens=index, stop_condition=stop)
        for index, stop in enumerate(
            ("max_tokens", "environment_done", "agent_completed", "max_turns", "context_length"), start=1
        )
    ]
    group = training.score_group(
        Mock(id="task"),
        samples,
        length_penalty=training.LengthPenaltyConfig(reference_tokens=4, penalty=penalty),
        taskset_id="alfworld",
    )
    assert group.training_rewards == [reward] * len(samples)
    assert all(sample.advantage == 0.0 for sample in group.rollouts)
    assert group.is_degenerate


def test_score_group_length_penalty_partial_credit(rollout: training.Rollout) -> None:
    group = training.score_group(
        Mock(id="task"),
        [replace(rollout, reward=0.5, num_output_tokens=length) for length in (4096, 8192, 9000)],
        length_penalty=training.LengthPenaltyConfig(penalty=0.05),
        taskset_id="reasoning_gym",
    )
    assert group.training_rewards == pytest.approx([0.4875, 0.475, 0.475])
    assert [sample.reward for sample in group.rollouts] == [0.5] * 3
    assert [sample.advantage for sample in group.rollouts] == pytest.approx([1 / 120, -1 / 240, -1 / 240])
    assert not group.is_degenerate


@pytest.mark.parametrize("reward", [-1.0, 2.0], ids=["negative", "above-one"])
def test_length_penalty_requires_unit_interval_rewards(reward: float, rollout: training.Rollout) -> None:
    with pytest.raises(ValueError, match=r"requires rewards in \[0, 1\]"):
        training.score_group(
            Mock(id="task"),
            [rollout, replace(rollout, reward=reward)],
            length_penalty=training.LengthPenaltyConfig(penalty=0.05),
            taskset_id="alfworld",
        )


@pytest.mark.parametrize(("preset", "penalty"), [("alfworld", 0.05), ("reasoning_gym", 0.0), ("mix", 0.05)])
def test_length_penalty_presets(preset: str, penalty: float) -> None:
    path = Path(__file__).parents[1] / "configs" / "grpo_v2" / f"{preset}.yaml"
    config = training.load_config(path, training.Config)
    assert config.length_penalty == training.LengthPenaltyConfig(reference_tokens=8192, penalty=penalty)
    assert config.run_id.endswith(f"_success_lp8192_{penalty}")


def test_reward_penalties_actor_batcher_integration(
    actor: Any, rollout: training.Rollout, monkeypatch: pytest.MonkeyPatch
) -> None:
    actor.config = replace(
        actor.config,
        length_penalty=training.LengthPenaltyConfig(reference_tokens=2, penalty=0.1),
        max_tokens=4,
        drop_degenerate_groups=True,
    )
    longer = replace(
        rollout,
        num_output_tokens=2,
        stop_condition="max_tokens",
        sequence=training.TrainingSequence(
            token_ids=np.array([1, 2, 3], dtype=np.int32),  # (T=3,)
            loss_mask=np.array([False, True, True]),  # (T=3,)
            old_log_probs=np.array([0.0, -0.2, -0.2], dtype=np.float32),  # (T=3,)
            weight_versions=np.where(np.array([False, True, True]), 0, -1).astype(np.int32),  # (T,)
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
            return await batcher.fetch_next_batch(batch_size=2, group_size=2, context_length=8)

    batch = asyncio.run(collect())
    assert handle.fetch_next_group.remote.call_count == 1
    assert batch.training_batch.advantages.shape == (2,)
    np.testing.assert_allclose(batch.training_batch.advantages, [0.025, -0.025])
    assert batch.rollout_metrics["rollout/degenerate_group_fraction"] == 0.0
    for suffix, raw, adjusted in (("mean", 1.0, 0.925), ("std", 0.0, 0.025), ("min", 1.0, 0.9), ("max", 1.0, 0.95)):
        assert batch.rollout_metrics[f"rollout/reward_{suffix}"] == raw
        assert batch.rollout_metrics[f"rollout/adjusted_reward_{suffix}"] == pytest.approx(adjusted)


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

    async def collect() -> training.RolloutBatch:
        async with aclosing(actor):
            with training.span("test.batch") as parent:
                async with asyncio.timeout(5):
                    result = await batcher.fetch_next_batch(
                        batch_size=4, group_size=2, context_length=16, include_examples=True
                    )
                assert otel_trace.get_current_span() is parent
                return result

    batch = asyncio.run(collect())
    assert started == 4
    assert f"Example 1 | task=alfworld:{task_discovery[1].id}" in batch.examples
    assert batch.examples.count("Full token sequence") == 1
    assert batch.training_batch.token_ids.shape == (4, 16)
    assert all(isinstance(value, np.ndarray) for value in jax.tree.leaves(batch.training_batch))
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


@pytest.mark.parametrize(
    ("phase", "error"),
    [
        pytest.param("setup", training.ActorDiedError(), id="worker-setup"),
        pytest.param("rollout", training.ActorDiedError(), id="worker-rollout"),
        pytest.param("setup", httpx.ConnectError("connection refused"), id="http-setup"),
        pytest.param("rollout", httpx.ReadError("connection lost"), id="http-read"),
        pytest.param("rollout", httpx.RemoteProtocolError("server disconnected"), id="http-protocol"),
    ],
)
def test_rollout_recovers_from_infrastructure_errors(
    actor: Any, rollout: training.Rollout, monkeypatch: pytest.MonkeyPatch, phase: str, error: Exception
) -> None:
    environments: list[AsyncMock] = []

    def init_env() -> AsyncMock:
        environment = AsyncMock()
        environment.__aenter__.return_value = environment
        if phase == "setup" and len(environments) < 2:
            environment.__aenter__.side_effect = error
        environments.append(environment)
        return environment

    task = Mock(id="fixed-task/123", init_env=Mock(side_effect=init_env))
    agent = AsyncMock()
    agent.__aenter__.return_value = agent
    agent.run.side_effect = [error, error, rollout, rollout] if phase == "rollout" else [rollout, rollout]
    monkeypatch.setattr(training, "create_agent", Mock(return_value=agent))

    group = asyncio.run(collect_and_close(actor, task))

    assert len(group.rollouts) == 2
    assert [item.reward for item in group.rollouts] == [rollout.reward, rollout.reward]
    assert len(environments) == 4
    assert len({id(item) for item in environments}) == 4
    for index, environment in enumerate(environments):
        environment.__aenter__.assert_awaited_once()
        assert environment.__aexit__.await_count == (0 if phase == "setup" and index < 2 else 1)
    assert not actor._runs


@pytest.mark.parametrize(
    "error", [training.ActorDiedError(), httpx.ReadError("connection lost")], ids=["worker", "http-read"]
)
def test_rollout_retry_exhaustion(
    actor: Any, monkeypatch: pytest.MonkeyPatch, task_discovery: list[ALFWorldTask], error: Exception
) -> None:
    agent = AsyncMock()
    agent.__aenter__.return_value = agent
    environments: list[Environment] = []

    async def run(env: Environment) -> training.Rollout:
        environments.append(env)
        if len(environments) <= 3:
            raise error
        return await asyncio.Future[training.Rollout]()

    agent.run.side_effect = run
    monkeypatch.setattr(training, "create_agent", Mock(return_value=agent))
    with pytest.raises(ExceptionGroup) as raised:
        asyncio.run(collect_and_close(actor, task_discovery[0]))

    assert len(raised.value.exceptions) == 1
    assert raised.value.exceptions[0] is error
    assert len(environments) == 4  # Three failed attempts and the cancelled sibling.
    for environment in environments:
        environment.__aexit__.assert_awaited_once()
    assert not actor._runs


@pytest.mark.parametrize(
    "error",
    [
        ValueError("invalid rollout"),
        training.ActorDiedError(cause=RayTaskError("__init__", "constructor failed", ValueError("invalid config"))),
        httpx.ReadTimeout("step timed out"),
        httpx.LocalProtocolError("invalid request"),
        httpx.HTTPStatusError(
            "bad request", request=httpx.Request("POST", "http://codegym/step"), response=httpx.Response(400)
        ),
    ],
    ids=["application-error", "actor-constructor-error", "http-timeout", "http-local-protocol", "http-status"],
)
def test_rollout_does_not_retry_application_errors(
    actor: Any, monkeypatch: pytest.MonkeyPatch, task_discovery: list[ALFWorldTask], error: Exception
) -> None:
    agent = AsyncMock()
    agent.__aenter__.return_value = agent
    agent.run.side_effect = error
    monkeypatch.setattr(training, "create_agent", Mock(return_value=agent))

    with pytest.raises(ExceptionGroup) as raised:
        asyncio.run(collect_and_close(actor, task_discovery[0]))

    assert agent.run.await_count == 2  # One attempt per rollout.
    assert all(item is error for item in raised.value.exceptions)


def test_rollout_retries_share_original_deadline(
    actor: Any, rollout: training.Rollout, monkeypatch: pytest.MonkeyPatch, task_discovery: list[ALFWorldTask]
) -> None:
    agent = AsyncMock()
    agent.__aenter__.return_value = agent
    monkeypatch.setattr(training, "create_agent", Mock(return_value=agent))

    async def collect() -> None:
        loop = asyncio.get_running_loop()
        now = loop.time()
        monkeypatch.setattr(loop, "time", lambda: now)
        attempts = 0

        async def run(env: Environment) -> training.Rollout:
            nonlocal now, attempts
            attempts += 1
            if attempts == 1:
                now += actor.config.rollout_timeout * 0.6
                raise httpx.ReadError("connection lost")
            if attempts == 2:
                now += actor.config.rollout_timeout * 0.6
                await asyncio.sleep(0)
                await asyncio.sleep(0)
            return rollout

        agent.run.side_effect = run
        await collect_and_close(actor, task_discovery[0])

    with pytest.raises(ExceptionGroup) as raised:
        asyncio.run(collect())
    assert len(raised.value.exceptions) == 1
    assert isinstance(raised.value.exceptions[0], TimeoutError)


def test_rollout_recovers_after_real_ray_worker_killed(
    actor: Any, rollout: training.Rollout, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Local workers use the existing environment without packaging the repository.
    monkeypatch.setattr(ray_constants, "RAY_ENABLE_UV_RUN_RUNTIME_ENV", False)
    actor.config = replace(actor.config, rollout_timeout=30)
    task = ReasoningGymTask(config=ReasoningGymConfig(task_config={"min_animals": 1, "max_animals": 1}), seed=42)
    agent = AsyncMock()
    agent.__aenter__.return_value = agent
    monkeypatch.setattr(training, "create_agent", Mock(return_value=agent))
    sessions: list[ReasoningGymSession] = []
    prompts: list[str] = []
    killed = False

    async def run(env: ReasoningGymSession) -> training.Rollout:
        nonlocal killed
        sessions.append(env)
        initial = await env.reset()
        prompts.append(initial.prompt)
        if not killed:
            killed = True
            await asyncio.to_thread(training.ray.kill, env.actor)
            await env.reset()  # Surface the actual ActorDiedError from the dead worker.
            pytest.fail("Killed worker unexpectedly answered reset")
        return rollout

    agent.run.side_effect = run
    with training.ray.init(
        address="local", num_cpus=2, num_gpus=0, include_dashboard=False, object_store_memory=80 * 1024**2
    ):
        group = asyncio.run(collect_and_close(actor, task))

    assert killed
    assert len(sessions) == len({id(session) for session in sessions}) == 3
    assert len(prompts) == 3 and len(set(prompts)) == 1
    assert len(group.rollouts) == 2
    assert [item.reward for item in group.rollouts] == [rollout.reward, rollout.reward]


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
            return await actor.collect_group(task, group_size=2, taskset_id="alfworld")

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
            await actor.collect_group(task, group_size=2, taskset_id="alfworld")

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
                    await actor.collect_group(task_discovery[0], group_size=2, taskset_id="alfworld")
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
        model="Qwen/custom",
        max_tokens=16,
        max_turns=2,
        temperature=0.7,
        rollout_timeout=300,
        finetuning=training.LoRAConfig() if mode == "lora" else training.FullFinetuningConfig(),
        loss=training.StableReinforceConfig() if mode == "full" else training.GRPOLossConfig(),
    )
    tokenizer = Mock()
    tokenizer.vocab_size = 128
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
            assert agent.sampler_top_k == (32 if mode == "full" else 0)

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


@pytest.mark.parametrize(
    ("inactive_window", "sync_steps"), [(False, 1), (True, 3)], ids=["every-update", "skip-inactive-window"]
)
def test_learner_accumulates_microbatches_and_publishes_on_updates(
    learner: Any, monkeypatch: pytest.MonkeyPatch, inactive_window: bool, sync_steps: int
) -> None:
    learner.config = replace(
        learner.config,
        batch_size=32,
        grad_accum_steps=16,
        parameter_sync_steps=sync_steps,
        total_iterations=4,
        checkpoint_interval=2,
        example_log_interval=2,
    )
    writer = MagicMock()
    writer.__enter__.return_value = writer
    monkeypatch.setattr(training, "MLflowLogger", Mock(return_value=writer))
    prepared = Mock(spec=training.GRPOBatch)
    prepared.token_ids = Mock(shape=(2, 2))  # (B=2, T=2)
    monkeypatch.setattr(training.ray, "get", lambda value: value)
    learner.batcher.fetch_next_batch.remote = Mock(
        return_value=training.RolloutBatch(
            training_batch=prepared,
            rollout_metrics={
                "rollout/reward_mean": 1.0,
                "rollout/degenerate_group_fraction": 0.0,
                "batcher/skipped_groups": 0,
                "batcher/stale_tokens": 0,
            },
            stop_counts={},
            num_output_tokens=2,
            examples="",
            timing_metrics={},
        )
    )
    del learner.apply_update  # Use the real optimizer update after the inner loop.
    published_steps: list[int] = []
    calls = 0

    def optimize(
        batch: training.GRPOBatch[np.ndarray],  # (B, T), (B,) leaves.
        gradient_sum: training.TrainParameters,  # Parameter shapes.
    ) -> training.BatchGradientResult:
        nonlocal calls
        calls += 1
        active = not (inactive_window and calls <= 16)
        return training.BatchGradientResult(
            gradient_sum=jax.tree.map(lambda value: value + float(active), gradient_sum),  # Parameter shapes.
            metrics={"train/policy_loss": 0.0},
            has_signal=active,
        )

    def publish() -> None:
        published_steps.append(int(learner.state.step))

    learner.optimize = optimize
    learner.publish_weights = publish
    learner.start()
    assert published_steps == ([0, 3] if inactive_window else [0, 1, 2, 3, 4])
    metrics = [call.args[0] for call in writer.log_metrics.call_args_list[1:]]
    assert [values["train/updates_total"] for values in metrics] == [
        index - int(inactive_window) for index in range(1, 5)
    ]
    assert [index for index, values in enumerate(metrics, start=1) if "learner/weight_transfer_seconds" in values] == (
        [4] if inactive_window else [1, 2, 3, 4]
    )
    fetches = learner.batcher.fetch_next_batch.remote.call_args_list
    assert [call.kwargs["iteration"] for call in fetches] == list(range(1, 65))
    assert [call.kwargs["iteration"] for call in fetches if call.kwargs["include_examples"]] == [32, 64]
    assert all(call.kwargs["batch_size"] == 2 for call in fetches)
    assert [values["train/rollouts_total"] for values in metrics] == [32, 64, 96, 128]
    saves = [call.args[1] for call in training.save_checkpoint.call_args_list]
    assert [progress.iteration for progress in saves] == [2, 4]
    assert int(learner.state.step) == 4 - int(inactive_window)


def test_iteration_rollout_metrics_pool_batches() -> None:
    summaries = [
        training.BatchSummary(
            metrics={
                "rollout/reward_mean": 0.0,
                "rollout/reward_std": 0.0,
                "rollout/reward_min": 0.0,
                "rollout/reward_max": 0.0,
                "rollout/reward_mean/first_max": 0.0,
                "rollout/reward_mean/shared_max": 0.0,
                "rollout/total_tokens_mean": 2.0,
                "batcher/padded_sequence_tokens": 4.0,
                "batcher/token_utilization_fraction": 0.5,
                "batcher/context_transcript_token_fraction": 0.5,
                "batcher/weight_age_transcript_token_fraction/0": 0.5,
                "batcher/skipped_groups": 1,
                "batcher/stale_tokens": 1,
                "actor/rollout_seconds_mean": 1.0,
            },
            num_rollouts=2,
            taskset_counts={"first_max": 1, "shared_max": 1},
        ),
        training.BatchSummary(
            metrics={
                "rollout/reward_mean": 3.0,
                "rollout/reward_std": 1.0,
                "rollout/reward_min": 2.0,
                "rollout/reward_max": 4.0,
                "rollout/reward_mean/second": 4.0,
                "rollout/reward_mean/shared_max": 2.0,
                "rollout/total_tokens_mean": 6.0,
                "batcher/padded_sequence_tokens": 8.0,
                "batcher/token_utilization_fraction": 0.75,
                "batcher/context_transcript_token_fraction": 0.75,
                "batcher/weight_age_transcript_token_fraction/1": 0.25,
                "batcher/skipped_groups": 2,
                "batcher/stale_tokens": 3,
                "actor/rollout_seconds_mean": 3.0,
            },
            num_rollouts=2,
            taskset_counts={"second": 1, "shared_max": 1},
        ),
    ]
    assert training.summarize_iteration(summaries) == pytest.approx(
        {
            "rollout/reward_mean": 1.5,
            "rollout/reward_std": 2.75**0.5,
            "rollout/reward_min": 0.0,
            "rollout/reward_max": 4.0,
            "rollout/reward_mean/first_max": 0.0,
            "rollout/reward_mean/second": 4.0,
            "rollout/reward_mean/shared_max": 1.0,
            "rollout/total_tokens_mean": 4.0,
            "batcher/padded_sequence_tokens": 8.0,
            "batcher/token_utilization_fraction": 2 / 3,
            "batcher/context_transcript_token_fraction": 11 / 16,
            "batcher/weight_age_transcript_token_fraction/0": 2 / 16,
            "batcher/weight_age_transcript_token_fraction/1": 3 / 16,
            "batcher/skipped_groups": 3,
            "batcher/stale_tokens": 4,
            "actor/rollout_seconds_mean": 2.0,
        }
    )


def test_batcher_drains_on_outer_iteration_boundaries(rollout: training.Rollout) -> None:
    actor = Mock()
    actor.fetch_next_group.remote = AsyncMock(
        return_value=training.RolloutGroup(task_id="0", rollouts=[rollout] * 2, taskset_id="test")
    )
    batcher = training.Batcher.__ray_metadata__.modified_class(
        actor=actor,
        config=training.Config(
            batch_size=8,
            grad_accum_steps=4,
            parameter_sync_steps=3,
            checkpoint_interval=2,
            total_iterations=3,
            prefetch_batches=3,
        ),
    )

    async def collect() -> None:
        async with asyncio.timeout(5):
            for iteration in range(1, 13):
                await batcher.fetch_next_batch(batch_size=2, group_size=2, context_length=8, iteration=iteration)
                await asyncio.sleep(0)
                if iteration in (8, 12):
                    assert not batcher._pending_groups
                    assert actor.fetch_next_group.remote.call_count == iteration
                else:
                    assert batcher._pending_groups

    asyncio.run(collect())


def test_batch_size_is_independent_of_group_size() -> None:
    config = training.Config(batch_size=2, group_size=7, learner_gpus=2)
    assert config.batch_size == 2
    with pytest.raises(ValueError, match="learner_gpus must divide microbatch_size"):
        training.Config(batch_size=3, group_size=4, learner_gpus=2)


@pytest.mark.parametrize(
    ("batch_size", "grad_accum_steps", "learner_gpus", "expected"),
    [(32, 16, 2, 2), (4, 1, 1, 4), (4, 4, 1, 1)],
    ids=["accumulated-data-parallel", "no-accumulation", "single-rollout"],
)
def test_microbatch_size(batch_size: int, grad_accum_steps: int, learner_gpus: int, expected: int) -> None:
    config = training.Config(batch_size=batch_size, grad_accum_steps=grad_accum_steps, learner_gpus=learner_gpus)
    assert config.microbatch_size == expected


@pytest.mark.parametrize(
    ("batch_size", "grad_accum_steps", "learner_gpus", "message"),
    [(3, 2, 1, "grad_accum_steps"), (2, 4, 1, "grad_accum_steps"), (4, 4, 2, "learner_gpus")],
    ids=["unequal-microbatches", "empty-microbatch", "microbatch-smaller-than-device-count"],
)
def test_invalid_microbatch_size(batch_size: int, grad_accum_steps: int, learner_gpus: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        training.Config(batch_size=batch_size, grad_accum_steps=grad_accum_steps, learner_gpus=learner_gpus)


def test_prefetch_capacity_uses_full_batch_size(rollout: training.Rollout) -> None:
    config = training.Config(batch_size=32, grad_accum_steps=16, group_size=8, prefetch_batches=2)
    actor = Mock()
    actor.fetch_next_group.remote = AsyncMock(
        return_value=training.RolloutGroup(task_id="0", taskset_id="test", rollouts=[rollout] * 8)
    )
    batcher = training.Batcher.__ray_metadata__.modified_class(actor=actor, config=config)

    async def collect() -> None:
        async with asyncio.timeout(5):
            for iteration in range(1, 17):
                batch = await batcher.fetch_next_batch(
                    batch_size=config.microbatch_size, group_size=8, context_length=8, iteration=iteration
                )
                assert batch.training_batch.token_ids.shape == (2, 8)
                assert batch.training_batch.advantages.shape == (2,)
                await asyncio.sleep(0)
                assert len(batcher._pending_groups) * config.group_size == 64
            assert actor.fetch_next_group.remote.call_count * config.group_size == 32 + 64
            assert not batcher._completed_groups
            await batcher.cancel_pending_groups()

    asyncio.run(collect())


@pytest.mark.parametrize("prefetch_batches", [0, 1])
def test_batcher_splits_groups_and_restores_leftovers(rollout: training.Rollout, prefetch_batches: int) -> None:
    actor = Mock()
    groups = [
        training.score_group(
            task=Mock(name="task", id=str(index)),
            taskset_id=f"taskset_{index}",
            rollouts=[
                replace(
                    rollout,
                    reward=reward,
                    interaction=f"rollout-{index * 4 + row}",
                    sequence=replace(
                        rollout.sequence,
                        token_ids=np.array([1, index * 4 + row + 2], dtype=np.int32),  # (T=2,)
                    ),
                )
                for row, reward in enumerate([0.0, 1.0, 1.0, 1.0])
            ],
        )
        for index in range(3)
    ]
    actor.fetch_next_group.remote = AsyncMock(side_effect=groups)
    config = training.Config(
        batch_size=3,
        group_size=4,
        prefetch_batches=prefetch_batches,
        checkpoint_interval=2,
        max_staleness=0,
        drop_degenerate_groups=True,
    )
    batcher_class = training.Batcher.__ray_metadata__.modified_class

    async def collect() -> None:
        batcher = batcher_class(actor=actor, config=config)
        consumed: list[int] = []
        for iteration in range(1, 5):
            batch = await batcher.fetch_next_batch(
                batch_size=3,
                group_size=4,
                context_length=8,
                iteration=iteration,
                consumption_weight_version=0 if iteration <= 2 else 1,
                include_examples=True,
            )
            assert batch.training_batch.token_ids.shape == (3, 8)
            consumed.extend(batch.training_batch.token_ids[:, 1].tolist())
            expected = [r.advantage for g in groups for r in g.rollouts][(iteration - 1) * 3 : iteration * 3]
            np.testing.assert_array_equal(batch.training_batch.advantages, expected)
            assert batch.rollout_metrics["rollout/degenerate_group_fraction"] == 0
            assert batch.rollout_metrics["batcher/skipped_groups"] == 0
            if iteration == 2:
                assert batch.taskset_counts == {"taskset_0": 1, "taskset_1": 2}
                assert batch.rollout_metrics["rollout/reward_mean/taskset_0"] == 1
                assert not batcher._pending_groups
                snapshot = json.loads(json.dumps(batcher.get_pool_state()))
                assert len(snapshot[0]["rollouts"]) == 2
                batcher = batcher_class(actor=actor, config=config)
                batcher.set_pool_state(snapshot)
            if iteration == 3:
                assert batch.rollout_metrics["batcher/stale_tokens"] == 3
                assert not batch.training_batch.loss_mask.any()
                assert "rollout-7" in batch.examples
        assert consumed == list(range(2, 14))
        assert actor.fetch_next_group.remote.call_count == 3
        assert not batcher._pending_groups
        assert batcher.get_pool_state() == []

    asyncio.run(collect())


def test_batcher_honors_each_requested_size(rollout: training.Rollout) -> None:
    actor = Mock()
    actor.fetch_next_group.remote = AsyncMock(
        side_effect=[
            training.RolloutGroup(
                task_id="mixed",
                taskset_id="test",
                rollouts=[replace(rollout, reward=0.0)] + [rollout] * 3,
            ),
            training.RolloutGroup(task_id="equal", taskset_id="test", rollouts=[rollout] * 4),
        ]
    )
    batcher = training.Batcher.__ray_metadata__.modified_class(actor=actor)

    async def collect() -> None:
        for iteration, (size, calls) in enumerate([(1, 1), (2, 1), (5, 2)], start=1):
            batch = await batcher.fetch_next_batch(
                batch_size=size,
                group_size=4,
                context_length=8,
                iteration=iteration,
            )
            assert batch.training_batch.token_ids.shape == (size, 8)
            assert batch.training_batch.advantages.shape == (size,)
            assert batch.rollout_metrics["rollout/degenerate_group_fraction"] == (4 / 5 if iteration == 3 else 0)
            assert actor.fetch_next_group.remote.call_count == calls
        assert batcher.get_pool_state() == []

    asyncio.run(collect())
