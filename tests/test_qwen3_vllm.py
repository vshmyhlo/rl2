"""Opt-in CUDA/NCCL parity: QWEN3_VLLM_TESTS=1; optionally QWEN3_CHECKPOINT=/local/snapshot."""

import json
import os
import subprocess
import time
from collections.abc import Generator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from openai import OpenAI
from safetensors.flax import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from vllm import _custom_ops as ops

from rl2 import qwen3
from rl2 import train_grpo_v2 as training
from rl2.agent_loop import completion_tokens

pytestmark = pytest.mark.skipif(os.environ.get("QWEN3_VLLM_TESTS") != "1", reason="Set QWEN3_VLLM_TESTS=1")


def torch_array(value: jax.Array) -> torch.Tensor:  # (*shape) -> (*shape)
    return torch.tensor(np.asarray(value.astype(jnp.float32)), device="cuda", dtype=getattr(torch, str(value.dtype)))


def errors(actual: jax.Array, expected: np.ndarray) -> dict[str, float]:  # Both (*shape).
    delta = np.abs(np.asarray(actual, dtype=np.float32) - expected)  # (*shape)
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean())}


@pytest.fixture(scope="module", autouse=True)
def require_cuda() -> None:
    assert jax.default_backend() == "gpu" and torch.cuda.is_available()
    assert not jax.config.jax_disable_jit


@pytest.mark.parametrize("operation", ["norm", "residual_norm", "rope", "swiglu"])
def test_cuda_operations(operation: str) -> None:
    x = jax.random.normal(jax.random.key(1), (2, 17, 4, 128)).astype(jnp.bfloat16)  # (B, T, N, H)
    other = jax.random.normal(jax.random.key(2), x.shape).astype(jnp.bfloat16)  # (B, T, N, H)
    weight = (1 + 0.1 * jax.random.normal(jax.random.key(3), (128,))).astype(jnp.bfloat16)  # (H,)
    expected = torch.empty_like(torch_array(x))  # (B, T, N, H)
    if operation == "norm":
        ops.rms_norm(expected, torch_array(x), torch_array(weight), 1e-6)
        actual = jax.jit(qwen3._rms_norm)(x, weight, 1e-6)  # (B, T, N, H)
    elif operation == "residual_norm":
        expected = torch_array(x)  # (B, T, N, H)
        ops.fused_add_rms_norm(expected, torch_array(other), torch_array(weight), 1e-6)
        actual = jax.jit(lambda a, b, w: qwen3._rms_norm(a + b, w, 1e-6))(x, other, weight)  # (B, T, N, H)
    elif operation == "swiglu":
        torch.ops._C.silu_and_mul(expected, torch.cat((torch_array(x), torch_array(other)), dim=-1))
        actual = jax.jit(lambda a, b: qwen3._silu(a) * b)(x, other)  # (B, T, N, H)
    else:
        positions = jnp.arange(34).reshape(2, 17)  # (B, T)
        angles = jnp.arange(34)[:, None] * 1e6 ** (-jnp.arange(0, 128, 2, dtype=jnp.float32) / 128)  # (B*T, H/2)
        cache = jnp.concatenate((jnp.cos(angles), jnp.sin(angles)), axis=-1).astype(jnp.bfloat16)  # (B*T, H)
        expected = torch_array(x).reshape(34, 512)  # (B*T, N*H)
        ops.rotary_embedding(torch.arange(34, device="cuda"), expected, expected.clone(), 128, torch_array(cache), True)
        expected = expected.reshape(x.shape)  # (B, T, N, H)
        actual = jax.jit(qwen3._rope)(x, positions, 1e6)  # (B, T, N, H)
    report = errors(actual, expected.float().cpu().numpy())
    print(operation, report)
    np.testing.assert_allclose(actual.astype(jnp.float32), expected.float().cpu(), atol=0.002, rtol=0.008)


@dataclass
class Engine:
    model: qwen3.Qwen3LM
    params: qwen3.Parameters  # HF parameter shapes.
    learner: Any
    client: OpenAI
    directory: Path
    real: bool


@pytest.fixture(scope="module", params=["tied", "untied", *(["checkpoint"] if os.environ.get("QWEN3_CHECKPOINT") else [])])
def engine(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Generator[Engine]:
    directory = tmp_path_factory.mktemp(f"vllm-{request.param}")
    real = request.param == "checkpoint"
    if real:
        source = Path(os.environ["QWEN3_CHECKPOINT"]).resolve()
        model, variables = qwen3.load_qwen3_checkpoint(str(source), dtype="float32")
        model = model.clone(dtype="bfloat16")
    else:
        source = directory / "model"
        source.mkdir()
        config = qwen3.Qwen3Config(
            vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=32,
            max_position_embeddings=256, eos_token_id=2, tie_word_embeddings=request.param == "tied",
        )
        model = qwen3.Qwen3LM(config=config)
        variables = model.init(jax.random.key(42), jnp.ones((1, 2), jnp.int32))
        # Trained norm scales expose rounding differences hidden by unit initialization.
        variables["params"] = {
            name: value * jnp.linspace(0.8, 1.2, value.size).reshape(value.shape) if "norm" in name else value
            for name, value in variables["params"].items()
        }  # HF parameter shapes.
        (source / "config.json").write_text(json.dumps({**asdict(config), "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}))
        save_file(jax.tree.map(lambda v: v.astype(jnp.bfloat16), variables["params"]), str(source / "model.safetensors"))
        tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(config.vocab_size)}, unk_token="t0"))
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0", eos_token="t2").save_pretrained(source)
    port = training.get_open_port()
    url = f"http://127.0.0.1:{port}"
    popen = subprocess.Popen

    def launch(command: list[str], *, env: dict[str, str]) -> subprocess.Popen[bytes]:
        command = list(command)
        command[command.index("--port") + 1] = str(port)
        command += ["--max-model-len", "256", "--max-num-seqs", "4", "--gpu-memory-utilization", "0.3", "--enforce-eager"]
        return popen(command, env={**env, "CUDA_VISIBLE_DEVICES": os.environ.get("QWEN3_VLLM_DEVICE", "1")}, stdout=log, stderr=subprocess.STDOUT)

    def post(path: str, body: dict[str, Any]) -> None:
        response = http.post(path, json=body)
        response.raise_for_status()

    def receive(info: dict[str, Any]) -> None:
        post("/pause", {"mode": "keep"})
        post("/start_weight_update", {})
        post("/update_weights", {"update_info": info})
        post("/finish_weight_update", {})
        post("/resume", {})

    def resolve(pending: Future[None]) -> None:
        pending.result(timeout=120)

    with ExitStack() as stack:
        log = stack.enter_context((directory / "server.log").open("wb"))
        with patch.object(training.subprocess, "Popen", launch):
            server = stack.enter_context(training.serve_vllm(str(source)))
        http = stack.enter_context(httpx.Client(base_url=url, timeout=120))
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            assert server.poll() is None, (directory / "server.log").read_text()
            try:
                if http.get("/health").status_code == 200:
                    break
            except httpx.ConnectError:
                pass
            time.sleep(1)
        else:
            pytest.fail(f"vLLM startup timeout: {directory / 'server.log'}")
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=1))
        learner_cls = training.Learner.__ray_metadata__.modified_class
        learner = object.__new__(learner_cls)
        learner.actor = SimpleNamespace(
            initialize_weight_transfer=SimpleNamespace(remote=lambda info: pool.submit(post, "/init_weight_transfer_engine", {"init_info": info})),
            update_weights=SimpleNamespace(remote=lambda info: pool.submit(receive, info)),
        )
        stack.enter_context(patch.object(training.ray, "get", resolve))
        learner.initialize_weight_transfer(stack)
        client = stack.enter_context(OpenAI(base_url=f"{url}/v1", api_key="unused"))
        yield Engine(model=model, params=variables["params"], learner=learner, client=client, directory=directory, real=real)


def test_published_rollout_parity(engine: Engine) -> None:
    model, master = engine.model, engine.params
    prompts = [[1, 7, 11], [1, 5, 9, 12, 17, 23, 31]]
    if engine.real:
        tokenizer = Tokenizer.from_file(str(Path(os.environ["QWEN3_CHECKPOINT"]) / "tokenizer.json"))
        prompts = [tokenizer.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).ids for text in ("What is 17 + 23?", "Name the capital of France.")]
    reports: list[dict[str, Any]] = []
    previous: np.ndarray | None = None  # (B, T-1)
    for version in range(2):
        # Deliberately change published values, including tied embeddings and norm scales.
        params = {name: value + version * 0.01 * jnp.sin(jnp.arange(value.size).reshape(value.shape)) for name, value in master.items()}  # HF parameter shapes.
        state = training.create_state(model, params, training.Config())
        engine.learner.state = state
        engine.learner.publish_weights()
        for temperature in (1.0, 0.7):
            sequences: list[list[int]] = []
            sampled: list[list[float]] = []
            for prompt in prompts:
                result = engine.client.completions.create(
                    model=engine.client.models.list().data[0].id, prompt=prompt, max_tokens=8,
                    temperature=temperature, top_p=1, logprobs=0, seed=123,
                    extra_body={"return_token_ids": True, "return_tokens_as_token_ids": True, "ignore_eos": True},
                )
                ids, log_probs = completion_tokens(result.choices[0])
                assert len(ids) == 8
                sequences.append(prompt + ids)
                sampled.append(log_probs)
            width = max(map(len, sequences))
            tokens = jnp.asarray([ids + [0] * (width - len(ids)) for ids in sequences], jnp.int32)  # (B, T)
            attention = jnp.arange(width)[None, :] < jnp.asarray(list(map(len, sequences)))[:, None]  # (B, T)
            loss_mask = attention & (jnp.arange(width)[None, :] >= jnp.asarray(list(map(len, prompts)))[:, None])  # (B, T)
            batch = training.GRPOBatch(token_ids=tokens, attention_mask=attention, loss_mask=loss_mask,
                old_log_probs=jnp.zeros(tokens.shape), advantages=jnp.ones((len(sequences),)))
            for backend in ("xla", "cudnn", "tokamax"):
                backend_state = state.replace(apply_fn=model.clone(attention_backend=backend).apply)
                actual = jax.jit(training.token_log_probs, static_argnames="temperature")(backend_state, params, batch, temperature)  # (B, T-1)
                assert actual.shape == (len(sequences), width - 1)
                selected = actual[loss_mask[:, 1:]]  # (sampled tokens,)
                expected = np.asarray(sampled, dtype=np.float32).reshape(-1)  # (sampled tokens,)
                report = {"version": version, "temperature": temperature, "backend": backend, **errors(selected, expected)}
                reports.append(report)
                print(report)
                (engine.directory / "parity.json").write_text(json.dumps(reports, indent=2))
                np.testing.assert_allclose(selected, expected, atol=0.1 if engine.real else 0.02, rtol=0)
            # A second publication must change observable scores; unchanged weights cannot pass silently.
            probe = jax.jit(model.apply)({"params": params}, jnp.asarray([prompts[0]], jnp.int32)).logits  # (1, P, V)
            if temperature == 1.0:
                if previous is not None:
                    assert not np.array_equal(previous, probe)
                previous = np.asarray(probe)
