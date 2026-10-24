"""Opt-in CUDA/NCCL parity: QWEN3_VLLM_TESTS=1; optionally QWEN3_CHECKPOINT=/local/snapshot."""

import asyncio
import json
import os
import subprocess
import time
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from importlib.metadata import version as package_version
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
from vllm.ir.ops.layernorm import fused_add_rms_norm, rms_norm
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

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
    assert len(jax.devices()) == 1, (
        "Expose one learner GPU with CUDA_VISIBLE_DEVICES; QWEN3_VLLM_DEVICE selects the other"
    )


@pytest.mark.parametrize("operation", ["norm", "residual_norm", "rope", "qk_norm_rope", "swiglu"])
def test_compiled_operations(operation: str) -> None:
    x = jax.random.normal(jax.random.key(1), (2, 17, 4, 128)).astype(jnp.bfloat16)  # (B, T, N, H)
    other = jax.random.normal(jax.random.key(2), x.shape).astype(jnp.bfloat16)  # (B, T, N, H)
    weight = (1 + 0.1 * jax.random.normal(jax.random.key(3), (128,))).astype(jnp.bfloat16)  # (H,)
    if operation == "norm":
        expected = torch.compile(rms_norm.impls["native"].impl_fn)(
            torch_array(x), torch_array(weight), 1e-6
        )  # (B, T, N, H)
        actual = jax.jit(qwen3._rms_norm)(x, weight, 1e-6)  # (B, T, N, H)
    elif operation == "residual_norm":
        expected, _ = torch.compile(fused_add_rms_norm.impls["native"].impl_fn)(
            torch_array(x), torch_array(other), torch_array(weight), 1e-6
        )  # Each (B, T, N, H).
        actual = jax.jit(lambda a, b, w: qwen3._rms_norm(a + b, w, 1e-6))(x, other, weight)  # (B, T, N, H)
    elif operation == "swiglu":
        expected = torch.compile(SiluAndMul.forward_native)(
            torch.cat((torch_array(x), torch_array(other)), dim=-1)
        )  # (B, T, N, H)
        actual = jax.jit(lambda a, b: qwen3._silu(a) * b)(x, other)  # (B, T, N, H)
    else:
        positions = jnp.arange(34).reshape(2, 17)  # (B, T)
        angles = jnp.arange(34)[:, None] * 1e6 ** (-jnp.arange(0, 128, 2, dtype=jnp.float32) / 128)  # (B*T, H/2)
        cache = jnp.concatenate((jnp.cos(angles), jnp.sin(angles)), axis=-1).astype(jnp.bfloat16)  # (B*T, H)

        def reference(
            values: torch.Tensor, scales: torch.Tensor, trig: torch.Tensor
        ) -> torch.Tensor:  # (B, T, N, H), (H,), (B*T, H)
            if operation == "qk_norm_rope":
                values = rms_norm.impls["native"].impl_fn(values, scales, 1e-6)  # (B, T, N, H)
            rotated, _ = RotaryEmbedding.forward_static(
                torch.arange(34, device=values.device), values.reshape(34, 512), None, 128, 128, trig, True
            )  # (B*T, N*H), None.
            return rotated.reshape(values.shape)  # (B, T, N, H)

        def forward(
            values: jax.Array, scales: jax.Array, positions: jax.Array
        ) -> jax.Array:  # (B, T, N, H), (H,), (B, T)
            if operation == "qk_norm_rope":
                values = qwen3._rms_norm(values, scales, 1e-6)  # (B, T, N, H)
            return qwen3._rope(values, positions, 1e6)  # (B, T, N, H)

        expected = torch.compile(reference)(torch_array(x), torch_array(weight), torch_array(cache))  # (B, T, N, H)
        actual = jax.jit(forward)(x, weight, positions)  # (B, T, N, H)
    expected_array = expected.float().cpu().numpy()  # (B, T, N, H)
    print(operation, errors(actual, expected_array))
    assert actual.shape == expected_array.shape == x.shape
    # Reduction/exp implementations can differ at an occasional BF16 rounding tie.
    assert np.mean(np.asarray(actual, dtype=np.float32) != expected_array) < 0.001
    np.testing.assert_allclose(actual.astype(jnp.float32), expected_array, atol=0.0001, rtol=0.008)


@dataclass
class Engine:
    model: qwen3.Qwen3LM
    params: qwen3.Parameters  # HF parameter shapes.
    learner: Any
    client: OpenAI
    directory: Path
    real: bool


def attention_forward(model: qwen3.Qwen3LM, implementation: str) -> Callable[..., qwen3.Qwen3Output]:
    original = qwen3.tokamax.dot_product_attention

    def attention(*args: Any, **kwargs: Any) -> jax.Array:  # Q/K/V shapes -> (B, T, N, H).
        return original(*args, **{**kwargs, "implementation": implementation})  # (B, T, N, H)

    def forward(
        variables: qwen3.Variables,
        tokens: jax.Array,  # (B, T)
        attention_mask: jax.Array,  # (B, T)
    ) -> qwen3.Qwen3Output:
        # Each closure gets its own JIT trace; the implementation cannot leak through JIT caches.
        with patch.object(qwen3.tokamax, "dot_product_attention", attention):
            return model.apply(variables, tokens, attention_mask=attention_mask)

    return forward


@pytest.fixture(
    scope="module", params=["tied", "untied", *(["checkpoint"] if os.environ.get("QWEN3_CHECKPOINT") else [])]
)
def engine(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Generator[Engine]:
    directory = tmp_path_factory.mktemp(f"vllm-{request.param}")
    if destination := os.environ.get("QWEN3_VLLM_REPORT_DIR"):
        directory = Path(destination) / request.param
        directory.mkdir(parents=True, exist_ok=True)
    real = request.param == "checkpoint"
    if real:
        source = Path(os.environ["QWEN3_CHECKPOINT"]).resolve()
        model, variables = qwen3.load_qwen3_checkpoint(str(source), dtype="float32")
        model = model.clone(dtype="bfloat16")
    else:
        source = directory / "model"
        source.mkdir(exist_ok=True)
        config = qwen3.Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=256,
            eos_token_id=2,
            tie_word_embeddings=request.param == "tied",
        )
        model = qwen3.Qwen3LM(config=config)
        variables = model.init(jax.random.key(42), jnp.ones((1, 2), jnp.int32))
        # Trained norm scales expose rounding differences hidden by unit initialization.
        variables["params"] = {
            name: value * jnp.linspace(0.8, 1.2, value.size).reshape(value.shape) if "norm" in name else value
            for name, value in variables["params"].items()
        }  # HF parameter shapes.
        (source / "config.json").write_text(
            json.dumps({**asdict(config), "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]})
        )
        save_file(
            jax.tree.map(lambda v: v.astype(jnp.bfloat16), variables["params"]), str(source / "model.safetensors")
        )
        tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(config.vocab_size)}, unk_token="t0"))
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0", eos_token="t2").save_pretrained(source)
    port = training.get_open_port()
    url = f"http://127.0.0.1:{port}"
    popen = subprocess.Popen
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "jax": jax.__version__,
                "torch": torch.__version__,
                "vllm": package_version("vllm"),
                "tokamax": package_version("tokamax"),
                "device": jax.devices()[0].device_kind,
                "dtype": "bfloat16",
                "master_dtype": "float32",
                "padding_multiple": training.Config().padding_multiple if real else 1,
                "eager": os.environ.get("QWEN3_VLLM_EAGER") == "1",
                "optimization_level": jax.config.jax_optimization_level,
                "matmul_precision": jax.config.jax_default_matmul_precision,
                "environment": {
                    key: os.environ.get(key)
                    for key in (
                        "XLA_FLAGS",
                        "XLA_PYTHON_CLIENT_PREALLOCATE",
                        "XLA_CLIENT_MEM_FRACTION",
                        "JAX_COMPILATION_CACHE_DIR",
                        "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES",
                        "NVIDIA_TF32_OVERRIDE",
                        "NCCL_NET",
                        "NCCL_PROFILER_PLUGIN",
                    )
                },
            },
            indent=2,
        )
    )
    print(f"vLLM parity artifacts: {directory}", flush=True)

    def launch(command: list[str], *, env: dict[str, str]) -> subprocess.Popen[bytes]:
        command = list(command)
        command[command.index("--port") + 1] = str(port)
        command += [
            "--max-model-len",
            "256",
            "--max-num-seqs",
            "4",
            "--gpu-memory-utilization",
            "0.3",
            "--max-logprobs",
            "128",
        ]
        if os.environ.get("QWEN3_VLLM_EAGER") == "1":
            command.append("--enforce-eager")
        return popen(
            command,
            env={
                **env,
                "CUDA_VISIBLE_DEVICES": os.environ.get("QWEN3_VLLM_DEVICE", "1"),
                "JAX_PLATFORMS": "cpu",  # Actor's production Ray runtime environment.
            },
            stdout=log,
            stderr=subprocess.STDOUT,
        )

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
        actor = object.__new__(training.Actor.__ray_metadata__.modified_class)
        actor.config = training.Config()
        stack.enter_context(patch.object(training, "SERVER_URL", url))

        def initialize(info: dict[str, Any]) -> Future[None]:
            return pool.submit(asyncio.run, actor.initialize_weight_transfer(info))

        def receive(info: dict[str, Any]) -> Future[None]:
            return pool.submit(asyncio.run, actor.update_weights(info))

        learner_cls = training.Learner.__ray_metadata__.modified_class
        learner = object.__new__(learner_cls)
        learner.actor = SimpleNamespace(
            initialize_weight_transfer=SimpleNamespace(remote=initialize),
            update_weights=SimpleNamespace(remote=receive),
        )
        stack.enter_context(patch.object(training.ray, "get", resolve))
        learner.initialize_weight_transfer(stack)
        client = stack.enter_context(OpenAI(base_url=f"{url}/v1", api_key="unused"))
        yield Engine(
            model=model, params=variables["params"], learner=learner, client=client, directory=directory, real=real
        )


def test_published_rollout_parity(engine: Engine) -> None:
    model, master = engine.model, engine.params
    forwards = {backend: model.clone(attention_backend=backend).apply for backend in ("xla", "cudnn")}
    implementations = ["triton", "cudnn", "xla", "xla_chunked"]
    capability = torch.cuda.get_device_capability()
    if capability[0] in (9, 10):
        implementations.append("mosaic")
    else:
        print(f"Tokamax Mosaic unavailable: requires SM90/SM100; this GPU is SM{capability[0]}{capability[1]}.")
    forwards.update(
        {
            f"tokamax-{implementation}": attention_forward(model.clone(attention_backend="tokamax"), implementation)
            for implementation in implementations
        }
    )
    prompts = [[1], [1, 5, 9, 12, 17, 23, 31]]
    observation = [3, 6, 4]
    if engine.real:
        tokenizer = Tokenizer.from_file(str(Path(os.environ["QWEN3_CHECKPOINT"]) / "tokenizer.json"))
        prompts = [
            tokenizer.encode(
                f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                add_special_tokens=False,
            ).ids
            for text in (
                "What is 17 + 23?",
                "Describe an unusual use for an ordinary household object. Explain why it could work, including one limitation.",
            )
        ]
        observation = tokenizer.encode(
            "<|im_end|>\n<|im_start|>user\nPlease continue with one more detail.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
        ).ids
    reports: list[dict[str, Any]] = []
    previous: dict[str, float] | None = None
    model_id = engine.client.models.list().data[0].id
    state = training.create_state(model, master, training.Config())
    for version in range(2):
        params = {
            name: value + version * (0.01 if "norm" in name else 0.0001) * jnp.sin(jnp.arange(value.shape[-1]))
            for name, value in master.items()
        }  # HF parameter shapes; FP32 updates must be rounded identically on both sides.
        state = state.replace(params=params)
        engine.learner.state = state
        engine.learner.publish_weights()
        for temperature in (1.0, 0.7):
            sequences = [list(prompt) for prompt in prompts]
            masks = [[False] * len(prompt) for prompt in prompts]
            sampled = [[0.0] * len(prompt) for prompt in prompts]
            candidates: dict[tuple[int, int, int], float] = {}
            for turn in range(2):
                result = engine.client.completions.create(
                    model=model_id,
                    prompt=sequences,
                    max_tokens=8,
                    temperature=temperature,
                    top_p=1,
                    logprobs=10 if engine.real else model.config.vocab_size,
                    seed=123,
                    extra_body={"return_token_ids": True, "return_tokens_as_token_ids": True, "ignore_eos": True},
                )
                assert len(result.choices) == len(prompts)
                for choice in result.choices:
                    row = choice.index
                    ids, log_probs = completion_tokens(choice)
                    assert len(ids) == 8
                    assert choice.logprobs is not None and choice.logprobs.top_logprobs is not None
                    for offset, alternatives in enumerate(choice.logprobs.top_logprobs):
                        if engine.real:
                            assert (
                                10 <= len(alternatives) <= 11
                            )  # vLLM includes the sampled token if outside the top ten.
                        else:
                            assert len(alternatives) == model.config.vocab_size
                        for token, score in alternatives.items():
                            candidates[row, len(sequences[row]) + offset - 1, int(token.removeprefix("token_id:"))] = (
                                score
                            )
                    # Reused prefixes must see the new publication, including prefix-cache invalidation.
                    if turn == 0 and row == 0 and temperature == 1.0:
                        current = choice.logprobs.top_logprobs[0]
                        if previous is not None:
                            common = previous.keys() & current.keys()
                            assert common and max(abs(current[key] - previous[key]) for key in common) > 1e-4
                        previous = current
                    sequences[row].extend(ids)
                    masks[row].extend([True] * len(ids))
                    sampled[row].extend(log_probs)
                    if turn == 0:
                        sequences[row].extend(observation)
                        masks[row].extend([False] * len(observation))
                        sampled[row].extend([0.0] * len(observation))
            rollouts = [
                training.Rollout(
                    sequence=training.TrainingSequence(
                        token_ids=np.asarray(ids, dtype=np.int32),  # (L,)
                        loss_mask=np.asarray(mask, dtype=np.bool_),  # (L,)
                        old_log_probs=np.asarray(scores, dtype=np.float32),  # (L,)
                    ),
                    prompt="",
                    reward=1.0,
                    advantage=1.0,
                    completion="",
                    interaction="",
                    num_turns=2,
                    num_input_tokens=len(ids) - sum(mask),
                    num_output_tokens=sum(mask),
                    rollout_seconds=0.0,
                    stop_condition="max_turns",
                )
                for ids, mask, scores in zip(sequences, masks, sampled, strict=True)
            ]
            batcher = object.__new__(training.Batcher.__ray_metadata__.modified_class)
            batcher.config = training.Config() if engine.real else training.Config(padding_multiple=1)
            batch = jax.tree.map(
                jnp.asarray,
                batcher.prepare_batch(
                    [training.RolloutGroup(task_id="parity", rollouts=rollouts)],
                    context_length=model.config.max_position_embeddings,
                ),
            )  # (B, T) leaves, except advantages: (B,).
            tokens, attention, loss_mask, old = batch[:4]  # Each (B, T).
            width = tokens.shape[1]
            if engine.real:
                assert width == training.Config().padding_multiple
            expected = np.asarray(old[:, 1:][loss_mask[:, 1:]])  # (sampled tokens,)
            indices = tuple(
                jnp.asarray(index) for index in zip(*candidates, strict=True)
            )  # Three (candidate count,) indices.
            expected_candidates = np.asarray(list(candidates.values()), dtype=np.float32)  # (candidate count,)
            (engine.directory / f"rollout-v{version}-t{temperature}.json").write_text(
                json.dumps(
                    {
                        "token_ids": sequences,
                        "loss_mask": masks,
                        "old_log_probs": sampled,
                        "candidates": [[*key, score] for key, score in candidates.items()],
                    }
                )
            )
            if temperature == 1.0:
                replay = engine.client.completions.create(
                    model=model_id,
                    prompt=sequences,
                    echo=True,
                    max_tokens=1,
                    temperature=1.0,
                    logprobs=0,
                    extra_body={"return_tokens_as_token_ids": True},
                )
                replay_scores = []
                for choice in replay.choices:
                    assert choice.logprobs is not None
                    replay_scores.extend(
                        score
                        for score, mask in zip(
                            choice.logprobs.token_logprobs[: len(sequences[choice.index])],
                            masks[choice.index],
                            strict=True,
                        )
                        if mask
                    )
                reports.append(
                    {
                        "version": version,
                        "temperature": 1.0,
                        "backend": "vllm-replay",
                        **errors(jnp.asarray(replay_scores), expected),
                    }
                )
            for backend, forward in forwards.items():
                backend_state = state.replace(apply_fn=forward)
                actual = jax.jit(training.token_log_probs, static_argnames="temperature")(
                    backend_state, params, batch, temperature
                )  # (B, T-1)
                assert actual.shape == (len(sequences), width - 1)
                selected = actual[loss_mask[:, 1:]]  # (sampled tokens,)
                logits = jax.jit(forward)(
                    {"params": params}, tokens[:, :-1], attention_mask=attention[:, :-1]
                ).logits  # (B, T-1, V)
                distribution = jax.nn.log_softmax(logits / temperature, axis=-1)  # (B, T-1, V)
                _, metrics = training.objective(actual, batch, clip_coef=0.2)
                report = {
                    "version": version,
                    "temperature": temperature,
                    "backend": backend,
                    **errors(selected, expected),
                    "candidates": errors(distribution[indices], expected_candidates),
                    "candidate_probs": errors(jnp.exp(distribution[indices]), np.exp(expected_candidates)),
                    "approx_kl": float(metrics["policy/approx_kl"]),
                    "clip_fraction": float(metrics["policy/clip_fraction"]),
                }
                reports.append(report)
                print(report, flush=True)
                (engine.directory / "parity.json").write_text(json.dumps(reports, indent=2))
    checked = [report for report in reports if report["backend"] != "vllm-replay"]
    # BF16 kernels differ even between vLLM prefill and decode. Bound both individual
    # outliers and aggregate policy drift; near-zero alternative probabilities need absolute probability bounds.
    for report in checked:
        assert report["max_abs"] < (0.35 if engine.real else 0.02), report
        assert report["mean_abs"] < (0.03 if engine.real else 0.005), report
        assert report["candidate_probs"]["max_abs"] < (0.1 if engine.real else 0.002), report
        assert report["candidate_probs"]["mean_abs"] < (0.0025 if engine.real else 0.0001), report
        assert report["approx_kl"] < (0.0025 if engine.real else 0.0001), report
        assert report["clip_fraction"] <= (0.05 if engine.real else 0), report
        if report["backend"] == "tokamax-triton":
            assert report["clip_fraction"] == 0, report
        if not engine.real:
            assert report["candidates"]["max_abs"] < 0.025, report
