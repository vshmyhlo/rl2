"""Full-weight and LoRA publication from a JAX learner to vLLM."""

import json
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import httpx
import jax
import numpy as np
import torch
from safetensors.numpy import save_file
from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo, NCCLTrainerWeightTransferEngine
from vllm.utils.network_utils import get_open_port

from rl2.qwen3 import LoRAParameters, Parameters
from rl2.tracing import span


@contextmanager
def paused_generation(client: httpx.Client, pause_mode: Literal["wait", "abort", "keep"]) -> Generator[None]:
    with span("learner.publish_weights.pause", {"mode": pause_mode}):
        response = client.post("/pause", params={"mode": pause_mode, "clear_cache": "true"})
        response.raise_for_status()
    yield
    # Failed publications leave generation paused until training cleanup.
    with span("learner.publish_weights.resume"):
        response = client.post("/resume")
        response.raise_for_status()


@contextmanager
def lora_weight_publisher(
    get_params: Callable[[], LoRAParameters],
    *,
    server_url: str,
    model: str,
    adapter_name: str,
    lora_rank: int,
    lora_alpha: float,
    pause_mode: Literal["wait", "abort", "keep"],
) -> Generator[Callable[[], None]]:
    """Export the current adapters and retain their files until vLLM finishes loading."""
    with httpx.Client(base_url=server_url, timeout=600) as client:

        def publish() -> None:
            with span("learner.publish_weights"), TemporaryDirectory(prefix="rl2-lora-") as temporary_directory:
                adapters = jax.tree.map(
                    lambda value: np.asarray(value, dtype=np.float32), jax.device_get(get_params())
                )  # Host (R, I) and (O, R) leaves.
                weights = {
                    f"base_model.model.{name.removesuffix('.weight')}.lora_{factor}.weight": value
                    for name, adapter in adapters.items()
                    for factor, value in (("A", adapter.a), ("B", adapter.b))
                }  # PEFT (R, I) and (O, R) leaves.
                directory = Path(temporary_directory)
                save_file(weights, directory / "adapter_model.safetensors")
                (directory / "adapter_config.json").write_text(
                    json.dumps(
                        {
                            "peft_type": "LORA",
                            "task_type": "CAUSAL_LM",
                            "base_model_name_or_path": model,
                            "r": lora_rank,
                            "lora_alpha": lora_alpha,
                            "target_modules": sorted({name.split(".")[-2] for name in adapters}),
                            "bias": "none",
                            "lora_dropout": 0.0,
                            "inference_mode": True,
                            "use_rslora": False,
                        }
                    )
                )
                with paused_generation(client, pause_mode), span("learner.publish_weights.load_adapter"):
                    response = client.post(
                        "/v1/load_lora_adapter",
                        json={"lora_name": adapter_name, "lora_path": temporary_directory, "load_inplace": True},
                    )
                    response.raise_for_status()

        yield publish


@dataclass
class JAXWeightSource(WeightSource):
    get_params: Callable[[], Parameters]  # Current HF parameter shapes, replicated across learner devices.

    def metadata(self) -> list[ParamMeta]:
        metadata: list[ParamMeta] = []
        for name, value in sorted(self.get_params().items()):  # HF parameter shapes.
            if not value.is_fully_replicated:
                raise ValueError("Weight publication requires replicated learner parameters")
            metadata.append(ParamMeta(name=name, dtype=torch.bfloat16, shape=value.shape))
        return metadata

    def __iter__(self) -> Generator[tuple[str, torch.Tensor]]:  # HF parameter shapes.
        for name, value in sorted(self.get_params().items()):  # HF parameter shapes.
            replica = value.addressable_data(0)  # Full HF shape on the first local device.
            # DLPack orders the JAX producer against the current Torch stream.
            # Cast into Torch-owned storage so NCCL's stream lifetime tracking applies.
            weight = torch.from_dlpack(replica).to(dtype=torch.bfloat16, copy=True)  # Same HF shape.
            yield name, weight


@contextmanager
def full_weight_publisher(
    get_params: Callable[[], Parameters],
    *,
    server_url: str,
    device_index: int,
    pause_mode: Literal["wait", "abort", "keep"],
) -> Generator[Callable[[], None]]:
    """Keep one NCCL group alive across publications; HTTP carries only metadata."""
    with (
        torch.cuda.device(device_index),
        httpx.Client(base_url=server_url, timeout=600) as client,
        ExitStack() as resources,
    ):
        sync_client = HTTPVLLMWeightSyncClient(base_url=server_url, timeout=600)
        engine = NCCLTrainerWeightTransferEngine.trainer_init(
            NCCLTrainerInitInfo(
                master_address="127.0.0.1",
                master_port=get_open_port(),
                world_size=2,  # One learner replica and the single vLLM worker.
                rank=0,
                packed=True,
                packed_buffer_size_bytes=64 * 1024 * 1024,
                packed_num_buffers=2,
            ),
            client=sync_client,
            source=JAXWeightSource(get_params=get_params),
        )
        resources.callback(engine.shutdown)
        resources.callback(sync_client.shutdown_weight_transfer_engine)

        def publish() -> None:
            with (
                span("learner.publish_weights"),
                paused_generation(client, pause_mode),
                span("learner.publish_weights.transfer"),
            ):
                engine.send_weights()

        yield publish
