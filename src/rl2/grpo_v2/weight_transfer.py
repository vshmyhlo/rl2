"""Full-weight and LoRA publication from a JAX learner to vLLM."""

import json
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import Literal

import httpx
import jax
import numpy as np
import torch
from safetensors.numpy import save_file
from vllm.distributed.weight_transfer import nccl_common
from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerInitInfo,
    NCCLTrainerWeightTransferEngine,
    NCCLWeightTransferInitInfo,
)
from vllm.utils.network_utils import get_open_port

from rl2.qwen3 import LoRAParameters, Parameters
from rl2.tracing import span


@contextmanager
def paused_generation(client: httpx.Client, pause_mode: Literal["wait", "keep"]) -> Generator[None]:
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
    pause_mode: Literal["wait", "keep"],
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
    pause_mode: Literal["wait", "keep"],
) -> Generator[Callable[[], None]]:
    """Keep one NCCL group alive across publications; HTTP carries only metadata."""
    with (
        torch.cuda.device(device_index),
        httpx.Client(base_url=server_url, timeout=600) as client,
        ExitStack() as resources,
    ):
        sync_client = HTTPVLLMWeightSyncClient(base_url=server_url, timeout=600)
        info = NCCLTrainerInitInfo(
            master_address="127.0.0.1",
            master_port=get_open_port(),
            world_size=2,  # One learner replica and the single vLLM worker.
            rank=0,
            packed=True,
            packed_buffer_size_bytes=64 * 1024 * 1024,
            packed_num_buffers=2,
        )
        engine = NCCLTrainerWeightTransferEngine(
            client=sync_client,
            source=JAXWeightSource(get_params=get_params),
            packed=info.packed,
            packed_buffer_size_bytes=info.packed_buffer_size_bytes,
            packed_num_buffers=info.packed_num_buffers,
        )

        def shutdown_worker(
            _exc_type: type[BaseException] | None, error: BaseException | None, _traceback: TracebackType | None
        ) -> None:
            try:
                sync_client.shutdown_weight_transfer_engine()
            except Exception as shutdown_error:
                if error is None:
                    raise
                error.add_note(f"Worker weight-transfer shutdown failed: {shutdown_error!r}")

        resources.push(shutdown_worker)
        # Abort locally first so a worker blocked receiving weights can service shutdown.
        resources.callback(engine.shutdown)

        # Own the engine before rendezvous so partial initialization is cleaned up.
        worker_info = NCCLWeightTransferInitInfo(
            master_address=info.master_address,
            master_port=info.master_port,
            rank_offset=1,
            world_size=info.world_size,
            packed=info.packed,
            packed_buffer_size_bytes=info.packed_buffer_size_bytes,
            packed_num_buffers=info.packed_num_buffers,
        )
        with ExitStack() as pending:
            executor = ThreadPoolExecutor(max_workers=1)
            # A failed local rendezvous must reach cleanup before waiting on the worker.
            pending.callback(executor.shutdown, wait=False)
            future = executor.submit(
                sync_client.init_weight_transfer_engine, nccl_common.worker_init_payload(worker_info)
            )
            engine.model_update_group = nccl_common.trainer_init(info)
            future.result()

        def publish() -> None:
            with (
                span("learner.publish_weights"),
                paused_generation(client, pause_mode),
                span("learner.publish_weights.transfer"),
            ):
                engine.send_weights()

        yield publish
