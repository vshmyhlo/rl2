"""Compile GRPO v2 gradient accumulation and optimizer update on an available GPU without loading weights."""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from rl2.configuration import load_config
from rl2.qwen3 import Qwen3Config, Qwen3LM
from rl2.train_grpo_v2 import Config, GRPOBatch, accumulate_gradients, create_state, update


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/grpo_v2/codegym.yaml"))
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--length", type=int, required=True, help="Padded transcript width, including the first token")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--donate-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Donate running gradients during accumulation and optimizer state during updates",
    )
    args = parser.parse_args()
    config = load_config(args.config, Config)
    model_config = Qwen3Config.from_dict(json.loads(args.model_config.read_text()))
    if not 2 <= args.length <= model_config.max_position_embeddings:
        parser.error("length must be between 2 and the model context length")
    device = jax.devices("gpu")[0]  # Fail rather than silently compiling for CPU.
    mesh = Mesh(devices=[device], axis_names=("batch",))
    replicated = NamedSharding(mesh, PartitionSpec())
    batched = NamedSharding(mesh, PartitionSpec("batch"))
    model = Qwen3LM(
        config=model_config,
        dtype="bfloat16",
        param_dtype=jnp.bfloat16,
        attention_backend=config.attention_backend,
        gradient_checkpointing=config.gradient_checkpointing,
        scan_layers=config.scan_layers,
    )
    params = {
        name: jax.ShapeDtypeStruct(shape, jnp.bfloat16) for name, shape in model_config.parameter_shapes().items()
    }  # HF parameter shapes; abstract inputs never allocate model weights.
    state = jax.eval_shape(lambda weights: create_state(model, weights, config), params)
    state = jax.tree.map(lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=replicated), state)
    batch_size = config.num_groups * config.group_size
    shape = (batch_size, args.length)
    batch = GRPOBatch(
        token_ids=jax.ShapeDtypeStruct(shape, jnp.int32, sharding=batched),  # (B, T)
        attention_mask=jax.ShapeDtypeStruct(shape, jnp.bool_, sharding=batched),  # (B, T)
        loss_mask=jax.ShapeDtypeStruct(shape, jnp.bool_, sharding=batched),  # (B, T)
        old_log_probs=jax.ShapeDtypeStruct(shape, jnp.float32, sharding=batched),  # (B, T)
        advantages=jax.ShapeDtypeStruct((batch_size,), jnp.float32, sharding=batched),  # (B,)
    )
    gradient_sum = jax.tree.map(
        lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=replicated), state.params
    )  # Trainable parameter shapes.
    accumulation = accumulate_gradients
    optimizer_update = update
    if not args.donate_state:
        accumulation = jax.jit(
            accumulate_gradients.__wrapped__,
            static_argnames=("temperature", "clip_coef", "rollout_is_threshold"),
            in_shardings=(PartitionSpec(), PartitionSpec("batch"), PartitionSpec("batch"), PartitionSpec()),
            out_shardings=PartitionSpec(),
        )
        optimizer_update = jax.jit(
            update.__wrapped__,
            static_argnames=("grad_accum_steps", "has_signal"),
            in_shardings=(PartitionSpec(), PartitionSpec()),
            out_shardings=PartitionSpec(),
        )
    with jax.set_mesh(mesh):
        computations = {
            "accumulate": accumulation.lower(
                state,
                batch,
                jax.ShapeDtypeStruct((batch_size, args.length - 1), jnp.float32, sharding=batched),  # (B, T-1) anchors.
                gradient_sum,
                config.temperature,
                config.clip_coef,
                config.rollout_is_threshold,
            ),
            "update": optimizer_update.lower(state, gradient_sum, config.grad_accum_steps, True),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, lowered in computations.items():
        started = perf_counter()
        (args.output / f"{name}.stablehlo.mlir").write_text(str(lowered.compiler_ir()))
        compiled = lowered.compile()
        hlo = compiled.as_text()
        (args.output / f"{name}.optimized.hlo.txt").write_text(hlo)
        operations = Counter(re.findall(r"^\s*(?:ROOT )?%[^\n=]+ = [^\n]*? ([a-z][a-z0-9-]+)\(", hlo, re.MULTILINE))
        collectives = {
            op: count
            for op, count in operations.items()
            if any(term in op for term in ("all-reduce", "all-gather", "reduce-scatter", "all-to-all", "collective"))
        }
        memory = compiled.memory_analysis()
        report = {
            "jax": jax.__version__,
            "device": device.device_kind,
            "model": config.model,
            "batch_size": batch_size,
            "grad_accum_steps": config.grad_accum_steps,
            "length": args.length,
            "gradient_checkpointing": config.gradient_checkpointing,
            "donate_buffers": args.donate_state,
            "executed": False,
            "compile_seconds": perf_counter() - started,
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "jax_optimization_level": os.environ.get("JAX_OPTIMIZATION_LEVEL", ""),
            "collectives": collectives,
            "operations": dict(operations),
            "custom_call_targets": dict(Counter(re.findall(r'custom_call_target="([^"]+)"', hlo))),
            "memory_bytes": {
                name: getattr(memory, name)
                for name in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "temp_size_in_bytes",
                    "alias_size_in_bytes",
                )
            },
        }
        (args.output / f"{name}.summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if collectives or re.search(r"host.callback|host_callback|xla_python|pure_callback|io_callback", hlo):
            raise RuntimeError("Unexpected communication or host callback in single-GPU update")


if __name__ == "__main__":
    main()
