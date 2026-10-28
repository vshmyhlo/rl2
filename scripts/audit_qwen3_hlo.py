"""Compile data-parallel Qwen3 GPU HLO and summarize collectives and memory."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import topologies
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxlib import xla_client
from tokenizers import Tokenizer

from rl2.qwen3 import ATTENTION_BACKENDS, Qwen3Config, Qwen3LM, Variables, load_qwen3_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-backend", choices=ATTENTION_BACKENDS, default="xla")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--logits-to-keep", type=int, default=0)
    parser.add_argument("--sharding", choices=("manual", "auto"), default="manual")
    parser.add_argument("--mode", choices=("forward", "backward", "train"), default="forward")
    parser.add_argument("--platform", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--execute", action="store_true", help="Load real weights and benchmark synchronized execution")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if args.devices < 1 or args.batch < 1 or args.batch % args.devices:
        parser.error("batch must be a positive multiple of devices")
    if args.length < 1 or args.warmup < 1 or args.runs < 1:
        parser.error("length, warmup, and runs must be positive")
    if args.mode != "forward" and args.length < 2:
        parser.error("backward/train require at least two tokens")
    if args.mode != "forward" and args.logits_to_keep:
        parser.error("backward/train require all logits for next-token loss")
    devices = jax.devices(args.platform)
    compile_only = len(devices) < args.devices
    if compile_only and (args.platform == "cpu" or args.execute):
        parser.error("Requested execution requires enough physical devices")
    if compile_only:
        target = xla_client.get_topology_for_devices(devices).target_config
        devices = topologies.get_topology_desc(
            platform="cuda",
            target_config=target,
            topology=f"1x1x{args.devices}",
        ).devices
    mesh = Mesh(np.array(devices[: args.devices]), ("data",))
    replicated = NamedSharding(mesh, P())
    batched = NamedSharding(mesh, P("data", None))
    output_sharding = NamedSharding(mesh, P("data", None, None))
    config = Qwen3Config.from_dict(json.loads((args.checkpoint / "config.json").read_text()))
    model = Qwen3LM(config=config, dtype=args.dtype, attention_backend=args.attention_backend)
    variables = {
        "params": {
            name: jax.ShapeDtypeStruct(shape, jnp.dtype(args.dtype), sharding=replicated)
            for name, shape in config.parameter_shapes().items()
        }
    }  # Configured parameter shapes; weights remain runtime arguments.
    tokens = jax.ShapeDtypeStruct((args.batch, args.length), jnp.int32, sharding=batched)  # (B, T)
    mask = jax.ShapeDtypeStruct((args.batch, args.length), jnp.bool_, sharding=batched)  # (B, T)

    def forward(weights: Variables, ids: jax.Array, valid: jax.Array) -> jax.Array:  # (B, T), (B, T)
        return model.apply(weights, ids, attention_mask=valid, logits_to_keep=args.logits_to_keep).logits  # (B, U, V)

    if args.sharding == "manual":
        forward = jax.shard_map(
            forward,
            mesh=mesh,
            in_specs=(P(), P("data", None), P("data", None)),
            out_specs=P("data", None, None),
            check_vma=False,  # Tokamax 0.0.15 Pallas outputs lack manual-axis annotations.
        )

    def loss(weights: Variables, ids: jax.Array, valid: jax.Array) -> jax.Array:  # (B, T), (B, T)
        logits = forward(weights, ids, valid)[:, :-1]  # (B, T-1, V)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, ids[:, 1:])  # (B, T-1)
        selected = valid[:, :-1] & valid[:, 1:]  # (B, T-1)
        return jnp.sum(losses * selected) / jnp.maximum(selected.sum(), 1)  # ()

    def train(weights: Variables, ids: jax.Array, valid: jax.Array) -> tuple[jax.Array, Variables]:  # (B, T), (B, T)
        value, gradients = jax.value_and_grad(loss)(weights, ids, valid)  # (), parameter shapes.
        updated = jax.tree.map(lambda weight, grad: weight - 1e-3 * grad, weights, gradients)
        return value, updated  # (), parameter shapes; one functional SGD step.

    computation = {"forward": forward, "backward": jax.value_and_grad(loss), "train": train}[args.mode]
    compile_start = perf_counter()
    lowered = jax.jit(
        computation,
        in_shardings=(replicated, batched, batched),
        out_shardings=output_sharding if args.mode == "forward" else (replicated, replicated),
    ).lower(variables, tokens, mask)
    compiled = lowered.compile()
    compile_seconds = perf_counter() - compile_start
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.platform}-{args.mode}-{args.dtype}-{args.attention_backend}-{args.sharding}-dp{args.devices}-t{args.length}-keep{args.logits_to_keep}"
    (args.output / f"{stem}.stablehlo.mlir").write_text(str(lowered.compiler_ir()))
    hlo = compiled.as_text()
    (args.output / f"{stem}.hlo.txt").write_text(hlo)
    operations = Counter(re.findall(r"^\s*(?:ROOT )?%[^\n=]+ = [^\n]*? ([a-z][a-z0-9-]+)\(", hlo, re.MULTILINE))
    collectives = {
        op: count
        for op, count in operations.items()
        if any(
            term in op for term in ("all-reduce", "all-gather", "reduce-scatter", "all-to-all", "collective-permute")
        )
    }
    memory = compiled.memory_analysis()
    report = {
        "jax": jax.__version__,
        "platform": args.platform,
        "mode": args.mode,
        "executed": False,
        "device": devices[0].device_kind,
        "compile_only": compile_only,
        "compile_seconds": compile_seconds,
        "devices": args.devices,
        "batch": args.batch,
        "length": args.length,
        "dtype": args.dtype,
        "backend": args.attention_backend,
        "logits_to_keep": args.logits_to_keep,
        "sharding": args.sharding,
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
    if args.execute:
        _, weights = load_qwen3_checkpoint(
            str(args.checkpoint), dtype=args.dtype, attention_backend=args.attention_backend
        )
        tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
        prompt = "Explain reinforcement learning in one sentence."
        encoded = tokenizer.encode(prompt, add_special_tokens=False).ids
        ids = jnp.tile(jnp.asarray(encoded, jnp.int32), (args.batch, (args.length + len(encoded) - 1) // len(encoded)))[
            :, : args.length
        ]  # (B, T), repeated real text for the requested benchmark length.
        weights = jax.device_put(weights, replicated)
        ids = jax.device_put(ids, batched)  # (B, T)
        valid = jax.device_put(jnp.ones(ids.shape, jnp.bool_), batched)  # (B, T)
        for _ in range(args.warmup):
            jax.block_until_ready(compiled(weights, ids, valid))
        durations: list[float] = []
        for _ in range(args.runs):
            start = perf_counter()
            jax.block_until_ready(compiled(weights, ids, valid))
            durations.append(perf_counter() - start)
        median = float(np.median(durations))
        report.update(
            executed=True,
            median_seconds=median,
            tokens_per_second=args.batch * args.length / median,
            samples_seconds=durations,
            prompt=prompt,
            warmup=args.warmup,
            device_memory_stats=[device.memory_stats() for device in jax.devices(args.platform)[: args.devices]],
        )
    (args.output / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if collectives and args.mode == "forward":
        raise RuntimeError(f"Unexpected communication in replicated-weight DP forward: {collectives}")
    if re.search(r"host.callback|host_callback|xla_python|pure_callback|io_callback", hlo):
        raise RuntimeError("Unexpected host callback in model HLO")


if __name__ == "__main__":
    main()
