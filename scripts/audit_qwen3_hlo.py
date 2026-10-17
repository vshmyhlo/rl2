"""Compile data-parallel Qwen3 GPU HLO and summarize collectives and memory."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import topologies
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxlib import xla_client

from rl2.qwen3 import Qwen3Config, Qwen3LM, Variables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-backend", choices=("xla", "tokamax", "cudnn"), default="xla")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--logits-to-keep", type=int, default=0)
    parser.add_argument("--sharding", choices=("manual", "auto"), default="manual")
    args = parser.parse_args()
    if args.devices < 1 or args.batch < 1 or args.batch % args.devices:
        parser.error("batch must be a positive multiple of devices")
    devices = jax.devices("gpu")
    compile_only = len(devices) < args.devices
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
    lowered = jax.jit(
        forward,
        in_shardings=(replicated, batched, batched),
        out_shardings=output_sharding,
    ).lower(variables, tokens, mask)
    compiled = lowered.compile()
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dtype}-{args.attention_backend}-{args.sharding}-dp{args.devices}-t{args.length}-keep{args.logits_to_keep}"
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
        "device": devices[0].device_kind,
        "compile_only": compile_only,
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
    (args.output / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if collectives:
        raise RuntimeError(f"Unexpected communication in replicated-weight DP forward: {collectives}")
    if re.search(r"host.callback|host_callback|xla_python|pure_callback|io_callback", hlo):
        raise RuntimeError("Unexpected host callback in model HLO")


if __name__ == "__main__":
    main()
