"""Compile AlphaZero collection and updates without executing training.

Example on the training host:
    JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python \
        scripts/audit_alphazero_hlo.py --output /tmp/alphazero-hlo

Uses the configuration's exact shapes and training mesh over all visible devices.
Set CUDA_VISIBLE_DEVICES before starting to inspect a smaller GPU allocation.
Artifacts include optimized HLO, compiler memory estimates, and collective sites;
these are not measurements of runtime throughput or peak device memory.
"""

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import jax
import jax.numpy as jnp
import optax
import pgx
from flax.training.train_state import TrainState
from jax.sharding import NamedSharding, PartitionSpec

from rl2.alphazero import train as az
from rl2.configuration import load_config


def compile_steps(config: az.Config, output: Path, step: Literal["collect", "update", "both"] = "both") -> None:
    """Save optimized executables for (B, T, ...) rollouts and replicated updates."""
    env = pgx.make(config.env_id)
    mesh = az.training_mesh(config)
    replicated = NamedSharding(mesh, PartitionSpec())
    batched = NamedSharding(mesh, PartitionSpec("batch"))
    key = jax.ShapeDtypeStruct((2,), jnp.uint32, sharding=replicated)  # (2,)
    observation = jax.eval_shape(env.init, key).observation  # (*O,)
    model = az.PolicyValueNet(
        num_actions=env.num_actions, channels=config.model.channels, num_blocks=config.model.num_blocks
    )
    params = jax.eval_shape(model.init, key, jax.ShapeDtypeStruct((1, *observation.shape), jnp.float32))[
        "params"
    ]  # Parameter shapes.

    def create_state(params: az.Parameters) -> TrainState:
        return TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
            ),
        )

    def replicate(leaf: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:  # (...) -> (...)
        return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=replicated)

    state = jax.tree.map(replicate, jax.eval_shape(create_state, params))
    shape = (config.num_envs, config.max_moves)
    batch = az.Batch(
        observation=jax.ShapeDtypeStruct((*shape, *observation.shape), jnp.float32, sharding=batched),  # (B, T, *O)
        action_weights=jax.ShapeDtypeStruct((*shape, env.num_actions), jnp.float32, sharding=batched),  # (B, T, A)
        value_targets=jax.ShapeDtypeStruct(shape, jnp.float32, sharding=batched),  # (B, T)
        policy_mask=jax.ShapeDtypeStruct(shape, jnp.bool_, sharding=batched),  # (B, T)
        value_mask=jax.ShapeDtypeStruct(shape, jnp.bool_, sharding=batched),  # (B, T)
    )
    output.mkdir(parents=True, exist_ok=True)
    with jax.set_mesh(mesh):
        for name in ("collect", "update") if step == "both" else (step,):
            started = perf_counter()
            print(f"Lowering {name} on {mesh.size} {jax.default_backend()} devices", flush=True)
            if name == "collect":
                lowered = az.collect_selfplay.lower(state.params, key, env=env, apply_fn=model.apply, config=config)
            else:
                lowered = az.train_step.lower(
                    state, key, batch, batch_size=config.batch_size, updates_per_iteration=config.updates_per_iteration
                )
            (output / f"{name}.stablehlo.mlir").write_text(str(lowered.compiler_ir()))
            compiled = lowered.compile()
            hlo = compiled.as_text()
            (output / f"{name}.optimized.hlo.txt").write_text(hlo)
            operations = Counter(re.findall(r"^\s*(?:ROOT )?%[^\n=]+ = [^\n]*? ([a-z][a-z0-9-]+)\(", hlo, re.MULTILINE))
            collective_sites = [
                line.strip()
                for line in hlo.splitlines()
                if re.search(
                    r"= .* (?:all-reduce|all-gather|reduce-scatter|all-to-all|collective-permute)(?:-start|-done)?\(",
                    line,
                )
            ]
            memory = compiled.memory_analysis()
            report: dict[str, Any] = {
                "step": name,
                "config": asdict(config),
                "jax": jax.__version__,
                "train_source_sha256": hashlib.sha256(Path(az.__file__).read_bytes()).hexdigest(),
                "backend": jax.default_backend(),
                "devices": [device.device_kind for device in mesh.devices.flat],
                "mesh": str(mesh),
                "executed": False,
                "compile_seconds": perf_counter() - started,
                "xla_flags": os.environ.get("XLA_FLAGS", ""),
                "operations": dict(operations),
                "collective_sites": collective_sites,
                "custom_call_targets": dict(Counter(re.findall(r'custom_call_target="([^"]+)"', hlo))),
                "host_callbacks": bool(
                    re.search(r"host.callback|host_callback|xla_python|pure_callback|io_callback", hlo)
                ),
                "memory_bytes": {
                    field: getattr(memory, field)
                    for field in (
                        "argument_size_in_bytes",
                        "output_size_in_bytes",
                        "temp_size_in_bytes",
                        "alias_size_in_bytes",
                    )
                },
                "input_shardings": str(compiled.input_shardings),
                "output_shardings": str(compiled.output_shardings),
            }
            (output / f"{name}.summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps({key: report[key] for key in ("step", "compile_seconds", "memory_bytes", "operations")}),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/alphazero/chess.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", choices=("collect", "update", "both"), default="both")
    args = parser.parse_args()
    compile_steps(load_config(args.config, az.Config), args.output, args.step)


if __name__ == "__main__":
    main()
