"""Compile MAPPO's exact configuration without executing training.

Example for an allocated pair of GPUs:
    JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python \
        scripts/audit_mappo_hlo.py --output /tmp/mappo-hlo

Uses all visible devices, like training. CPU devices can check partitioning but
cannot establish GPU efficiency. Memory estimates and static collective sites
are compiler diagnostics, not runtime peak memory or communication counts.
"""

import argparse
import hashlib
import inspect
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
from flax.training.train_state import TrainState
from jax.sharding import NamedSharding, PartitionSpec

from rl2 import train_mappo as ppo
from rl2.configuration import load_config

type Step = Literal["reset", "rollout", "train", "all"]


def compile_steps(config: ppo.Config, output: Path, step: Step = "all") -> None:
    """Export reset, rollout, and donated training executables at configured shapes."""
    env = ppo.make_env(config)
    mesh = ppo.training_mesh(config, env.num_agents)
    replicated = NamedSharding(mesh, PartitionSpec())
    batched = NamedSharding(mesh, PartitionSpec("batch"))
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path(ppo.__file__).parent
    source_paths = [Path(ppo.__file__), Path(inspect.getfile(type(env)))]
    source_paths.extend(source_root / name for name in ("actions.py", "lstm.py", "utils.py"))
    sources = output / "sources"
    sources.mkdir(exist_ok=True)
    source_sha256: dict[str, str] = {}
    for path in source_paths:
        content = path.read_bytes()
        (sources / path.name).write_bytes(content)
        source_sha256[path.name] = hashlib.sha256(content).hexdigest()

    def shape(leaf: jax.ShapeDtypeStruct, sharding: NamedSharding) -> jax.ShapeDtypeStruct:  # (...) -> (...)
        return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding, weak_type=leaf.weak_type)

    with jax.set_mesh(mesh):
        key = jax.ShapeDtypeStruct((), jax.random.key(0).dtype, sharding=replicated)  # () typed PRNG key
        reset_keys = jax.ShapeDtypeStruct((config.num_envs,), key.dtype, sharding=batched)  # (E,)
        reset = jax.jit(jax.vmap(env.reset), in_shardings=batched, out_shardings=batched)
        observation, env_state = jax.tree.map(
            lambda leaf: shape(leaf, batched), jax.eval_shape(reset, reset_keys)
        )  # observation: (E, N, ...)/(E, N, *C, A); state: (E, ...)
        model = ppo.ActorCritic(
            action_space=env.action_space,
            model=config.model,
            dtype=jnp.bfloat16 if config.bf16 else jnp.float32,
        )
        actor_count = config.num_envs * env.num_agents

        def create_state(key: jax.Array) -> TrainState:  # () typed PRNG key
            return TrainState.create(
                apply_fn=model.apply,
                params=model.init(
                    key,
                    jax.tree.map(
                        lambda leaf: jnp.zeros((1, 1, *leaf.shape[2:]), leaf.dtype), observation.observation
                    ),  # observation leaves: (B=1, T=1, ...)
                    model.initial_carry(1),  # leaves: (B=1, H)
                    jnp.ones((1, 1), bool),  # (B=1, T=1)
                    jnp.ones((1, 1, *observation.actions_mask.shape[2:]), bool),  # (B=1, T=1, *C, A)
                )["params"],
                tx=optax.inject_hyperparams(
                    lambda learning_rate: optax.chain(
                        optax.clip_by_global_norm(config.max_grad_norm), optax.adam(learning_rate, eps=1e-5)
                    )
                )(config.learning_rate),
            )

        state = jax.tree.map(lambda leaf: shape(leaf, replicated), jax.eval_shape(create_state, key))
        carry = ppo.RolloutCarry(
            env_state=env_state,  # leaves: (E, ...)
            observation=observation,  # (E, N, ...)/(E, N, *C, A)
            model_carry=jax.tree.map(
                lambda leaf: shape(leaf, batched), jax.eval_shape(lambda: model.initial_carry(actor_count))
            ),  # leaves: (B, H)
            episode_start=jax.ShapeDtypeStruct((actor_count,), jnp.bool_, sharding=batched),  # (B,)
            episode_return=jax.ShapeDtypeStruct((config.num_envs,), jnp.float32, sharding=batched),  # (E,)
            episode_step_count=jax.ShapeDtypeStruct((config.num_envs,), jnp.int32, sharding=batched),  # (E,)
        )
        # Actual scalar leaves retain 64-bit counters with global x64 disabled, as in training.
        metrics = jax.device_put(ppo.init_metrics(), replicated)  # scalar leaves
        schedule_step = jax.ShapeDtypeStruct((), jnp.float32, sharding=replicated)  # ()
        rollout = jax.jit(
            ppo.collect_rollout,
            static_argnames=("env", "config"),
            in_shardings=(batched, replicated, replicated),
            out_shardings=(
                batched,
                ppo.RolloutResult(
                    batch=batched,
                    key=replicated,
                    episode_return_sum=replicated,
                    episode_length_sum=replicated,
                    episode_count=replicated,
                    terminated_episode_count=replicated,
                ),
            ),
        )
        for name in ("reset", "rollout", "train") if step == "all" else (step,):
            started = perf_counter()
            print(f"Lowering {name} on {mesh.size} {jax.default_backend()} devices", flush=True)
            if name == "reset":
                lowered = reset.lower(reset_keys)
            elif name == "rollout":
                lowered = rollout.lower(carry, state, env, config, key)
            else:
                lowered = ppo.train_step_donated.lower(carry, state, env, config, key, metrics, schedule_step)
            (output / f"{name}.stablehlo.mlir").write_text(str(lowered.compiler_ir()))
            compiled = lowered.compile()
            hlo = compiled.as_text()
            (output / f"{name}.optimized.hlo.txt").write_text(hlo)
            operations = Counter(re.findall(r"^\s*(?:ROOT )?%[^\n=]+ = [^\n]*? ([a-z][a-z0-9-]+)\(", hlo, re.MULTILINE))
            memory = compiled.memory_analysis()
            report: dict[str, Any] = {
                "step": name,
                "config": asdict(config),
                "jax": jax.__version__,
                "source_sha256": source_sha256,
                "backend": jax.default_backend(),
                "devices": [device.device_kind for device in mesh.devices.flat],
                "mesh": str(mesh),
                "executed": False,
                "compile_seconds": perf_counter() - started,
                "xla_flags": os.environ.get("XLA_FLAGS", ""),
                "operations": dict(operations),
                "collective_sites": [
                    line.strip()
                    for line in hlo.splitlines()
                    if re.search(
                        r"= .* (?:all-reduce|all-gather|reduce-scatter|all-to-all|collective-permute)(?:-start|-done)?\(",
                        line,
                    )
                ],
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
    parser.add_argument("--config", type=Path, default=Path("configs/mappo/lux.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", choices=("reset", "rollout", "train", "all"), default="all")
    args = parser.parse_args()
    compile_steps(load_config(args.config, ppo.Config), args.output, args.step)


if __name__ == "__main__":
    main()
