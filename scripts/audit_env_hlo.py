"""Compile batched environment transitions and save backend-optimized HLO.

Run from the repository root, for example:
    uv run python scripts/audit_env_hlo.py --backend cpu --output /tmp/env-hlo

Each case runs in a separate process. Inputs are abstract: no reset, rollout, or
training executes. Reports describe compiler estimates, not measured throughput.
Uses PPO's training mesh over all visible devices: state, keys, actions, and
transition outputs are sharded over batch; environment parameters are replicated.
Use --backend gpu on the training host; CPU HLO cannot establish GPU efficiency.
Cases use representative adapter defaults. Use --env config --config PATH to
compile the exact environment and batch size from a PPO configuration instead.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from rl2.actions import ContinuousSpace
from rl2.configuration import load_config
from rl2.envs.base import BaseEnv
from rl2.envs.brax_env import BraxConfig, BraxEnv
from rl2.envs.craftax_env import CraftaxConfig, CraftaxEnv
from rl2.envs.foragax_env import ForagaxConfig, ForagaxEnv
from rl2.envs.gymnax_env import GymnaxConfig, GymnaxEnv
from rl2.envs.jaxnav_env import JaxNavConfig, JaxNavEnv
from rl2.envs.jaxtari_env import JAXtariConfig, JAXtariEnv
from rl2.envs.kinetix_env import KinetixConfig, KinetixEnv
from rl2.envs.mpe_env import MPEConfig, MPEEnv
from rl2.envs.mujoco_playground_env import PlaygroundConfig, PlaygroundEnv
from rl2.envs.multi_agent import BaseMultiAgentEnv
from rl2.envs.pgx_env import PgxConfig, PgxEnv
from rl2.envs.robotarium_env import RobotariumConfig, RobotariumEnv
from rl2.envs.smax_env import SMAXConfig, SMAXEnv
from rl2.envs.storm_env import STORMConfig, STORMEnv
from rl2.envs.xminigrid_env import XMiniGridConfig, XMiniGridEnv
from rl2.jax_cache import configure_compilation_cache
from rl2.ppo import Config, make_env, training_mesh

type Environment = BaseEnv[Any, Any] | BaseMultiAgentEnv[Any]

logger = logging.getLogger(__name__)

CASES: dict[str, Callable[[], Environment]] = {
    "brax": lambda: BraxEnv(BraxConfig()),
    "craftax": lambda: CraftaxEnv(CraftaxConfig()),
    "craftax_classic": lambda: CraftaxEnv(CraftaxConfig(env_id="Craftax-Classic-Symbolic-v1")),
    "foragax": lambda: ForagaxEnv(ForagaxConfig()),
    "gymnax": lambda: GymnaxEnv(GymnaxConfig()),
    "jaxnav": lambda: JaxNavEnv(JaxNavConfig()),
    "jaxtari": lambda: JAXtariEnv(JAXtariConfig()),
    "kinetix": lambda: KinetixEnv(KinetixConfig()),
    "mpe": lambda: MPEEnv(MPEConfig()),
    "playground": lambda: PlaygroundEnv(PlaygroundConfig()),
    "playground_cartpole": lambda: PlaygroundEnv(PlaygroundConfig(task="CartpoleSwingup")),
    "pgx": lambda: PgxEnv(PgxConfig()),
    "robotarium": lambda: RobotariumEnv(RobotariumConfig()),
    "smax": lambda: SMAXEnv(SMAXConfig()),
    "storm": lambda: STORMEnv(STORMConfig()),
    "xminigrid": lambda: XMiniGridEnv(XMiniGridConfig()),
}


def compile_case(name: str, config: Config, backend: str, mode: str, output: Path) -> dict[str, Any]:
    """Compile one adapter at B=config.num_envs, preserving every transition output."""
    configure_compilation_cache()
    devices = jax.devices(backend)
    if jax.default_backend() != backend:
        raise ValueError(f"Set JAX_PLATFORMS={backend} to match PPO's default backend")
    mesh = training_mesh(config)
    batch_size = config.num_envs
    batched = PartitionSpec("batch")
    replicated = PartitionSpec()
    with jax.default_device(devices[0]), jax.set_mesh(mesh):
        env = make_env(config) if name == "config" else CASES[name]()
        keys = jax.ShapeDtypeStruct((batch_size,), jax.random.key(0).dtype)  # (B,) typed keys
        if isinstance(env, BaseEnv):
            _, state = jax.eval_shape(jax.vmap(env.reset, in_axes=(0, None)), keys, env.params)  # (B, ...) leaves
            action_shape = env.action_space.shape if isinstance(env.action_space, ContinuousSpace) else ()
            dtype = jnp.float32 if isinstance(env.action_space, ContinuousSpace) else jnp.int32
            actions = jax.ShapeDtypeStruct((batch_size, *action_shape), dtype)  # (B, *action_shape)
            step = env.step_batch if mode == "batch" else jax.vmap(env.step, in_axes=(0, 0, 0, None))
            arguments = (keys, state, actions, env.params)
            in_shardings = (batched, batched, batched, replicated)
        else:
            _, state = jax.eval_shape(jax.vmap(env.reset), keys)  # state leaves: (B, ...)
            actions = jax.ShapeDtypeStruct((batch_size, env.num_agents), jnp.int32)  # (B, N)
            step = jax.vmap(env.step)
            arguments = (keys, state, actions)
            in_shardings = (batched, batched, batched)
        started = perf_counter()
        lowered = jax.jit(step, in_shardings=in_shardings, out_shardings=batched).lower(*arguments)
        (output / "step.stablehlo.mlir").write_text(str(lowered.compiler_ir()))
        compiled = lowered.compile()
        compile_seconds = perf_counter() - started
        hlo = compiled.as_text()
        (output / "step.optimized.hlo.txt").write_text(hlo)
        operations = Counter(re.findall(r"^\s*(?:ROOT )?%[^\n=]+ = [^\n]*? ([a-z][a-z0-9-]+)\(", hlo, re.MULTILINE))
        memory = compiled.memory_analysis()
        return {
            "case": name,
            "adapter": type(env).__name__,
            **({"environment": asdict(config.env)} if name == "config" else {}),
            "status": "compiled",
            "jax": jax.__version__,
            "backend": devices[0].platform,
            "devices": [device.device_kind for device in devices],
            "device_count": len(devices),
            "batch_size": batch_size,
            "batch_size_per_device": batch_size // len(devices),
            "num_minibatches": config.num_minibatches,
            "mesh": str(mesh),
            "input_shardings": str(compiled.input_shardings),
            "output_shardings": str(compiled.output_shardings),
            "mode": mode,
            "executed": False,
            "compile_seconds": compile_seconds,
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "operations": dict(operations),
            "collectives": {
                operation: count
                for operation, count in operations.items()
                if any(
                    term in operation
                    for term in ("all-reduce", "all-gather", "reduce-scatter", "all-to-all", "collective")
                )
            },
            "custom_call_targets": dict(Counter(re.findall(r'custom_call_target="([^"]+)"', hlo))),
            "host_callbacks": bool(re.search(r"host.callback|host_callback|xla_python|pure_callback|io_callback", hlo)),
            "cost_analysis": compiled.cost_analysis(),
            "memory_bytes": {
                field: getattr(memory, field)
                for field in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "temp_size_in_bytes",
                    "alias_size_in_bytes",
                )
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=[*CASES, "config"])
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/ppo.yaml")
    parser.add_argument("--batch-size", type=int, help="Override the config's global environment count")
    parser.add_argument("--num-minibatches", type=int, help="Override PPO's mesh divisibility constraint")
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--mode", choices=("batch", "vmap"), default="batch")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900, help="Maximum seconds per isolated case")
    args = parser.parse_args()
    if any(value is not None and value < 1 for value in (args.batch_size, args.num_minibatches, args.timeout)):
        parser.error("batch-size, num-minibatches, and timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.env is not None:
        for filename in ("summary.json", "step.optimized.hlo.txt", "step.stablehlo.mlir"):
            (args.output / filename).unlink(missing_ok=True)
        try:
            config = load_config(args.config, Config)
            config = replace(
                config,
                num_envs=args.batch_size if args.batch_size is not None else config.num_envs,
                num_minibatches=args.num_minibatches if args.num_minibatches is not None else config.num_minibatches,
            )
            report = compile_case(args.env, config, args.backend, args.mode, args.output)
        except Exception as error:
            logger.exception("Compilation failed for %s", args.env)
            report = {"case": args.env, "status": "failed", "error": f"{type(error).__name__}: {error}"}
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        sys.exit(0 if report["status"] == "compiled" else 1)
    reports = []
    for name in CASES:
        directory = args.output / name
        directory.mkdir(exist_ok=True)
        command = [
            sys.executable,
            __file__,
            "--env",
            name,
            "--backend",
            args.backend,
            "--config",
            str(args.config),
            "--mode",
            args.mode,
            "--output",
            str(directory),
        ]
        if args.batch_size is not None:
            command.extend(("--batch-size", str(args.batch_size)))
        if args.num_minibatches is not None:
            command.extend(("--num-minibatches", str(args.num_minibatches)))
        summary = directory / "summary.json"
        summary.unlink(missing_ok=True)
        with (directory / "compile.log").open("w") as log:
            try:
                process = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=False
                )
                error = f"Process exited with status {process.returncode}; see compile.log"
            except subprocess.TimeoutExpired:
                error = f"Compilation exceeded {args.timeout} seconds; see compile.log"
        report = (
            json.loads(summary.read_text()) if summary.exists() else {"case": name, "status": "failed", "error": error}
        )
        reports.append(report)
        (args.output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(f"{name}: {report['status']}", flush=True)
    sys.exit(0 if all(report["status"] == "compiled" for report in reports) else 1)


if __name__ == "__main__":
    main()
