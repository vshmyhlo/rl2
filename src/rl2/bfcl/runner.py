"""Run the pinned BFCL protocol against a local vLLM server."""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
REGISTRY_NAME = f"{MODEL}-FC"
CATEGORIES = "single_turn,multi_turn"
TEMPERATURE = 0.001
CONTEXT_LENGTH = 40960
SERVER_ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
SMOKE_IDS = {
    "simple_python": ["simple_python_0"],
    "multiple": ["multiple_0"],
    "irrelevance": ["irrelevance_0"],
    "multi_turn_base": ["multi_turn_base_0"],
}


def server_command(checkpoint: Path, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(checkpoint),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(CONTEXT_LENGTH),
        "--gpu-memory-utilization",
        "0.9",
        "--max-num-seqs",
        "32",
        "--enable-prefix-caching",
        "--generation-config",
        "auto",
        "--seed",
        "0",
    ]


def evaluation_command(action: str, checkpoint: Path, smoke: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "rl2.bfcl.qwen",
        action,
        "--model",
        REGISTRY_NAME,
        "--test-category",
        ",".join(SMOKE_IDS) if smoke else CATEGORIES,
    ]
    if action == "generate":
        command += [
            "--backend",
            "vllm",
            "--skip-server-setup",
            "--local-model-path",
            str(checkpoint),
            "--temperature",
            str(TEMPERATURE),
            "--num-threads",
            "32",
            "--include-input-log",
        ]
        if smoke:
            command += ["--run-ids"]
    elif smoke:
        command += ["--partial-eval"]
    return command


def protocol(checkpoint: Path, port: int, smoke: bool) -> dict:
    return {
        "model": MODEL,
        "revision": REVISION,
        "bfcl_registry_name": REGISTRY_NAME,
        "categories": list(SMOKE_IDS) if smoke else CATEGORIES.split(","),
        "smoke_ids": SMOKE_IDS if smoke else None,
        "temperature": TEMPERATURE,
        "effective_temperature": 0.01,  # vLLM's floor for nonzero temperatures.
        "context_length": CONTEXT_LENGTH,
        "thinking": True,
        "max_tokens_per_response": 4096,
        "num_threads": 32,
        "server_command": server_command(checkpoint, port),
        "server_env": SERVER_ENV,
        "versions": {name: version(name) for name in ("bfcl-eval", "vllm", "transformers", "torch", "numpy")},
        "sha256": {
            str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                ROOT / "uv.lock",
                HERE / "runner.py",
                HERE / "qwen.py",
                checkpoint / "config.json",
                checkpoint / "tokenizer_config.json",
                checkpoint / "generation_config.json",
            )
        },
    }


def write_manifest(run_dir: Path, manifest: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Run configuration changed; choose a fresh --run-dir.")
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")


def check_server(checkpoint: Path, port: int) -> None:
    with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
        models = json.load(response)["data"]
    if not any(model["id"] == str(checkpoint) and model["max_model_len"] == CONTEXT_LENGTH for model in models):
        raise ValueError(
            f"Server must serve the pinned checkpoint with a {CONTEXT_LENGTH}-token context; use rl2-bfcl serve."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "serve", "generate", "evaluate"))
    parser.add_argument("--run-dir", type=Path, help="Default: runs/bfcl/qwen3-0.6b[-smoke] at repository root")
    parser.add_argument("--port", type=int, default=1053)
    parser.add_argument("--smoke", action="store_true", help="Four fixed cases; never a benchmark score")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    checkpoint = Path(
        snapshot_download(
            repo_id=MODEL,
            revision=REVISION,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "merges.txt", "vocab.json"],
            local_files_only=args.action != "download",
        )
    ).resolve()
    if args.action == "download":
        print(checkpoint)
        return

    if args.action == "serve":
        command = server_command(checkpoint, args.port)
        print(shlex.join(command), flush=True)
        os.execve(command[0], command, {**os.environ, **SERVER_ENV})
        return

    run_dir = args.run_dir or ROOT / "runs" / "bfcl" / ("qwen3-0.6b-smoke" if args.smoke else "qwen3-0.6b")
    run_dir = run_dir.resolve()
    manifest = protocol(checkpoint, args.port, args.smoke)
    if args.action == "generate":
        check_server(checkpoint, args.port)
    if args.action == "evaluate" and not (run_dir / "protocol.json").exists():
        parser.error("Generate results in this run directory before evaluating.")
    write_manifest(run_dir, manifest)
    if args.smoke:
        (run_dir / "test_case_ids_to_generate.json").write_text(json.dumps(SMOKE_IDS, indent=2) + "\n")
    command = evaluation_command(args.action, checkpoint, args.smoke)
    print(shlex.join(command), flush=True)
    subprocess.run(
        command,
        check=True,
        env={
            **os.environ,
            "BFCL_PROJECT_ROOT": str(run_dir),
            "LOCAL_SERVER_ENDPOINT": "127.0.0.1",
            "LOCAL_SERVER_PORT": str(args.port),
        },
    )


if __name__ == "__main__":
    main()
