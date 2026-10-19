"""Opt-in CUDA interoperability check: RL2_CUDA_TESTS=1 uv run pytest <this-file>."""

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RL2_CUDA_TESTS") != "1", reason="Requires CUDA toolkit and GPU")


def test_jax_torch_transformers_and_vllm_sampling(tmp_path: Path) -> None:
    # A fresh process/cache checks shared-library loading and the sampler's JIT compiler.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            dedent("""
                import jax
                import jax.numpy as jnp
                import torch
                from transformers import Qwen3Config, Qwen3ForCausalLM
                from vllm.v1.sample.ops.topk_topp_sampler import flashinfer_sample

                assert jax.default_backend() == "gpu"
                assert torch.version.cuda == "13.0"
                inputs = jnp.ones((1, 1, 3, 3), dtype=jnp.float32)  # (N, C, H, W)
                kernel = jnp.ones((1, 1, 2, 2), dtype=jnp.float32)  # (O, I, H, W)
                @jax.jit
                def convolve(
                    x: jax.Array,  # (N, C, H, W)
                    w: jax.Array,  # (O, I, KH, KW)
                ) -> jax.Array:
                    return jax.lax.conv_general_dilated(x, w, (1, 1), "VALID")  # (N, O, H-KH+1, W-KW+1)

                output = convolve(inputs, kernel)  # (1, 1, 2, 2)
                assert output.shape == (1, 1, 2, 2)
                assert output.tolist() == [[[[4.0, 4.0], [4.0, 4.0]]]]
                torch_output = torch.nn.functional.conv2d(
                    torch.ones((1, 1, 3, 3), device="cuda"),  # (N, C, H, W)
                    torch.ones((1, 1, 2, 2), device="cuda"),  # (O, I, H, W)
                )  # (1, 1, 2, 2)
                assert torch_output.tolist() == output.tolist()
                config = Qwen3Config(
                    vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                )
                model = Qwen3ForCausalLM(config).cuda()
                tokens = torch.tensor([[1, 2, 3]], device="cuda")  # (B=1, T=3)
                prediction = model(tokens, labels=tokens)
                assert prediction.logits.shape == (1, 3, 32)  # (B, T, V)
                prediction.loss.backward()
                assert model.lm_head.weight.grad is not None
                logits = torch.zeros((1, 8), device="cuda")  # (B=1, V=8)
                top_k = torch.tensor([4], device="cuda", dtype=torch.int32)  # (B,)
                top_p = torch.tensor([0.9], device="cuda")  # (B,)
                sample = flashinfer_sample(logits, top_k, top_p)  # (B,)
                assert sample.shape == (1,)
                assert 0 <= sample.item() < 8
                print("cuda-stack-ok")
            """),
        ],
        env={
            **os.environ,
            "JAX_PLATFORMS": "cuda",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "FLASHINFER_WORKSPACE_BASE": str(tmp_path),
            "MAX_JOBS": "2",
        },
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cuda-stack-ok" in result.stdout
