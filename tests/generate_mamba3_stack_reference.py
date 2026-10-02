"""Generate fixtures from official Block/GatedMLP and Mamba3 CPU references.

Usage (requires torch, numpy, einops outside the normal test environment):
    python tests/generate_mamba3_stack_reference.py /path/to/unpacked/mamba

Use revision e9594ce1c732d97440f0332fdc43170a2294dbfa. The original upstream
Block and GatedMLP definitions run unchanged, using unfused normalization.
The continuous-input wrapper follows MixerModel.forward, including its final
residual addition and norm, but omits token embeddings. Parameters are perturbed
to exercise nonuniform norm weights, gates, and substantial rotary phases.
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from generate_mamba3_reference import UPSTREAM_REVISION, load_definitions, reference_namespace
from torch import nn


def generate(root: Path, destination: Path) -> None:
    namespace, captured = reference_namespace(root)
    namespace.update(RMSNorm=nn.RMSNorm)
    load_definitions(root, "mamba_ssm/modules/block.py", {"Block"}, namespace)
    load_definitions(root, "mamba_ssm/modules/mlp.py", {"GatedMLP"}, namespace)
    arrays: dict[str, np.ndarray] = {"upstream_revision": np.asarray(UPSTREAM_REVISION)}

    def save(name: str, tensor: torch.Tensor) -> None:
        arrays[name] = tensor.detach().numpy().copy()

    for rank, rms_norm, width in ((1, True, 16), (2, True, 16), (1, False, 0), (4, True, 16), (2, False, 16)):
        torch.manual_seed(703 + rank + width)
        prefix = f"r{rank}_rms{int(rms_norm)}_w{width}"
        norm_cls = partial(nn.RMSNorm if rms_norm else nn.LayerNorm, eps=1e-5)
        mixer_cls = partial(
            namespace["Mamba3"],
            d_state=8,
            expand=2,
            headdim=4,
            ngroups=1,
            is_mimo=rank > 1,
            mimo_rank=rank,
            rope_fraction=0.5,
            is_outproj_norm=False,
            chunk_size=2,
        )
        mlp_cls = partial(namespace["GatedMLP"], hidden_features=width, multiple_of=1) if width else nn.Identity
        layers = nn.ModuleList(
            namespace["Block"](8, mixer_cls, mlp_cls, norm_cls=norm_cls, residual_in_fp32=True) for _ in range(2)
        )
        norm_f = norm_cls(8)
        parameters = [
            (f"layers_{i}.{name}", param) for i, layer in enumerate(layers) for name, param in layer.named_parameters()
        ]
        parameters.extend((f"norm_f.{name}", param) for name, param in norm_f.named_parameters())
        with torch.no_grad():
            for name, param in parameters:
                if name.endswith("dt_bias"):
                    param.copy_(torch.linspace(-1.5, 0.5, 4))
                else:
                    param.add_(0.15 * torch.randn_like(param))
        x = torch.randn(2, 6, 8, requires_grad=True)
        hidden, residual = x, None
        for i, layer in enumerate(layers):
            hidden, residual = layer(hidden, residual)
            for name, tensor in captured.items():
                save(f"{prefix}/carry/{i}/{name}", tensor)
        output = norm_f(hidden + residual)
        probe = torch.randn_like(output)
        (output * probe).sum().backward()
        for name, tensor in (("x", x), ("y", output), ("probe", probe), ("dx", x.grad)):
            save(f"{prefix}/{name}", tensor.transpose(0, 1))
        for name, param in parameters:
            name = name.replace(".mlp.", ".")
            gradient = param.grad
            if name.endswith(("in_proj.weight", "out_proj.weight", "fc1.weight", "fc2.weight")):
                name = name.removesuffix("weight") + "kernel"
                param, gradient = param.T, gradient.T
            elif name.endswith(".weight"):
                name = name.removesuffix("weight") + "scale"
            name = name.replace(".", "/")
            save(f"{prefix}/params/{name}", param)
            save(f"{prefix}/grads/{name}", gradient)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    print(f"Wrote {destination} ({len(arrays)} arrays)")


if __name__ == "__main__":
    generate(Path(sys.argv[1]), Path(__file__).parent / "data" / "mamba3_stack_reference.npz")
