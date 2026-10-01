"""Regenerate CPU reference fixtures using the official module and test kernels.

Requires torch, numpy, einops in a separate environment, not rl2 dependencies.
Usage: python tests/generate_mamba3_reference.py /path/to/unpacked/mamba
Use upstream revision e9594ce1c732d97440f0332fdc43170a2294dbfa.

Extracts the original AST definitions without modifying their function bodies.
Only CUDA entry points are replaced by the upstream CPU reference functions:
tests/ops/triton/test_mamba3_siso.py:mamba3_siso_step_ref and
tests/ops/tilelang/test_mamba3_mimo.py:mamba3_MIMO_chunk_ref (rotate_pairwise=False).
RMSNorm uses upstream rms_norm_ref. This runs the actual Mamba3 constructor
and forward method. Fixtures include trained-like nonuniform parameters,
outputs, all carry leaves, input gradients, and every parameter gradient.
"""

import ast
import math
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

UPSTREAM_REVISION = "e9594ce1c732d97440f0332fdc43170a2294dbfa"
type Namespace = dict[str, Any]


def load_definitions(root: Path, path: str, names: set[str], namespace: Namespace) -> None:
    source = ast.parse((root / path).read_text())
    definitions = [
        node for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert {node.name for node in definitions} == names
    # Load only reviewed definitions; importing the modules requires CUDA DSLs.
    exec(compile(ast.Module(body=definitions, type_ignores=[]), path, "exec"), namespace)  # noqa: S102


def generate(root: Path, destination: Path) -> None:
    namespace: Namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "Tensor": torch.Tensor,
        "Optional": Optional,
        "Tuple": tuple,
        "rearrange": rearrange,
        "repeat": repeat,
    }
    load_definitions(root, "mamba_ssm/ops/triton/layernorm_gated.py", {"rms_norm_ref"}, namespace)
    load_definitions(root, "tests/ops/triton/test_mamba3_siso.py", {"mamba3_siso_step_ref"}, namespace)
    load_definitions(root, "tests/ops/tilelang/test_mamba3_mimo.py", {"mamba3_MIMO_chunk_ref", "_pad_zeros"}, namespace)

    class ReferenceRMSNorm(nn.Module):
        def __init__(
            self,
            hidden_size: int,
            eps: float,
            group_size: int | None = None,
            norm_before_gate: bool = True,
            **kwargs: Any,
        ) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size, **kwargs))
            self.eps = eps
            self.group_size = group_size
            self.norm_before_gate = norm_before_gate

        def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
            return namespace["rms_norm_ref"](
                x,
                self.weight,
                None,
                z=z,
                eps=self.eps,
                group_size=self.group_size,
                norm_before_gate=self.norm_before_gate,
            )

    captured: Namespace = {}

    def siso(**kwargs: Any) -> torch.Tensor:
        names = ("Q", "K", "V", "ADT", "DT", "Trap", "Q_bias", "K_bias", "Angles", "D", "Z")
        output, states = namespace["mamba3_siso_step_ref"](**{name: kwargs[name] for name in names})
        angle, state, key, value = states
        captured.update(angle=angle, state=state, key=key.unsqueeze(2), value=value)
        return output

    def mimo(**kwargs: Any) -> torch.Tensor:
        dt, adt, chunk = kwargs["DT"], kwargs["ADT"], kwargs["chunk_size"]
        batch, heads, steps = dt.shape
        cumsum = adt.reshape(batch, heads, -1, chunk).cumsum(-1)
        reverse = cumsum[..., -1:] - cumsum
        angles = (kwargs["Angles"].tanh() * math.pi * dt.transpose(1, 2).unsqueeze(-1)).cumsum(1)
        angles = angles.remainder(2 * math.pi)
        output, state, key = namespace["mamba3_MIMO_chunk_ref"](
            kwargs["Q"],
            kwargs["K"],
            kwargs["V"],
            kwargs["Q_bias"],
            kwargs["K_bias"],
            kwargs["MIMO_V"],
            kwargs["MIMO_Out"],
            kwargs["Z"],
            kwargs["MIMO_Z"],
            angles,
            cumsum.reshape(batch, heads, steps),
            reverse.reshape(batch, heads, steps),
            dt,
            kwargs["Trap"],
            kwargs["D"],
            chunk_size=chunk,
            rotary_dim_divisor=kwargs["rotary_dim_divisor"],
            return_final_state=True,
            dtype=torch.float32,
            rotate_pairwise=False,
            fused_norm=kwargs["fuse_pregate_headwise_rms_norm"],
            outproj_norm_weight=kwargs["outproj_norm_weight"],
            outproj_norm_eps=kwargs["outproj_norm_eps"],
        )
        captured.update(
            angle=angles[:, -1], state=state.transpose(-1, -2), key=key.transpose(1, 2), value=kwargs["V"][:, -1]
        )
        return output

    namespace.update(RMSNormGated=ReferenceRMSNorm, mamba3_siso_combined=siso, mamba3_mimo_combined=mimo)
    load_definitions(root, "mamba_ssm/modules/mamba3.py", {"heavy_tail_activation", "Mamba3"}, namespace)
    arrays: dict[str, np.ndarray] = {"upstream_revision": np.asarray(UPSTREAM_REVISION)}

    def save(name: str, tensor: torch.Tensor) -> None:
        arrays[name] = tensor.detach().numpy().copy()

    for rank in (1, 2, 4):
        for fraction in (0.5, 1.0):
            for norm in (False, True):
                torch.manual_seed(2026 + rank + int(fraction * 10) + int(norm))
                groups = 2 if norm else 1
                prefix = f"r{rank}_f{int(fraction * 100)}_n{int(norm)}"
                model = namespace["Mamba3"](
                    d_model=8,
                    d_state=8,
                    expand=2,
                    headdim=4,
                    ngroups=groups,
                    is_mimo=rank > 1,
                    mimo_rank=rank,
                    rope_fraction=fraction,
                    is_outproj_norm=norm,
                    chunk_size=2,
                )
                # Exercise rotation and normalization with nontrivial learned
                # weights; tiny initial dt otherwise hides layout mistakes.
                with torch.no_grad():
                    model.dt_bias.copy_(torch.linspace(-1.5, 0.5, 4))
                    for name, parameter in model.named_parameters():
                        if name != "dt_bias":
                            parameter.add_(0.15 * torch.randn_like(parameter))
                x = torch.randn(2, 6, 8, requires_grad=True)
                output = model(x)
                probe = torch.randn_like(output)
                (output * probe).sum().backward()
                save(f"{prefix}/x", x.transpose(0, 1))
                save(f"{prefix}/y", output.transpose(0, 1))
                save(f"{prefix}/probe", probe.transpose(0, 1))
                save(f"{prefix}/dx", x.grad.transpose(0, 1))
                for name, tensor in captured.items():
                    save(f"{prefix}/carry/{name}", tensor)
                for name, parameter in model.named_parameters():
                    gradient = parameter.grad
                    if name in ("in_proj.weight", "out_proj.weight"):
                        local_name = name.replace(".weight", "/kernel")
                        value, gradient = parameter.T, gradient.T
                    elif name in ("B_norm.weight", "C_norm.weight"):
                        local_name, value = name.replace(".weight", "/scale"), parameter
                    elif name == "norm.weight":
                        local_name, value = "out_norm_scale", parameter.reshape(4, 4)
                        gradient = gradient.reshape(4, 4)
                    else:
                        local_name, value = name, parameter
                    save(f"{prefix}/params/{local_name}", value)
                    save(f"{prefix}/grads/{local_name}", gradient)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    print(f"Wrote {destination} ({len(arrays)} arrays)")


if __name__ == "__main__":
    generate(Path(sys.argv[1]), Path(__file__).parent / "data" / "mamba3_reference.npz")
