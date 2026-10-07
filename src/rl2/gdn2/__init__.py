"""JAX/Flax Gated DeltaNet-2; see README.md for interfaces and upstream provenance."""

from rl2.gdn2.core import delta_rule_step, gated_delta_rule
from rl2.gdn2.model import (
    GatedDeltaNet2,
    GatedDeltaNet2Carry,
    GatedDeltaNet2Config,
    GatedDeltaNet2LM,
    GatedDeltaNet2Stack,
    GatedDeltaNet2StackCarry,
)

__all__ = [
    "GatedDeltaNet2",
    "GatedDeltaNet2Carry",
    "GatedDeltaNet2Config",
    "GatedDeltaNet2LM",
    "GatedDeltaNet2Stack",
    "GatedDeltaNet2StackCarry",
    "delta_rule_step",
    "gated_delta_rule",
]
