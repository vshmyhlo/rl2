"""Autoregressive AST edit model."""

from typing import TypeAlias

import jax
from flax import linen as nn

from rl2.karel_ast import KarelAST as AST

Feedback: TypeAlias = jax.Array


class ARASTEditModel(nn.Module):
    """Autoregressive AST edit model."""

    def step(self, program: AST, feedback: Feedback, active: jax.Array) -> None:
        """Run one autoregressive AST edit step."""
        pass
