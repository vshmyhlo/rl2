"""Barebones reinforcement learning experiments."""

import os
import sys

# Crazyflow uses SciPy rotations on JAX arrays; enable before SciPy is imported.
if "scipy" not in sys.modules:
    os.environ.setdefault("SCIPY_ARRAY_API", "1")
