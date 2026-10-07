# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Portable, differentiable Gated Delta Rule-2 recurrence.

Based on NVlabs/GatedDeltaNet-2 at a5552fe3c67e0ebc7ef1220df68ae8896ec62d56.
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Distributed under the NVIDIA Source Code License-NC; see LICENSE in this directory.
"""

import jax
import jax.numpy as jnp

from rl2.shape_checker import ShapeChecker

type DeltaInputs = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


def delta_rule_step(
    state: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_decay: jax.Array,
    erase: jax.Array,
    write: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return (state, output), accumulating in float32.

    Inputs are float32: state [B,H,K,V], q/k/log_decay/erase [B,H,K],
    v/write [B,H,V]. q and k must already be normalized as desired, and q
    already scaled (the mixer uses K**-0.5). log_decay is a natural logarithm.
    Gates are supplied after activation: erase in [0,1] (or [0,2] for negative
    eigenvalues), write in [0,1], log_decay <= 0. The primitive also accepts
    unconstrained gates for differentiation and mathematical experiments.
    """
    sc = ShapeChecker()
    sc.check(state, "BHKV", jnp.float32)
    sc.check([q, k, log_decay, erase], "BHK", jnp.float32)
    sc.check([v, write], "BHV", jnp.float32)
    decayed = jnp.exp(log_decay)[..., None] * state
    removed = jnp.einsum("bhk,bhkv->bhv", erase * k, decayed)
    state = decayed + k[..., None] * (write * v - removed)[..., None, :]
    output = jnp.einsum("bhk,bhkv->bhv", q, state)
    sc.check(state, "BHKV", jnp.float32)
    sc.check(output, "BHV", jnp.float32)
    return state, output


def gated_delta_rule(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_decay: jax.Array,
    erase: jax.Array,
    write: jax.Array,
    initial_state: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Scan time-major [T,B,H,K/V] inputs; return (final state, [T,B,H,V]).

    Normalization and scaling are the caller's responsibility, as in
    :func:`delta_rule_step`. Empty inputs preserve the initial state.
    """
    sc = ShapeChecker()
    sc.check([q, k, log_decay, erase], "TBHK", jnp.float32)
    sc.check([v, write], "TBHV", jnp.float32)
    state = jnp.zeros(sc["BHKV"], jnp.float32) if initial_state is None else initial_state
    sc.check(state, "BHKV", jnp.float32)

    def step(carry: jax.Array, inputs: DeltaInputs) -> tuple[jax.Array, jax.Array]:
        return delta_rule_step(carry, *inputs)

    state, output = jax.lax.scan(step, state, (q, k, v, log_decay, erase, write))
    sc.check(state, "BHKV", jnp.float32)
    sc.check(output, "TBHV", jnp.float32)
    return state, output
