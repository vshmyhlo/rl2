"""Fused causal depthwise convolution/SiLU and gated RMSNorm for JAX.

These kernels use this repository's float32 state and parameter conventions.
"""

import triton
import triton.language as tl


@triton.jit
def conv_forward(
    x: tl.tensor,
    weight: tl.tensor,
    bias: tl.tensor,
    y: tl.tensor,
    z: tl.tensor,
    T: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    BD: tl.constexpr,
) -> None:
    row, block = tl.program_id(0), tl.program_id(1)
    batch, time = row // T, row % T
    d = block * BD + tl.arange(0, BD)
    accum = tl.load(bias + d, d < D, 0)
    for tap in tl.static_range(C):
        value = tl.load(x + (batch * (T + C - 1) + time + tap) * D + d, d < D, 0)
        w = tl.load(weight + tap * D + d, d < D, 0)
        accum += value * w
    tl.store(z + row * D + d, accum, d < D)
    tl.store(y + row * D + d, accum * tl.sigmoid(accum), d < D)


@triton.jit
def conv_backward_input(
    dz: tl.tensor,
    weight: tl.tensor,
    dx: tl.tensor,
    T: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    BD: tl.constexpr,
) -> None:
    row, block = tl.program_id(0), tl.program_id(1)
    batch, time = row // (T + C - 1), row % (T + C - 1)
    d = block * BD + tl.arange(0, BD)
    accum = tl.full((BD,), 0, tl.float32)
    for tap in tl.static_range(C):
        t = time - tap
        grad = tl.load(dz + (batch * T + t) * D + d, (d < D) & (t >= 0) & (t < T), 0)
        w = tl.load(weight + tap * D + d, d < D, 0)
        accum += grad * w
    tl.store(dx + row * D + d, accum, d < D)


@triton.jit
def conv_backward_weight(
    x: tl.tensor,
    dz: tl.tensor,
    dw: tl.tensor,
    db: tl.tensor,
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    BD: tl.constexpr,
    BR: tl.constexpr,
) -> None:
    tap, block = tl.program_id(0), tl.program_id(1)
    d = block * BD + tl.arange(0, BD)
    rows = tl.arange(0, BR)
    accum = tl.full((BR, BD), 0, tl.float32)
    bias = tl.full((BR, BD), 0, tl.float32)
    for start in range(tl.cdiv(B * T, BR)):
        r = start * BR + rows
        batch, time = r // T, r % T
        valid = (r[:, None] < B * T) & (d[None, :] < D)
        value = tl.load(x + (batch[:, None] * (T + C - 1) + time[:, None] + tap) * D + d[None, :], valid, 0)
        grad = tl.load(dz + r[:, None] * D + d[None, :], valid, 0)
        accum += value * grad
        if tap == 0:
            bias += grad
    tl.store(dw + tap * D + d, tl.sum(accum, axis=0), d < D)
    if tap == 0:
        tl.store(db + d, tl.sum(bias, axis=0), d < D)


@triton.jit
def norm_forward(
    x: tl.tensor,
    gate: tl.tensor,
    weight: tl.tensor,
    y: tl.tensor,
    V: tl.constexpr,
    BV: tl.constexpr,
    EPS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    v = tl.arange(0, BV)
    value = tl.load(x + row * V + v, v < V, 0)
    g = tl.load(gate + row * V + v, v < V, 0)
    w = tl.load(weight + v, v < V, 0)
    inv = tl.rsqrt(tl.sum(value * value, axis=0) / V + EPS)
    yv = value * inv * w * g * tl.sigmoid(g)
    tl.store(y + row * V + v, yv, v < V)


@triton.jit
def norm_backward(
    x: tl.tensor,
    gate: tl.tensor,
    weight: tl.tensor,
    dy: tl.tensor,
    dx: tl.tensor,
    dg: tl.tensor,
    dw: tl.tensor,
    V: tl.constexpr,
    BV: tl.constexpr,
    EPS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    v = tl.arange(0, BV)
    value = tl.load(x + row * V + v, v < V, 0)
    g = tl.load(gate + row * V + v, v < V, 0)
    w = tl.load(weight + v, v < V, 0)
    grad = tl.load(dy + row * V + v, v < V, 0)
    inv = tl.rsqrt(tl.sum(value * value, axis=0) / V + EPS)
    sig = tl.sigmoid(g)
    silu = g * sig
    direct = grad * w * silu
    dxv = inv * direct - value * inv * inv * inv * tl.sum(direct * value, axis=0) / V
    dgv = grad * value * inv * w * sig * (1 + g * (1 - sig))
    dwv = grad * value * inv * silu
    tl.store(dx + row * V + v, dxv, v < V)
    tl.store(dg + row * V + v, dgv, v < V)
    tl.store(dw + row * V + v, dwv, v < V)
