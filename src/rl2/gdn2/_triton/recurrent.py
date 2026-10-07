# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Distributed under the NVIDIA Source Code License-NC; see ../LICENSE.
"""Dense-state adaptation of NVlabs' fused_recurrent_gdn2_fwd_kernel.

Pinned source and adaptation details are in README.md.
"""

import triton
import triton.language as tl


@triton.jit
def recurrent_kernel(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    g: tl.tensor,
    b: tl.tensor,
    w: tl.tensor,
    h0: tl.tensor,
    ht: tl.tensor,
    out: tl.tensor,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
) -> None:
    iv, ibh = tl.program_id(0), tl.program_id(1)
    ib, ih = ibh // H, ibh % H
    ik = tl.arange(0, BK)
    jv = iv * BV + tl.arange(0, BV)
    offset = ibh * K * V + ik[:, None] * V + jv[None, :]
    valid = (ik[:, None] < K) & (jv[None, :] < V)
    state = tl.load(h0 + offset, valid, 0).to(tl.float32)
    for t in range(T):
        pk = (ib * T * H + t * H + ih) * K + ik
        pv = (ib * T * H + t * H + ih) * V + jv
        qt = tl.load(q + pk, ik < K, 0).to(tl.float32)
        kt = tl.load(k + pk, ik < K, 0).to(tl.float32)
        gt = tl.load(g + pk, ik < K, 0).to(tl.float32)
        bt = tl.load(b + pk, ik < K, 0).to(tl.float32)
        vt = tl.load(v + pv, jv < V, 0).to(tl.float32)
        wt = tl.load(w + pv, jv < V, 0).to(tl.float32)
        state *= tl.exp(gt)[:, None]
        removed = tl.sum(state * (bt * kt)[:, None], axis=0)
        state += kt[:, None] * (wt * vt - removed)[None, :]
        y = tl.sum(state * qt[:, None], axis=0)
        tl.store(out + pv, y, jv < V)
    tl.store(ht + offset, state, valid)
