# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


"""Adapted Triton kernels; see README.md for provenance and changes."""

import triton
import triton.language as tl

exp2 = tl.exp2


@triton.jit
def unflatten_program_id(X: tl.constexpr) -> tuple[tl.tensor, tl.tensor]:
    pid = tl.program_id(0).to(tl.int64)
    return pid % X, pid // X


@triton.jit
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k: tl.tensor,
    v: tl.tensor,
    w: tl.tensor,
    gk: tl.tensor,
    h0: tl.tensor,
    h: tl.tensor,
    v_new: tl.tensor,
    ht: tl.tensor,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
) -> None:
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v, i_nh = (pid % NV, (pid // NV).to(tl.int64))
    i_n, i_h = (i_nh // HV, i_nh % HV)
    bos = i_n * T
    NT = tl.cdiv(T, BT)
    boh = i_n * NT
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)
    h += (boh * HV + i_h).to(tl.int64) * K * V
    v += (bos * HV + i_h).to(tl.int64) * V
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    v_new += (bos * HV + i_h).to(tl.int64) * V
    h0 = h0 + i_nh * K * V
    ht = ht + i_nh * K * V
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K
    p_h0_1 = h0 + o_k1[:, None] * V + o_v[None, :]
    m_h0_1 = m_k1[:, None] & m_v[None, :]
    b_h1 += tl.load(p_h0_1, mask=m_h0_1, other=0.0).to(tl.float32)
    if K > 64:
        p_h0_2 = h0 + o_k2[:, None] * V + o_v[None, :]
        m_h0_2 = m_k2[:, None] & m_v[None, :]
        b_h2 += tl.load(p_h0_2, mask=m_h0_2, other=0.0).to(tl.float32)
    if K > 128:
        p_h0_3 = h0 + o_k3[:, None] * V + o_v[None, :]
        m_h0_3 = m_k3[:, None] & m_v[None, :]
        b_h3 += tl.load(p_h0_3, mask=m_h0_3, other=0.0).to(tl.float32)
    if K > 192:
        p_h0_4 = h0 + o_k4[:, None] * V + o_v[None, :]
        m_h0_4 = m_k4[:, None] & m_v[None, :]
        b_h4 += tl.load(p_h0_4, mask=m_h0_4, other=0.0).to(tl.float32)
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t_int64 * BT + tl.arange(0, BT)
        m_t = o_t < T
        p_h1 = h + i_t_int64 * HV * K * V + o_k1[:, None] * V + o_v[None, :]
        m_h1 = m_k1[:, None] & m_v[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
        if K > 64:
            p_h2 = h + i_t_int64 * HV * K * V + o_k2[:, None] * V + o_v[None, :]
            m_h2 = m_k2[:, None] & m_v[None, :]
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_h2)
        if K > 128:
            p_h3 = h + i_t_int64 * HV * K * V + o_k3[:, None] * V + o_v[None, :]
            m_h3 = m_k3[:, None] & m_v[None, :]
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_h3)
        if K > 192:
            p_h4 = h + i_t_int64 * HV * K * V + o_k4[:, None] * V + o_v[None, :]
            m_h4 = m_k4[:, None] & m_v[None, :]
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_h4)
        p_w = w + o_t[:, None] * (HV * K) + o_k1[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype), input_precision="tf32x3")
        if K > 64:
            p_w = w + o_t[:, None] * (HV * K) + o_k2[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            b_v = tl.dot(b_w, b_h2.to(b_w.dtype), b_v, input_precision="tf32x3")
        if K > 128:
            p_w = w + o_t[:, None] * (HV * K) + o_k3[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            b_v = tl.dot(b_w, b_h3.to(b_w.dtype), b_v, input_precision="tf32x3")
        if K > 192:
            p_w = w + o_t[:, None] * (HV * K) + o_k4[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            b_v = tl.dot(b_w, b_h4.to(b_w.dtype), b_v, input_precision="tf32x3")
        p_v = v + o_t[:, None] * (HV * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0) - b_v
        p_v = v_new + o_t[:, None] * (HV * V) + o_v[None, :]
        tl.store(p_v, b_v.to(p_v.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])
        last_idx = min((i_t + 1) * BT, T) - 1
        o_k1 = tl.arange(0, 64)
        b_gk_last1 = tl.load(gk + (bos + last_idx) * HV * K + i_h * K + o_k1, mask=o_k1 < K, other=0.0).to(tl.float32)
        b_h1 *= exp2(b_gk_last1)[:, None]
        if K > 64:
            o_k2 = 64 + o_k1
            b_gk_last2 = tl.load(gk + (bos + last_idx) * HV * K + i_h * K + o_k2, mask=o_k2 < K, other=0.0).to(
                tl.float32
            )
            b_h2 *= exp2(b_gk_last2)[:, None]
        if K > 128:
            o_k3 = 128 + o_k1
            b_gk_last3 = tl.load(gk + (bos + last_idx) * HV * K + i_h * K + o_k3, mask=o_k3 < K, other=0.0).to(
                tl.float32
            )
            b_h3 *= exp2(b_gk_last3)[:, None]
        if K > 192:
            o_k4 = 192 + o_k1
            b_gk_last4 = tl.load(gk + (bos + last_idx) * HV * K + i_h * K + o_k4, mask=o_k4 < K, other=0.0).to(
                tl.float32
            )
            b_h4 *= exp2(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)
        p_k = k + o_k1[:, None] + o_t[None, :] * (H * K)
        b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_h1 = tl.dot(b_k, b_v, b_h1, input_precision="tf32x3")
        if K > 64:
            p_k = k + o_k2[:, None] + o_t[None, :] * (H * K)
            b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            b_h2 = tl.dot(b_k, b_v, b_h2, input_precision="tf32x3")
        if K > 128:
            p_k = k + o_k3[:, None] + o_t[None, :] * (H * K)
            b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            b_h3 = tl.dot(b_k, b_v, b_h3, input_precision="tf32x3")
        if K > 192:
            p_k = k + o_k4[:, None] + o_t[None, :] * (H * K)
            b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            b_h4 = tl.dot(b_k, b_v, b_h4, input_precision="tf32x3")
    p_ht = ht + o_k1[:, None] * V + o_v[None, :]
    m_ht = m_k1[:, None] & m_v[None, :]
    tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), mask=m_ht)
    if K > 64:
        p_ht = ht + o_k2[:, None] * V + o_v[None, :]
        m_ht = m_k2[:, None] & m_v[None, :]
        tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), mask=m_ht)
    if K > 128:
        p_ht = ht + o_k3[:, None] * V + o_v[None, :]
        m_ht = m_k3[:, None] & m_v[None, :]
        tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), mask=m_ht)
    if K > 192:
        p_ht = ht + o_k4[:, None] * V + o_v[None, :]
        m_ht = m_k4[:, None] & m_v[None, :]
        tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), mask=m_ht)


@triton.jit
def chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64(
    q: tl.tensor,
    k: tl.tensor,
    w: tl.tensor,
    gk: tl.tensor,
    dht: tl.tensor,
    do: tl.tensor,
    dv: tl.tensor,
    dh: tl.tensor,
    dh0: tl.tensor,
    dv2: tl.tensor,
    scale: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
) -> None:
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v, i_nh = (pid % NV, (pid // NV).to(tl.int64))
    i_n, i_h = (i_nh // HV, i_nh % HV)
    bos = i_n * T
    NT = tl.cdiv(T, BT)
    boh = i_n * NT
    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_dh4 = tl.zeros([64, BV], dtype=tl.float32)
    q += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    do += (bos * HV + i_h).to(tl.int64) * V
    dv += (bos * HV + i_h).to(tl.int64) * V
    dv2 += (bos * HV + i_h).to(tl.int64) * V
    dh += (boh * HV + i_h).to(tl.int64) * K * V
    gk += (bos * HV + i_h).to(tl.int64) * K
    dh0 += i_nh * K * V
    dht += i_nh * K * V
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K
    p_dht1 = dht + o_k1[:, None] * V + o_v[None, :]
    m_dht1 = m_k1[:, None] & m_v[None, :]
    b_dh1 += tl.load(p_dht1, mask=m_dht1, other=0.0)
    if K > 64:
        p_dht2 = dht + o_k2[:, None] * V + o_v[None, :]
        m_dht2 = m_k2[:, None] & m_v[None, :]
        b_dh2 += tl.load(p_dht2, mask=m_dht2, other=0.0)
    if K > 128:
        p_dht3 = dht + o_k3[:, None] * V + o_v[None, :]
        m_dht3 = m_k3[:, None] & m_v[None, :]
        b_dh3 += tl.load(p_dht3, mask=m_dht3, other=0.0)
    if K > 192:
        p_dht4 = dht + o_k4[:, None] * V + o_v[None, :]
        m_dht4 = m_k4[:, None] & m_v[None, :]
        b_dh4 += tl.load(p_dht4, mask=m_dht4, other=0.0)
    for i_t in range(NT - 1, -1, -1):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t_int64 * BT + tl.arange(0, BT)
        m_t = o_t < T
        p_dh1 = dh + i_t_int64 * HV * K * V + o_k1[:, None] * V + o_v[None, :]
        m_dh1 = m_k1[:, None] & m_v[None, :]
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), mask=m_dh1)
        if K > 64:
            p_dh2 = dh + i_t_int64 * HV * K * V + o_k2[:, None] * V + o_v[None, :]
            m_dh2 = m_k2[:, None] & m_v[None, :]
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), mask=m_dh2)
        if K > 128:
            p_dh3 = dh + i_t_int64 * HV * K * V + o_k3[:, None] * V + o_v[None, :]
            m_dh3 = m_k3[:, None] & m_v[None, :]
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), mask=m_dh3)
        if K > 192:
            p_dh4 = dh + i_t_int64 * HV * K * V + o_k4[:, None] * V + o_v[None, :]
            m_dh4 = m_k4[:, None] & m_v[None, :]
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), mask=m_dh4)
        last_idx = min((i_t_int64 + 1) * BT, T) - 1
        p_dv = dv + o_t[:, None] * (HV * V) + o_v[None, :]
        p_dv2 = dv2 + o_t[:, None] * (HV * V) + o_v[None, :]
        p_do = do + o_t[:, None] * (HV * V) + o_v[None, :]
        b_do = tl.load(p_do, mask=m_t[:, None] & m_v[None, :], other=0.0)
        p_k = k + o_t[:, None] * (H * K) + o_k1[None, :]
        b_k = tl.load(p_k, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        o_k1 = tl.arange(0, 64)
        b_gk_last1 = tl.load(gk + last_idx * HV * K + o_k1, mask=o_k1 < K, other=0.0).to(tl.float32)
        b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype), input_precision="tf32x3")
        if K > 64:
            p_k = k + o_t[:, None] * (H * K) + o_k2[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            b_gk_last2 = tl.load(gk + last_idx * HV * K + o_k2, mask=o_k2 < K, other=0.0).to(tl.float32)
            b_dv = tl.dot(b_k, b_dh2.to(b_k.dtype), b_dv, input_precision="tf32x3")
        if K > 128:
            p_k = k + o_t[:, None] * (H * K) + o_k3[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            b_gk_last3 = tl.load(gk + last_idx * HV * K + o_k3, mask=o_k3 < K, other=0.0).to(tl.float32)
            b_dv = tl.dot(b_k, b_dh3.to(b_k.dtype), b_dv, input_precision="tf32x3")
        if K > 192:
            p_k = k + o_t[:, None] * (H * K) + o_k4[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            b_gk_last4 = tl.load(gk + last_idx * HV * K + o_k4, mask=o_k4 < K, other=0.0).to(tl.float32)
            b_dv = tl.dot(b_k, b_dh4.to(b_k.dtype), b_dv, input_precision="tf32x3")
        b_dv += tl.load(p_dv, mask=m_t[:, None] & m_v[None, :], other=0.0)
        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])
        p_w = w + o_k1[:, None] + o_t[None, :] * (HV * K)
        p_q = q + o_k1[:, None] + o_t[None, :] * (H * K)
        b_w = tl.load(p_w, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_q = tl.load(p_q, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_dh1 *= exp2(b_gk_last1[:, None])
        b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), input_precision="tf32x3") * scale - tl.dot(
            b_w, b_dv.to(b_w.dtype), input_precision="tf32x3"
        )
        if K > 64:
            p_q = q + o_k2[:, None] + o_t[None, :] * (H * K)
            p_w = w + o_k2[:, None] + o_t[None, :] * (HV * K)
            b_q = tl.load(p_q, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            b_w = tl.load(p_w, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            b_dh2 *= exp2(b_gk_last2[:, None])
            b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), input_precision="tf32x3") * scale - tl.dot(
                b_w, b_dv.to(b_w.dtype), input_precision="tf32x3"
            )
        if K > 128:
            p_q = q + o_k3[:, None] + o_t[None, :] * (H * K)
            p_w = w + o_k3[:, None] + o_t[None, :] * (HV * K)
            b_q = tl.load(p_q, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            b_w = tl.load(p_w, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            b_dh3 *= exp2(b_gk_last3[:, None])
            b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), input_precision="tf32x3") * scale - tl.dot(
                b_w, b_dv.to(b_w.dtype), input_precision="tf32x3"
            )
        if K > 192:
            p_q = q + o_k4[:, None] + o_t[None, :] * (H * K)
            p_w = w + o_k4[:, None] + o_t[None, :] * (HV * K)
            b_q = tl.load(p_q, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            b_w = tl.load(p_w, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            b_dh4 *= exp2(b_gk_last4[:, None])
            b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), input_precision="tf32x3") * scale - tl.dot(
                b_w, b_dv.to(b_w.dtype), input_precision="tf32x3"
            )
    p_dh0 = dh0 + o_k1[:, None] * V + o_v[None, :]
    m_dh0 = m_k1[:, None] & m_v[None, :]
    tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), mask=m_dh0)
    if K > 64:
        p_dh1 = dh0 + o_k2[:, None] * V + o_v[None, :]
        m_dh1 = m_k2[:, None] & m_v[None, :]
        tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), mask=m_dh1)
    if K > 128:
        p_dh2 = dh0 + o_k3[:, None] * V + o_v[None, :]
        m_dh2 = m_k3[:, None] & m_v[None, :]
        tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), mask=m_dh2)
    if K > 192:
        p_dh3 = dh0 + o_k4[:, None] * V + o_v[None, :]
        m_dh3 = m_k4[:, None] & m_v[None, :]
        tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), mask=m_dh3)


@triton.jit
def chunk_gla_fwd_kernel_o(
    q: tl.tensor,
    v: tl.tensor,
    g: tl.tensor,
    h: tl.tensor,
    A: tl.tensor,
    o: tl.tensor,
    scale: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
) -> None:
    i_v, i_t = unflatten_program_id(tl.cdiv(V, BV))
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hv = (i_bh // HV, i_bh % HV)
    i_h = i_hv // (HV // H)
    NT = tl.cdiv(T, BT)
    i_tg = (i_b * NT + i_t).to(tl.int64)
    bos = (i_b * T).to(tl.int64)
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    q += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    v += (bos * HV + i_hv) * V
    o += (bos * HV + i_hv) * V
    h += (i_tg * HV + i_hv).to(tl.int64) * K * V
    A += (bos * HV + i_hv) * BT
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    o_t = i_t * BT + tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)
    o_i = tl.arange(0, BT)
    m_t = o_t < T
    m_v = o_v < V
    m_tv = m_t[:, None] & m_v[None, :]
    m_A = m_t[:, None] & (o_i[None, :] < BT)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_qk = m_t[:, None] & m_k[None, :]
        p_q = q + o_t[:, None] * (H * K) + o_k[None, :]
        p_g = g + o_t[:, None] * (HV * K) + o_k[None, :]
        p_h = h + o_k[:, None] * V + o_v[None, :]
        m_h = m_k[:, None] & m_v[None, :]
        b_q = tl.load(p_q, mask=m_qk, other=0.0)
        b_g = tl.load(p_g, mask=m_qk, other=0.0).to(tl.float32)
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        b_h = tl.load(p_h, mask=m_h, other=0.0)
        if i_k >= 0:
            b_o = tl.dot(b_qg, b_h.to(b_qg.dtype), b_o, input_precision="tf32x3")
    b_o *= scale
    p_v = v + o_t[:, None] * (HV * V) + o_v[None, :]
    p_o = o + o_t[:, None] * (HV * V) + o_v[None, :]
    p_A = A + o_t[:, None] * (HV * BT) + o_i[None, :]
    b_v = tl.load(p_v, mask=m_tv, other=0.0)
    b_A = tl.load(p_A, mask=m_A, other=0.0)
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o = tl.dot(b_A, b_v, b_o, input_precision="tf32x3")
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_tv)


@triton.jit
def chunk_kda_bwd_kernel_dAv(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    A: tl.tensor,
    do: tl.tensor,
    dv: tl.tensor,
    dA: tl.tensor,
    scale: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
) -> None:
    i_t, i_bh = (tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64))
    i_b, i_hv = (i_bh // HV, i_bh % HV)
    i_h = i_hv // (HV // H)
    bos = i_b * T
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * HV + i_hv) * V
    do += (bos * HV + i_hv) * V
    dv += (bos * HV + i_hv) * V
    dA += (bos * HV + i_hv) * BT
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_A = tl.arange(0, BT)
    m_AT = (o_A[:, None] < BT) & m_t[None, :]
    p_A = A + (bos * HV + i_hv) * BT + o_A[:, None] + o_t[None, :] * (HV * BT)
    b_A = tl.load(p_A, mask=m_AT, other=0.0)
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0).to(do.dtype.element_ty)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_vT = m_v[:, None] & m_t[None, :]
        m_tv = m_t[:, None] & m_v[None, :]
        p_v = v + o_v[:, None] + o_t[None, :] * (HV * V)
        p_do = do + o_t[:, None] * (HV * V) + o_v[None, :]
        p_dv = dv + o_t[:, None] * (HV * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_vT, other=0.0)
        b_do = tl.load(p_do, mask=m_tv, other=0.0)
        b_dA = tl.dot(b_do, b_v, b_dA, input_precision="tf32x3")
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do, input_precision="tf32x3")
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_tv)
    m_dA = m_t[:, None] & (o_A[None, :] < BT)
    p_dA = dA + o_t[:, None] * (HV * BT) + o_A[None, :]
    b_dA = tl.where(o_t[:, None] >= o_t, b_dA * scale, 0.0)
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), mask=m_dA)
