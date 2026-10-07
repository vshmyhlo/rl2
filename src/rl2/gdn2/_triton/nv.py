# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Adapted Triton kernels; see README.md for provenance and changes."""

import triton
import triton.language as tl

exp2 = tl.exp2
SOLVE_TRIL_DOT_PRECISION = tl.constexpr("tf32x3")


@triton.jit
def chunk_gdn2_fwd_kernel_intra_token_parallel(
    q: tl.tensor,
    k: tl.tensor,
    g: tl.tensor,
    b: tl.tensor,
    Aqk: tl.tensor,
    Akk: tl.tensor,
    scale: tl.constexpr,
    N: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
) -> None:
    i_tg, i_hg = (tl.program_id(0), tl.program_id(1))
    bos = i_tg // T * T
    i_t = i_tg % T
    if i_t >= T:
        return
    i_c = i_t // BT
    i_s = i_t % BT // BC
    i_tc = i_c * BT
    i_ts = i_tc + i_s * BC
    q += bos * H * K
    k += bos * H * K
    g += bos * H * K
    Aqk += bos * H * BT
    Akk += bos * H * BC
    b += bos * H * K
    BK: tl.constexpr = triton.next_power_of_2(K)
    o_h = tl.arange(0, BH)
    o_k = tl.arange(0, BK)
    m_h = i_hg * BH + o_h < H
    m_k = o_k < K
    p_q = tl.make_block_ptr(q + i_t * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_t * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_t * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_b = tl.make_block_ptr(b + i_t * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_b = tl.load(p_b, boundary_check=(0, 1)).to(tl.float32)
    b_k = b_k * b_b
    for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):
        p_kj = tl.make_block_ptr(k + j * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
        p_gj = tl.make_block_ptr(g + j * H * K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
        b_kj = tl.load(p_kj, boundary_check=(0, 1)).to(tl.float32)
        b_gj = tl.load(p_gj, boundary_check=(0, 1)).to(tl.float32)
        b_kgj = b_kj * exp2(b_g - b_gj)
        b_kgj = tl.where(m_k[None, :], b_kgj, 0.0)
        b_Aqk = tl.sum(b_q * b_kgj, axis=1) * scale
        b_Akk = tl.sum(b_k * b_kgj, axis=1) * tl.where(j < i_t, 1.0, 0.0)
        tl.store(Aqk + i_t * H * BT + (i_hg * BH + o_h) * BT + j % BT, b_Aqk.to(Aqk.dtype.element_ty), mask=m_h)
        tl.store(Akk + i_t * H * BC + (i_hg * BH + o_h) * BC + j - i_ts, b_Akk.to(Akk.dtype.element_ty), mask=m_h)


@triton.jit
def chunk_gdn2_fwd_kernel_inter_solve_fused(
    q: tl.tensor,
    k: tl.tensor,
    g: tl.tensor,
    b: tl.tensor,
    Akkd: tl.tensor,
    Aqk: tl.tensor,
    Akk: tl.tensor,
    scale: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
) -> None:
    i_t, i_bh = (tl.program_id(0), tl.program_id(1))
    i_b, i_h = (i_bh // H, i_bh % H)
    bos = i_b * T
    if i_t * BT >= T:
        return
    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    b += (bos * H + i_h) * K
    Aqk += (bos * H + i_h) * BT
    Akk += (bos * H + i_h) * BT
    Akkd += (bos * H + i_h) * BC
    o_i = tl.arange(0, BC)
    m_tc1 = i_tc1 + o_i < T
    m_tc2 = i_tc2 + o_i < T
    m_tc3 = i_tc3 + o_i < T
    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_k0 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        p_g0 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1)).to(tl.float32)
        b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)
        if i_tc1 < T:
            p_q1 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_k1 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_g1 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_b1 = tl.make_block_ptr(b, (T, K), (H * K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            b_q1 = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32)
            b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
            b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
            b_b1 = tl.load(p_b1, boundary_check=(0, 1)).to(tl.float32)
            b_gn1 = tl.load(g + i_tc1 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
            b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))
            b_bk1 = b_b1 * b_k1
            b_Aqk10 += tl.dot(b_q1 * b_gqn, b_kgt, input_precision="tf32x3")
            b_Akk10 += tl.dot(b_bk1 * b_gqn, b_kgt, input_precision="tf32x3")
            if i_tc2 < T:
                p_q2 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                p_k2 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                p_g2 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                p_b2 = tl.make_block_ptr(b, (T, K), (H * K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
                b_q2 = tl.load(p_q2, boundary_check=(0, 1)).to(tl.float32)
                b_k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
                b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
                b_b2 = tl.load(p_b2, boundary_check=(0, 1)).to(tl.float32)
                b_gn2 = tl.load(g + i_tc2 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
                b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
                b_qg2 = b_q2 * b_gqn2
                b_bkg2 = b_b2 * b_k2 * b_gqn2
                b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0))
                b_Aqk20 += tl.dot(b_qg2, b_kgt, input_precision="tf32x3")
                b_Akk20 += tl.dot(b_bkg2, b_kgt, input_precision="tf32x3")
                b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1))
                b_Aqk21 += tl.dot(b_qg2, b_kgt, input_precision="tf32x3")
                b_Akk21 += tl.dot(b_bkg2, b_kgt, input_precision="tf32x3")
                if i_tc3 < T:
                    p_q3 = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    p_k3 = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    p_g3 = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    p_b3 = tl.make_block_ptr(b, (T, K), (H * K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
                    b_q3 = tl.load(p_q3, boundary_check=(0, 1)).to(tl.float32)
                    b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
                    b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
                    b_b3 = tl.load(p_b3, boundary_check=(0, 1)).to(tl.float32)
                    b_gn3 = tl.load(g + i_tc3 * H * K + o_k, mask=m_k, other=0).to(tl.float32)
                    b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                    b_qg3 = b_q3 * b_gqn3
                    b_bkg3 = b_b3 * b_k3 * b_gqn3
                    b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0))
                    b_Aqk30 += tl.dot(b_qg3, b_kgt, input_precision="tf32x3")
                    b_Akk30 += tl.dot(b_bkg3, b_kgt, input_precision="tf32x3")
                    b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1))
                    b_Aqk31 += tl.dot(b_qg3, b_kgt, input_precision="tf32x3")
                    b_Akk31 += tl.dot(b_bkg3, b_kgt, input_precision="tf32x3")
                    b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2))
                    b_Aqk32 += tl.dot(b_qg3, b_kgt, input_precision="tf32x3")
                    b_Akk32 += tl.dot(b_bkg3, b_kgt, input_precision="tf32x3")
    if i_tc1 < T:
        p_Aqk10 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
        tl.store(p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    if i_tc2 < T:
        p_Aqk20 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
        p_Aqk21 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
        tl.store(p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    if i_tc3 < T:
        p_Aqk30 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
        p_Aqk31 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
        p_Aqk32 = tl.make_block_ptr(Aqk, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
        tl.store(p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
    p_Akk00 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk22 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Akk33 = tl.make_block_ptr(Akkd, (T, BC), (H * BC, 1), (i_tc3, 0), (BC, BC), (1, 0))
    b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)
    if not False:
        m_A = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]
        b_Ai00 = -tl.where(m_A, b_Ai00, 0)
        b_Ai11 = -tl.where(m_A, b_Ai11, 0)
        b_Ai22 = -tl.where(m_A, b_Ai22, 0)
        b_Ai33 = -tl.where(m_A, b_Ai33, 0)
        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a00 = tl.where(o_i < i, b_a00, 0.0)
            b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        for i in range(BC + 2, min(2 * BC, T - i_tc0)):
            b_a11 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a11 = tl.where(o_i < i - BC, b_a11, 0.0)
            b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
        for i in range(2 * BC + 2, min(3 * BC, T - i_tc0)):
            b_a22 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a22 = tl.where(o_i < i - 2 * BC, b_a22, 0.0)
            b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
            b_Ai22 = tl.where((o_i == i - 2 * BC)[:, None], b_a22, b_Ai22)
        for i in range(3 * BC + 2, min(4 * BC, T - i_tc0)):
            b_a33 = -tl.load(Akkd + (i_tc0 + i) * H * BC + o_i)
            b_a33 = tl.where(o_i < i - 3 * BC, b_a33, 0.0)
            b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
            b_Ai33 = tl.where((o_i == i - 3 * BC)[:, None], b_a33, b_Ai33)
        b_Ai00 += m_I
        b_Ai11 += m_I
        b_Ai22 += m_I
        b_Ai33 += m_I
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    p_Akk00 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk10 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
    p_Akk20 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Akk21 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
    p_Akk22 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc2, 2 * BC), (BC, BC), (1, 0))
    p_Akk30 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    p_Akk31 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
    p_Akk32 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 2 * BC), (BC, BC), (1, 0))
    p_Akk33 = tl.make_block_ptr(Akk, (T, BT), (H * BT, 1), (i_tc3, 3 * BC), (BC, BC), (1, 0))
    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def recompute_w_u_fwd_gdn2_kernel(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    b: tl.tensor,
    wg: tl.tensor,
    A: tl.tensor,
    gk: tl.tensor,
    w: tl.tensor,
    u: tl.tensor,
    qg: tl.tensor,
    kg: tl.tensor,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
) -> None:
    i_t, i_bh = (tl.program_id(0), tl.program_id(1))
    i_b, i_h = (i_bh // H, i_bh % H)
    bos = i_b * T
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_wg = tl.make_block_ptr(wg + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_wg = tl.load(p_wg, boundary_check=(0, 1))
        b_vb = (b_v * b_wg).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, input_precision="tf32x3")
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))
    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_b = tl.make_block_ptr(b + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_b = tl.load(p_b, boundary_check=(0, 1))
        b_kb = b_k * b_b
        p_gk = tl.make_block_ptr(gk + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
        b_kb *= exp2(b_gk)
        p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_qg = tl.make_block_ptr(qg + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_qg = b_q * exp2(b_gk)
        tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))
        last_idx = min(i_t * BT + BT, T) - 1
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_gn = tl.load(gk + ((bos + last_idx) * H + i_h) * K + o_k, mask=m_k, other=0.0).to(tl.float32)
        b_kg = b_k * tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], exp2(b_gn[None, :] - b_gk), 0)
        p_kg = tl.make_block_ptr(kg + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))
        b_w = tl.dot(b_A, b_kb.to(b_k.dtype), input_precision="tf32x3")
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gdn2_bwd_kernel_wy_dqkg_fused(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    v_new: tl.tensor,
    g: tl.tensor,
    b: tl.tensor,
    wg: tl.tensor,
    A: tl.tensor,
    h: tl.tensor,
    do: tl.tensor,
    dh: tl.tensor,
    dv: tl.tensor,
    dq: tl.tensor,
    dk: tl.tensor,
    dv2: tl.tensor,
    dg: tl.tensor,
    db: tl.tensor,
    dw: tl.tensor,
    dA: tl.tensor,
    scale: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
) -> None:
    i_t, i_bh = (tl.program_id(0), tl.program_id(1))
    i_b, i_h = (i_bh // H, i_bh % H)
    NT = tl.cdiv(T, BT)
    i_tg = (i_b * NT + i_t).to(tl.int64)
    bos = (i_b * T).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_last = o_t == min(T, i_t * BT + BT) - 1
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    v_new += (bos * H + i_h) * V
    g += (bos * H + i_h) * K
    b += (bos * H + i_h) * K
    wg += (bos * H + i_h) * V
    A += (bos * H + i_h) * BT
    h += (i_tg * H + i_h) * K * V
    do += (bos * H + i_h) * V
    dh += (i_tg * H + i_h) * K * V
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V
    dg += (bos * H + i_h) * K
    db += (bos * H + i_h) * K
    dw += (bos * H + i_h) * V
    dA += (bos * H + i_h) * BT
    p_A = tl.make_block_ptr(A, (BT, T), (1, H * BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_b = tl.make_block_ptr(b, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_b = tl.load(p_b, boundary_check=(0, 1))
        p_gn = g + (min(T, i_t * BT + BT) - 1).to(tl.int64) * H * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)
        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dw_flow = tl.zeros([BT, BK], dtype=tl.float32)
        b_dgk = tl.zeros([BK], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            p_v_new = tl.make_block_ptr(v_new, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_v_new = tl.load(p_v_new, boundary_check=(0, 1))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_dgk += tl.sum(b_h * b_dh, axis=0)
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype), input_precision="tf32x3")
            b_dk += tl.dot(b_v_new, b_dh.to(b_v_new.dtype), input_precision="tf32x3")
            b_dw_flow += tl.dot(b_dv.to(b_v_new.dtype), b_h.to(b_v_new.dtype), input_precision="tf32x3")
            tl.debug_barrier()
            if i_k == 0:
                p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                p_dv2 = tl.make_block_ptr(dv2, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                p_wg = tl.make_block_ptr(wg, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                p_dw_gate = tl.make_block_ptr(dw, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                b_v = tl.load(p_v, boundary_check=(0, 1))
                b_wg = tl.load(p_wg, boundary_check=(0, 1))
                b_dA += tl.dot(b_dv, tl.trans(b_v * b_wg), input_precision="tf32x3")
                b_dvb = tl.dot(b_A, b_dv, input_precision="tf32x3")
                b_dv2 = b_dvb * b_wg
                b_dw_gate = b_dvb * b_v
                tl.store(p_dv2, b_dv2.to(p_dv2.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_dw_gate, b_dw_gate.to(p_dw_gate.dtype.element_ty), boundary_check=(0, 1))
        b_gk_exp = exp2(b_g)
        b_gb = b_gk_exp * b_b
        b_dgk *= exp2(b_gn)
        b_dq = b_dq * b_gk_exp * scale
        b_dk = b_dk * tl.where(m_t[:, None], exp2(b_gn[None, :] - b_g), 0)
        b_kg = b_k * b_gk_exp
        b_dw_flow = -b_dw_flow.to(b_A.dtype)
        b_dA += tl.dot(b_dw_flow, tl.trans((b_kg * b_b).to(b_A.dtype)), input_precision="tf32x3")
        b_dkgb = tl.dot(b_A, b_dw_flow, input_precision="tf32x3")
        p_db = tl.make_block_ptr(db, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_db_partial = b_dkgb * b_kg
        tl.store(p_db, b_db_partial.to(p_db.dtype.element_ty), boundary_check=(0, 1))
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_kdk = b_k * b_dk
        b_dgk += tl.sum(b_kdk, axis=0)
        b_dg = b_q * b_dq - b_kdk + m_last[:, None] * b_dgk + b_kg * b_dkgb * b_b
        b_dk = b_dk + b_dkgb * b_gb
        p_dq = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dg = tl.make_block_ptr(dg, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A, input_precision="tf32x3")
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype), input_precision="tf32x3")
    b_dA = tl.where(m_A, -b_dA, 0)
    p_dA = tl.make_block_ptr(dA, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gdn2_bwd_kernel_intra(
    q: tl.tensor,
    k: tl.tensor,
    g: tl.tensor,
    b: tl.tensor,
    dAqk: tl.tensor,
    dAkk: tl.tensor,
    dq: tl.tensor,
    dk: tl.tensor,
    dg: tl.tensor,
    dq2: tl.tensor,
    dk2: tl.tensor,
    dg2: tl.tensor,
    db: tl.tensor,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
) -> None:
    i_kc, i_t, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_b, i_h = (i_bh // H, i_bh % H)
    i_k, i_i = (i_kc // NC, i_kc % NC)
    all = B * T
    bos = i_b * T
    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    b += (bos * H + i_h) * K
    dAqk += (bos * H + i_h) * BT
    dAkk += (bos * H + i_h) * BT
    dq += (bos * H + i_h) * K
    dq2 += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dk2 += (bos * H + i_h) * K
    dg += (bos * H + i_h) * K
    dg2 += (bos * H + i_h) * K
    db += (i_k * all + bos) * H * BK + i_h * BK
    p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    p_b = tl.make_block_ptr(b, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_b = tl.load(p_b, boundary_check=(0, 1))
    b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = g + i_ti * H * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(i_i):
            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (H * BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
            p_dAkk = tl.make_block_ptr(dAkk, (T, BT), (H * BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
            b_kj = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = b_kj * exp2(b_gn - b_gk)
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))
            b_dq2 += tl.dot(b_dAqk, b_kg, input_precision="tf32x3")
            b_dk2 += tl.dot(b_dAkk, b_kg, input_precision="tf32x3")
        b_gqn = exp2(b_g - b_gn)
        b_dq2 *= b_gqn
        b_dk2 *= b_gqn
    o_i = tl.arange(0, BC)
    m_dA = i_ti + o_i < T
    o_dA = (i_ti + o_i) * H * BT + i_i * BC
    p_kj = k + i_ti * H * K + o_k
    p_gkj = g + i_ti * H * K + o_k
    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    for j in range(min(BC, T - i_t * BT - i_i * BC)):
        b_dAqk = tl.load(dAqk + o_dA + j, mask=m_dA, other=0)
        b_dAkk = tl.load(dAkk + o_dA + j, mask=m_dA, other=0)
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        m_i = o_i[:, None] >= j
        b_gqk = exp2(b_g - b_gkj[None, :])
        b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_gqk, 0.0)
        b_dk2 += tl.where(m_i, b_dAkk[:, None] * b_kj[None, :] * b_gqk, 0.0)
        p_kj += H * K
        p_gkj += H * K
    b_db_tile = b_dk2 * b_k
    b_dk2 = b_dk2 * b_b
    p_dq = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dq2 = tl.make_block_ptr(dq2, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_db = tl.make_block_ptr(db, (T, BK), (H * BK, 1), (i_ti, 0), (BC, BK), (1, 0))
    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db_tile.to(p_db.dtype.element_ty), boundary_check=(0, 1))
    tl.debug_barrier()
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)
    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = g + (min(i_ti + BC, T) - 1) * H * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(i_i + 1, NC):
            p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_bj = tl.make_block_ptr(b, (T, K), (H * K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dAqk = tl.make_block_ptr(dAqk, (BT, T), (1, H * BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            p_dAkk = tl.make_block_ptr(dAkk, (BT, T), (1, H * BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            b_bj = tl.load(p_bj, boundary_check=(0, 1))
            b_qj = tl.load(p_q, boundary_check=(0, 1))
            b_kbj = tl.load(p_k, boundary_check=(0, 1)) * b_bj
            b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))
            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            b_gkn = exp2(b_gk - b_gn)
            b_qg = b_qj * tl.where(m_j[:, None], b_gkn, 0)
            b_kbg = b_kbj * tl.where(m_j[:, None], b_gkn, 0)
            b_dkt += tl.dot(b_dAqk, b_qg, input_precision="tf32x3")
            b_dkt += tl.dot(b_dAkk, b_kbg, input_precision="tf32x3")
        b_dkt *= exp2(b_gn - b_g)
    o_dA = i_ti * H * BT + i_i * BC + o_i
    p_qj = q + i_ti * H * K + o_k
    p_kj = k + i_ti * H * K + o_k
    p_gkj = g + i_ti * H * K + o_k
    p_bj_ptr = b + i_ti * H * K + o_k
    for j in range(min(BC, T - i_t * BT - i_i * BC)):
        b_dAqk = tl.load(dAqk + o_dA + j * H * BT)
        b_dAkk = tl.load(dAkk + o_dA + j * H * BT)
        b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_bj = tl.load(p_bj_ptr, mask=m_k, other=0).to(tl.float32)
        b_kbj = b_kj * b_bj
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        m_i = o_i[:, None] <= j
        b_gkq = exp2(b_gkj[None, :] - b_g)
        b_dkt += tl.where(m_i, b_dAqk[:, None] * b_qj[None, :] * b_gkq, 0.0)
        b_dkt += tl.where(m_i, b_dAkk[:, None] * b_kbj[None, :] * b_gkq, 0.0)
        p_qj += H * K
        p_kj += H * K
        p_gkj += H * K
        p_bj_ptr += H * K
    p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dk2 = tl.make_block_ptr(dk2, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dg2 = tl.make_block_ptr(dg2, (T, K), (H * K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_dg2 += (b_dk2 - b_dkt) * b_k + tl.load(p_dg, boundary_check=(0, 1))
    b_dk2 += tl.load(p_dk, boundary_check=(0, 1))
    b_dk2 += b_dkt
    tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg2, b_dg2.to(p_dg2.dtype.element_ty), boundary_check=(0, 1))
