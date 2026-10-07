"""Optional JAX bindings for NVIDIA's GDN-2 Triton kernels.

Imports no PyTorch. Arrays use batch-first BTD/BTHD layouts and float32,
matching the portable core and the kernels' blocks of 64 time steps.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from rl2.gdn2.masking import prefix_mask
from rl2.shape_checker import ShapeChecker

try:
    import jax_triton as jt
    import triton
except ImportError as error:
    raise ImportError("Install the GPU backend with uv sync --extra cuda12 --extra gdn2-triton") from error

from rl2.gdn2._triton import fla, nv

type Arrays = tuple[jax.Array, ...]
type Result = tuple[jax.Array, jax.Array]


class ChunkResidual(NamedTuple):
    inputs: Arrays
    g: jax.Array
    aqk: jax.Array
    akk: jax.Array


def _shape(shape: tuple[int, ...]) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(shape, jnp.float32)


def _cumsum(g: jax.Array, *, reverse: bool = False) -> jax.Array:
    sc = ShapeChecker()
    sc.check(g, "BTHK", jnp.float32)
    b, t, h, k = g.shape
    blocks = g.reshape(b, t // 64, 64, h, k)
    if reverse:
        return jnp.cumsum(blocks[:, :, ::-1], axis=2)[:, :, ::-1].reshape(g.shape)
    return jnp.cumsum(blocks, axis=2).reshape(g.shape) * 1.4426950408889634


def _wy(inputs: Arrays, g: jax.Array, akk: jax.Array) -> tuple[Arrays, Arrays]:
    q, k, v, _, b, wg, state = inputs
    sc = ShapeChecker(C=64)
    sc.check([q, k, g, b], "BTHK", jnp.float32)
    sc.check([v, wg], "BTHV", jnp.float32)
    sc.check(state, "BHKV", jnp.float32)
    sc.check(akk, "BTHC", jnp.float32)
    batch, time, heads, key = q.shape
    value = v.shape[-1]
    dims = {"T": time, "H": heads, "K": key, "V": value, "BT": 64, "BV": 32}
    w, u, qg, kg = jt.triton_call(
        q,
        k,
        v,
        b,
        wg,
        akk,
        g,
        kernel=nv.recompute_w_u_fwd_gdn2_kernel,
        out_type=(_shape(q.shape), _shape(v.shape), _shape(q.shape), _shape(k.shape)),
        grid=(time // 64, batch * heads),
        BK=32,
        **dims,
    )
    h, v_new, ht = jt.triton_call(
        kg,
        u,
        w,
        g,
        state,
        kernel=fla.chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
        num_stages=1,  # Keep multi-chunk K=128 state propagation within A100 shared memory.
        out_type=(_shape((batch, time // 64, heads, key, value)), _shape(v.shape), _shape(state.shape)),
        grid=(triton.cdiv(value, 32) * batch * heads,),
        HV=heads,
        **dims,
    )
    sc.check([w, qg, kg], "BTHK", jnp.float32)
    sc.check([u, v_new], "BTHV", jnp.float32)
    sc.check(ht, "BHKV", jnp.float32)
    return (w, qg, kg), (h, v_new, ht)


def _chunk_forward(*inputs: jax.Array) -> tuple[Result, ChunkResidual]:
    q, k, v, log_decay, b, wg, state = inputs
    sc = ShapeChecker(C=64)
    sc.check([q, k, log_decay, b], "BTHK", jnp.float32)
    sc.check([v, wg], "BTHV", jnp.float32)
    sc.check(state, "BHKV", jnp.float32)
    batch, time, heads, key = q.shape
    value = v.shape[-1]
    g = _cumsum(log_decay)
    dims = {"T": time, "H": heads, "K": key, "BT": 64}
    aqk_diag, akkd = jt.triton_call(
        q,
        k,
        g,
        b,
        kernel=nv.chunk_gdn2_fwd_kernel_intra_token_parallel,
        out_type=(_shape((batch, time, heads, 64)), _shape((batch, time, heads, 16))),
        zeroed_outputs=(0, 1),
        grid=(batch * time, heads),
        scale=1.0,
        N=batch,
        BC=16,
        BH=1,
        **dims,
    )
    aqk_offdiag, akk = jt.triton_call(
        q,
        k,
        g,
        b,
        akkd,
        kernel=nv.chunk_gdn2_fwd_kernel_inter_solve_fused,
        out_type=(_shape(aqk_diag.shape), _shape(aqk_diag.shape)),
        zeroed_outputs=(0, 1),
        grid=(time // 64, batch * heads),
        scale=1.0,
        BC=16,
        BK=32,
        **dims,
    )
    aqk = aqk_diag + aqk_offdiag
    _, (h, v_new, ht) = _wy(inputs, g, akk)
    out = jt.triton_call(
        q,
        v_new,
        g,
        h,
        aqk,
        kernel=fla.chunk_gla_fwd_kernel_o,
        out_type=_shape(v.shape),
        grid=(triton.cdiv(value, 32) * (time // 64), batch * heads),
        scale=1.0,
        HV=heads,
        V=value,
        BK=32,
        BV=32,
        **dims,
    )
    return (ht, out), ChunkResidual(inputs, g, aqk, akk)


@jax.custom_vjp
def _chunk(*inputs: jax.Array) -> Result:
    return _chunk_forward(*inputs)[0]


def _chunk_backward(residual: ChunkResidual, cotangent: Result) -> Arrays:
    inputs, g, aqk, akk = residual
    q, k, v, _, b, wg, state = inputs
    dht, do = cotangent
    sc = ShapeChecker()
    sc.check([q, k, g, b], "BTHK", jnp.float32)
    sc.check([v, wg, do], "BTHV", jnp.float32)
    sc.check([state, dht], "BHKV", jnp.float32)
    batch, time, heads, key = q.shape
    value = v.shape[-1]
    dims = {"T": time, "H": heads, "K": key, "V": value, "BT": 64, "BV": 32}
    grid = (time // 64, batch * heads)
    (w, qg, kg), (h, v_new, _) = _wy(inputs, g, akk)
    dv, daqk = jt.triton_call(
        q,
        k,
        v_new,
        aqk,
        do,
        kernel=fla.chunk_kda_bwd_kernel_dAv,
        out_type=(_shape(v.shape), _shape(aqk.shape)),
        grid=grid,
        scale=1.0,
        HV=heads,
        BK=32,
        **dims,
    )
    dh, dh0, dv = jt.triton_call(
        qg,
        kg,
        w,
        g,
        dht,
        do,
        dv,
        kernel=fla.chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
        num_stages=1,  # Keep multi-chunk K=128 state propagation within A100 shared memory.
        out_type=(_shape(h.shape), _shape(state.shape), _shape(v.shape)),
        grid=(triton.cdiv(value, 32) * batch * heads,),
        scale=1.0,
        HV=heads,
        **dims,
    )
    dq, dk, dv, dg, db, dw, dakk = jt.triton_call(
        q,
        k,
        v,
        v_new,
        g,
        b,
        wg,
        akk,
        h,
        do,
        dh,
        dv,
        kernel=nv.chunk_gdn2_bwd_kernel_wy_dqkg_fused,
        out_type=tuple(_shape(a.shape) for a in (q, k, v, g, b, wg, akk)),
        grid=grid,
        scale=1.0,
        BK=32,
        **dims,
    )
    bk = min(32, triton.next_power_of_2(key))
    nk = triton.cdiv(key, bk)
    dq, dk, dg, db_parts = jt.triton_call(
        q,
        k,
        g,
        b,
        daqk,
        dakk,
        dq,
        dk,
        dg,
        kernel=nv.chunk_gdn2_bwd_kernel_intra,
        out_type=(_shape(q.shape), _shape(k.shape), _shape(g.shape), _shape((nk, batch, time, heads, bk))),
        grid=(nk * 4, time // 64, batch * heads),
        B=batch,
        T=time,
        H=heads,
        K=key,
        BT=64,
        BC=16,
        BK=bk,
        NC=4,
    )
    db = db + db_parts.transpose(1, 2, 3, 0, 4).reshape(batch, time, heads, nk * bk)[..., :key]
    return dq, dk, dv, _cumsum(dg, reverse=True), db, dw, dh0


_chunk.defvjp(_chunk_forward, _chunk_backward)


def chunk_gated_delta_rule(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_decay: jax.Array,
    erase: jax.Array,
    write: jax.Array,
    initial_state: jax.Array | None = None,
) -> Result:
    """Chunked forward and first-order reverse-mode gradients on NVIDIA GPUs.

    Same batch-first float32 contract as core.gated_delta_rule. Key dimensions
    up to 256 are supported. Tail padding is internal and preserves final state.
    """
    sc = ShapeChecker()
    sc.check([q, k, log_decay, erase], "BTHK", jnp.float32)
    sc.check([v, write], "BTHV", jnp.float32)
    state = jnp.zeros(sc["BHKV"], jnp.float32) if initial_state is None else initial_state
    sc.check(state, "BHKV", jnp.float32)
    if q.shape[-1] > 256:
        raise ValueError("Triton chunk kernels support head_dim <= 256")
    if q.shape[1] == 0:
        return state, jnp.zeros_like(v)
    pad = (-q.shape[1]) % 64
    inputs = tuple(jnp.pad(a, ((0, 0), (0, pad), (0, 0), (0, 0))) for a in (q, k, v, log_decay, erase, write))
    final_state, output = _chunk(*inputs, state)
    output = output[:, : q.shape[1]]
    sc.check(final_state, "BHKV", jnp.float32)
    sc.check(output, "BTHV", jnp.float32)
    return final_state, output


@jax.custom_vjp
def _recurrent(*inputs: jax.Array) -> Result:
    from rl2.gdn2._triton.recurrent import recurrent_kernel

    q, k, v, g, b, w, state = inputs
    sc = ShapeChecker()
    sc.check([q, k, g, b], "BTHK", jnp.float32)
    sc.check([v, w], "BTHV", jnp.float32)
    sc.check(state, "BHKV", jnp.float32)
    batch, time, heads, key = q.shape
    value = v.shape[-1]
    return jt.triton_call(
        q,
        k,
        v,
        g,
        b,
        w,
        state,
        kernel=recurrent_kernel,
        out_type=(_shape(state.shape), _shape(v.shape)),
        grid=(triton.cdiv(value, 32), batch * heads),
        T=time,
        H=heads,
        K=key,
        V=value,
        BK=triton.next_power_of_2(key),
        BV=32,
    )


def _recurrent_forward(*inputs: jax.Array) -> tuple[Result, Arrays]:
    return _recurrent(*inputs), inputs


def _recurrent_backward(inputs: Arrays, cotangent: Result) -> Arrays:
    # Decoding normally has no gradient. If differentiated, recompute via the
    # chunked primitive and use its custom backward, including initial state.
    _, pullback = jax.vjp(chunk_gated_delta_rule, *inputs)
    return pullback(cotangent)


_recurrent.defvjp(_recurrent_forward, _recurrent_backward)


def fused_recurrent_gated_delta_rule(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_decay: jax.Array,
    erase: jax.Array,
    write: jax.Array,
    initial_state: jax.Array | None = None,
) -> Result:
    """Fused decoding/prefill recurrence; autodiff uses chunked backward."""
    sc = ShapeChecker()
    sc.check([q, k, log_decay, erase], "BTHK", jnp.float32)
    sc.check([v, write], "BTHV", jnp.float32)
    state = jnp.zeros(sc["BHKV"], jnp.float32) if initial_state is None else initial_state
    sc.check(state, "BHKV", jnp.float32)
    if q.shape[-1] > 256:
        raise ValueError("Triton kernels support head_dim <= 256")
    if q.shape[1] == 0:
        return state, jnp.zeros_like(v)
    final_state, output = _recurrent(q, k, v, log_decay, erase, write, state)
    sc.check(final_state, "BHKV", jnp.float32)
    sc.check(output, "BTHV", jnp.float32)
    return final_state, output


def _conv_forward(x: jax.Array, weight: jax.Array, bias: jax.Array) -> tuple[jax.Array, Arrays]:
    from rl2.gdn2._triton.pointwise import conv_forward

    sc = ShapeChecker()
    sc.check(x, "BLD", jnp.float32)
    sc.check(weight, "CD", jnp.float32)
    sc.check(bias, "D", jnp.float32)
    batch, length, width = x.shape
    size = weight.shape[0]
    time = length - size + 1
    y, z = jt.triton_call(
        x,
        weight,
        bias,
        kernel=conv_forward,
        out_type=(_shape((batch, time, width)), _shape((batch, time, width))),
        grid=(batch * time, triton.cdiv(width, 128)),
        T=time,
        D=width,
        C=size,
        BD=128,
    )
    return y, (x, weight, z)


@jax.custom_vjp
def _conv(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    return _conv_forward(x, weight, bias)[0]


def _conv_backward(residual: Arrays, dy: jax.Array) -> Arrays:
    from rl2.gdn2._triton.pointwise import conv_backward_input, conv_backward_weight

    x, weight, z = residual
    sc = ShapeChecker()
    sc.check([dy, z], "BTD", jnp.float32)
    sc.check(weight, "CD", jnp.float32)
    sc.check(x, "BLD", jnp.float32)
    batch, time, width = dy.shape
    size = weight.shape[0]
    sig = jax.nn.sigmoid(z)
    dz = dy * sig * (1 + z * (1 - sig))
    dx = jt.triton_call(
        dz,
        weight,
        kernel=conv_backward_input,
        out_type=_shape(x.shape),
        grid=(batch * x.shape[1], triton.cdiv(width, 128)),
        T=time,
        D=width,
        C=size,
        BD=128,
    )
    dw, db = jt.triton_call(
        x,
        dz,
        kernel=conv_backward_weight,
        out_type=(_shape(weight.shape), _shape((width,))),
        grid=(size, triton.cdiv(width, 32)),
        B=batch,
        T=time,
        D=width,
        C=size,
        BD=32,
        BR=32,
    )
    return dx, dw, db


_conv.defvjp(_conv_forward, _conv_backward)


def short_conv(
    x: jax.Array,
    x_len: jax.Array,
    history: jax.Array,
    weight: jax.Array,
    bias: jax.Array,
) -> Result:
    """Convolve [B,T,D] with required int32 valid-prefix lengths x_len[B].

    Return (final history, SiLU(depthwise causal convolution)). Right padding
    produces zero output and leaves history at the last valid token; zero
    lengths preserve the supplied history.
    """
    sc = ShapeChecker()
    sc.check(x, "BTD", jnp.float32)
    sc.check(weight, "CD", jnp.float32)
    sc.check(bias, "D", jnp.float32)
    sc.check(history, "BND", jnp.float32)
    valid = prefix_mask(x_len, x.shape[0], x.shape[1])
    if history.shape[1] != weight.shape[0] - 1:
        raise ValueError("Convolution history must contain conv_size - 1 tokens")
    if x.shape[1] == 0:
        return history, x
    x = jnp.where(valid[..., None], x, 0)
    joined = jnp.concatenate((history, x), axis=1)
    output = _conv(joined, weight, bias)
    output = jnp.where(valid[..., None], output, 0)
    # This also covers conv_size=1, where the history has length zero.
    indices = x_len[:, None] + jnp.arange(history.shape[1], dtype=jnp.int32)[None, :]
    sc.check(indices, "BN", jnp.int32)
    final_history = jnp.take_along_axis(joined, indices[..., None], axis=1)
    sc.check(final_history, "BND", jnp.float32)
    sc.check(output, "BTD", jnp.float32)
    return final_history, output


# Epsilon is a static kernel option, not a trainable parameter.


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def gated_rms_norm(x: jax.Array, gate: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    from rl2.gdn2._triton.pointwise import norm_forward

    sc = ShapeChecker()
    sc.check([x, gate], "BTHV", jnp.float32)
    sc.check(weight, "V", jnp.float32)
    value = x.shape[-1]
    if x.size == 0:
        return x
    return jt.triton_call(
        x,
        gate,
        weight,
        kernel=norm_forward,
        out_type=_shape(x.shape),
        grid=(x.size // value,),
        V=value,
        BV=triton.next_power_of_2(value),
        EPS=eps,
    )


def _norm_forward(x: jax.Array, gate: jax.Array, weight: jax.Array, eps: float) -> tuple[jax.Array, Arrays]:
    return gated_rms_norm(x, gate, weight, eps), (x, gate, weight)


def _norm_backward(eps: float, residual: Arrays, dy: jax.Array) -> Arrays:
    from rl2.gdn2._triton.pointwise import norm_backward

    x, gate, weight = residual
    sc = ShapeChecker()
    sc.check([x, gate, dy], "BTHV", jnp.float32)
    sc.check(weight, "V", jnp.float32)
    value = x.shape[-1]
    if x.size == 0:
        return dy, dy, jnp.zeros_like(weight)
    dx, dg, dw = jt.triton_call(
        x,
        gate,
        weight,
        dy,
        kernel=norm_backward,
        out_type=(_shape(x.shape), _shape(x.shape), _shape(x.shape)),
        grid=(x.size // value,),
        V=value,
        BV=triton.next_power_of_2(value),
        EPS=eps,
    )
    return dx, dg, jnp.sum(dw, axis=(0, 1, 2))


gated_rms_norm.defvjp(_norm_forward, _norm_backward)
