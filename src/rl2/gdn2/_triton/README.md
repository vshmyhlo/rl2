# Kernel provenance

`nv.py` adapts these kernels from
[NVlabs/GatedDeltaNet-2, revision a5552fe3c67e0ebc7ef1220df68ae8896ec62d56](https://github.com/NVlabs/GatedDeltaNet-2/blob/a5552fe3c67e0ebc7ef1220df68ae8896ec62d56/lit_gpt/gdn2_ops/chunk_gdn2.py):

- `chunk_gdn2_fwd_kernel_intra_token_parallel`
- `chunk_gdn2_fwd_kernel_inter_solve_fused`
- `recompute_w_u_fwd_gdn2_kernel`
- `chunk_gdn2_bwd_kernel_wy_dqkg_fused`
- `chunk_gdn2_bwd_kernel_intra`

`recurrent.py` adapts the same revision's
[`fused_recurrent_gdn2_fwd_kernel`](https://github.com/NVlabs/GatedDeltaNet-2/blob/a5552fe3c67e0ebc7ef1220df68ae8896ec62d56/lit_gpt/gdn2_ops/fused_recurrent_gdn2.py)
for contiguous dense states and pre-normalized/scaled queries and keys.
Both files retain the [NVIDIA Source Code License-NC](../LICENSE).

`fla.py` adapts kernels from
[Flash Linear Attention, revision 8926cd1cec6727df7fa0138640517fb9e2990d2f](https://github.com/fla-org/flash-linear-attention/tree/8926cd1cec6727df7fa0138640517fb9e2990d2f):

- `fla/ops/common/chunk_delta_h.py`: state propagation and its backward pass.
- `fla/ops/gla/chunk.py`: `chunk_gla_fwd_kernel_o`.
- `fla/ops/kda/chunk_bwd.py`: `chunk_kda_bwd_kernel_dAv`.
- `fla/ops/utils/op.py`: `unflatten_program_id`.

FLA copyright notices are preserved in `fla.py`; its MIT license is in
[LICENSE-FLA](LICENSE-FLA).

The adaptations remove PyTorch wrappers, autotuning, unused variable-length,
transposed-state, and graph-buffer branches. Pointer arguments are reordered
for `jax_triton.triton_call`. Chunk size is 64, subchunk size 16, and matrix tiles
are 32. State propagation uses one pipeline stage to fit the A100 shared-memory
limit at the checkpoint's 128-wide heads across multiple chunks. Float32 matrix products use `tf32x3` precision, including triangular
solves. `Aqk` diagonal and off-diagonal contributions use separate buffers to
preserve JAX input immutability. Partial chunks are padded with identity state
transitions by the JAX wrapper. Cumulative decay and its reverse scan remain
JAX operations. The backward recomputes WY factors and chunk-boundary states,
as in upstream, rather than retaining a state matrix for every token.

`pointwise.py` contains local implementations of depthwise convolution with
SiLU and gated RMS normalization, including their derivatives. These follow
our float32 history/parameter conventions and are not verbatim FLA kernels.

This version supports NVIDIA GPUs, dense batches, key dimensions up to 256,
and first-order reverse-mode autodiff. It was tested on an A100 with JAX 0.11.2,
jax-triton 0.4.1, and Triton 3.8.0. Higher-order derivatives, forward-mode JVP,
`vmap` over a custom call, multi-device sharding, and other GPU architectures
have not been validated. Batch examples through the existing batch axis.
