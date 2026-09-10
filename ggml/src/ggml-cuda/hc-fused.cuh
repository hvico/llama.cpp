#pragma once

#include "common.cuh"

// Fused kernels for the manifold hyper-connection (mHC) glue used by qwen4exp:
// the residual is kept as hc parallel streams [n_embd, hc, T] and every
// attention/FFN block is wrapped by a "mix" (streams -> one input) and a
// "combine" (block output -> streams). The graph builder emits them as chains
// of small elementwise ops; these entry points run each chain as one kernel.

// mix tail:  dst[e, t] = scale * sum_h xn[e + h*n_embd, t] * sigmoid(y[e + h*n_embd, t])
//   xn, y: [hc*n_embd, T] F32 contiguous, dst: [n_embd, T] F32 contiguous
void ggml_cuda_op_hc_mix_tail(ggml_backend_cuda_context & ctx,
                              const ggml_tensor * xn,
                              const ggml_tensor * y,
                              float               scale,
                              ggml_tensor *       dst);

// combine:   w[h, t]      = s_post * sigmoid(s_pre * sum_k w_inject[k, h] * xn[k, t])
//            dst[e, h, t] = residual[e, h, t] + block[e, t] * w[h, t]
//   residual/dst: [n_embd, hc, T] F32 contiguous, block: [n_embd, T] F32 contiguous,
//   xn: [hc_dim, T] F32 contiguous, w_inject: [hc_dim, hc] F32 contiguous
void ggml_cuda_op_hc_combine(ggml_backend_cuda_context & ctx,
                             const ggml_tensor * residual,
                             const ggml_tensor * block,
                             const ggml_tensor * xn,
                             const ggml_tensor * w_inject,
                             float               s_pre,
                             float               s_post,
                             ggml_tensor *       dst);

// dst = unary(scale * src + bias), unary in {SILU, SIGMOID}; F32 contiguous
void ggml_cuda_op_scale_unary(ggml_backend_cuda_context & ctx,
                              const ggml_tensor * src,
                              float               scale,
                              float               bias,
                              ggml_unary_op       op,
                              ggml_tensor *       dst);
