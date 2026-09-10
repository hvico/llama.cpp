#pragma once

#include "common.cuh"

// MUL_MAT_ID (MoE expert GEMMs) on FP16 tensor cores for Volta.
//
// Volta has FP16 HMMA but no int8 MMA, so MMQ falls back to dp4a there. For the MoE prefill
// shapes (few tokens per expert, 4-6 bit expert weights) this kernel dequantizes the expert
// weight tiles to FP16 in shared memory, converts the activations to FP16 once, and runs
// m16n16k16 WMMA with FP32 accumulation. Numerically it is equivalent to the cuBLAS
// dequantize-to-F16 path used for dense matrices on Volta.

bool ggml_cuda_mul_mat_id_hmma_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_id_hmma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
