#pragma once

#include "common.cuh"

// A chain of elementwise ops applied to one carried tensor, run as a single kernel:
//   v = x[i]; for each step: v = v OP operand[bcast(i)]  |  v = v*s + b  |  v = unary(v)
// Used for the small gating chains of a layer (e.g. add bias -> softplus -> mul, sigmoid -> mul) that
// would otherwise cost one launch per op.
#define GGML_CUDA_EWCHAIN_MAX_STEPS 6

enum ggml_cuda_ewchain_kind : int32_t {
    GGML_CUDA_EWCHAIN_ADD   = 0,
    GGML_CUDA_EWCHAIN_SUB   = 1,
    GGML_CUDA_EWCHAIN_MUL   = 2,
    GGML_CUDA_EWCHAIN_DIV   = 3,
    GGML_CUDA_EWCHAIN_SCALE = 4, // v*s + b
    GGML_CUDA_EWCHAIN_UNARY = 5,
};

struct ggml_cuda_ewchain_step {
    int32_t       kind;
    int32_t       unary;      // ggml_unary_op for UNARY steps
    const float * operand;    // binary steps
    // operand indexing per output dim: the stride in elements, 0 for a broadcast dim (size 1). The operand
    // must have size 1 or the output's size in every dim (no modulo broadcast).
    uint32_t      nb[4];
    float         s;
    float         b;
};

struct ggml_cuda_ewchain_params {
    int32_t                n_steps;
    uint3                  ne0_fd;
    uint3                  ne1_fd;
    uint3                  ne2_fd;
    ggml_cuda_ewchain_step steps[GGML_CUDA_EWCHAIN_MAX_STEPS];
};

// returns false when a unary op of the chain is not supported by the kernel
bool ggml_cuda_ewchain_unary_supported(ggml_unary_op op);

void ggml_cuda_op_ewchain(ggml_backend_cuda_context & ctx, const ggml_tensor * x, ggml_tensor * dst, const ggml_cuda_ewchain_params & params);
