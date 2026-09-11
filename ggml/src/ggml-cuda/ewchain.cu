#include "ewchain.cuh"
#include "unary.cuh"

static __device__ __forceinline__ float ewchain_unary(int op, float x) {
    switch (op) {
        case GGML_UNARY_OP_SILU:     return ggml_cuda_op_silu_single(x);
        case GGML_UNARY_OP_SIGMOID:  return 1.0f / (1.0f + expf(-x));
        case GGML_UNARY_OP_SOFTPLUS: return (x > 20.0f) ? x : logf(1.0f + expf(x)); // as unary.cu
        case GGML_UNARY_OP_RELU:     return fmaxf(x, 0.0f);
        case GGML_UNARY_OP_TANH:     return tanhf(x);
        case GGML_UNARY_OP_EXP:      return expf(x);
        case GGML_UNARY_OP_NEG:      return -x;
        case GGML_UNARY_OP_ABS:      return fabsf(x);
        case GGML_UNARY_OP_GELU:     return ggml_cuda_op_gelu_single(x);
        default:                     return x;
    }
}

bool ggml_cuda_ewchain_unary_supported(ggml_unary_op op) {
    switch (op) {
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_SIGMOID:
        case GGML_UNARY_OP_SOFTPLUS:
        case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_TANH:
        case GGML_UNARY_OP_EXP:
        case GGML_UNARY_OP_NEG:
        case GGML_UNARY_OP_ABS:
        case GGML_UNARY_OP_GELU:
            return true;
        default:
            return false;
    }
}

static __device__ __forceinline__ uint32_t ewchain_index(const ggml_cuda_ewchain_step & st, const uint32_t c[4]) {
    return c[0]*st.nb[0] + c[1]*st.nb[1] + c[2]*st.nb[2] + c[3]*st.nb[3];
}

static __global__ void ewchain_kernel(
        const float * __restrict__ x, float * __restrict__ dst, const uint32_t n,
        const ggml_cuda_ewchain_params params) {
    const uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    // output coordinates for the broadcast operands
    uint32_t c[4];
    const uint32_t r  = fastdiv(i, params.ne0_fd);
    c[0] = i - r*params.ne0_fd.z;
    const uint32_t r2 = fastdiv(r, params.ne1_fd);
    c[1] = r - r2*params.ne1_fd.z;
    c[3] = fastdiv(r2, params.ne2_fd);
    c[2] = r2 - c[3]*params.ne2_fd.z;

    float v = x[i];

    for (int s = 0; s < params.n_steps; ++s) {
        const ggml_cuda_ewchain_step & st = params.steps[s];
        switch (st.kind) {
            case GGML_CUDA_EWCHAIN_SCALE:
                v = v*st.s + st.b;
                break;
            case GGML_CUDA_EWCHAIN_UNARY:
                v = ewchain_unary(st.unary, v);
                break;
            default: {
                const float o = st.operand[ewchain_index(st, c)];
                switch (st.kind) {
                    case GGML_CUDA_EWCHAIN_ADD: v += o; break;
                    case GGML_CUDA_EWCHAIN_SUB: v -= o; break;
                    case GGML_CUDA_EWCHAIN_MUL: v *= o; break;
                    default:                    v /= o; break;
                }
            }
        }
    }

    dst[i] = v;
}

void ggml_cuda_op_ewchain(ggml_backend_cuda_context & ctx, const ggml_tensor * x, ggml_tensor * dst, const ggml_cuda_ewchain_params & params) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_nelements(x) == ggml_nelements(dst));

    const int64_t n = ggml_nelements(dst);
    GGML_ASSERT(n < INT32_MAX);
    const int block = 256;
    const int grid  = (int) ((n + block - 1)/block);

    ggml_cuda_ewchain_params p = params;
    p.ne0_fd = init_fastdiv_values((uint32_t) dst->ne[0]);
    p.ne1_fd = init_fastdiv_values((uint32_t) dst->ne[1]);
    p.ne2_fd = init_fastdiv_values((uint32_t) dst->ne[2]);

    ewchain_kernel<<<grid, block, 0, ctx.stream()>>>((const float *) x->data, (float *) dst->data, (uint32_t) n, p);
}
