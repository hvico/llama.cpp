#include "hc-fused.cuh"
#include "unary.cuh"

#define HC_MAX_STREAMS 8

// same formulas as unary.cu so fused and unfused paths stay bit-identical
static __device__ __forceinline__ float hc_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// one thread per (e, t)
static __global__ void hc_mix_tail_kernel(
        const float * __restrict__ xn,
        const float * __restrict__ y,
        float       * __restrict__ dst,
        const int   n_embd,
        const int   hc,
        const float scale,
        const int   total,
        const uint3 n_embd_fastdiv) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int t = (int) fastdiv((uint32_t) idx, n_embd_fastdiv);
    const int e = idx - t*n_embd;

    const int64_t base = (int64_t) t*hc*n_embd + e;

    float acc = 0.0f;
    for (int h = 0; h < hc; ++h) {
        const int64_t i = base + (int64_t) h*n_embd;
        // separate mul + add (no FMA contraction) to match the unfused mul -> add chain
        acc = __fadd_rn(acc, __fmul_rn(xn[i], hc_sigmoid(y[i])));
    }

    dst[idx] = __fmul_rn(acc, scale);
}

void ggml_cuda_op_hc_mix_tail(ggml_backend_cuda_context & ctx,
                              const ggml_tensor * xn,
                              const ggml_tensor * y,
                              float               scale,
                              ggml_tensor *       dst) {
    GGML_ASSERT(xn->type == GGML_TYPE_F32 && y->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(xn) && ggml_is_contiguous(y) && ggml_is_contiguous(dst));

    const int n_embd = (int) dst->ne[0];
    const int T      = (int) ggml_nrows(dst);
    const int hc     = (int) (xn->ne[0] / n_embd);
    GGML_ASSERT(hc*n_embd == xn->ne[0] && ggml_nelements(y) == ggml_nelements(xn));
    GGML_ASSERT(ggml_nrows(xn) == T);

    const int   total = n_embd*T;
    const int   block = 256;
    const int   grid  = (total + block - 1) / block;
    const uint3 n_embd_fastdiv = init_fastdiv_values((uint32_t) n_embd);

    hc_mix_tail_kernel<<<grid, block, 0, ctx.stream()>>>(
        (const float *) xn->data, (const float *) y->data, (float *) dst->data,
        n_embd, hc, scale, total, n_embd_fastdiv);
}

// one block per (stream h, token t): the block reduces w_inject[:, h] . xn[:, t]
// (float4 loads, several in flight — a single block is latency-bound on the
// ~80 KB it reads), then writes stream h of the token. Spreading the streams
// over blocks costs re-reading xn from L2 but keeps every SM busy at T = 1.
template <int BLOCK>
static __global__ void hc_combine_kernel(
        const float * __restrict__ residual,
        const float * __restrict__ block,
        const float * __restrict__ xn,
        const float * __restrict__ w_inject,
        float       * __restrict__ dst,
        const int   n_embd,
        const int   hc,
        const float s_pre,
        const float s_post) {
    const int h       = blockIdx.x;
    const int t       = blockIdx.y;
    const int tid     = threadIdx.x;
    const int hc_dim  = n_embd*hc;
    const int hc_dim4 = hc_dim/4;

    const float4 * xn_t = (const float4 *) (xn + (int64_t) t*hc_dim);
    const float4 * w_h  = (const float4 *) (w_inject + (int64_t) h*hc_dim);

    float acc = 0.0f;
#pragma unroll 8
    for (int k = tid; k < hc_dim4; k += BLOCK) {
        const float4 x = xn_t[k];
        const float4 w = w_h[k];
        acc += x.x*w.x + x.y*w.y + x.z*w.z + x.w*w.w;
    }

    __shared__ float s_red[BLOCK/WARP_SIZE];
    __shared__ float s_w;

    acc = warp_reduce_sum(acc);
    if (tid % WARP_SIZE == 0) {
        s_red[tid / WARP_SIZE] = acc;
    }
    __syncthreads();

    if (tid == 0) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < BLOCK/WARP_SIZE; ++w) {
            v += s_red[w];
        }
        s_w = s_post * hc_sigmoid(s_pre * v);
    }
    __syncthreads();

    const float  w      = s_w;
    const int    n4     = n_embd/4;
    const float4 * res4 = (const float4 *) (residual + (int64_t) t*hc_dim + (int64_t) h*n_embd);
    const float4 * blk4 = (const float4 *) (block    + (int64_t) t*n_embd);
    float4       * dst4 = (float4 *)       (dst      + (int64_t) t*hc_dim + (int64_t) h*n_embd);

#pragma unroll 4
    for (int e = tid; e < n4; e += BLOCK) {
        const float4 r = res4[e];
        const float4 b = blk4[e];
        float4 o;
        o.x = __fadd_rn(r.x, __fmul_rn(b.x, w));
        o.y = __fadd_rn(r.y, __fmul_rn(b.y, w));
        o.z = __fadd_rn(r.z, __fmul_rn(b.z, w));
        o.w = __fadd_rn(r.w, __fmul_rn(b.w, w));
        dst4[e] = o;
    }
}

void ggml_cuda_op_hc_combine(ggml_backend_cuda_context & ctx,
                             const ggml_tensor * residual,
                             const ggml_tensor * block,
                             const ggml_tensor * xn,
                             const ggml_tensor * w_inject,
                             float               s_pre,
                             float               s_post,
                             ggml_tensor *       dst) {
    GGML_ASSERT(residual->type == GGML_TYPE_F32 && block->type == GGML_TYPE_F32 &&
                xn->type == GGML_TYPE_F32 && w_inject->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(residual) && ggml_is_contiguous(block) &&
                ggml_is_contiguous(xn) && ggml_is_contiguous(w_inject) && ggml_is_contiguous(dst));

    const int n_embd = (int) dst->ne[0];
    const int hc     = (int) dst->ne[1];
    const int T      = (int) (dst->ne[2]*dst->ne[3]);
    GGML_ASSERT(ggml_are_same_shape(residual, dst));
    GGML_ASSERT(block->ne[0] == n_embd && ggml_nrows(block) == T);
    GGML_ASSERT(xn->ne[0] == (int64_t) n_embd*hc && ggml_nrows(xn) == T);
    GGML_ASSERT(w_inject->ne[0] == (int64_t) n_embd*hc && w_inject->ne[1] == hc);
    GGML_ASSERT(n_embd % 4 == 0); // float4 loads over xn / w_inject rows

    constexpr int BLOCK = 256;
    const dim3 grid(hc, T);
    hc_combine_kernel<BLOCK><<<grid, BLOCK, 0, ctx.stream()>>>(
        (const float *) residual->data, (const float *) block->data,
        (const float *) xn->data, (const float *) w_inject->data, (float *) dst->data,
        n_embd, hc, s_pre, s_post);
}

template <bool silu>
static __global__ void scale_unary_kernel(const float * __restrict__ src, float * __restrict__ dst,
                                          const float scale, const float bias, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float x = scale*src[i] + bias; // same expression as scale_f32 (contracts the same way)
    dst[i] = silu ? ggml_cuda_op_silu_single(x) : hc_sigmoid(x);
}

void ggml_cuda_op_scale_unary(ggml_backend_cuda_context & ctx,
                              const ggml_tensor * src,
                              float               scale,
                              float               bias,
                              ggml_unary_op       op,
                              ggml_tensor *       dst) {
    GGML_ASSERT(src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(dst));

    const int64_t n     = ggml_nelements(dst);
    const int     block = 256;
    const int     grid  = (int) ((n + block - 1) / block);

    switch (op) {
        case GGML_UNARY_OP_SILU:
            scale_unary_kernel<true><<<grid, block, 0, ctx.stream()>>>((const float *) src->data, (float *) dst->data, scale, bias, n);
            break;
        case GGML_UNARY_OP_SIGMOID:
            scale_unary_kernel<false><<<grid, block, 0, ctx.stream()>>>((const float *) src->data, (float *) dst->data, scale, bias, n);
            break;
        default:
            GGML_ABORT("scale_unary: unsupported unary op");
    }
}
