#include "common.cuh"
#include "mma.cuh"
#include "mmid.cuh"
#include "mmid-hmma.cuh"

#include <cstdlib>

using namespace ggml_cuda_mma;

#define MMID_HMMA_K        32                  // k elements per pipeline step (one dequantized group per row)
#define MMID_HMMA_LD       (MMID_HMMA_K + 8)   // shared memory row stride of the B tile in halfs

// ---------------------------------------------------------------------------------------------
// dequantization of 32 consecutive elements (k0 % 32 == 0) of one row into 16 half2

// 4 bytes holding small unsigned ints b0..b3 -> (b0, b1) and (b2, b3) as half2 values 1024 + b
static __device__ __forceinline__ half2 mmid_u32_as_half2(const uint32_t v) {
    half2 h;
    memcpy(&h, &v, sizeof(h));
    return h;
}
static __device__ __forceinline__ uint32_t mmid_half2_as_u32(const half2 h) {
    uint32_t v;
    memcpy(&v, &h, sizeof(v));
    return v;
}
static __device__ __forceinline__ half2 mmid_u8x2_to_half2_lo(const uint32_t w) {
    return mmid_u32_as_half2(__byte_perm(w, 0x64646464, 0x4140));
}
static __device__ __forceinline__ half2 mmid_u8x2_to_half2_hi(const uint32_t w) {
    return mmid_u32_as_half2(__byte_perm(w, 0x64646464, 0x4342));
}

// v = (b - bias) * d + m for 4 bytes b of w (bias is exact in half: 1024 + integer offset); returns the
// 4 halfs packed as (lo pair, hi pair)
static __device__ __forceinline__ uint2 mmid_dq4(const uint32_t w, const half2 bias, const half2 d, const half2 m) {
    const half2 t0 = __hsub2(mmid_u8x2_to_half2_lo(w), bias);
    const half2 t1 = __hsub2(mmid_u8x2_to_half2_hi(w), bias);
    return make_uint2(mmid_half2_as_u32(__hfma2(t0, d, m)), mmid_half2_as_u32(__hfma2(t1, d, m)));
}
static __device__ __forceinline__ uint4 mmid_pack8(const uint2 a, const uint2 b) {
    return make_uint4(a.x, a.y, b.x, b.y);
}

// Global loads through inline asm: keeps the prefetch loads where they are written and their results in
// the registers that consume them (ptxas otherwise hoists the load and inserts a copy that waits on it).
static __device__ __forceinline__ uint32_t mmid_ldg_u32(const void * p) {
    uint32_t v;
    asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
static __device__ __forceinline__ uint2 mmid_ldg_u64(const void * p) {
    uint2 v;
    asm volatile("ld.global.nc.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}
static __device__ __forceinline__ uint4 mmid_ldg_u128(const void * p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

// spread bits i..i+3 of qh to the byte lanes (as value 16 for a set bit)
static __device__ __forceinline__ uint32_t mmid_qh4(const uint32_t qh, const int shift) {
    const uint32_t b = qh >> shift;
    return ((b & 1) << 4) | ((b & 2) << 11) | ((b & 4) << 18) | ((b & 8) << 25);
}

// Raw quantized data of 64 consecutive elements (k0 % 64 == 0) of one row, loaded into registers with as
// few (wide) loads as the block layout allows, two pipeline steps ahead of the dequantization. store_lo/hi
// dequantize elements [k0, k0+32) / [k0+32, k0+64) into 4 uint4 (16 half2, k order).

template <ggml_type type>
struct mmid_raw;

// 32-element (block size 32) formats: two consecutive blocks per load set
template <>
struct mmid_raw<GGML_TYPE_Q8_0> { // 2 x 34 bytes, 4-byte aligned (68*ib)
    uint32_t w[17];
    __device__ __forceinline__ void load(const char * __restrict__ row, const int k0) {
        const char * blk = row + (k0/QK8_0)*sizeof(block_q8_0);
#pragma unroll
        for (int i = 0; i < 17; ++i) {
            w[i] = mmid_ldg_u32(blk + 4*i);
        }
    }
    // block b (0/1): d = low/high half of the 34-byte block start; qs follow
    template <int b>
    __device__ __forceinline__ void store(uint4 * __restrict__ o4) const {
        // block 0 starts at byte 0: d = w[0] & 0xFFFF, qs bytes 2..33 -> words: (w[0] >> 16 | w[1] << 16), ...
        // block 1 starts at byte 34: d = w[8] >> 16, qs bytes 36..67 -> words w[9..16]
        uint32_t d16;
        uint32_t q[8];
        if constexpr (b == 0) {
            d16 = w[0] & 0xFFFFu;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                q[i] = __funnelshift_r(w[i], w[i + 1], 16);
            }
        } else {
            d16 = w[8] >> 16;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                q[i] = w[9 + i];
            }
        }
        const half2 dd   = __half2half2(__ushort_as_half((unsigned short) d16));
        const half2 bias = __float2half2_rn(1024.0f + 128.0f);
        const half2 zero = __float2half2_rn(0.0f);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            o4[i] = mmid_pack8(mmid_dq4(q[2*i] ^ 0x80808080u, bias, dd, zero), mmid_dq4(q[2*i + 1] ^ 0x80808080u, bias, dd, zero));
        }
    }
    __device__ __forceinline__ void store_lo(uint4 * __restrict__ o4) const { store<0>(o4); }
    __device__ __forceinline__ void store_hi(uint4 * __restrict__ o4) const { store<1>(o4); }
};

template <>
struct mmid_raw<GGML_TYPE_Q4_0> { // 2 x 18 bytes, 4-byte aligned (36*ib)
    uint32_t w[9];
    __device__ __forceinline__ void load(const char * __restrict__ row, const int k0) {
        const char * blk = row + (k0/QK4_0)*sizeof(block_q4_0);
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            w[i] = mmid_ldg_u32(blk + 4*i);
        }
    }
    template <int b>
    __device__ __forceinline__ void store(uint4 * __restrict__ o4) const {
        // block 0: d = w[0] & 0xFFFF, qs bytes 2..17; block 1 starts at byte 18: d = w[4] >> 16, qs bytes 20..35 = w[5..8]
        uint32_t d16;
        uint32_t q[4];
        if constexpr (b == 0) {
            d16 = w[0] & 0xFFFFu;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                q[i] = __funnelshift_r(w[i], w[i + 1], 16);
            }
        } else {
            d16 = w[4] >> 16;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                q[i] = w[5 + i];
            }
        }
        const half2 dd   = __half2half2(__ushort_as_half((unsigned short) d16));
        const half2 bias = __float2half2_rn(1024.0f + 8.0f);
        const half2 zero = __float2half2_rn(0.0f);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            o4[i]     = mmid_pack8(mmid_dq4( q[2*i]           & 0x0F0F0F0Fu, bias, dd, zero), mmid_dq4( q[2*i + 1]       & 0x0F0F0F0Fu, bias, dd, zero));
            o4[i + 2] = mmid_pack8(mmid_dq4((q[2*i]     >> 4) & 0x0F0F0F0Fu, bias, dd, zero), mmid_dq4((q[2*i + 1] >> 4) & 0x0F0F0F0Fu, bias, dd, zero));
        }
    }
    __device__ __forceinline__ void store_lo(uint4 * __restrict__ o4) const { store<0>(o4); }
    __device__ __forceinline__ void store_hi(uint4 * __restrict__ o4) const { store<1>(o4); }
};

template <>
struct mmid_raw<GGML_TYPE_Q4_1> { // 2 x 20 bytes, 8-byte aligned (40*ib)
    uint2 w[5];
    __device__ __forceinline__ void load(const char * __restrict__ row, const int k0) {
        const char * blk = row + (k0/QK4_1)*sizeof(block_q4_1);
#pragma unroll
        for (int i = 0; i < 5; ++i) {
            w[i] = mmid_ldg_u64(blk + 8*i);
        }
    }
    template <int b>
    __device__ __forceinline__ void store(uint4 * __restrict__ o4) const {
        // words: block 0 = [dm, q0, q1, q2, q3], block 1 = [dm, q0, q1, q2, q3] starting at word 5
        uint32_t dm;
        uint32_t q[4];
        if constexpr (b == 0) {
            dm = w[0].x; q[0] = w[0].y; q[1] = w[1].x; q[2] = w[1].y; q[3] = w[2].x;
        } else {
            dm = w[2].y; q[0] = w[3].x; q[1] = w[3].y; q[2] = w[4].x; q[3] = w[4].y;
        }
        const half2 h    = mmid_u32_as_half2(dm);
        const half2 dd   = __low2half2(h);
        const half2 mm   = __high2half2(h);
        const half2 bias = __float2half2_rn(1024.0f);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            o4[i]     = mmid_pack8(mmid_dq4( q[2*i]           & 0x0F0F0F0Fu, bias, dd, mm), mmid_dq4( q[2*i + 1]       & 0x0F0F0F0Fu, bias, dd, mm));
            o4[i + 2] = mmid_pack8(mmid_dq4((q[2*i]     >> 4) & 0x0F0F0F0Fu, bias, dd, mm), mmid_dq4((q[2*i + 1] >> 4) & 0x0F0F0F0Fu, bias, dd, mm));
        }
    }
    __device__ __forceinline__ void store_lo(uint4 * __restrict__ o4) const { store<0>(o4); }
    __device__ __forceinline__ void store_hi(uint4 * __restrict__ o4) const { store<1>(o4); }
};

template <>
struct mmid_raw<GGML_TYPE_Q5_0> { // 2 x 22 bytes, 4-byte aligned (44*ib)
    uint32_t w[11];
    __device__ __forceinline__ void load(const char * __restrict__ row, const int k0) {
        const char * blk = row + (k0/QK5_0)*sizeof(block_q5_0);
#pragma unroll
        for (int i = 0; i < 11; ++i) {
            w[i] = mmid_ldg_u32(blk + 4*i);
        }
    }
    template <int b>
    __device__ __forceinline__ void store(uint4 * __restrict__ o4) const {
        // block 0: d = bytes 0-1, qh = bytes 2-5, qs = bytes 6-21; block 1 (byte 22): d = w[5] >> 16, qh = w[6], qs = w[7..10]
        uint32_t d16, qh;
        uint32_t q[4];
        if constexpr (b == 0) {
            d16 = w[0] & 0xFFFFu;
            qh  = __funnelshift_r(w[0], w[1], 16);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                q[i] = __funnelshift_r(w[i + 1], w[i + 2], 16);
            }
        } else {
            d16 = w[5] >> 16;
            qh  = w[6];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                q[i] = w[7 + i];
            }
        }
        const half2 dd   = __half2half2(__ushort_as_half((unsigned short) d16));
        const half2 bias = __float2half2_rn(1024.0f + 16.0f);
        const half2 zero = __float2half2_rn(0.0f);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            o4[i]     = mmid_pack8(mmid_dq4(( q[2*i]           & 0x0F0F0F0Fu) | mmid_qh4(qh,      8*i), bias, dd, zero), mmid_dq4(( q[2*i + 1]       & 0x0F0F0F0Fu) | mmid_qh4(qh,      8*i + 4), bias, dd, zero));
            o4[i + 2] = mmid_pack8(mmid_dq4(((q[2*i]     >> 4) & 0x0F0F0F0Fu) | mmid_qh4(qh, 16 + 8*i), bias, dd, zero), mmid_dq4(((q[2*i + 1] >> 4) & 0x0F0F0F0Fu) | mmid_qh4(qh, 16 + 8*i + 4), bias, dd, zero));
        }
    }
    __device__ __forceinline__ void store_lo(uint4 * __restrict__ o4) const { store<0>(o4); }
    __device__ __forceinline__ void store_hi(uint4 * __restrict__ o4) const { store<1>(o4); }
};

template <>
struct mmid_raw<GGML_TYPE_Q5_1> { // 2 x 24 bytes, 16-byte aligned (48*ib)
    uint4 w[3];
    __device__ __forceinline__ void load(const char * __restrict__ row, const int k0) {
        const char * blk = row + (k0/QK5_1)*sizeof(block_q5_1);
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            w[i] = mmid_ldg_u128(blk + 16*i);
        }
    }
    template <int b>
    __device__ __forceinline__ void store(uint4 * __restrict__ o4) const {
        // words: block 0 = [dm, qh, q0..q3], block 1 = [dm, qh, q0..q3] from word 6
        uint32_t dm, qh;
        uint32_t q[4];
        if constexpr (b == 0) {
            dm = w[0].x; qh = w[0].y; q[0] = w[0].z; q[1] = w[0].w; q[2] = w[1].x; q[3] = w[1].y;
        } else {
            dm = w[1].z; qh = w[1].w; q[0] = w[2].x; q[1] = w[2].y; q[2] = w[2].z; q[3] = w[2].w;
        }
        const half2 h    = mmid_u32_as_half2(dm);
        const half2 dd   = __low2half2(h);
        const half2 mm   = __high2half2(h);
        const half2 bias = __float2half2_rn(1024.0f);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            o4[i]     = mmid_pack8(mmid_dq4(( q[2*i]           & 0x0F0F0F0Fu) | mmid_qh4(qh,      8*i), bias, dd, mm), mmid_dq4(( q[2*i + 1]       & 0x0F0F0F0Fu) | mmid_qh4(qh,      8*i + 4), bias, dd, mm));
            o4[i + 2] = mmid_pack8(mmid_dq4(((q[2*i]     >> 4) & 0x0F0F0F0Fu) | mmid_qh4(qh, 16 + 8*i), bias, dd, mm), mmid_dq4(((q[2*i + 1] >> 4) & 0x0F0F0F0Fu) | mmid_qh4(qh, 16 + 8*i + 4), bias, dd, mm));
        }
    }
    __device__ __forceinline__ void store_lo(uint4 * __restrict__ o4) const { store<0>(o4); }
    __device__ __forceinline__ void store_hi(uint4 * __restrict__ o4) const { store<1>(o4); }
};

// 6-bit scale/min of sub-block j from the 12 packed bytes held in 3 words (no array indexing: the
// words stay in registers)
static __device__ __forceinline__ void mmid_scale_min_k4(const int j, const uint32_t w0, const uint32_t w1, const uint32_t w2, int & d, int & m) {
    if (j < 4) {
        const int sh = 8*j;
        d = (w0 >> sh) & 63;
        m = (w1 >> sh) & 63;
    } else {
        const int sh = 8*(j - 4);
        const int b0 = (w0 >> sh) & 0xFF; // scales[j-4]
        const int b1 = (w1 >> sh) & 0xFF; // scales[j]
        const int b2 = (w2 >> sh) & 0xFF; // scales[j+4]
        d = (b2 & 0xF) | ((b0 >> 6) << 4);
        m = (b2 >>  4) | ((b1 >> 6) << 4);
    }
}

// K-quants: 64 consecutive elements = the low and high nibbles of the same 32 bytes
template <>
struct mmid_raw<GGML_TYPE_Q4_K> {
    uint4 dmsc;   // dm + 12 scale bytes
    uint4 q0, q1; // 32 bytes of nibbles
    int   e0;
    __device__ __forceinline__ void load(const char * __restrict__ row, const int k0) {
        const char * blk = row + (k0/QK_K)*sizeof(block_q4_K);
        e0   = k0 % QK_K;
        dmsc = mmid_ldg_u128(blk);
        q0   = mmid_ldg_u128(blk + 16 + 32*(e0/64));
        q1   = mmid_ldg_u128(blk + 32 + 32*(e0/64));
    }
    template <int hi>
    __device__ __forceinline__ void store(uint4 * __restrict__ o4) const {
        const half2 h    = mmid_u32_as_half2(dmsc.x);
        const float dall = __low2float(h);
        const float dmin = __high2float(h);
        int s, m;
        mmid_scale_min_k4(e0/32 + hi, dmsc.y, dmsc.z, dmsc.w, s, m);
        const half2 dd   = __float2half2_rn(dall*s);
        const half2 mm   = __float2half2_rn(-dmin*m);
        const half2 bias = __float2half2_rn(1024.0f);
        constexpr int sh = hi ? 4 : 0;
        o4[0] = mmid_pack8(mmid_dq4((q0.x >> sh) & 0x0F0F0F0Fu, bias, dd, mm), mmid_dq4((q0.y >> sh) & 0x0F0F0F0Fu, bias, dd, mm));
        o4[1] = mmid_pack8(mmid_dq4((q0.z >> sh) & 0x0F0F0F0Fu, bias, dd, mm), mmid_dq4((q0.w >> sh) & 0x0F0F0F0Fu, bias, dd, mm));
        o4[2] = mmid_pack8(mmid_dq4((q1.x >> sh) & 0x0F0F0F0Fu, bias, dd, mm), mmid_dq4((q1.y >> sh) & 0x0F0F0F0Fu, bias, dd, mm));
        o4[3] = mmid_pack8(mmid_dq4((q1.z >> sh) & 0x0F0F0F0Fu, bias, dd, mm), mmid_dq4((q1.w >> sh) & 0x0F0F0F0Fu, bias, dd, mm));
    }
    __device__ __forceinline__ void store_lo(uint4 * __restrict__ o4) const { store<0>(o4); }
    __device__ __forceinline__ void store_hi(uint4 * __restrict__ o4) const { store<1>(o4); }
};

template <>
struct mmid_raw<GGML_TYPE_Q5_K> {
    uint4 dmsc;
    uint4 h0, h1; // 32 bytes of high bits (whole super-block)
    uint4 q0, q1;
    int   e0;
    __device__ __forceinline__ void load(const char * __restrict__ row, const int k0) {
        const char * blk = row + (k0/QK_K)*sizeof(block_q5_K);
        e0   = k0 % QK_K;
        dmsc = mmid_ldg_u128(blk);
        h0   = mmid_ldg_u128(blk + 16);
        h1   = mmid_ldg_u128(blk + 32);
        q0   = mmid_ldg_u128(blk + 48 + 32*(e0/64));
        q1   = mmid_ldg_u128(blk + 64 + 32*(e0/64));
    }
    template <int hi>
    __device__ __forceinline__ void store(uint4 * __restrict__ o4) const {
        const half2 h    = mmid_u32_as_half2(dmsc.x);
        const float dall = __low2float(h);
        const float dmin = __high2float(h);
        int s, m;
        mmid_scale_min_k4(e0/32 + hi, dmsc.y, dmsc.z, dmsc.w, s, m);
        const half2 dd   = __float2half2_rn(dall*s);
        const half2 mm   = __float2half2_rn(-dmin*m);
        const half2 bias = __float2half2_rn(1024.0f);
        constexpr int sh = hi ? 4 : 0;
        const int hs = 2*(e0/64) + hi;
#define MMID_Q5K(w, x) mmid_dq4((((w) >> sh) & 0x0F0F0F0Fu) | ((((x) >> hs) & 0x01010101u) << 4), bias, dd, mm)
        o4[0] = mmid_pack8(MMID_Q5K(q0.x, h0.x), MMID_Q5K(q0.y, h0.y));
        o4[1] = mmid_pack8(MMID_Q5K(q0.z, h0.z), MMID_Q5K(q0.w, h0.w));
        o4[2] = mmid_pack8(MMID_Q5K(q1.x, h1.x), MMID_Q5K(q1.y, h1.y));
        o4[3] = mmid_pack8(MMID_Q5K(q1.z, h1.z), MMID_Q5K(q1.w, h1.w));
#undef MMID_Q5K
    }
    __device__ __forceinline__ void store_lo(uint4 * __restrict__ o4) const { store<0>(o4); }
    __device__ __forceinline__ void store_hi(uint4 * __restrict__ o4) const { store<1>(o4); }
};

// ---------------------------------------------------------------------------------------------

// src1 rows (F32) -> F16 in compact expert order: y[c] = src1[ids_src1[c]]
static __global__ void mmid_hmma_gather_y(const float * __restrict__ src, const int32_t * __restrict__ ids_src1, half * __restrict__ dst,
                                          const int64_t ne10, const int64_t s11, const int64_t nrows) {
    const int64_t i  = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t n8 = ne10/8;
    if (i >= nrows*n8) {
        return;
    }
    const int64_t c = i / n8;
    const int64_t k = (i - c*n8)*8;
    const int64_t r = ids_src1[c];
    const float4 a = *(const float4 *) (src + r*s11 + k);
    const float4 b = *(const float4 *) (src + r*s11 + k + 4);
    uint4 o;
    o.x = mmid_half2_as_u32(__floats2half2_rn(a.x, a.y));
    o.y = mmid_half2_as_u32(__floats2half2_rn(a.z, a.w));
    o.z = mmid_half2_as_u32(__floats2half2_rn(b.x, b.y));
    o.w = mmid_half2_as_u32(__floats2half2_rn(b.z, b.w));
    *(uint4 *) (dst + c*ne10 + k) = o;
}

#define MMID_HMMA_WARPS_PER_BLOCK 4
#define MMID_HMMA_NTHREADS        (MMID_HMMA_WARPS_PER_BLOCK*WARP_SIZE)
#define MMID_HMMA_RW              32                                     // rows per warp: one per lane
#define MMID_HMMA_ROWS            (MMID_HMMA_WARPS_PER_BLOCK*MMID_HMMA_RW) // rows of the expert matrix per block
#define MMID_HMMA_MAX_EXPERTS     4096                                   // expert_bounds is staged in shared memory

typedef tile<32, 4, half2>                               mmid_tile_A; // 32 rows x 8 k: lane -> row, x[l] = k 2l, 2l+1
typedef tile< 8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> mmid_tile_B; //  8 cols x 8 k: x[l] = k 2l, 2l+1 of col get_i()
typedef tile<32, 8, float>                               mmid_tile_C; // 32 rows x 8 cols

// B (activation) raw data of one step: the tile is N_TILE tokens x 32 k halfs = N_TILE*4 uint4
template <int N_TILE>
struct mmid_braw {
    static constexpr int NL = (N_TILE*4 + MMID_HMMA_NTHREADS - 1)/MMID_HMMA_NTHREADS; // 1 for N_TILE <= 32
    uint4 v0, v1;

    // rows beyond the expert's tokens read the (allocated) slack rows: their columns are never stored
    __device__ __forceinline__ void load(const half * __restrict__ yb, const int K, const int tid, const int kp) {
        {
            const int i = tid;
            v0 = mmid_ldg_u128(yb + (int64_t) (i/4)*K + kp + (i % 4)*8);
        }
        if constexpr (NL > 1) {
            const int i = tid + MMID_HMMA_NTHREADS;
            v1 = mmid_ldg_u128(yb + (int64_t) (i/4)*K + kp + (i % 4)*8);
        }
    }
    __device__ __forceinline__ void store(half * __restrict__ cB, const int tid) const {
        constexpr int LD = MMID_HMMA_LD;
        {
            const int i = tid;
            if (i < N_TILE*4) { *(uint4 *) (cB + (i/4)*LD + (i % 4)*8) = v0; }
        }
        if constexpr (NL > 1) {
            const int i = tid + MMID_HMMA_NTHREADS;
            if (i < N_TILE*4) { *(uint4 *) (cB + (i/4)*LD + (i % 4)*8) = v1; }
        }
    }
};

template <int N_TILE>
struct mmid_hmma_cfg {
    static constexpr int B_ELEMS    = N_TILE*MMID_HMMA_LD;
    static constexpr int SMEM_TILES = 2*B_ELEMS*sizeof(half);
    static constexpr int OCC        = 3;
};

// One block computes 128 rows x N_TILE columns of one expert with Volta mma.m8n8k4. Each lane owns one
// row: the 32 weights of its row for a step are dequantized straight into the A tiles (no shared memory
// round trip). B (activations, F16, compact expert order) is staged through a double-buffered shared tile.
// The raw global loads of A and B run 2-3 steps ahead in explicit register sets (inline-asm loads so
// ptxas keeps them in place), the first loads of the next work item are issued before the epilogue of
// the current one, one barrier per step.
template <ggml_type type, int N_TILE>
__launch_bounds__(MMID_HMMA_NTHREADS, mmid_hmma_cfg<N_TILE>::OCC)
static __global__ void mmid_hmma_kernel(
        const char    * __restrict__ x,             // expert weights [K, M, n_expert]
        const half    * __restrict__ y,             // activations F16, compact expert order [K, ncompact (+pad)]
        const int32_t * __restrict__ ids_dst,       // compact row -> dst row
        const int32_t * __restrict__ expert_bounds, // [n_expert + 1]
        float         * __restrict__ dst,
        const int K, const int M, const int n_expert, const int ntx, const int nty,
        const int64_t stride_row_x, const int64_t stride_expert_x, const int64_t stride_row_dst) {
    using cfg = mmid_hmma_cfg<N_TILE>;
    constexpr int LD = MMID_HMMA_LD;
    constexpr int NT = N_TILE/8; // B/C tiles per step

    extern __shared__ __align__(128) char mmid_smem[];
    half * sB       = (half *) mmid_smem;                       // 2 x [N_TILE][LD]
    int  * s_bounds = (int *) (sB + 2*cfg::B_ELEMS);            // [n_expert + 1]

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp*WARP_SIZE + lane;

    for (int i = tid; i < n_expert + 1; i += MMID_HMMA_NTHREADS) {
        s_bounds[i] = expert_bounds[i];
    }
    __syncthreads();

    const int a_r = warp*MMID_HMMA_RW + lane; // row within the block

    // per-lane B tile source offsets (in halfs) within the shared B tile: token get_i(), k-slice s
    const int b_i = mmid_tile_B::get_i(0);

    const long n_items = (long) nty*n_expert*ntx;

    // work item -> (expert, column tile, row tile) and the per-lane source pointers; rows beyond M are
    // clamped to a valid row and never stored
    auto decode = [&](const long item, int & e, int & jt, int & it, int & col_low, int & ncols, const char *& xrow, const half *& yb) -> bool {
        it = (int) (item % nty);
        const long t2 = item / nty;
        e  = (int) (t2 % n_expert);
        jt = (int) (t2 / n_expert);
        col_low = s_bounds[e];
        const int col_diff = s_bounds[e + 1] - col_low;
        if (jt*N_TILE >= col_diff) {
            return false;
        }
        ncols = min(N_TILE, col_diff - jt*N_TILE);
        const int row = it*MMID_HMMA_ROWS + a_r;
        xrow = x + (int64_t) e*stride_expert_x + (int64_t) (row < M ? row : M - 1)*stride_row_x;
        yb   = y + (int64_t) (col_low + jt*N_TILE)*K;
        return true;
    };

    // the raw data of the first two steps of an item is loaded before the previous item's epilogue
    mmid_raw<type>     raw0, raw1;
    mmid_braw<N_TILE>  braw0, braw1;

    long item = blockIdx.x;
    int e, jt, it, col_low, ncols;
    const char * xrow;
    const half * yb;
    while (item < n_items && !decode(item, e, jt, it, col_low, ncols, xrow, yb)) {
        item += gridDim.x;
    }
    if (item < n_items) {
        raw0.load(xrow, 0);
        raw1.load(xrow, 2*MMID_HMMA_K); // K >= 4*MMID_HMMA_K is checked on the host
        braw0.load(yb, K, tid, 0);
        braw1.load(yb, K, tid, MMID_HMMA_K);
    }

    while (item < n_items) {
        const int  row0 = it*MMID_HMMA_ROWS;

        __syncthreads(); // the previous item may still read the B tiles

        mmid_tile_C C[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int l = 0; l < mmid_tile_C::ne; ++l) {
                C[t].x[l] = 0.0f;
            }
        }

        // two explicit register sets for the raw data of steps 2i and 2i+1
        // raw A data covers 2 steps (64 k); raw0 serves steps 4i, 4i+1 and raw1 steps 4i+2, 4i+3, each
        // refilled right after its second use (3 steps ahead). B raw data alternates per step (2 ahead).
#define MMID_HMMA_STEP(RAW, HI, BRAW, CB, K0)                                                                 \
        do {                                                                                                  \
            mmid_tile_A A[4];                                                                                 \
            if (HI) {                                                                                         \
                RAW.store_hi((uint4 *) A);                                                                    \
            } else {                                                                                          \
                RAW.store_lo((uint4 *) A);                                                                    \
            }                                                                                                 \
            BRAW.store(CB, tid);                                                                              \
            {                                                                                                 \
                const int kp = (K0) + 2*MMID_HMMA_K;                                                          \
                if (kp < K) {                                                                                 \
                    BRAW.load(yb, K, tid, kp);                                                                \
                }                                                                                             \
                if (HI && kp + MMID_HMMA_K < K) {                                                             \
                    RAW.load(xrow, kp + MMID_HMMA_K);                                                         \
                }                                                                                             \
            }                                                                                                 \
            __syncthreads();                                                                                  \
            _Pragma("unroll")                                                                                 \
            for (int s = 0; s < 4; ++s) {                                                                     \
                _Pragma("unroll")                                                                             \
                for (int t = 0; t < NT; ++t) {                                                                \
                    mmid_tile_B B;                                                                            \
                    *(uint4 *) B.x = *(const uint4 *) (CB + (t*8 + b_i)*LD + 8*s);                            \
                    mma(C[t], A[s], B);                                                                       \
                }                                                                                             \
            }                                                                                                 \
        } while (0)

        const int nsteps = K/MMID_HMMA_K; // even
        for (int step = 0; step < nsteps; step += 4) {
            MMID_HMMA_STEP(raw0, false, braw0, sB,                  (step + 0)*MMID_HMMA_K);
            MMID_HMMA_STEP(raw0, true,  braw1, (sB + cfg::B_ELEMS), (step + 1)*MMID_HMMA_K);
            if (step + 2 < nsteps) {
                MMID_HMMA_STEP(raw1, false, braw0, sB,                  (step + 2)*MMID_HMMA_K);
                MMID_HMMA_STEP(raw1, true,  braw1, (sB + cfg::B_ELEMS), (step + 3)*MMID_HMMA_K);
            }
        }
#undef MMID_HMMA_STEP

        // next item: issue its first loads now so they overlap with the epilogue
        const int32_t * ids_dst_tile = ids_dst + col_low + jt*N_TILE;
        const int ncols_cur = ncols;
        {
            item += gridDim.x;
            while (item < n_items && !decode(item, e, jt, it, col_low, ncols, xrow, yb)) {
                item += gridDim.x;
            }
            if (item < n_items) {
                raw0.load(xrow, 0);
                raw1.load(xrow, 2*MMID_HMMA_K);
                braw0.load(yb, K, tid, 0);
                braw1.load(yb, K, tid, MMID_HMMA_K);
            }
        }

        // epilogue: C[t].x[l] -> dst[ids_dst[col]][row]
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int l = 0; l < mmid_tile_C::ne; ++l) {
                const int row = row0 + warp*MMID_HMMA_RW + mmid_tile_C::get_i(l);
                const int col = t*8 + mmid_tile_C::get_j(l);
                if (row < M && col < ncols_cur) {
                    dst[(int64_t) ids_dst_tile[col]*stride_row_dst + row] = C[t].x[l];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------

// Column tile width from the average number of tokens per expert (n_tokens*n_expert_used/n_expert).
// Measured on V100 with Qwen3.5-style MoE (10 of 512 experts): below ~2 tokens per expert the dp4a MMQ
// kernel is as fast or faster (returns 0), N=32 wins up to ~16 tokens per expert, N=64 above.
static int mmid_hmma_n_tile(const int64_t n_tokens, const int64_t n_expert_used, const int64_t n_expert) {
    static const int env = [] {
        const char * s = getenv("GGML_MMID_HMMA_N"); // debug override
        return s ? atoi(s) : 0;
    }();
    if (env == 16 || env == 32 || env == 64) {
        return env;
    }
    const int64_t tpe2 = 2*n_tokens*n_expert_used/n_expert; // 2x tokens per expert
    if (tpe2 < 4) {
        return 0;
    }
    return tpe2 < 32 ? 32 : 64;
}

bool ggml_cuda_mul_mat_id_hmma_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst, const int cc) {
    static const bool disabled = [] {
        const char * s = getenv("GGML_CUDA_MMID_HMMA");
        return s && atoi(s) == 0;
    }();
    if (disabled) {
        return false;
    }
    if (!volta_mma_available(cc) || turing_mma_available(cc)) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        return false;
    }
    switch (src0->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            break;
        default:
            return false;
    }
    if (src0->ne[0] % (2*MMID_HMMA_K) != 0 || src0->ne[0] < 4*MMID_HMMA_K || src0->ne[3] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->ne[2] > MMID_HMMA_MAX_EXPERTS) {
        return false;
    }
    if (src1->nb[0] != sizeof(float) || src1->nb[1] % 16 != 0 || dst->nb[0] != sizeof(float)) {
        return false;
    }
    if (src1->nb[2] % src1->nb[1] != 0 || dst->nb[2] % dst->nb[1] != 0) {
        return false;
    }
    if (ids->nb[0] != sizeof(int32_t)) {
        return false;
    }
    if (mmid_hmma_n_tile(src1->ne[2], ids->ne[0], src0->ne[2]) == 0) {
        return false;
    }
    return true;
}

template <ggml_type type, int N_TILE>
static void mmid_hmma_launch_n(const int grid, const int smem, cudaStream_t stream,
        const char * x, const half * y, const int32_t * ids_dst, const int32_t * expert_bounds, float * dst,
        const int K, const int M, const int n_expert, const int ntx, const int nty,
        const int64_t s01, const int64_t s02, const int64_t s1) {
    const int smem_n = mmid_hmma_cfg<N_TILE>::SMEM_TILES + smem; // smem: bounds bytes
    GGML_ASSERT(smem_n <= 48*1024); // no opt-in needed below 48 KiB (n_expert is bounded in the support check)
    mmid_hmma_kernel<type, N_TILE><<<grid, dim3(WARP_SIZE, MMID_HMMA_WARPS_PER_BLOCK), smem_n, stream>>>(x, y, ids_dst, expert_bounds, dst, K, M, n_expert, ntx, nty, s01, s02, s1);
}

template <ggml_type type>
static void mmid_hmma_launch(const int N_TILE, const int grid, const int smem, cudaStream_t stream,
        const char * x, const half * y, const int32_t * ids_dst, const int32_t * expert_bounds, float * dst,
        const int K, const int M, const int n_expert, const int ntx, const int nty,
        const int64_t s01, const int64_t s02, const int64_t s1) {
    switch (N_TILE) {
        case 16: mmid_hmma_launch_n<type, 16>(grid, smem, stream, x, y, ids_dst, expert_bounds, dst, K, M, n_expert, ntx, nty, s01, s02, s1); break;
        case 32: mmid_hmma_launch_n<type, 32>(grid, smem, stream, x, y, ids_dst, expert_bounds, dst, K, M, n_expert, ntx, nty, s01, s02, s1); break;
        case 64: mmid_hmma_launch_n<type, 64>(grid, smem, stream, x, y, ids_dst, expert_bounds, dst, K, M, n_expert, ntx, nty, s01, s02, s1); break;
        default: GGML_ABORT("mmid_hmma: bad N_TILE");
    }
}

void ggml_cuda_mul_mat_id_hmma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const int64_t n_expert      = ne02;
    const int64_t n_tokens      = ne12;
    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows   = n_tokens*n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);
    GGML_ASSERT(ne13 == 1);

    const int64_t K = ne00;
    const int64_t M = ne01;

    const int N_TILE = mmid_hmma_n_tile(n_tokens, n_expert_used, n_expert);
    const int ntx    = (n_tokens + N_TILE - 1)/N_TILE;
    const int nty    = (M + MMID_HMMA_ROWS - 1)/MMID_HMMA_ROWS;

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), n_expert + 1);
    {
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            n_expert, n_tokens, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ false, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    // activations -> F16 in compact expert order (+ N_TILE rows of slack for the last partial tile)
    ggml_cuda_pool_alloc<half> y_f16(ctx.pool(), (ne_get_rows + N_TILE)*K);
    {
        const int64_t n     = ne_get_rows*(K/8);
        const int     block = 256;
        const int64_t grid  = (n + block - 1)/block;
        mmid_hmma_gather_y<<<grid, block, 0, stream>>>((const float *) src1->data, ids_src1.get(), y_f16.get(), K, nb11/sizeof(float), ne_get_rows);
        CUDA_CHECK(cudaGetLastError());
    }

    const int nsm  = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int smem = (n_expert + 1)*sizeof(int); // expert bounds; the tile buffers are added per N_TILE
    const int grid = 3*nsm; // resident blocks; each loops over its share of the (row tile, expert, column tile) items

    const int64_t s01 = nb01;
    const int64_t s02 = nb02;
    const int64_t s1  = nb1/sizeof(float);

    switch (src0->type) {
        case GGML_TYPE_Q4_0: mmid_hmma_launch<GGML_TYPE_Q4_0>(N_TILE, grid, smem, stream, (const char *) src0->data, y_f16.get(), ids_dst.get(), expert_bounds.get(), (float *) dst->data, K, M, n_expert, ntx, nty, s01, s02, s1); break;
        case GGML_TYPE_Q4_1: mmid_hmma_launch<GGML_TYPE_Q4_1>(N_TILE, grid, smem, stream, (const char *) src0->data, y_f16.get(), ids_dst.get(), expert_bounds.get(), (float *) dst->data, K, M, n_expert, ntx, nty, s01, s02, s1); break;
        case GGML_TYPE_Q5_0: mmid_hmma_launch<GGML_TYPE_Q5_0>(N_TILE, grid, smem, stream, (const char *) src0->data, y_f16.get(), ids_dst.get(), expert_bounds.get(), (float *) dst->data, K, M, n_expert, ntx, nty, s01, s02, s1); break;
        case GGML_TYPE_Q5_1: mmid_hmma_launch<GGML_TYPE_Q5_1>(N_TILE, grid, smem, stream, (const char *) src0->data, y_f16.get(), ids_dst.get(), expert_bounds.get(), (float *) dst->data, K, M, n_expert, ntx, nty, s01, s02, s1); break;
        case GGML_TYPE_Q8_0: mmid_hmma_launch<GGML_TYPE_Q8_0>(N_TILE, grid, smem, stream, (const char *) src0->data, y_f16.get(), ids_dst.get(), expert_bounds.get(), (float *) dst->data, K, M, n_expert, ntx, nty, s01, s02, s1); break;
        case GGML_TYPE_Q4_K: mmid_hmma_launch<GGML_TYPE_Q4_K>(N_TILE, grid, smem, stream, (const char *) src0->data, y_f16.get(), ids_dst.get(), expert_bounds.get(), (float *) dst->data, K, M, n_expert, ntx, nty, s01, s02, s1); break;
        case GGML_TYPE_Q5_K: mmid_hmma_launch<GGML_TYPE_Q5_K>(N_TILE, grid, smem, stream, (const char *) src0->data, y_f16.get(), ids_dst.get(), expert_bounds.get(), (float *) dst->data, K, M, n_expert, ntx, nty, s01, s02, s1); break;
        default:
            GGML_ABORT("mmid_hmma: unsupported type");
    }
    CUDA_CHECK(cudaGetLastError());
}
