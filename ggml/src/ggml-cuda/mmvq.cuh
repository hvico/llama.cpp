#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

// MUL_MAT_ID with more than MMVQ_MAX_BATCH_SIZE tokens can still run as MMVQ, in chunks of MMVQ_MAX_BATCH_SIZE tokens,
// when there is (almost) no weight reuse between tokens anyway: with ~1 token per expert the tiled kernels only add
// staging overhead (Volta, 512-expert MoE: 20 tokens 94 ms with MMQ vs ~75 ms as MMVQ chunks). Returns the maximum
// number of tokens for which MMVQ is used for this MUL_MAT_ID.
int get_mmvq_mmid_max_batch_chunked(ggml_type type, int cc, int64_t n_expert, int64_t n_expert_used);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
