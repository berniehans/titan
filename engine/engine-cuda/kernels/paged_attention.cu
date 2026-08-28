// Port of vLLM PagedAttention decode kernel (csrc/pos_encoding_kernels.cu, paged attention v1/v2, Apache-2.0)
//
// PagedAttention decode CUDA kernel for single-pass scaled dot-product attention
// over a paged key-value cache pool.
//
// Dynamic shared memory layout:
//   s_acc:  head_dim floats (accumulated weighted values)
//   s_part: 32 floats (thread partial dot products / reduction scratch / broadcast)
//   s_m:    1 float (online softmax running maximum)
//   s_l:    1 float (online softmax running normalizer sum)
//   Total smem: (head_dim + 34) * sizeof(float)

extern "C" __global__ void paged_attention_decode_kernel(
    const float* __restrict__ q,
    const float* __restrict__ pool,
    const unsigned* __restrict__ block_table,
    float* __restrict__ out,
    int n_head,
    int n_head_kv,
    int head_dim,
    int block_tokens,
    int seq_tokens,
    int query_pos,
    int causal,
    float scale,
    const unsigned int* __restrict__ pos_ptr)
{
    if (pos_ptr != nullptr) {
        query_pos = (int)*pos_ptr;
        seq_tokens = (int)*pos_ptr + 1;
    }

    const int qh = blockIdx.x;
    const int tid = threadIdx.x;

    const int group = n_head / n_head_kv;
    const int hk = qh / group;

    const size_t row_len = (size_t)n_head_kv * (size_t)head_dim;
    const size_t floats_per_token = 2 * row_len;
    const size_t floats_per_block = (size_t)block_tokens * floats_per_token;

    const float* qrow = q + (size_t)qh * (size_t)head_dim;

    // Cache Q head (128 floats = 32 float4 vectors across 32 threads)
    const float4 q_val = (tid < 32 && (tid * 4 + 3) < head_dim) ? ((const float4*)qrow)[tid] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    float s_m = -__int_as_float(0x7f800000u);
    float s_l = 0.0f;

    const int n_blocks = (seq_tokens + block_tokens - 1) / block_tokens;

    for (int b = 0; b < n_blocks; ++b) {
        const int in_block = (seq_tokens - b * block_tokens < block_tokens)
                             ? (seq_tokens - b * block_tokens)
                             : block_tokens;
        int valid = in_block;
        if (causal) {
            const int first_invalid = query_pos + 1 - b * block_tokens;
            if (in_block > first_invalid) {
                valid = (first_invalid > 0 ? first_invalid : 0);
            }
        }
        if (valid <= 0) continue;

        const unsigned phys = block_table[b];
        const float* krow = pool + (size_t)phys * floats_per_block + (size_t)hk * (size_t)head_dim;
        const float* vrow = krow + row_len;

        for (int t = 0; t < valid; ++t) {
            const float* k = krow + (size_t)t * floats_per_token;
            const float* v = vrow + (size_t)t * floats_per_token;

            const float4 k_val = ((const float4*)k)[tid];
            const float4 v_val = ((const float4*)v)[tid];

            float part = __fmaf_rn(q_val.x, k_val.x, __fmaf_rn(q_val.y, k_val.y, __fmaf_rn(q_val.z, k_val.z, q_val.w * k_val.w)));

            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                part += __shfl_down_sync(0xffffffff, part, mask);
            }

            const float total_score = __shfl_sync(0xffffffff, part, 0) * scale;
            const float m_new = fmaxf(s_m, total_score);
            const float corr = __expf(s_m - m_new);
            const float p_new = __expf(total_score - m_new);

            s_l = __fmaf_rn(s_l, corr, p_new);
            s_m = m_new;

            acc0 = __fmaf_rn(acc0, corr, v_val.x * p_new);
            acc1 = __fmaf_rn(acc1, corr, v_val.y * p_new);
            acc2 = __fmaf_rn(acc2, corr, v_val.z * p_new);
            acc3 = __fmaf_rn(acc3, corr, v_val.w * p_new);
        }
    }

    float* out_head = out + (size_t)qh * (size_t)head_dim;
    const float inv_l = 1.0f / s_l;

    ((float4*)out_head)[tid] = make_float4(acc0 * inv_l, acc1 * inv_l, acc2 * inv_l, acc3 * inv_l);
}
