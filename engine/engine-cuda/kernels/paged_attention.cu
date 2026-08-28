// Port of vLLM PagedAttention & FlashDecoding Attention Kernels (Apache-2.0).
//
// PagedAttention decode CUDA kernel for single-pass scaled dot-product attention
// over a paged key-value cache pool with Split-KV FlashDecoding acceleration for long contexts.

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

            const float4 k_val = (tid * 4 + 3 < head_dim) ? ((const float4*)k)[tid] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float4 v_val = (tid * 4 + 3 < head_dim) ? ((const float4*)v)[tid] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

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

    if (tid * 4 + 3 < head_dim) {
        ((float4*)out_head)[tid] = make_float4(acc0 * inv_l, acc1 * inv_l, acc2 * inv_l, acc3 * inv_l);
    }
}

// FlashDecoding Split-KV Kernel:
// gridDim.x = n_head
// gridDim.y = num_splits
// blockDim.x = 32 (1 warp per head_dim=128)
extern "C" __global__ void flash_decoding_split_kernel(
    const float* __restrict__ q,
    const float* __restrict__ pool,
    const unsigned* __restrict__ block_table,
    float* __restrict__ partial_acc,  // [n_head, max_splits, head_dim]
    float* __restrict__ partial_m,    // [n_head, max_splits]
    float* __restrict__ partial_l,    // [n_head, max_splits]
    int n_head,
    int n_head_kv,
    int head_dim,
    int block_tokens,
    int seq_tokens,
    int query_pos,
    int causal,
    float scale,
    const unsigned int* __restrict__ pos_ptr,
    int tokens_per_split,
    int max_splits
) {
    if (pos_ptr != nullptr) {
        query_pos = (int)*pos_ptr;
        seq_tokens = (int)*pos_ptr + 1;
    }

    const int qh = blockIdx.x;
    const int split_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (qh >= n_head || split_idx >= max_splits) return;

    const int group = n_head / n_head_kv;
    const int hk = qh / group;

    const size_t row_len = (size_t)n_head_kv * (size_t)head_dim;
    const size_t floats_per_token = 2 * row_len;
    const size_t floats_per_block = (size_t)block_tokens * floats_per_token;

    const float* qrow = q + (size_t)qh * (size_t)head_dim;
    const float4 q_val = (tid < 32 && (tid * 4 + 3) < head_dim) ? ((const float4*)qrow)[tid] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float s_m = -__int_as_float(0x7f800000u);
    float s_l = 0.0f;

    const int token_start = split_idx * tokens_per_split;
    const int token_end = (token_start + tokens_per_split < seq_tokens) ? (token_start + tokens_per_split) : seq_tokens;

    if (token_start < seq_tokens) {
        const int b_start = token_start / block_tokens;
        const int b_end = (token_end + block_tokens - 1) / block_tokens;

        for (int b = b_start; b < b_end; ++b) {
            const int block_token_start = b * block_tokens;
            const int in_block_start = (token_start > block_token_start) ? (token_start - block_token_start) : 0;
            const int in_block_end = (token_end < block_token_start + block_tokens) ? (token_end - block_token_start) : block_tokens;

            int valid_end = in_block_end;
            if (causal) {
                const int first_invalid = query_pos + 1 - block_token_start;
                if (valid_end > first_invalid) {
                    valid_end = (first_invalid > in_block_start ? first_invalid : in_block_start);
                }
            }
            if (valid_end <= in_block_start) continue;

            const unsigned phys = block_table[b];
            const float* krow = pool + (size_t)phys * floats_per_block + (size_t)hk * (size_t)head_dim;
            const float* vrow = krow + row_len;

            for (int t = in_block_start; t < valid_end; ++t) {
                const float* k = krow + (size_t)t * floats_per_token;
                const float* v = vrow + (size_t)t * floats_per_token;

                const float4 k_val = (tid * 4 + 3 < head_dim) ? ((const float4*)k)[tid] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                const float4 v_val = (tid * 4 + 3 < head_dim) ? ((const float4*)v)[tid] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

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
    }

    const size_t split_offset = ((size_t)qh * (size_t)max_splits + (size_t)split_idx);
    float* acc_head_split = partial_acc + split_offset * (size_t)head_dim;
    if (tid * 4 + 3 < head_dim) {
        ((float4*)acc_head_split)[tid] = make_float4(acc0, acc1, acc2, acc3);
    }

    if (tid == 0) {
        partial_m[split_offset] = s_m;
        partial_l[split_offset] = s_l;
    }
}

// FlashDecoding Online Softmax Reduction Kernel across splits:
// gridDim.x = n_head
// blockDim.x = 32 (1 warp per head_dim=128)
extern "C" __global__ void flash_decoding_reduce_kernel(
    const float* __restrict__ partial_acc, // [n_head, max_splits, head_dim]
    const float* __restrict__ partial_m,   // [n_head, max_splits]
    const float* __restrict__ partial_l,   // [n_head, max_splits]
    float* __restrict__ out,               // [n_head, head_dim]
    int n_head,
    int head_dim,
    int num_splits,
    int max_splits
) {
    const int qh = blockIdx.x;
    const int tid = threadIdx.x;

    if (qh >= n_head) return;

    // 1. Find global maximum score m_global across all splits
    float m_global = -__int_as_float(0x7f800000u);
    for (int s = 0; s < num_splits; ++s) {
        const size_t split_offset = (size_t)qh * (size_t)max_splits + (size_t)s;
        const float ms = partial_m[split_offset];
        m_global = fmaxf(m_global, ms);
    }

    // 2. Compute global normalizer sum l_global and combine acc vectors
    float l_global = 0.0f;
    float final_acc0 = 0.0f;
    float final_acc1 = 0.0f;
    float final_acc2 = 0.0f;
    float final_acc3 = 0.0f;

    for (int s = 0; s < num_splits; ++s) {
        const size_t split_offset = (size_t)qh * (size_t)max_splits + (size_t)s;
        const float ms = partial_m[split_offset];
        const float ls = partial_l[split_offset];

        if (ls > 0.0f) {
            const float weight = __expf(ms - m_global);
            l_global += ls * weight;

            const float* acc_head_split = partial_acc + split_offset * (size_t)head_dim;
            const float4 acc_val = (tid * 4 + 3 < head_dim) ? ((const float4*)acc_head_split)[tid] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            final_acc0 = __fmaf_rn(acc_val.x, weight, final_acc0);
            final_acc1 = __fmaf_rn(acc_val.y, weight, final_acc1);
            final_acc2 = __fmaf_rn(acc_val.z, weight, final_acc2);
            final_acc3 = __fmaf_rn(acc_val.w, weight, final_acc3);
        }
    }

    float* out_head = out + (size_t)qh * (size_t)head_dim;
    const float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;

    if (tid * 4 + 3 < head_dim) {
        ((float4*)out_head)[tid] = make_float4(final_acc0 * inv_l, final_acc1 * inv_l, final_acc2 * inv_l, final_acc3 * inv_l);
    }
}