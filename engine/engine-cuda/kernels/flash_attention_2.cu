// flash_attention_2.cu — FlashAttention-2 Causal Kernel for Prefill (Phase 11, Task 2.1).
//
// Computes causal self-attention over resident paged KV blocks with online softmax.
// Q: [q_tokens, n_head, head_dim]
// Pool: Paged KV cache floats
// Block Table: physical block indices
// Out: [q_tokens, n_head, head_dim]
//
// Uses warp-level cooperative reduction and online softmax scaling:
//   - 32 threads per warp process head_dim = 128 (4 floats per thread).
//   - Running row maximum (m) and sum of exponentials (l) kept in registers.
//   - Output vector (O) accumulated directly in registers and normalized in-place.

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return __shfl_sync(0xffffffff, val, 0);
}

extern "C" __global__ void flash_attention_2_kernel(
    const float* __restrict__ q,
    const float* __restrict__ pool,
    const unsigned* __restrict__ block_table,
    float* __restrict__ out,
    int n_head,
    int n_head_kv,
    int head_dim,
    int block_tokens,
    int q_tokens,
    int pos_offset,
    float scale,
    const unsigned int* __restrict__ pos_ptr)
{
    // Grid: blockIdx.x = q_head (0..n_head-1), blockIdx.y = q_pos_in_chunk (0..q_tokens-1)
    // Block: threadIdx.x = 0..31 (warp of 32 threads for head_dim = 128)
    const int qh = blockIdx.x;
    const int q_pos_in_chunk = blockIdx.y;
    const int tid = threadIdx.x;

    if (qh >= n_head || q_pos_in_chunk >= q_tokens || tid >= 32) return;

    const int cur_offset = (pos_ptr != nullptr) ? (int)*pos_ptr : pos_offset;
    const int global_q_pos = cur_offset + q_pos_in_chunk;

    const int gqa_group = n_head / n_head_kv;
    const int kh = qh / gqa_group;

    const int floats_per_token = 2 * n_head_kv * head_dim;
    const int row_len = n_head_kv * head_dim;
    const int floats_per_block = block_tokens * floats_per_token;

    // Pointer to this query vector Q[q_pos_in_chunk, qh, :]
    const size_t q_stride = (size_t)n_head * (size_t)head_dim;
    const float* q_vec = q + (size_t)q_pos_in_chunk * q_stride + (size_t)qh * (size_t)head_dim;

    // Each thread in warp loads 4 elements (32 * 4 = 128)
    const int elem_idx = tid * 4;
    const float4 q_local = (elem_idx + 3 < head_dim) ? *(const float4*)(q_vec + elem_idx) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Online softmax state in registers
    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float4 o_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Causal attention: attend to all positions k_pos in [0, global_q_pos]
    for (int k_pos = 0; k_pos <= global_q_pos; ++k_pos) {
        const int b = k_pos / block_tokens;
        const int in_block = k_pos % block_tokens;
        const unsigned phys = block_table[b];

        const float4* k_ptr4 = (const float4*)(pool + (size_t)phys * (size_t)floats_per_block + (size_t)kh * (size_t)head_dim + (size_t)in_block * (size_t)floats_per_token + (size_t)elem_idx);
        const float4* v_ptr4 = (const float4*)(pool + (size_t)phys * (size_t)floats_per_block + (size_t)kh * (size_t)head_dim + (size_t)row_len + (size_t)in_block * (size_t)floats_per_token + (size_t)elem_idx);

        const float4 k_val = (elem_idx + 3 < head_dim) ? *k_ptr4 : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // 1. Compute dot product Q . K
        float local_dot = __fmaf_rn(q_local.x, k_val.x, __fmaf_rn(q_local.y, k_val.y, __fmaf_rn(q_local.z, k_val.z, q_local.w * k_val.w)));

        float score = warp_reduce_sum(local_dot) * scale;

        // 2. Online softmax update
        float m_curr = (score > m_prev) ? score : m_prev;
        float alpha = __expf(m_prev - m_curr);
        float beta  = __expf(score - m_curr);

        l_prev = __fmaf_rn(l_prev, alpha, beta);

        const float4 v_val = (elem_idx + 3 < head_dim) ? *v_ptr4 : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        o_acc.x = __fmaf_rn(o_acc.x, alpha, beta * v_val.x);
        o_acc.y = __fmaf_rn(o_acc.y, alpha, beta * v_val.y);
        o_acc.z = __fmaf_rn(o_acc.z, alpha, beta * v_val.z);
        o_acc.w = __fmaf_rn(o_acc.w, alpha, beta * v_val.w);

        m_prev = m_curr;
    }

    // 3. Final normalization
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    if (elem_idx + 3 < head_dim) {
        float4* out_vec4 = (float4*)(out + (size_t)q_pos_in_chunk * q_stride + (size_t)qh * (size_t)head_dim + (size_t)elem_idx);
        *out_vec4 = make_float4(o_acc.x * inv_l, o_acc.y * inv_l, o_acc.z * inv_l, o_acc.w * inv_l);
    }
}
