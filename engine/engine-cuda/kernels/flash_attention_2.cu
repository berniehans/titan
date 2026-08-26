// flash_attention_2.cu — FlashAttention-2 Causal Kernel for Prefill (Phase 11, Task 2.1).
//
// Computes causal self-attention over resident paged KV blocks with online softmax.
// Q: [seq_tokens, n_head, head_dim]
// Pool: Paged KV cache floats
// Block Table: physical block indices
// Out: [seq_tokens, n_head, head_dim]
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
    int seq_tokens,
    float scale)
{
    // Grid: blockIdx.x = q_head (0..n_head-1), blockIdx.y = q_pos (0..seq_tokens-1)
    // Block: threadIdx.x = 0..31 (warp of 32 threads for head_dim = 128)
    const int qh = blockIdx.x;
    const int q_pos = blockIdx.y;
    const int tid = threadIdx.x;

    if (qh >= n_head || q_pos >= seq_tokens || tid >= 32) return;

    const int gqa_group = n_head / n_head_kv;
    const int kh = qh / gqa_group;

    const int floats_per_token = 2 * n_head_kv * head_dim;
    const int row_len = n_head_kv * head_dim;
    const int floats_per_block = block_tokens * floats_per_token;

    // Pointer to this query vector Q[q_pos, qh, :]
    const size_t q_stride = (size_t)n_head * (size_t)head_dim;
    const float* q_vec = q + (size_t)q_pos * q_stride + (size_t)qh * (size_t)head_dim;

    // Each thread in warp loads 4 elements (32 * 4 = 128)
    const int elem_idx = tid * 4;
    float q_local[4];
    q_local[0] = q_vec[elem_idx + 0];
    q_local[1] = q_vec[elem_idx + 1];
    q_local[2] = q_vec[elem_idx + 2];
    q_local[3] = q_vec[elem_idx + 3];

    // Online softmax state in registers
    float m_prev = -1e30f;
    float l_prev = 0.0f;
    float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Causal attention: attend to all positions k_pos in [0, q_pos]
    for (int k_pos = 0; k_pos <= q_pos; ++k_pos) {
        const int b = k_pos / block_tokens;
        const int in_block = k_pos % block_tokens;
        const unsigned phys = block_table[b];

        const float* k_ptr = pool + (size_t)phys * (size_t)floats_per_block + (size_t)kh * (size_t)head_dim + (size_t)in_block * (size_t)floats_per_token;
        const float* v_ptr = pool + (size_t)phys * (size_t)floats_per_block + (size_t)kh * (size_t)head_dim + (size_t)row_len + (size_t)in_block * (size_t)floats_per_token;

        // 1. Compute dot product Q . K
        float local_dot = q_local[0] * k_ptr[elem_idx + 0]
                        + q_local[1] * k_ptr[elem_idx + 1]
                        + q_local[2] * k_ptr[elem_idx + 2]
                        + q_local[3] * k_ptr[elem_idx + 3];

        float score = warp_reduce_sum(local_dot) * scale;

        // 2. Online softmax update
        float m_curr = (score > m_prev) ? score : m_prev;
        float alpha = __expf(m_prev - m_curr);
        float beta  = __expf(score - m_curr);

        l_prev = l_prev * alpha + beta;

        float v0 = v_ptr[elem_idx + 0];
        float v1 = v_ptr[elem_idx + 1];
        float v2 = v_ptr[elem_idx + 2];
        float v3 = v_ptr[elem_idx + 3];

        o_acc[0] = o_acc[0] * alpha + beta * v0;
        o_acc[1] = o_acc[1] * alpha + beta * v1;
        o_acc[2] = o_acc[2] * alpha + beta * v2;
        o_acc[3] = o_acc[3] * alpha + beta * v3;

        m_prev = m_curr;
    }

    // 3. Final normalization
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    float* out_vec = out + (size_t)q_pos * q_stride + (size_t)qh * (size_t)head_dim;

    out_vec[elem_idx + 0] = o_acc[0] * inv_l;
    out_vec[elem_idx + 1] = o_acc[1] * inv_l;
    out_vec[elem_idx + 2] = o_acc[2] * inv_l;
    out_vec[elem_idx + 3] = o_acc[3] * inv_l;
}
