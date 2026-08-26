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

    extern __shared__ float smem[];
    float* s_acc = smem;
    float* s_part = smem + head_dim;
    float* s_m = smem + head_dim + 32;
    float* s_l = s_m + 1;

    for (int j = tid; j < head_dim; j += 32) {
        s_acc[j] = 0.0f;
    }
    if (tid == 0) {
        s_m[0] = -__int_as_float(0x7f800000u);
        s_l[0] = 0.0f;
    }
    __syncthreads();

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

            float part = 0.0f;
            for (int j = tid; j < head_dim; j += 32) {
                part += qrow[j] * k[j];
            }
            s_part[tid] = part;
            __syncthreads();

            for (int stride = 16; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    s_part[tid] += s_part[tid + stride];
                }
                __syncthreads();
            }

            if (tid == 0) {
                float score = s_part[0] * scale;
                float m_old = s_m[0];
                float m_new = fmaxf(m_old, score);
                float corr = expf(m_old - m_new);
                float p_new = expf(score - m_new);
                s_l[0] = s_l[0] * corr + p_new;
                s_m[0] = m_new;
                s_part[0] = corr;
                s_part[1] = p_new;
            }
            __syncthreads();

            const float corr = s_part[0];
            const float p_new = s_part[1];
            for (int j = tid; j < head_dim; j += 32) {
                s_acc[j] = s_acc[j] * corr + v[j] * p_new;
            }
            __syncthreads();
        }
    }

    const float l = s_l[0];
    float* out_head = out + (size_t)qh * (size_t)head_dim;
    for (int j = tid; j < head_dim; j += 32) {
        out_head[j] = s_acc[j] / l;
    }
}
