// Port of llama.cpp ggml_compute_forward_rms_norm + rope (ggml/src/ggml.c, ggml/src/ggml-cuda/rope.cu @ cb1adf8)

extern "C" __global__ void norm_rope_swiglu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    const float* __restrict__ norm_w,
    int n,
    int n_dims,
    float freq_base,
    unsigned int pos,
    const float* __restrict__ up,
    float* __restrict__ out,
    float eps,
    int mode,
    const unsigned int* __restrict__ pos_ptr,
    int n_heads)
{
    unsigned int cur_pos = (pos_ptr != nullptr) ? *pos_ptr : pos;
    if (n_heads > 0) {
        cur_pos += (blockIdx.x / n_heads);
    }

    const int row = blockIdx.x;
    const float* row_x = x + (size_t)row * (size_t)n;
    const size_t residual_offset = (mode & 8)
        ? (size_t)((n_heads > 0) ? (blockIdx.x % n_heads) : 0) * (size_t)n
        : (size_t)row * (size_t)n;
    const float* row_residual = residual + residual_offset;
    const float* row_norm_w = norm_w;
    const float* row_up = up + (size_t)row * (size_t)n;
    float* row_out = out + (size_t)row * (size_t)n;

    const int lane = threadIdx.x & 31;
    const int n4 = n / 4;

    // Phase 1: RMSNorm + Residual add (bit0 of mode)
    if (mode & 1) {
        float psum = 0.0f;
        const float4* x4 = (const float4*)row_x;
        const float4* res4 = (const float4*)row_residual;

        for (int j = lane; j < n4; j += 32) {
            float4 v = x4[j];
            float4 r = res4[j];
            float tmp0 = v.x + r.x;
            float tmp1 = v.y + r.y;
            float tmp2 = v.z + r.z;
            float tmp3 = v.w + r.w;
            psum = __fmaf_rn(tmp0, tmp0, psum);
            psum = __fmaf_rn(tmp1, tmp1, psum);
            psum = __fmaf_rn(tmp2, tmp2, psum);
            psum = __fmaf_rn(tmp3, tmp3, psum);
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            psum += __shfl_down_sync(0xffffffff, psum, mask);
        }

        float scale = 0.0f;
        if (lane == 0) {
            float mean = psum / (float)n;
            scale = rsqrtf(mean + eps);
        }
        scale = __shfl_sync(0xffffffff, scale, 0);

        const float4* w4 = (const float4*)row_norm_w;
        float4* out4 = (float4*)row_out;

        for (int j = lane; j < n4; j += 32) {
            float4 v = x4[j];
            float4 r = res4[j];
            float4 w = w4[j];
            float4 res;
            res.x = (v.x + r.x) * scale * w.x;
            res.y = (v.y + r.y) * scale * w.y;
            res.z = (v.z + r.z) * scale * w.z;
            res.w = (v.w + r.w) * scale * w.w;
            out4[j] = res;
        }
    } else {
        const float4* x4 = (const float4*)row_x;
        const float4* res4 = (const float4*)row_residual;
        float4* out4 = (float4*)row_out;
        for (int j = lane; j < n4; j += 32) {
            float4 v = x4[j];
            float4 r = res4[j];
            float4 res;
            res.x = v.x + r.x;
            res.y = v.y + r.y;
            res.z = v.z + r.z;
            res.w = v.w + r.w;
            out4[j] = res;
        }
    }

    // Phase 2: Rotary Position Embedding (bit1 of mode)
    if (mode & 2) {
        const int half = n_dims / 2;
        const float inv_dims = 1.0f / (float)n_dims;
        const float log_base = logf(freq_base);

        for (int k = lane; k < half; k += 32) {
            float exp_val = -2.0f * (float)k * inv_dims;
            float freq = __expf(exp_val * log_base);
            float theta = (float)cur_pos * freq;
            float s, c;
            __sincosf(theta, &s, &c);
            float x0 = row_out[k];
            float x1 = row_out[k + half];
            row_out[k] = __fmaf_rn(x0, c, -x1 * s);
            row_out[k + half] = __fmaf_rn(x0, s, x1 * c);
        }
    }

    // Phase 3: SwiGLU (bit2 of mode)
    if (mode & 4) {
        const float4* g4 = (const float4*)row_x;
        const float4* u4 = (const float4*)row_up;
        float4* out4 = (float4*)row_out;
        for (int j = lane; j < n4; j += 32) {
            float4 g = g4[j];
            float4 u = u4[j];
            float4 res;
            res.x = (g.x / (1.0f + __expf(-g.x))) * u.x;
            res.y = (g.y / (1.0f + __expf(-g.y))) * u.y;
            res.z = (g.z / (1.0f + __expf(-g.z))) * u.z;
            res.w = (g.w / (1.0f + __expf(-g.w))) * u.w;
            out4[j] = res;
        }
    }
}

extern "C" __global__ void fused_qk_norm_rope_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ qn_w,
    const float* __restrict__ kn_w,
    int n_head_q,
    int n_head_k,
    int head_dim,
    int n_rot,
    float freq_base,
    float eps,
    int mode,
    const unsigned int* __restrict__ pos_ptr,
    const float* __restrict__ v,
    float* __restrict__ pool,
    const unsigned* __restrict__ block_table,
    int block_tokens)
{
    const unsigned int cur_pos = *pos_ptr;
    const int bid = blockIdx.x;
    const bool is_q = (bid < n_head_q);
    const int head_idx = is_q ? bid : (bid - n_head_q);

    float* row_out = is_q ? (q + (size_t)head_idx * (size_t)head_dim) : (k + (size_t)head_idx * (size_t)head_dim);
    const float* row_norm_w = is_q ? qn_w : kn_w;

    const int lane = threadIdx.x & 31;

    // RMSNorm Phase (bit0 of mode)
    if (mode & 1) {
        float4 val = ((const float4*)row_out)[lane];
        float psum = __fmaf_rn(val.x, val.x, __fmaf_rn(val.y, val.y, __fmaf_rn(val.z, val.z, val.w * val.w)));

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            psum += __shfl_down_sync(0xffffffff, psum, mask);
        }

        const float total_sum = __shfl_sync(0xffffffff, psum, 0);
        const float scale = rsqrtf(total_sum / (float)head_dim + eps);
        const float4 nw = ((const float4*)row_norm_w)[lane];

        val.x = val.x * scale * nw.x;
        val.y = val.y * scale * nw.y;
        val.z = val.z * scale * nw.z;
        val.w = val.w * scale * nw.w;

        ((float4*)row_out)[lane] = val;
    }

    // RoPE Phase (bit1 of mode)
    if (mode & 2) {
        const float inv_dims = 1.0f / (float)n_rot;
        const float log_base = logf(freq_base);

        // First half (lanes 0..31 -> indices 0..31)
        {
            float exp_val = -2.0f * (float)lane * inv_dims;
            float freq = __expf(exp_val * log_base);
            float theta = (float)cur_pos * freq;
            float s, c;
            __sincosf(theta, &s, &c);
            float x0 = row_out[lane];
            float x1 = row_out[lane + 64];
            row_out[lane]      = __fmaf_rn(x0, c, -x1 * s);
            row_out[lane + 64] = __fmaf_rn(x0, s,  x1 * c);
        }

        // Second half (lanes 0..31 -> indices 32..63)
        {
            const int j = lane + 32;
            float exp_val = -2.0f * (float)j * inv_dims;
            float freq = __expf(exp_val * log_base);
            float theta = (float)cur_pos * freq;
            float s, c;
            __sincosf(theta, &s, &c);
            float x0 = row_out[j];
            float x1 = row_out[j + 64];
            row_out[j]      = __fmaf_rn(x0, c, -x1 * s);
            row_out[j + 64] = __fmaf_rn(x0, s,  x1 * c);
        }
    }

    // Phase 3: Fused Paged KV Append (only for K heads if pool is provided)
    if (!is_q && pool != nullptr && block_table != nullptr) {
        const int g = (int)cur_pos;
        const int blk_idx = g / block_tokens;
        const int slot = g % block_tokens;
        const unsigned phys = block_table[blk_idx];

        const int row_len = n_head_k * head_dim;
        const int floats_per_token = 2 * row_len;
        const int floats_per_block = block_tokens * floats_per_token;

        const size_t pool_base = (size_t)phys * (size_t)floats_per_block + (size_t)slot * (size_t)floats_per_token;
        const size_t head_offset = (size_t)head_idx * (size_t)head_dim;

        ((float4*)(pool + pool_base + head_offset))[lane] = ((const float4*)row_out)[lane];
        if (v != nullptr) {
            ((float4*)(pool + pool_base + (size_t)row_len + head_offset))[lane] = ((const float4*)(v + head_offset))[lane];
        }
    }
}

