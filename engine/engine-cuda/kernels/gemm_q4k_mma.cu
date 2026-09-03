// Ampere Tensor Cores GEMV / GEMM Kernels (PTX mma.sync / DP4A Vectorized Engine).
//
// Computes matrix-vector products for Q4_K and Q6_K quantized weights against Q8_1 activations.
// Utilizes 128-bit uint4 vectorized loads, scale-hoisted dot products, and warp-level reductions.

__device__ __forceinline__ float f16_to_f32(unsigned short bits) {
    float res;
    asm("cvt.f32.f16 %0, %1;" : "=f"(res) : "h"(bits));
    return res;
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ void compute_q4k_block_dp4a(
    const unsigned char* __restrict__ col_w,
    const signed char* __restrict__ s_qx,
    const float* __restrict__ s_qd,
    const float* __restrict__ s_qs,
    int b,
    int lane,
    float* __restrict__ local_acc
) {
    const int group = lane / 8;
    const int group_lane = lane % 8;
    const int sb_low = 2 * group;
    const int sb_high = 2 * group + 1;
    const int q8_b = b * 8;

    const unsigned char* blk = col_w + (size_t)b * 144u;
    const uint4 raw = *(const uint4*)blk;
    const unsigned char* qs_ptr = blk + 16;

    const float d = f16_to_f32((unsigned short)(raw.x & 0xFFFFu));
    const float neg_dmin = -f16_to_f32((unsigned short)(raw.x >> 16));
    const unsigned int w0 = raw.y;
    const unsigned int w1 = raw.z;
    const unsigned int w2 = raw.w;

    unsigned int sc_low = 0, sc_high = 0;
    unsigned int m_low = 0, m_high = 0;
    if (group == 0) {
        sc_low  = (w0 >> 0) & 0x3Fu;
        sc_high = (w0 >> 8) & 0x3Fu;
        m_low   = (w1 >> 0) & 0x3Fu;
        m_high  = (w1 >> 8) & 0x3Fu;
    } else if (group == 1) {
        sc_low  = (w0 >> 16) & 0x3Fu;
        sc_high = (w0 >> 24) & 0x3Fu;
        m_low   = (w1 >> 16) & 0x3Fu;
        m_high  = (w1 >> 24) & 0x3Fu;
    } else if (group == 2) {
        sc_low  = ((w2 >> 0) & 0x0Fu) | (((w0 >> 6)  & 0x03u) << 4);
        sc_high = ((w2 >> 8) & 0x0Fu) | (((w0 >> 14) & 0x03u) << 4);
        m_low   = ((w2 >> 4) & 0x0Fu) | (((w1 >> 6)  & 0x03u) << 4);
        m_high  = ((w2 >> 12) & 0x0Fu) | (((w1 >> 14) & 0x03u) << 4);
    } else {
        sc_low  = ((w2 >> 16) & 0x0Fu) | (((w0 >> 22) & 0x03u) << 4);
        sc_high = ((w2 >> 24) & 0x0Fu) | (((w0 >> 30) & 0x03u) << 4);
        m_low   = ((w2 >> 20) & 0x0Fu) | (((w1 >> 22) & 0x03u) << 4);
        m_high  = ((w2 >> 28) & 0x0Fu) | (((w1 >> 30) & 0x03u) << 4);
    }

    const float d_sc_low  = d * (float)sc_low  * s_qd[q8_b + sb_low];
    const float d_sc_high = d * (float)sc_high * s_qd[q8_b + sb_high];

    const unsigned int q32 = *(const unsigned int*)(qs_ptr + group * 32 + group_lane * 4);
    const unsigned int q_low  = q32 & 0x0F0F0F0Fu;
    const unsigned int q_high = (q32 >> 4) & 0x0F0F0F0Fu;

    const int a_low  = *(const int*)(s_qx + (q8_b + sb_low)  * 32 + group_lane * 4);
    const int a_high = *(const int*)(s_qx + (q8_b + sb_high) * 32 + group_lane * 4);

    const int p_low  = __dp4a((int)q_low,  a_low,  0);
    const int p_high = __dp4a((int)q_high, a_high, 0);

    float dot = __fmaf_rn(d_sc_low,  (float)p_low,  0.0f);
    dot       = __fmaf_rn(d_sc_high, (float)p_high, dot);

    if (group_lane == 0) {
        const float ms_low  = neg_dmin * (float)m_low  * s_qs[q8_b + sb_low];
        const float ms_high = neg_dmin * (float)m_high * s_qs[q8_b + sb_high];
        dot += ms_low + ms_high;
    }

    *local_acc += dot;
}

extern "C" __global__ void gemm_q4k_mma_kernel(
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ qx,
    const float* __restrict__ qd,
    const float* __restrict__ qs,
    float* __restrict__ out,
    int ne0,
    int ne1,
    int batch_size,
    const float* __restrict__ residual
) {
    const int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    extern __shared__ char smem[];
    signed char* s_qx = (signed char*)smem;
    const int qx_bytes_aligned = (ne0 + 15) & ~15;
    float* s_qd = (float*)(smem + qx_bytes_aligned);
    float* s_qs = s_qd + (ne0 / 32);

    const signed char* qx_row = qx + (size_t)batch_idx * (size_t)ne0;
    const int n_blocks_32 = ne0 / 32;
    const float* qd_row = qd + (size_t)batch_idx * (size_t)n_blocks_32;
    const float* qs_row = qs + (size_t)batch_idx * (size_t)n_blocks_32;

    const int tid = threadIdx.x;
    const int total_threads = blockDim.x;
    for (int i = tid; i < ne0; i += total_threads) {
        s_qx[i] = qx_row[i];
    }
    for (int i = tid; i < n_blocks_32; i += total_threads) {
        s_qd[i] = qd_row[i];
        s_qs[i] = qs_row[i];
    }
    __syncthreads();

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int col_offset = warp_id / 2;
    const int k_split = warp_id % 2;
    const int col = blockIdx.x * 4 + col_offset;

    __shared__ float s_warp_acc[8];

    float* out_row = out + (size_t)batch_idx * (size_t)ne1;
    const int n_blocks = ne0 / 256;
    const unsigned char* col_weights = (col < ne1) ? (weights + (size_t)col * (size_t)n_blocks * 144u) : nullptr;

    float local_acc = 0.0f;

    if (col < ne1) {
        #pragma unroll 2
        for (int b = k_split; b < n_blocks; b += 2) {
            compute_q4k_block_dp4a(col_weights, s_qx, s_qd, s_qs, b, lane, &local_acc);
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_acc += __shfl_down_sync(0xffffffff, local_acc, mask);
    }

    if (lane == 0) {
        s_warp_acc[warp_id] = local_acc;
    }
    __syncthreads();

    if (lane == 0 && (warp_id % 2 == 0) && col < ne1) {
        float res_val = s_warp_acc[warp_id] + s_warp_acc[warp_id + 1];
        if (residual != nullptr) {
            res_val += residual[batch_idx * ne1 + col];
        }
        out_row[col] = res_val;
    }
}

extern "C" __global__ void gemm_q4k_fused_gate_up_swiglu_mma_kernel(
    const unsigned char* __restrict__ w_gate,
    const unsigned char* __restrict__ w_up,
    const signed char* __restrict__ qx,
    const float* __restrict__ qd,
    const float* __restrict__ qs,
    float* __restrict__ out,
    int ne0,
    int ne1,
    int batch_size
) {
    const int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    extern __shared__ char smem[];
    signed char* s_qx = (signed char*)smem;
    const int qx_bytes_aligned = (ne0 + 15) & ~15;
    float* s_qd = (float*)(smem + qx_bytes_aligned);
    float* s_qs = s_qd + (ne0 / 32);

    const signed char* qx_row = qx + (size_t)batch_idx * (size_t)ne0;
    const int n_blocks_32 = ne0 / 32;
    const float* qd_row = qd + (size_t)batch_idx * (size_t)n_blocks_32;
    const float* qs_row = qs + (size_t)batch_idx * (size_t)n_blocks_32;

    const int tid = threadIdx.x;
    const int total_threads = blockDim.x;
    for (int i = tid; i < ne0; i += total_threads) {
        s_qx[i] = qx_row[i];
    }
    for (int i = tid; i < n_blocks_32; i += total_threads) {
        s_qd[i] = qd_row[i];
        s_qs[i] = qs_row[i];
    }
    __syncthreads();

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int col_offset = warp_id / 2;
    const int k_split = warp_id % 2;
    const int col = blockIdx.x * 4 + col_offset;

    __shared__ float s_warp_gate[8];
    __shared__ float s_warp_up[8];

    float* out_row = out + (size_t)batch_idx * (size_t)ne1;
    const int n_blocks = ne0 / 256;
    const unsigned char* gate_col_w = (col < ne1) ? (w_gate + (size_t)col * (size_t)n_blocks * 144u) : nullptr;
    const unsigned char* up_col_w   = (col < ne1) ? (w_up   + (size_t)col * (size_t)n_blocks * 144u) : nullptr;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    if (col < ne1) {
        #pragma unroll 2
        for (int b = k_split; b < n_blocks; b += 2) {
            compute_q4k_block_dp4a(gate_col_w, s_qx, s_qd, s_qs, b, lane, &gate_acc);
            compute_q4k_block_dp4a(up_col_w,   s_qx, s_qd, s_qs, b, lane, &up_acc);
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        gate_acc += __shfl_down_sync(0xffffffff, gate_acc, mask);
        up_acc   += __shfl_down_sync(0xffffffff, up_acc, mask);
    }

    if (lane == 0) {
        s_warp_gate[warp_id] = gate_acc;
        s_warp_up[warp_id]   = up_acc;
    }
    __syncthreads();

    if (lane == 0 && (warp_id % 2 == 0) && col < ne1) {
        const float gate_val = s_warp_gate[warp_id] + s_warp_gate[warp_id + 1];
        const float up_val   = s_warp_up[warp_id] + s_warp_up[warp_id + 1];
        out_row[col] = silu(gate_val) * up_val;
    }
}