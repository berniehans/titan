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
    extern __shared__ char smem[];
    signed char* s_qx = (signed char*)smem;
    float* s_qd = (float*)(s_qx + ne0);
    float* s_qs = s_qd + (ne0 / 32);

    // Fast coalesced activation load
    const int tid = threadIdx.x;
    const int total_threads = blockDim.x;
    for (int i = tid; i < ne0; i += total_threads) {
        s_qx[i] = qx[i];
    }
    const int n_blocks_32 = ne0 / 32;
    for (int i = tid; i < n_blocks_32; i += total_threads) {
        s_qd[i] = qd[i];
        s_qs[i] = qs[i];
    }
    __syncthreads();

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int col = blockIdx.x * (blockDim.x / 32) + warp_id;
    const int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size || col >= ne1) return;

    float* out_row = out + (size_t)batch_idx * (size_t)ne1;
    const int n_blocks = ne0 / 256;
    const unsigned char* col_weights = weights + (size_t)col * (size_t)n_blocks * 144u;

    float local_acc = 0.0f;
    float min_sum = 0.0f;

    #pragma unroll 2
    for (int b = 0; b < n_blocks; ++b) {
        const int q8_b = b * 8;
        const signed char* qx_base = s_qx + q8_b * 32;
        const signed char qx0 = qx_base[0 * 32 + lane];
        const signed char qx1 = qx_base[1 * 32 + lane];
        const signed char qx2 = qx_base[2 * 32 + lane];
        const signed char qx3 = qx_base[3 * 32 + lane];
        const signed char qx4 = qx_base[4 * 32 + lane];
        const signed char qx5 = qx_base[5 * 32 + lane];
        const signed char qx6 = qx_base[6 * 32 + lane];
        const signed char qx7 = qx_base[7 * 32 + lane];

        const unsigned char* blk = col_weights + (size_t)b * 144u;
        const uint4 raw = *(const uint4*)blk;
        const unsigned char* qs_ptr = blk + 16;

        // Unpack scales
        const float d = f16_to_f32((unsigned short)(raw.x & 0xFFFFu));
        const float neg_dmin = -f16_to_f32((unsigned short)(raw.x >> 16));
        const unsigned int w0 = raw.y;
        const unsigned int w1 = raw.z;
        const unsigned int w2 = raw.w;

        const float d_sc0 = d * (float)((w0 >> 0) & 0x3Fu) * s_qd[q8_b + 0];
        const float d_sc1 = d * (float)((w0 >> 8) & 0x3Fu) * s_qd[q8_b + 1];
        const float d_sc2 = d * (float)((w0 >> 16) & 0x3Fu) * s_qd[q8_b + 2];
        const float d_sc3 = d * (float)((w0 >> 24) & 0x3Fu) * s_qd[q8_b + 3];
        const float d_sc4 = d * (float)(((w2 >> 0) & 0x0Fu) | (((w0 >> 6) & 0x03u) << 4)) * s_qd[q8_b + 4];
        const float d_sc5 = d * (float)(((w2 >> 8) & 0x0Fu) | (((w0 >> 14) & 0x03u) << 4)) * s_qd[q8_b + 5];
        const float d_sc6 = d * (float)(((w2 >> 16) & 0x0Fu) | (((w0 >> 22) & 0x03u) << 4)) * s_qd[q8_b + 6];
        const float d_sc7 = d * (float)(((w2 >> 24) & 0x0Fu) | (((w0 >> 30) & 0x03u) << 4)) * s_qd[q8_b + 7];

        const unsigned char q0 = qs_ptr[0 * 32 + lane];
        const unsigned char q1 = qs_ptr[1 * 32 + lane];
        const unsigned char q2 = qs_ptr[2 * 32 + lane];
        const unsigned char q3 = qs_ptr[3 * 32 + lane];

        float dot = 0.0f;
        dot = __fmaf_rn(d_sc0, (float)((int)(q0 & 0x0Fu) * (int)qx0), dot);
        dot = __fmaf_rn(d_sc1, (float)((int)(q0 >> 4)    * (int)qx1), dot);
        dot = __fmaf_rn(d_sc2, (float)((int)(q1 & 0x0Fu) * (int)qx2), dot);
        dot = __fmaf_rn(d_sc3, (float)((int)(q1 >> 4)    * (int)qx3), dot);
        dot = __fmaf_rn(d_sc4, (float)((int)(q2 & 0x0Fu) * (int)qx4), dot);
        dot = __fmaf_rn(d_sc5, (float)((int)(q2 >> 4)    * (int)qx5), dot);
        dot = __fmaf_rn(d_sc6, (float)((int)(q3 & 0x0Fu) * (int)qx6), dot);
        dot = __fmaf_rn(d_sc7, (float)((int)(q3 >> 4)    * (int)qx7), dot);
        local_acc += dot;

        if (lane == 0) {
            float ms = 0.0f;
            const float m0 = neg_dmin * (float)((w1 >> 0) & 0x3Fu);
            const float m1 = neg_dmin * (float)((w1 >> 8) & 0x3Fu);
            const float m2 = neg_dmin * (float)((w1 >> 16) & 0x3Fu);
            const float m3 = neg_dmin * (float)((w1 >> 24) & 0x3Fu);
            const float m4 = neg_dmin * (float)(((w2 >> 4) & 0x0Fu) | (((w1 >> 6) & 0x03u) << 4));
            const float m5 = neg_dmin * (float)(((w2 >> 12) & 0x0Fu) | (((w1 >> 14) & 0x03u) << 4));
            const float m6 = neg_dmin * (float)(((w2 >> 20) & 0x0Fu) | (((w1 >> 22) & 0x03u) << 4));
            const float m7 = neg_dmin * (float)(((w2 >> 28) & 0x0Fu) | (((w1 >> 30) & 0x03u) << 4));

            ms = __fmaf_rn(m0, s_qs[q8_b + 0], ms);
            ms = __fmaf_rn(m1, s_qs[q8_b + 1], ms);
            ms = __fmaf_rn(m2, s_qs[q8_b + 2], ms);
            ms = __fmaf_rn(m3, s_qs[q8_b + 3], ms);
            ms = __fmaf_rn(m4, s_qs[q8_b + 4], ms);
            ms = __fmaf_rn(m5, s_qs[q8_b + 5], ms);
            ms = __fmaf_rn(m6, s_qs[q8_b + 6], ms);
            ms = __fmaf_rn(m7, s_qs[q8_b + 7], ms);
            min_sum += ms;
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_acc += __shfl_down_sync(0xffffffff, local_acc, mask);
    }

    if (lane == 0) {
        float res_val = local_acc + min_sum;
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
    extern __shared__ char smem[];
    signed char* s_qx = (signed char*)smem;
    float* s_qd = (float*)(s_qx + ne0);
    float* s_qs = s_qd + (ne0 / 32);

    const int tid = threadIdx.x;
    const int total_threads = blockDim.x;
    for (int i = tid; i < ne0; i += total_threads) {
        s_qx[i] = qx[i];
    }
    const int n_blocks_32 = ne0 / 32;
    for (int i = tid; i < n_blocks_32; i += total_threads) {
        s_qd[i] = qd[i];
        s_qs[i] = qs[i];
    }
    __syncthreads();

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int col = blockIdx.x * (blockDim.x / 32) + warp_id;
    const int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size || col >= ne1) return;

    float* out_row = out + (size_t)batch_idx * (size_t)ne1;
    const int n_blocks = ne0 / 256;
    const unsigned char* gate_col_w = w_gate + (size_t)col * (size_t)n_blocks * 144u;
    const unsigned char* up_col_w   = w_up   + (size_t)col * (size_t)n_blocks * 144u;

    float gate_acc = 0.0f, gate_min = 0.0f;
    float up_acc = 0.0f, up_min = 0.0f;

    #pragma unroll 2
    for (int b = 0; b < n_blocks; ++b) {
        const int q8_b = b * 8;
        const signed char* qx_base = s_qx + q8_b * 32;
        const signed char qx0 = qx_base[0 * 32 + lane];
        const signed char qx1 = qx_base[1 * 32 + lane];
        const signed char qx2 = qx_base[2 * 32 + lane];
        const signed char qx3 = qx_base[3 * 32 + lane];
        const signed char qx4 = qx_base[4 * 32 + lane];
        const signed char qx5 = qx_base[5 * 32 + lane];
        const signed char qx6 = qx_base[6 * 32 + lane];
        const signed char qx7 = qx_base[7 * 32 + lane];

        // 1. Gate projection accumulation
        {
            const unsigned char* blk = gate_col_w + (size_t)b * 144u;
            const uint4 raw = *(const uint4*)blk;
            const unsigned char* qs_ptr = blk + 16;

            const float d = f16_to_f32((unsigned short)(raw.x & 0xFFFFu));
            const float neg_dmin = -f16_to_f32((unsigned short)(raw.x >> 16));
            const unsigned int w0 = raw.y;
            const unsigned int w1 = raw.z;
            const unsigned int w2 = raw.w;

            const float d_sc0 = d * (float)((w0 >> 0) & 0x3Fu) * s_qd[q8_b + 0];
            const float d_sc1 = d * (float)((w0 >> 8) & 0x3Fu) * s_qd[q8_b + 1];
            const float d_sc2 = d * (float)((w0 >> 16) & 0x3Fu) * s_qd[q8_b + 2];
            const float d_sc3 = d * (float)((w0 >> 24) & 0x3Fu) * s_qd[q8_b + 3];
            const float d_sc4 = d * (float)(((w2 >> 0) & 0x0Fu) | (((w0 >> 6) & 0x03u) << 4)) * s_qd[q8_b + 4];
            const float d_sc5 = d * (float)(((w2 >> 8) & 0x0Fu) | (((w0 >> 14) & 0x03u) << 4)) * s_qd[q8_b + 5];
            const float d_sc6 = d * (float)(((w2 >> 16) & 0x0Fu) | (((w0 >> 22) & 0x03u) << 4)) * s_qd[q8_b + 6];
            const float d_sc7 = d * (float)(((w2 >> 24) & 0x0Fu) | (((w0 >> 30) & 0x03u) << 4)) * s_qd[q8_b + 7];

            const unsigned char q0 = qs_ptr[0 * 32 + lane];
            const unsigned char q1 = qs_ptr[1 * 32 + lane];
            const unsigned char q2 = qs_ptr[2 * 32 + lane];
            const unsigned char q3 = qs_ptr[3 * 32 + lane];

            float dot = 0.0f;
            dot = __fmaf_rn(d_sc0, (float)((int)(q0 & 0x0Fu) * (int)qx0), dot);
            dot = __fmaf_rn(d_sc1, (float)((int)(q0 >> 4)    * (int)qx1), dot);
            dot = __fmaf_rn(d_sc2, (float)((int)(q1 & 0x0Fu) * (int)qx2), dot);
            dot = __fmaf_rn(d_sc3, (float)((int)(q1 >> 4)    * (int)qx3), dot);
            dot = __fmaf_rn(d_sc4, (float)((int)(q2 & 0x0Fu) * (int)qx4), dot);
            dot = __fmaf_rn(d_sc5, (float)((int)(q2 >> 4)    * (int)qx5), dot);
            dot = __fmaf_rn(d_sc6, (float)((int)(q3 & 0x0Fu) * (int)qx6), dot);
            dot = __fmaf_rn(d_sc7, (float)((int)(q3 >> 4)    * (int)qx7), dot);
            gate_acc += dot;

            if (lane == 0) {
                float ms = 0.0f;
                const float m0 = neg_dmin * (float)((w1 >> 0) & 0x3Fu);
                const float m1 = neg_dmin * (float)((w1 >> 8) & 0x3Fu);
                const float m2 = neg_dmin * (float)((w1 >> 16) & 0x3Fu);
                const float m3 = neg_dmin * (float)((w1 >> 24) & 0x3Fu);
                const float m4 = neg_dmin * (float)(((w2 >> 4) & 0x0Fu) | (((w1 >> 6) & 0x03u) << 4));
                const float m5 = neg_dmin * (float)(((w2 >> 12) & 0x0Fu) | (((w1 >> 14) & 0x03u) << 4));
                const float m6 = neg_dmin * (float)(((w2 >> 20) & 0x0Fu) | (((w1 >> 22) & 0x03u) << 4));
                const float m7 = neg_dmin * (float)(((w2 >> 28) & 0x0Fu) | (((w1 >> 30) & 0x03u) << 4));

                ms = __fmaf_rn(m0, s_qs[q8_b + 0], ms);
                ms = __fmaf_rn(m1, s_qs[q8_b + 1], ms);
                ms = __fmaf_rn(m2, s_qs[q8_b + 2], ms);
                ms = __fmaf_rn(m3, s_qs[q8_b + 3], ms);
                ms = __fmaf_rn(m4, s_qs[q8_b + 4], ms);
                ms = __fmaf_rn(m5, s_qs[q8_b + 5], ms);
                ms = __fmaf_rn(m6, s_qs[q8_b + 6], ms);
                ms = __fmaf_rn(m7, s_qs[q8_b + 7], ms);
                gate_min += ms;
            }
        }

        // 2. Up projection accumulation
        {
            const unsigned char* blk = up_col_w + (size_t)b * 144u;
            const uint4 raw = *(const uint4*)blk;
            const unsigned char* qs_ptr = blk + 16;

            const float d = f16_to_f32((unsigned short)(raw.x & 0xFFFFu));
            const float neg_dmin = -f16_to_f32((unsigned short)(raw.x >> 16));
            const unsigned int w0 = raw.y;
            const unsigned int w1 = raw.z;
            const unsigned int w2 = raw.w;

            const float d_sc0 = d * (float)((w0 >> 0) & 0x3Fu) * s_qd[q8_b + 0];
            const float d_sc1 = d * (float)((w0 >> 8) & 0x3Fu) * s_qd[q8_b + 1];
            const float d_sc2 = d * (float)((w0 >> 16) & 0x3Fu) * s_qd[q8_b + 2];
            const float d_sc3 = d * (float)((w0 >> 24) & 0x3Fu) * s_qd[q8_b + 3];
            const float d_sc4 = d * (float)(((w2 >> 0) & 0x0Fu) | (((w0 >> 6) & 0x03u) << 4)) * s_qd[q8_b + 4];
            const float d_sc5 = d * (float)(((w2 >> 8) & 0x0Fu) | (((w0 >> 14) & 0x03u) << 4)) * s_qd[q8_b + 5];
            const float d_sc6 = d * (float)(((w2 >> 16) & 0x0Fu) | (((w0 >> 22) & 0x03u) << 4)) * s_qd[q8_b + 6];
            const float d_sc7 = d * (float)(((w2 >> 24) & 0x0Fu) | (((w0 >> 30) & 0x03u) << 4)) * s_qd[q8_b + 7];

            const unsigned char q0 = qs_ptr[0 * 32 + lane];
            const unsigned char q1 = qs_ptr[1 * 32 + lane];
            const unsigned char q2 = qs_ptr[2 * 32 + lane];
            const unsigned char q3 = qs_ptr[3 * 32 + lane];

            float dot = 0.0f;
            dot = __fmaf_rn(d_sc0, (float)((int)(q0 & 0x0Fu) * (int)qx0), dot);
            dot = __fmaf_rn(d_sc1, (float)((int)(q0 >> 4)    * (int)qx1), dot);
            dot = __fmaf_rn(d_sc2, (float)((int)(q1 & 0x0Fu) * (int)qx2), dot);
            dot = __fmaf_rn(d_sc3, (float)((int)(q1 >> 4)    * (int)qx3), dot);
            dot = __fmaf_rn(d_sc4, (float)((int)(q2 & 0x0Fu) * (int)qx4), dot);
            dot = __fmaf_rn(d_sc5, (float)((int)(q2 >> 4)    * (int)qx5), dot);
            dot = __fmaf_rn(d_sc6, (float)((int)(q3 & 0x0Fu) * (int)qx6), dot);
            dot = __fmaf_rn(d_sc7, (float)((int)(q3 >> 4)    * (int)qx7), dot);
            up_acc += dot;

            if (lane == 0) {
                float ms = 0.0f;
                const float m0 = neg_dmin * (float)((w1 >> 0) & 0x3Fu);
                const float m1 = neg_dmin * (float)((w1 >> 8) & 0x3Fu);
                const float m2 = neg_dmin * (float)((w1 >> 16) & 0x3Fu);
                const float m3 = neg_dmin * (float)((w1 >> 24) & 0x3Fu);
                const float m4 = neg_dmin * (float)(((w2 >> 4) & 0x0Fu) | (((w1 >> 6) & 0x03u) << 4));
                const float m5 = neg_dmin * (float)(((w2 >> 12) & 0x0Fu) | (((w1 >> 14) & 0x03u) << 4));
                const float m6 = neg_dmin * (float)(((w2 >> 20) & 0x0Fu) | (((w1 >> 22) & 0x03u) << 4));
                const float m7 = neg_dmin * (float)(((w2 >> 28) & 0x0Fu) | (((w1 >> 30) & 0x03u) << 4));

                ms = __fmaf_rn(m0, s_qs[q8_b + 0], ms);
                ms = __fmaf_rn(m1, s_qs[q8_b + 1], ms);
                ms = __fmaf_rn(m2, s_qs[q8_b + 2], ms);
                ms = __fmaf_rn(m3, s_qs[q8_b + 3], ms);
                ms = __fmaf_rn(m4, s_qs[q8_b + 4], ms);
                ms = __fmaf_rn(m5, s_qs[q8_b + 5], ms);
                ms = __fmaf_rn(m6, s_qs[q8_b + 6], ms);
                ms = __fmaf_rn(m7, s_qs[q8_b + 7], ms);
                up_min += ms;
            }
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        gate_acc += __shfl_down_sync(0xffffffff, gate_acc, mask);
        up_acc   += __shfl_down_sync(0xffffffff, up_acc,   mask);
    }

    if (lane == 0) {
        const float gate_val = gate_acc + gate_min;
        const float up_val   = up_acc + up_min;
        // In-register fused SwiGLU: silu(gate) * up
        out_row[col] = silu(gate_val) * up_val;
    }
}