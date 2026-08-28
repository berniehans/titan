// gemm_quant.cu — Batched Quantized Matrix Multiplication Kernels (Phase 11, Task 1.1).
//
// Computes Y = X * W^T for arbitrary batch size M (1 <= M <= 512).
// X has shape [M, ne0] (FP32 activation matrix).
// W has shape [ne1, ne0] (Quantized weight matrix, column-major per block).
// Out has shape [M, ne1] (FP32 output matrix).
//
// Supported formats:
//   Q4_K: 256 weights per 144-byte super-block
//   Q6_K: 256 weights per 210-byte super-block
//   Q8_0: 32 weights per 34-byte block ([fp16 d][int8 qs x32])
//   F16:  ne0 * 2 bytes (fp16 LE)
//   F32:  ne0 * 4 bytes (fp32 LE)

__device__ __forceinline__ float f16_to_f32(unsigned short bits) {
    float res;
    asm("cvt.f32.f16 %0, %1;" : "=f"(res) : "h"(bits));
    return res;
}



struct Q4K_Header {
    float d1[4];
    float d2[4];
    float neg_m1[4];
    float neg_m2[4];
};

struct Q4K_BlockScales {
    float d_sc[8];
    float m[8];
};

__device__ __forceinline__ void unpack_q4k_scales(const uint4 raw, Q4K_BlockScales* s) {
    const float d        = f16_to_f32((unsigned short)(raw.x & 0xFFFFu));
    const float neg_dmin = -f16_to_f32((unsigned short)(raw.x >> 16));
    const unsigned int w0 = raw.y;
    const unsigned int w1 = raw.z;
    const unsigned int w2 = raw.w;

    // chunk 0
    s->d_sc[0] = d * (float)((w0 >> 0) & 0x3Fu);
    s->d_sc[1] = d * (float)((w0 >> 8) & 0x3Fu);
    s->m[0]    = neg_dmin * (float)((w1 >> 0) & 0x3Fu);
    s->m[1]    = neg_dmin * (float)((w1 >> 8) & 0x3Fu);

    // chunk 1
    s->d_sc[2] = d * (float)((w0 >> 16) & 0x3Fu);
    s->d_sc[3] = d * (float)((w0 >> 24) & 0x3Fu);
    s->m[2]    = neg_dmin * (float)((w1 >> 16) & 0x3Fu);
    s->m[3]    = neg_dmin * (float)((w1 >> 24) & 0x3Fu);

    // chunk 2
    s->d_sc[4] = d * (float)(((w2 >> 0) & 0x0Fu) | (((w0 >> 6) & 0x03u) << 4));
    s->d_sc[5] = d * (float)(((w2 >> 8) & 0x0Fu) | (((w0 >> 14) & 0x03u) << 4));
    s->m[4]    = neg_dmin * (float)(((w2 >> 4) & 0x0Fu) | (((w1 >> 6) & 0x03u) << 4));
    s->m[5]    = neg_dmin * (float)(((w2 >> 12) & 0x0Fu) | (((w1 >> 14) & 0x03u) << 4));

    // chunk 3
    s->d_sc[6] = d * (float)(((w2 >> 16) & 0x0Fu) | (((w0 >> 22) & 0x03u) << 4));
    s->d_sc[7] = d * (float)(((w2 >> 24) & 0x0Fu) | (((w0 >> 30) & 0x03u) << 4));
    s->m[6]    = neg_dmin * (float)(((w2 >> 20) & 0x0Fu) | (((w1 >> 22) & 0x03u) << 4));
    s->m[7]    = neg_dmin * (float)(((w2 >> 28) & 0x0Fu) | (((w1 >> 30) & 0x03u) << 4));
}

__device__ __forceinline__ void compute_q4k_chunk_pair(
    const uint4 raw,
    unsigned char q,
    int chunk_idx,
    float* w_lo,
    float* w_hi)
{
    const float d        = f16_to_f32((unsigned short)(raw.x & 0xFFFFu));
    const float neg_dmin = -f16_to_f32((unsigned short)(raw.x >> 16));
    const unsigned int w0 = raw.y;
    const unsigned int w1 = raw.z;
    const unsigned int w2 = raw.w;

    unsigned sc_lo, sc_hi, m_lo, m_hi;
    if (chunk_idx == 0) {
        sc_lo = (w0 >> 0) & 0x3Fu;
        sc_hi = (w0 >> 8) & 0x3Fu;
        m_lo  = (w1 >> 0) & 0x3Fu;
        m_hi  = (w1 >> 8) & 0x3Fu;
    } else if (chunk_idx == 1) {
        sc_lo = (w0 >> 16) & 0x3Fu;
        sc_hi = (w0 >> 24) & 0x3Fu;
        m_lo  = (w1 >> 16) & 0x3Fu;
        m_hi  = (w1 >> 24) & 0x3Fu;
    } else if (chunk_idx == 2) {
        sc_lo = ((w2 >> 0) & 0x0Fu) | (((w0 >> 6) & 0x03u) << 4);
        sc_hi = ((w2 >> 8) & 0x0Fu) | (((w0 >> 14) & 0x03u) << 4);
        m_lo  = ((w2 >> 4) & 0x0Fu) | (((w1 >> 6) & 0x03u) << 4);
        m_hi  = ((w2 >> 12) & 0x0Fu) | (((w1 >> 14) & 0x03u) << 4);
    } else {
        sc_lo = ((w2 >> 16) & 0x0Fu) | (((w0 >> 22) & 0x03u) << 4);
        sc_hi = ((w2 >> 24) & 0x0Fu) | (((w0 >> 30) & 0x03u) << 4);
        m_lo  = ((w2 >> 20) & 0x0Fu) | (((w1 >> 22) & 0x03u) << 4);
        m_hi  = ((w2 >> 28) & 0x0Fu) | (((w1 >> 30) & 0x03u) << 4);
    }

    *w_lo = __fmaf_rn(d * (float)sc_lo, (float)(q & 0x0Fu), neg_dmin * (float)m_lo);
    *w_hi = __fmaf_rn(d * (float)sc_hi, (float)(q >> 4),   neg_dmin * (float)m_hi);
}

__device__ __forceinline__ void unpack_q4k_header(const unsigned char* blk, Q4K_Header* h) {
    const uint4 raw = *(const uint4*)blk;
    const float d        = f16_to_f32((unsigned short)(raw.x & 0xFFFFu));
    const float neg_dmin = -f16_to_f32((unsigned short)(raw.x >> 16));

    const unsigned int w0 = raw.y;
    const unsigned int w1 = raw.z;
    const unsigned int w2 = raw.w;

    const unsigned sc0 = (w0 >> 0) & 0x3Fu;
    const unsigned sc1 = (w0 >> 8) & 0x3Fu;
    const unsigned sc2 = (w0 >> 16) & 0x3Fu;
    const unsigned sc3 = (w0 >> 24) & 0x3Fu;

    const unsigned m0 = (w1 >> 0) & 0x3Fu;
    const unsigned m1 = (w1 >> 8) & 0x3Fu;
    const unsigned m2 = (w1 >> 16) & 0x3Fu;
    const unsigned m3 = (w1 >> 24) & 0x3Fu;

    const unsigned sc4 = ((w2 >> 0) & 0x0Fu) | (((w0 >> 6) & 0x03u) << 4);
    const unsigned sc5 = ((w2 >> 8) & 0x0Fu) | (((w0 >> 14) & 0x03u) << 4);
    const unsigned sc6 = ((w2 >> 16) & 0x0Fu) | (((w0 >> 22) & 0x03u) << 4);
    const unsigned sc7 = ((w2 >> 24) & 0x0Fu) | (((w0 >> 30) & 0x03u) << 4);

    const unsigned m4 = ((w2 >> 4) & 0x0Fu) | (((w1 >> 6) & 0x03u) << 4);
    const unsigned m5 = ((w2 >> 12) & 0x0Fu) | (((w1 >> 14) & 0x03u) << 4);
    const unsigned m6 = ((w2 >> 20) & 0x0Fu) | (((w1 >> 22) & 0x03u) << 4);
    const unsigned m7 = ((w2 >> 28) & 0x0Fu) | (((w1 >> 30) & 0x03u) << 4);

    h->d1[0] = d * (float)sc0;  h->neg_m1[0] = neg_dmin * (float)m0;
    h->d2[0] = d * (float)sc1;  h->neg_m2[0] = neg_dmin * (float)m1;
    h->d1[1] = d * (float)sc2;  h->neg_m1[1] = neg_dmin * (float)m2;
    h->d2[1] = d * (float)sc3;  h->neg_m2[1] = neg_dmin * (float)m3;
    h->d1[2] = d * (float)sc4;  h->neg_m1[2] = neg_dmin * (float)m4;
    h->d2[2] = d * (float)sc5;  h->neg_m2[2] = neg_dmin * (float)m5;
    h->d1[3] = d * (float)sc6;  h->neg_m1[3] = neg_dmin * (float)m6;
    h->d2[3] = d * (float)sc7;  h->neg_m2[3] = neg_dmin * (float)m7;
}

struct Q6K_Header {
    float ds[8];
};

__device__ __forceinline__ void unpack_q6k_header(const unsigned char* blk, int is, Q6K_Header* h) {
    const float d = f16_to_f32(*(const unsigned short*)(blk + 208));
    const signed char* scales = (const signed char*)(blk + 192);

    h->ds[0] = d * (float)scales[0  + is];
    h->ds[1] = d * (float)scales[2  + is];
    h->ds[2] = d * (float)scales[4  + is];
    h->ds[3] = d * (float)scales[6  + is];
    h->ds[4] = d * (float)scales[8  + is];
    h->ds[5] = d * (float)scales[10 + is];
    h->ds[6] = d * (float)scales[12 + is];
    h->ds[7] = d * (float)scales[14 + is];
}

__device__ __forceinline__ void get_scale_min(int j, const unsigned char* scales, unsigned* s, unsigned* m) {
    if (j < 4) {
        *s = (unsigned)(scales[j]     & 63u);
        *m = (unsigned)(scales[j + 4] & 63u);
    } else {
        *s = (unsigned)(scales[j + 4] & 0xFu) | (((unsigned)scales[j - 4] >> 6) << 4);
        *m = (unsigned)(scales[j + 4] >> 4)   | (((unsigned)scales[j]     >> 6) << 4);
    }
}

__device__ __forceinline__ void load_activation_smem(
    const signed char* __restrict__ qx,
    const float* __restrict__ qd,
    const float* __restrict__ qs,
    signed char* __restrict__ s_qx,
    float* __restrict__ s_qd,
    float* __restrict__ s_qs,
    int ne0)
{
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int ne0_16 = ne0 / 16;
    const int4* in_qx16 = (const int4*)qx;
    int4* out_qx16 = (int4*)s_qx;

    #pragma unroll
    for (int i = tid; i < ne0_16; i += num_threads) {
        out_qx16[i] = in_qx16[i];
    }

    const int num_blocks_32 = ne0 / 32;
    #pragma unroll
    for (int i = tid; i < num_blocks_32; i += num_threads) {
        s_qd[i] = qd[i];
        if (s_qs != nullptr && qs != nullptr) {
            s_qs[i] = qs[i];
        }
    }
    __syncthreads();
}

extern "C" __global__ void quantize_row_q8_1_kernel(
    const float* __restrict__ x,
    const float* __restrict__ norm_weight,
    signed char* __restrict__ out_qx,
    float* __restrict__ out_qd,
    float* __restrict__ out_qs,
    int ne0,
    float eps)
{
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = num_threads >> 5;
    const int num_blocks_32 = ne0 / 32;

    __shared__ float s_warp_sum[32];
    __shared__ float s_rms_scale;

    float rms_scale = 1.0f;
    if (norm_weight != nullptr) {
        const int ne0_4 = ne0 / 4;
        const float4* x4 = (const float4*)x;
        float sum_sq = 0.0f;
        for (int i = tid; i < ne0_4; i += num_threads) {
            const float4 xi = x4[i];
            sum_sq = __fmaf_rn(xi.x, xi.x, sum_sq);
            sum_sq = __fmaf_rn(xi.y, xi.y, sum_sq);
            sum_sq = __fmaf_rn(xi.z, xi.z, sum_sq);
            sum_sq = __fmaf_rn(xi.w, xi.w, sum_sq);
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, mask);
        }

        if (lane == 0) {
            s_warp_sum[warp_id] = sum_sq;
        }
        __syncthreads();

        if (tid == 0) {
            float total = 0.0f;
            for (int w = 0; w < num_warps; ++w) {
                total += s_warp_sum[w];
            }
            s_rms_scale = rsqrtf(total / (float)ne0 + eps);
        }
        __syncthreads();
        rms_scale = s_rms_scale;
    }

    for (int b = warp_id; b < num_blocks_32; b += num_warps) {
        const int elem_idx = b * 32 + lane;
        float val = x[elem_idx];
        if (norm_weight != nullptr) {
            val = val * rms_scale * norm_weight[elem_idx];
        }

        float amax = fabsf(val);
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, mask));
        }
        amax = __shfl_sync(0xffffffff, amax, 0);

        const float d = amax / 127.0f;
        const float id = (d > 0.0f) ? (1.0f / d) : 0.0f;

        const float qf = roundf(val * id);
        const int qi = (int)qf;
        const signed char q_clamped = (signed char)max(-128, min(127, qi));

        out_qx[elem_idx] = q_clamped;

        float bsum = val;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            bsum += __shfl_down_sync(0xffffffff, bsum, mask);
        }

        if (lane == 0) {
            out_qd[b] = d;
            out_qs[b] = bsum;
        }
    }
}

extern "C" __global__ void gemm_q4k_kernel(
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ qx,
    const float* __restrict__ qd,
    const float* __restrict__ qs,
    float* __restrict__ out,
    int ne0,
    int ne1,
    int batch_size,
    const float* __restrict__ residual)
{
    extern __shared__ char smem[];
    signed char* s_qx = (signed char*)smem;
    float* s_qd = (float*)(s_qx + ne0);
    float* s_qs = s_qd + (ne0 / 32);

    load_activation_smem(qx, qd, qs, s_qx, s_qd, s_qs, ne0);

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

        Q4K_BlockScales s;
        unpack_q4k_scales(raw, &s);

        const unsigned char q0 = qs_ptr[0 * 32 + lane];
        const unsigned char q1 = qs_ptr[1 * 32 + lane];
        const unsigned char q2 = qs_ptr[2 * 32 + lane];
        const unsigned char q3 = qs_ptr[3 * 32 + lane];

        const float d_sc0 = s.d_sc[0] * s_qd[q8_b + 0];
        const float d_sc1 = s.d_sc[1] * s_qd[q8_b + 1];
        const float d_sc2 = s.d_sc[2] * s_qd[q8_b + 2];
        const float d_sc3 = s.d_sc[3] * s_qd[q8_b + 3];
        const float d_sc4 = s.d_sc[4] * s_qd[q8_b + 4];
        const float d_sc5 = s.d_sc[5] * s_qd[q8_b + 5];
        const float d_sc6 = s.d_sc[6] * s_qd[q8_b + 6];
        const float d_sc7 = s.d_sc[7] * s_qd[q8_b + 7];

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
            #pragma unroll
            for (int sb = 0; sb < 8; ++sb) {
                ms = __fmaf_rn(s.m[sb], s_qs[q8_b + sb], ms);
            }
            min_sum += ms;
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_acc += __shfl_down_sync(0xffffffff, local_acc, mask);
    }

    if (lane == 0) {
        float acc = local_acc + min_sum;
        if (residual != nullptr) {
            acc += residual[col];
        }
        out_row[col] = acc;
    }
}

extern "C" __global__ void gemm_q4k_splitk_kernel(
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ qx,
    const float* __restrict__ qd,
    const float* __restrict__ qs,
    float* __restrict__ out,
    int ne0,
    int ne1,
    int batch_size,
    const float* __restrict__ residual)
{
    extern __shared__ char smem[];
    signed char* s_qx = (signed char*)smem;
    float* s_qd = (float*)(s_qx + ne0);
    float* s_qs = s_qd + (ne0 / 32);
    float* s_part = (float*)(s_qs + (ne0 / 32));

    load_activation_smem(qx, qd, qs, s_qx, s_qd, s_qs, ne0);

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int col = blockIdx.x;
    const int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size || col >= ne1) return;

    float* out_row = out + (size_t)batch_idx * (size_t)ne1;

    const int n_blocks = ne0 / 256;
    const int blocks_per_split = (n_blocks + 7) / 8;
    const int b_start = warp_id * blocks_per_split;
    const int b_end = (b_start + blocks_per_split < n_blocks) ? (b_start + blocks_per_split) : n_blocks;

    const unsigned char* col_weights = weights + (size_t)col * (size_t)n_blocks * 144u;

    float local_acc = 0.0f;
    float min_sum = 0.0f;

    for (int b = b_start; b < b_end; ++b) {
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

        Q4K_BlockScales s;
        unpack_q4k_scales(raw, &s);

        const unsigned char q0 = qs_ptr[0 * 32 + lane];
        const unsigned char q1 = qs_ptr[1 * 32 + lane];
        const unsigned char q2 = qs_ptr[2 * 32 + lane];
        const unsigned char q3 = qs_ptr[3 * 32 + lane];

        float dot = 0.0f;
        dot = __fmaf_rn(s.d_sc[0] * s_qd[q8_b + 0], (float)((int)(q0 & 0x0Fu) * (int)qx0), dot);
        dot = __fmaf_rn(s.d_sc[1] * s_qd[q8_b + 1], (float)((int)(q0 >> 4)    * (int)qx1), dot);
        dot = __fmaf_rn(s.d_sc[2] * s_qd[q8_b + 2], (float)((int)(q1 & 0x0Fu) * (int)qx2), dot);
        dot = __fmaf_rn(s.d_sc[3] * s_qd[q8_b + 3], (float)((int)(q1 >> 4)    * (int)qx3), dot);
        dot = __fmaf_rn(s.d_sc[4] * s_qd[q8_b + 4], (float)((int)(q2 & 0x0Fu) * (int)qx4), dot);
        dot = __fmaf_rn(s.d_sc[5] * s_qd[q8_b + 5], (float)((int)(q2 >> 4)    * (int)qx5), dot);
        dot = __fmaf_rn(s.d_sc[6] * s_qd[q8_b + 6], (float)((int)(q3 & 0x0Fu) * (int)qx6), dot);
        dot = __fmaf_rn(s.d_sc[7] * s_qd[q8_b + 7], (float)((int)(q3 >> 4)    * (int)qx7), dot);
        local_acc += dot;

        if (lane == 0) {
            float ms = 0.0f;
            #pragma unroll
            for (int sb = 0; sb < 8; ++sb) {
                ms = __fmaf_rn(s.m[sb], s_qs[q8_b + sb], ms);
            }
            min_sum += ms;
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_acc += __shfl_down_sync(0xffffffff, local_acc, mask);
    }

    if (lane == 0) {
        s_part[warp_id] = local_acc + min_sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = s_part[0] + s_part[1] + s_part[2] + s_part[3]
                    + s_part[4] + s_part[5] + s_part[6] + s_part[7];
        if (residual != nullptr) {
            total += residual[col];
        }
        out_row[col] = total;
    }
}

extern "C" __global__ void gemm_q6k_kernel(
    const unsigned char* __restrict__ weights,
    const signed char* __restrict__ qx,
    const float* __restrict__ qd,
    const float* __restrict__ qs,
    float* __restrict__ out,
    int ne0,
    int ne1,
    int batch_size,
    const float* __restrict__ residual)
{
    extern __shared__ char smem[];
    signed char* s_qx = (signed char*)smem;
    float* s_qd = (float*)(s_qx + ne0);
    float* s_qs = s_qd + (ne0 / 32);

    load_activation_smem(qx, qd, qs, s_qx, s_qd, s_qs, ne0);

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int col = blockIdx.x * (blockDim.x / 32) + warp_id;
    const int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size || col >= ne1) return;

    float* out_row = out + (size_t)batch_idx * (size_t)ne1;

    const int n_blocks = ne0 / 256;
    const unsigned char* col_w = weights + (size_t)col * (size_t)n_blocks * 210u;

    float local_acc = 0.0f;
    const int is = lane >> 4;

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

        const unsigned char* blk = col_w + (size_t)b * 210u;
        const signed char* sc = (const signed char*)(blk + 192);
        const float d = f16_to_f32(*(const unsigned short*)(blk + 208));

        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;

        const unsigned char ql0 = ql[lane];
        const unsigned char ql1 = ql[lane + 32];
        const unsigned char ql2 = ql[64 + lane];
        const unsigned char ql3 = ql[96 + lane];
        const unsigned char qh0 = qh[lane];
        const unsigned char qh1 = qh[32 + lane];

        const int q0 = (int)(ql0 & 0x0Fu) | (((int)(qh0 >> 0) & 3) << 4);
        const int q1 = (int)(ql1 & 0x0Fu) | (((int)(qh0 >> 2) & 3) << 4);
        const int q2 = (int)(ql0 >> 4)    | (((int)(qh0 >> 4) & 3) << 4);
        const int q3 = (int)(ql1 >> 4)    | (((int)(qh0 >> 6) & 3) << 4);
        const int q4 = (int)(ql2 & 0x0Fu) | (((int)(qh1 >> 0) & 3) << 4);
        const int q5 = (int)(ql3 & 0x0Fu) | (((int)(qh1 >> 2) & 3) << 4);
        const int q6 = (int)(ql2 >> 4)    | (((int)(qh1 >> 4) & 3) << 4);
        const int q7 = (int)(ql3 >> 4)    | (((int)(qh1 >> 6) & 3) << 4);

        const float d_qd0 = d * s_qd[q8_b + 0];
        const float d_qd1 = d * s_qd[q8_b + 1];
        const float d_qd2 = d * s_qd[q8_b + 2];
        const float d_qd3 = d * s_qd[q8_b + 3];
        const float d_qd4 = d * s_qd[q8_b + 4];
        const float d_qd5 = d * s_qd[q8_b + 5];
        const float d_qd6 = d * s_qd[q8_b + 6];
        const float d_qd7 = d * s_qd[q8_b + 7];

        float dot = 0.0f;
        dot = __fmaf_rn(d_qd0, (float)((q0 - 32) * (int)qx0 * (int)sc[0  + is]), dot);
        dot = __fmaf_rn(d_qd1, (float)((q1 - 32) * (int)qx1 * (int)sc[2  + is]), dot);
        dot = __fmaf_rn(d_qd2, (float)((q2 - 32) * (int)qx2 * (int)sc[4  + is]), dot);
        dot = __fmaf_rn(d_qd3, (float)((q3 - 32) * (int)qx3 * (int)sc[6  + is]), dot);
        dot = __fmaf_rn(d_qd4, (float)((q4 - 32) * (int)qx4 * (int)sc[8  + is]), dot);
        dot = __fmaf_rn(d_qd5, (float)((q5 - 32) * (int)qx5 * (int)sc[10 + is]), dot);
        dot = __fmaf_rn(d_qd6, (float)((q6 - 32) * (int)qx6 * (int)sc[12 + is]), dot);
        dot = __fmaf_rn(d_qd7, (float)((q7 - 32) * (int)qx7 * (int)sc[14 + is]), dot);
        local_acc += dot;
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_acc += __shfl_down_sync(0xffffffff, local_acc, mask);
    }

    if (lane == 0) {
        float acc = local_acc;
        if (residual != nullptr) {
            acc += residual[col];
        }
        out_row[col] = acc;
    }
}

extern "C" __global__ void gemm_fused_qkv_kernel(
    const unsigned char* __restrict__ wq,
    const unsigned char* __restrict__ wk,
    const unsigned char* __restrict__ wv,
    const signed char* __restrict__ qx,
    const float* __restrict__ qd,
    const float* __restrict__ qs,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    int ne0,
    int qdim,
    int kvd,
    int batch_size,
    const float* __restrict__ qb,
    const float* __restrict__ kb,
    const float* __restrict__ vb)
{
    extern __shared__ char smem[];
    signed char* s_qx = (signed char*)smem;
    float* s_qd = (float*)(s_qx + ne0);
    float* s_qs = s_qd + (ne0 / 32);

    load_activation_smem(qx, qd, qs, s_qx, s_qd, s_qs, ne0);

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int col = blockIdx.x * (blockDim.x / 32) + warp_id;
    const int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;

    const int total_cols = qdim + kvd + kvd;
    if (col >= total_cols) return;

    const int n_blocks = ne0 / 256;

    if (col < qdim) {
        // Q projection (Q4_K)
        float* out_row = q_out + (size_t)batch_idx * (size_t)qdim;
        const unsigned char* col_w = wq + (size_t)col * (size_t)n_blocks * 144u;

        float local_acc = 0.0f;
        float min_sum = 0.0f;

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

            const unsigned char* blk = col_w + (size_t)b * 144u;
            const uint4 raw = *(const uint4*)blk;
            const unsigned char* qs_ptr = blk + 16;
            Q4K_BlockScales s;
            unpack_q4k_scales(raw, &s);

            const unsigned char q0 = qs_ptr[0 * 32 + lane];
            const unsigned char q1 = qs_ptr[1 * 32 + lane];
            const unsigned char q2 = qs_ptr[2 * 32 + lane];
            const unsigned char q3 = qs_ptr[3 * 32 + lane];

            const float s_qd0 = s_qd[q8_b + 0];
            const float s_qd1 = s_qd[q8_b + 1];
            const float s_qd2 = s_qd[q8_b + 2];
            const float s_qd3 = s_qd[q8_b + 3];
            const float s_qd4 = s_qd[q8_b + 4];
            const float s_qd5 = s_qd[q8_b + 5];
            const float s_qd6 = s_qd[q8_b + 6];
            const float s_qd7 = s_qd[q8_b + 7];

            float dot = 0.0f;
            dot = __fmaf_rn(s.d_sc[0] * s_qd0, (float)((int)(q0 & 0x0Fu) * (int)qx0), dot);
            dot = __fmaf_rn(s.d_sc[1] * s_qd1, (float)((int)(q0 >> 4)    * (int)qx1), dot);
            dot = __fmaf_rn(s.d_sc[2] * s_qd2, (float)((int)(q1 & 0x0Fu) * (int)qx2), dot);
            dot = __fmaf_rn(s.d_sc[3] * s_qd3, (float)((int)(q1 >> 4)    * (int)qx3), dot);
            dot = __fmaf_rn(s.d_sc[4] * s_qd4, (float)((int)(q2 & 0x0Fu) * (int)qx4), dot);
            dot = __fmaf_rn(s.d_sc[5] * s_qd5, (float)((int)(q2 >> 4)    * (int)qx5), dot);
            dot = __fmaf_rn(s.d_sc[6] * s_qd6, (float)((int)(q3 & 0x0Fu) * (int)qx6), dot);
            dot = __fmaf_rn(s.d_sc[7] * s_qd7, (float)((int)(q3 >> 4)    * (int)qx7), dot);
            local_acc += dot;

            if (lane == 0) {
                float ms = 0.0f;
                #pragma unroll
                for (int sb = 0; sb < 8; ++sb) {
                    ms = __fmaf_rn(s.m[sb], s_qs[q8_b + sb], ms);
                }
                min_sum += ms;
            }
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            local_acc += __shfl_down_sync(0xffffffff, local_acc, mask);
        }

        if (lane == 0) {
            float acc = local_acc + min_sum;
            if (qb != nullptr) {
                acc += qb[col];
            }
            out_row[col] = acc;
        }
    } else if (col < qdim + kvd) {
        // K projection (Q4_K)
        const int k_col = col - qdim;
        float* out_row = k_out + (size_t)batch_idx * (size_t)kvd;
        const unsigned char* col_w = wk + (size_t)k_col * (size_t)n_blocks * 144u;

        float local_acc = 0.0f;
        float min_sum = 0.0f;

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

            const unsigned char* blk = col_w + (size_t)b * 144u;
            const uint4 raw = *(const uint4*)blk;
            const unsigned char* qs_ptr = blk + 16;
            Q4K_BlockScales s;
            unpack_q4k_scales(raw, &s);

            const unsigned char q0 = qs_ptr[0 * 32 + lane];
            const unsigned char q1 = qs_ptr[1 * 32 + lane];
            const unsigned char q2 = qs_ptr[2 * 32 + lane];
            const unsigned char q3 = qs_ptr[3 * 32 + lane];

            const float s_qd0 = s_qd[q8_b + 0];
            const float s_qd1 = s_qd[q8_b + 1];
            const float s_qd2 = s_qd[q8_b + 2];
            const float s_qd3 = s_qd[q8_b + 3];
            const float s_qd4 = s_qd[q8_b + 4];
            const float s_qd5 = s_qd[q8_b + 5];
            const float s_qd6 = s_qd[q8_b + 6];
            const float s_qd7 = s_qd[q8_b + 7];

            float dot = 0.0f;
            dot = __fmaf_rn(s.d_sc[0] * s_qd0, (float)((int)(q0 & 0x0Fu) * (int)qx0), dot);
            dot = __fmaf_rn(s.d_sc[1] * s_qd1, (float)((int)(q0 >> 4)    * (int)qx1), dot);
            dot = __fmaf_rn(s.d_sc[2] * s_qd2, (float)((int)(q1 & 0x0Fu) * (int)qx2), dot);
            dot = __fmaf_rn(s.d_sc[3] * s_qd3, (float)((int)(q1 >> 4)    * (int)qx3), dot);
            dot = __fmaf_rn(s.d_sc[4] * s_qd4, (float)((int)(q2 & 0x0Fu) * (int)qx4), dot);
            dot = __fmaf_rn(s.d_sc[5] * s_qd5, (float)((int)(q2 >> 4)    * (int)qx5), dot);
            dot = __fmaf_rn(s.d_sc[6] * s_qd6, (float)((int)(q3 & 0x0Fu) * (int)qx6), dot);
            dot = __fmaf_rn(s.d_sc[7] * s_qd7, (float)((int)(q3 >> 4)    * (int)qx7), dot);
            local_acc += dot;

            if (lane == 0) {
                float ms = 0.0f;
                #pragma unroll
                for (int sb = 0; sb < 8; ++sb) {
                    ms = __fmaf_rn(s.m[sb], s_qs[q8_b + sb], ms);
                }
                min_sum += ms;
            }
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            local_acc += __shfl_down_sync(0xffffffff, local_acc, mask);
        }

        if (lane == 0) {
            float acc = local_acc + min_sum;
            if (kb != nullptr) {
                acc += kb[k_col];
            }
            out_row[k_col] = acc;
        }
    } else {
        // V projection (Q6_K)
        const int v_col = col - qdim - kvd;
        float* out_row = v_out + (size_t)batch_idx * (size_t)kvd;
        const unsigned char* col_w = wv + (size_t)v_col * (size_t)n_blocks * 210u;

        float local_acc = 0.0f;
        const int is = lane >> 4;

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

            const unsigned char* blk = col_w + (size_t)b * 210u;
            const signed char* sc = (const signed char*)(blk + 192);
            const float d = f16_to_f32(*(const unsigned short*)(blk + 208));

            const unsigned char* ql = blk;
            const unsigned char* qh = blk + 128;

            const unsigned char ql0 = ql[lane];
            const unsigned char ql1 = ql[lane + 32];
            const unsigned char ql2 = ql[64 + lane];
            const unsigned char ql3 = ql[96 + lane];
            const unsigned char qh0 = qh[lane];
            const unsigned char qh1 = qh[32 + lane];

            const int q0 = (int)(ql0 & 0x0Fu) | (((int)(qh0 >> 0) & 3) << 4);
            const int q1 = (int)(ql1 & 0x0Fu) | (((int)(qh0 >> 2) & 3) << 4);
            const int q2 = (int)(ql0 >> 4)    | (((int)(qh0 >> 4) & 3) << 4);
            const int q3 = (int)(ql1 >> 4)    | (((int)(qh0 >> 6) & 3) << 4);
            const int q4 = (int)(ql2 & 0x0Fu) | (((int)(qh1 >> 0) & 3) << 4);
            const int q5 = (int)(ql3 & 0x0Fu) | (((int)(qh1 >> 2) & 3) << 4);
            const int q6 = (int)(ql2 >> 4)    | (((int)(qh1 >> 4) & 3) << 4);
            const int q7 = (int)(ql3 >> 4)    | (((int)(qh1 >> 6) & 3) << 4);

            float dot = 0.0f;
            dot = __fmaf_rn(d * s_qd[q8_b + 0], (float)((q0 - 32) * (int)qx0 * (int)sc[0  + is]), dot);
            dot = __fmaf_rn(d * s_qd[q8_b + 1], (float)((q1 - 32) * (int)qx1 * (int)sc[2  + is]), dot);
            dot = __fmaf_rn(d * s_qd[q8_b + 2], (float)((q2 - 32) * (int)qx2 * (int)sc[4  + is]), dot);
            dot = __fmaf_rn(d * s_qd[q8_b + 3], (float)((q3 - 32) * (int)qx3 * (int)sc[6  + is]), dot);
            dot = __fmaf_rn(d * s_qd[q8_b + 4], (float)((q4 - 32) * (int)qx4 * (int)sc[8  + is]), dot);
            dot = __fmaf_rn(d * s_qd[q8_b + 5], (float)((q5 - 32) * (int)qx5 * (int)sc[10 + is]), dot);
            dot = __fmaf_rn(d * s_qd[q8_b + 6], (float)((q6 - 32) * (int)qx6 * (int)sc[12 + is]), dot);
            dot = __fmaf_rn(d * s_qd[q8_b + 7], (float)((q7 - 32) * (int)qx7 * (int)sc[14 + is]), dot);
            local_acc += dot;
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            local_acc += __shfl_down_sync(0xffffffff, local_acc, mask);
        }

        if (lane == 0) {
            float acc = local_acc;
            if (vb != nullptr) {
                acc += vb[v_col];
            }
            out_row[v_col] = acc;
        }
    }
}

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));
}

extern "C" __global__ void gemm_q4k_fused_gate_up_swiglu_kernel(
    const unsigned char* __restrict__ wgate,
    const unsigned char* __restrict__ wup,
    const signed char* __restrict__ qx,
    const float* __restrict__ qd,
    const float* __restrict__ qs,
    float* __restrict__ out,
    int ne0,
    int ne1,
    int batch_size)
{
    extern __shared__ char smem[];
    signed char* s_qx = (signed char*)smem;
    float* s_qd = (float*)(s_qx + ne0);
    float* s_qs = s_qd + (ne0 / 32);

    load_activation_smem(qx, qd, qs, s_qx, s_qd, s_qs, ne0);

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int col = blockIdx.x * (blockDim.x / 32) + warp_id;
    const int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size || col >= ne1) return;

    float* out_row = out + (size_t)batch_idx * (size_t)ne1;

    const int n_blocks = ne0 / 256;
    const unsigned char* gate_w = wgate + (size_t)col * (size_t)n_blocks * 144u;
    const unsigned char* up_w   = wup   + (size_t)col * (size_t)n_blocks * 144u;

    float local_gate = 0.0f;
    float local_up   = 0.0f;
    float min_gate = 0.0f;
    float min_up   = 0.0f;

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

        const float s_qd0 = s_qd[q8_b + 0];
        const float s_qd1 = s_qd[q8_b + 1];
        const float s_qd2 = s_qd[q8_b + 2];
        const float s_qd3 = s_qd[q8_b + 3];
        const float s_qd4 = s_qd[q8_b + 4];
        const float s_qd5 = s_qd[q8_b + 5];
        const float s_qd6 = s_qd[q8_b + 6];
        const float s_qd7 = s_qd[q8_b + 7];

        // Gate
        {
            const unsigned char* blk = gate_w + (size_t)b * 144u;
            const uint4 raw = *(const uint4*)blk;
            const unsigned char* qs_ptr = blk + 16;
            Q4K_BlockScales s;
            unpack_q4k_scales(raw, &s);

            const unsigned char q0 = qs_ptr[0 * 32 + lane];
            const unsigned char q1 = qs_ptr[1 * 32 + lane];
            const unsigned char q2 = qs_ptr[2 * 32 + lane];
            const unsigned char q3 = qs_ptr[3 * 32 + lane];

            float dot = 0.0f;
            dot = __fmaf_rn(s.d_sc[0] * s_qd0, (float)((int)(q0 & 0x0Fu) * (int)qx0), dot);
            dot = __fmaf_rn(s.d_sc[1] * s_qd1, (float)((int)(q0 >> 4)    * (int)qx1), dot);
            dot = __fmaf_rn(s.d_sc[2] * s_qd2, (float)((int)(q1 & 0x0Fu) * (int)qx2), dot);
            dot = __fmaf_rn(s.d_sc[3] * s_qd3, (float)((int)(q1 >> 4)    * (int)qx3), dot);
            dot = __fmaf_rn(s.d_sc[4] * s_qd4, (float)((int)(q2 & 0x0Fu) * (int)qx4), dot);
            dot = __fmaf_rn(s.d_sc[5] * s_qd5, (float)((int)(q2 >> 4)    * (int)qx5), dot);
            dot = __fmaf_rn(s.d_sc[6] * s_qd6, (float)((int)(q3 & 0x0Fu) * (int)qx6), dot);
            dot = __fmaf_rn(s.d_sc[7] * s_qd7, (float)((int)(q3 >> 4)    * (int)qx7), dot);
            local_gate += dot;

            if (lane == 0) {
                float ms = 0.0f;
                #pragma unroll
                for (int sb = 0; sb < 8; ++sb) {
                    ms = __fmaf_rn(s.m[sb], s_qs[q8_b + sb], ms);
                }
                min_gate += ms;
            }
        }

        // Up
        {
            const unsigned char* blk = up_w + (size_t)b * 144u;
            const uint4 raw = *(const uint4*)blk;
            const unsigned char* qs_ptr = blk + 16;
            Q4K_BlockScales s;
            unpack_q4k_scales(raw, &s);

            const unsigned char q0 = qs_ptr[0 * 32 + lane];
            const unsigned char q1 = qs_ptr[1 * 32 + lane];
            const unsigned char q2 = qs_ptr[2 * 32 + lane];
            const unsigned char q3 = qs_ptr[3 * 32 + lane];

            float dot = 0.0f;
            dot = __fmaf_rn(s.d_sc[0] * s_qd0, (float)((int)(q0 & 0x0Fu) * (int)qx0), dot);
            dot = __fmaf_rn(s.d_sc[1] * s_qd1, (float)((int)(q0 >> 4)    * (int)qx1), dot);
            dot = __fmaf_rn(s.d_sc[2] * s_qd2, (float)((int)(q1 & 0x0Fu) * (int)qx2), dot);
            dot = __fmaf_rn(s.d_sc[3] * s_qd3, (float)((int)(q1 >> 4)    * (int)qx3), dot);
            dot = __fmaf_rn(s.d_sc[4] * s_qd4, (float)((int)(q2 & 0x0Fu) * (int)qx4), dot);
            dot = __fmaf_rn(s.d_sc[5] * s_qd5, (float)((int)(q2 >> 4)    * (int)qx5), dot);
            dot = __fmaf_rn(s.d_sc[6] * s_qd6, (float)((int)(q3 & 0x0Fu) * (int)qx6), dot);
            dot = __fmaf_rn(s.d_sc[7] * s_qd7, (float)((int)(q3 >> 4)    * (int)qx7), dot);
            local_up += dot;

            if (lane == 0) {
                float ms = 0.0f;
                #pragma unroll
                for (int sb = 0; sb < 8; ++sb) {
                    ms = __fmaf_rn(s.m[sb], s_qs[q8_b + sb], ms);
                }
                min_up += ms;
            }
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_gate += __shfl_down_sync(0xffffffff, local_gate, mask);
        local_up   += __shfl_down_sync(0xffffffff, local_up, mask);
    }

    if (lane == 0) {
        const float g = local_gate + min_gate;
        const float u = local_up   + min_up;
        out_row[col] = silu_f(g) * u;
    }
}

extern "C" __global__ void gemm_q6k_splitk_kernel(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ out,
    int ne0,
    int ne1,
    int batch_size,
    const float* __restrict__ residual)
{
    const int warp_id      = threadIdx.x / 32;
    const int lane         = threadIdx.x % 32;
    const int col_in_block = warp_id / 4;        // 0 or 1 (2 cols per block of 256 threads)
    const int warp_in_col  = warp_id % 4;        // 0, 1, 2, or 3 (4 warps per column)

    const int col = blockIdx.x * 2 + col_in_block;
    const int batch_idx = blockIdx.y;
    if (col >= ne1 || batch_idx >= batch_size) return;

    const float* x_row = x + (size_t)batch_idx * (size_t)ne0;
    float* out_row = out + (size_t)batch_idx * (size_t)ne1;

    const int n_blocks = ne0 / 256;
    const unsigned char* col_w = weights + (size_t)col * (size_t)n_blocks * 210u;

    const int quarter_blocks = (n_blocks + 3) / 4;
    const int b_start = warp_in_col * quarter_blocks;
    int b_end = b_start + quarter_blocks;
    if (b_end > n_blocks) b_end = n_blocks;

    float acc = 0.0f;
    const int is = lane >> 4;

    for (int b = b_start; b < b_end; ++b) {
        const unsigned char* blk = col_w + (size_t)b * 210u;
        const float* x_blk = x_row + (size_t)b * 256u;

        Q6K_Header h;
        unpack_q6k_header(blk, is, &h);

        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;

        const unsigned char ql0 = ql[lane];
        const unsigned char ql1 = ql[lane + 32];
        const unsigned char ql2 = ql[64 + lane];
        const unsigned char ql3 = ql[96 + lane];
        const unsigned char qh0 = qh[lane];
        const unsigned char qh1 = qh[32 + lane];

        const int q0 = (int)(ql0 & 0x0Fu) | (((int)(qh0 >> 0) & 3) << 4);
        const int q1 = (int)(ql1 & 0x0Fu) | (((int)(qh0 >> 2) & 3) << 4);
        const int q2 = (int)(ql0 >> 4)    | (((int)(qh0 >> 4) & 3) << 4);
        const int q3 = (int)(ql1 >> 4)    | (((int)(qh0 >> 6) & 3) << 4);
        const int q4 = (int)(ql2 & 0x0Fu) | (((int)(qh1 >> 0) & 3) << 4);
        const int q5 = (int)(ql3 & 0x0Fu) | (((int)(qh1 >> 2) & 3) << 4);
        const int q6 = (int)(ql2 >> 4)    | (((int)(qh1 >> 4) & 3) << 4);
        const int q7 = (int)(ql3 >> 4)    | (((int)(qh1 >> 6) & 3) << 4);

        const float x0 = x_blk[0 * 32 + lane];
        const float x1 = x_blk[1 * 32 + lane];
        const float x2 = x_blk[2 * 32 + lane];
        const float x3 = x_blk[3 * 32 + lane];
        const float x4 = x_blk[4 * 32 + lane];
        const float x5 = x_blk[5 * 32 + lane];
        const float x6 = x_blk[6 * 32 + lane];
        const float x7 = x_blk[7 * 32 + lane];

        const float w0 = h.ds[0] * (float)(q0 - 32);
        const float w1 = h.ds[1] * (float)(q1 - 32);
        const float w2 = h.ds[2] * (float)(q2 - 32);
        const float w3 = h.ds[3] * (float)(q3 - 32);
        const float w4 = h.ds[4] * (float)(q4 - 32);
        const float w5 = h.ds[5] * (float)(q5 - 32);
        const float w6 = h.ds[6] * (float)(q6 - 32);
        const float w7 = h.ds[7] * (float)(q7 - 32);

        acc = __fmaf_rn(w0, x0, acc);
        acc = __fmaf_rn(w1, x1, acc);
        acc = __fmaf_rn(w2, x2, acc);
        acc = __fmaf_rn(w3, x3, acc);
        acc = __fmaf_rn(w4, x4, acc);
        acc = __fmaf_rn(w5, x5, acc);
        acc = __fmaf_rn(w6, x6, acc);
        acc = __fmaf_rn(w7, x7, acc);
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, mask);
    }

    __shared__ float smem_split[2][4];
    if (lane == 0) {
        smem_split[col_in_block][warp_in_col] = acc;
    }
    __syncthreads();

    if (warp_id == 0 && lane < 2) {
        int c_idx = lane;
        int target_col = blockIdx.x * 2 + c_idx;
        if (target_col < ne1) {
            float val = smem_split[c_idx][0] + smem_split[c_idx][1] + smem_split[c_idx][2] + smem_split[c_idx][3];
            if (residual != nullptr) {
                val += residual[target_col];
            }
            out_row[target_col] = val;
        }
    }
}


extern "C" __global__ void gemm_q8_kernel(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ out,
    int ne0,
    int ne1,
    int batch_size)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int col = blockIdx.x * (blockDim.x / 32) + warp_id;
    const int batch_idx = blockIdx.y;
    if (col >= ne1 || batch_idx >= batch_size) return;

    const float* x_row = x + (size_t)batch_idx * (size_t)ne0;
    float* out_row = out + (size_t)batch_idx * (size_t)ne1;

    const int n_blocks = ne0 / 32;
    const unsigned char* col_weights = weights + (size_t)col * (size_t)n_blocks * 34u;

    float acc = 0.0f;

    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* blk = col_weights + (size_t)b * 34u;
        const unsigned d_bits = (unsigned)blk[0] | ((unsigned)blk[1] << 8);
        const float d = f16_to_f32(d_bits);
        const signed char* qs = (const signed char*)(blk + 2);
        const float* x_blk = x_row + (size_t)b * 32u;

        const float w = __fmul_rn(d, (float)qs[lane]);
        acc = __fadd_rn(acc, __fmul_rn(w, x_blk[lane]));
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, mask);
    }

    if (lane == 0) {
        out_row[col] = acc;
    }
}

extern "C" __global__ void gemm_f16_kernel(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ out,
    int ne0,
    int ne1,
    int batch_size)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int col = blockIdx.x * (blockDim.x / 32) + warp_id;
    const int batch_idx = blockIdx.y;
    if (col >= ne1 || batch_idx >= batch_size) return;

    const float* x_row = x + (size_t)batch_idx * (size_t)ne0;
    float* out_row = out + (size_t)batch_idx * (size_t)ne1;

    const unsigned short* col_w = (const unsigned short*)weights + (size_t)col * (size_t)ne0;

    float acc = 0.0f;
    for (int i = lane; i < ne0; i += 32) {
        const float w = f16_to_f32((unsigned)col_w[i]);
        acc = __fadd_rn(acc, __fmul_rn(w, x_row[i]));
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, mask);
    }

    if (lane == 0) {
        out_row[col] = acc;
    }
}

extern "C" __global__ void gemm_f32_kernel(
    const float* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ out,
    int ne0,
    int ne1,
    int batch_size)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int col = blockIdx.x * (blockDim.x / 32) + warp_id;
    const int batch_idx = blockIdx.y;
    if (col >= ne1 || batch_idx >= batch_size) return;

    const float* x_row = x + (size_t)batch_idx * (size_t)ne0;
    float* out_row = out + (size_t)batch_idx * (size_t)ne1;

    const float* col_w = weights + (size_t)col * (size_t)ne0;

    float acc = 0.0f;
    for (int i = lane; i < ne0; i += 32) {
        acc = __fadd_rn(acc, __fmul_rn(col_w[i], x_row[i]));
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, mask);
    }

    if (lane == 0) {
        out_row[col] = acc;
    }
}

extern "C" __global__ void get_rows_q4k_kernel(
    const unsigned char* __restrict__ emb_weights,
    const int* __restrict__ token_ids_ptr,
    float* __restrict__ x_out,
    int hidden_dim,
    int n_tokens)
{
    const int token_idx = blockIdx.x;
    if (token_idx >= n_tokens) return;

    const int tok = token_ids_ptr[token_idx];
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int n_blocks = hidden_dim / 256;
    const unsigned char* row_weights = emb_weights + (size_t)tok * (size_t)n_blocks * 144u;
    float* x_row = x_out + (size_t)token_idx * (size_t)hidden_dim;

    for (int b = warp_id; b < n_blocks; b += blockDim.x / 32) {
        const unsigned char* blk = row_weights + (size_t)b * 144u;
        float* x_blk = x_row + (size_t)b * 256u;

        Q4K_Header h;
        unpack_q4k_header(blk, &h);
        const unsigned char* qs = blk + 16;

        const unsigned char q0 = qs[0 * 32 + lane];
        const unsigned char q1 = qs[1 * 32 + lane];
        const unsigned char q2 = qs[2 * 32 + lane];
        const unsigned char q3 = qs[3 * 32 + lane];

        x_blk[0 * 64 + lane]      = __fmaf_rn(h.d1[0], (float)(q0 & 0x0Fu), h.neg_m1[0]);
        x_blk[0 * 64 + 32 + lane] = __fmaf_rn(h.d2[0], (float)(q0 >> 4),   h.neg_m2[0]);
        x_blk[1 * 64 + lane]      = __fmaf_rn(h.d1[1], (float)(q1 & 0x0Fu), h.neg_m1[1]);
        x_blk[1 * 64 + 32 + lane] = __fmaf_rn(h.d2[1], (float)(q1 >> 4),   h.neg_m2[1]);
        x_blk[2 * 64 + lane]      = __fmaf_rn(h.d1[2], (float)(q2 & 0x0Fu), h.neg_m1[2]);
        x_blk[2 * 64 + 32 + lane] = __fmaf_rn(h.d2[2], (float)(q2 >> 4),   h.neg_m2[2]);
        x_blk[3 * 64 + lane]      = __fmaf_rn(h.d1[3], (float)(q3 & 0x0Fu), h.neg_m1[3]);
        x_blk[3 * 64 + 32 + lane] = __fmaf_rn(h.d2[3], (float)(q3 >> 4),   h.neg_m2[3]);
    }
}

extern "C" __global__ void gpu_sample_greedy_kernel(
    const float* __restrict__ logits,
    int* __restrict__ selected_token,
    int vocab_size)
{
    float max_val = -1e30f;
    int max_idx = 0;

    const float4* __restrict__ logits4 = (const float4*)logits;
    const int n4 = vocab_size / 4;

    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        float4 v = logits4[i];
        int base_idx = i * 4;
        if (v.x > max_val) { max_val = v.x; max_idx = base_idx + 0; }
        if (v.y > max_val) { max_val = v.y; max_idx = base_idx + 1; }
        if (v.z > max_val) { max_val = v.z; max_idx = base_idx + 2; }
        if (v.w > max_val) { max_val = v.w; max_idx = base_idx + 3; }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, max_val, mask);
        int other_idx   = __shfl_down_sync(0xffffffff, max_idx, mask);
        if (other_val > max_val) {
            max_val = other_val;
            max_idx = other_idx;
        }
    }

    __shared__ float s_max_val[32];
    __shared__ int s_max_idx[32];

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        s_max_val[warp_id] = max_val;
        s_max_idx[warp_id] = max_idx;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float block_max_val = s_max_val[0];
        int block_max_idx   = s_max_idx[0];
        const int num_warps = blockDim.x / 32;
        for (int w = 1; w < num_warps; ++w) {
            if (s_max_val[w] > block_max_val) {
                block_max_val = s_max_val[w];
                block_max_idx = s_max_idx[w];
            }
        }
        *selected_token = block_max_idx;
    }
}

extern "C" __global__ void gpu_advance_token_step_kernel(
    const int* __restrict__ selected_token,
    int* __restrict__ token_id,
    unsigned int* __restrict__ pos_dev,
    int* __restrict__ output_tokens_history,
    int* __restrict__ step_counter)
{
    if (threadIdx.x == 0) {
        int tok = *selected_token;
        *token_id = tok;
        *pos_dev += 1;
        if (output_tokens_history != nullptr && step_counter != nullptr) {
            int step = *step_counter;
            output_tokens_history[step] = tok;
            *step_counter = step + 1;
        }
    }
}

extern "C" __global__ void gemm_q4k_batched_gate_up_swiglu_kernel(
    const unsigned char* __restrict__ wgate,
    const unsigned char* __restrict__ wup,
    const float* __restrict__ x,
    float* __restrict__ out,
    int ne0,
    int ne1,
    int batch_size)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int col0 = (blockIdx.x * (blockDim.x / 32) + warp_id) * 2;
    const int col1 = col0 + 1;
    const int batch_base = blockIdx.y * 8;
    if (batch_base >= batch_size) return;
    if (col0 >= ne1) return;

    const float* xs[8];
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        int b_idx = batch_base + t;
        xs[t] = (b_idx < batch_size) ? (x + (size_t)b_idx * (size_t)ne0) : nullptr;
    }

    const int n_blocks = ne0 / 256;
    const unsigned char* gate0_col_w = wgate + (size_t)col0 * (size_t)n_blocks * 144u;
    const unsigned char* up0_col_w   = wup   + (size_t)col0 * (size_t)n_blocks * 144u;
    const unsigned char* gate1_col_w = (col1 < ne1) ? (wgate + (size_t)col1 * (size_t)n_blocks * 144u) : nullptr;
    const unsigned char* up1_col_w   = (col1 < ne1) ? (wup   + (size_t)col1 * (size_t)n_blocks * 144u) : nullptr;

    float acc_g0[8] = {0.0f};
    float acc_u0[8] = {0.0f};
    float acc_g1[8] = {0.0f};
    float acc_u1[8] = {0.0f};

    for (int b = 0; b < n_blocks; ++b) {
        // --- Column 0: Gate & Up Weights ---
        {
            const unsigned char* blk_g = gate0_col_w + (size_t)b * 144u;
            Q4K_Header hg;
            unpack_q4k_header(blk_g, &hg);
            const unsigned char* qs_g = blk_g + 16;

            const unsigned char* blk_u = up0_col_w + (size_t)b * 144u;
            Q4K_Header hu;
            unpack_q4k_header(blk_u, &hu);
            const unsigned char* qs_u = blk_u + 16;

            const unsigned char q0_g = qs_g[0 * 32 + lane];
            const unsigned char q1_g = qs_g[1 * 32 + lane];
            const unsigned char q2_g = qs_g[2 * 32 + lane];
            const unsigned char q3_g = qs_g[3 * 32 + lane];

            const unsigned char q0_u = qs_u[0 * 32 + lane];
            const unsigned char q1_u = qs_u[1 * 32 + lane];
            const unsigned char q2_u = qs_u[2 * 32 + lane];
            const unsigned char q3_u = qs_u[3 * 32 + lane];

            const float wg0 = __fmaf_rn(hg.d1[0], (float)(q0_g & 0x0Fu), hg.neg_m1[0]);
            const float wg1 = __fmaf_rn(hg.d2[0], (float)(q0_g >> 4),   hg.neg_m2[0]);
            const float wg2 = __fmaf_rn(hg.d1[1], (float)(q1_g & 0x0Fu), hg.neg_m1[1]);
            const float wg3 = __fmaf_rn(hg.d2[1], (float)(q1_g >> 4),   hg.neg_m2[1]);
            const float wg4 = __fmaf_rn(hg.d1[2], (float)(q2_g & 0x0Fu), hg.neg_m1[2]);
            const float wg5 = __fmaf_rn(hg.d2[2], (float)(q2_g >> 4),   hg.neg_m2[2]);
            const float wg6 = __fmaf_rn(hg.d1[3], (float)(q3_g & 0x0Fu), hg.neg_m1[3]);
            const float wg7 = __fmaf_rn(hg.d2[3], (float)(q3_g >> 4),   hg.neg_m2[3]);

            const float wu0 = __fmaf_rn(hu.d1[0], (float)(q0_u & 0x0Fu), hu.neg_m1[0]);
            const float wu1 = __fmaf_rn(hu.d2[0], (float)(q0_u >> 4),   hu.neg_m2[0]);
            const float wu2 = __fmaf_rn(hu.d1[1], (float)(q1_u & 0x0Fu), hu.neg_m1[1]);
            const float wu3 = __fmaf_rn(hu.d2[1], (float)(q1_u >> 4),   hu.neg_m2[1]);
            const float wu4 = __fmaf_rn(hu.d1[2], (float)(q2_u & 0x0Fu), hu.neg_m1[2]);
            const float wu5 = __fmaf_rn(hu.d2[2], (float)(q2_u >> 4),   hu.neg_m2[2]);
            const float wu6 = __fmaf_rn(hu.d1[3], (float)(q3_u & 0x0Fu), hu.neg_m1[3]);
            const float wu7 = __fmaf_rn(hu.d2[3], (float)(q3_u >> 4),   hu.neg_m2[3]);

            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                if (xs[t] != nullptr) {
                    const float* x_blk = xs[t] + (size_t)b * 256u;
                    const float x0_lo = x_blk[0 * 64 + lane];
                    const float x0_hi = x_blk[0 * 64 + 32 + lane];
                    const float x1_lo = x_blk[1 * 64 + lane];
                    const float x1_hi = x_blk[1 * 64 + 32 + lane];
                    const float x2_lo = x_blk[2 * 64 + lane];
                    const float x2_hi = x_blk[2 * 64 + 32 + lane];
                    const float x3_lo = x_blk[3 * 64 + lane];
                    const float x3_hi = x_blk[3 * 64 + 32 + lane];

                    acc_g0[t] = __fmaf_rn(wg0, x0_lo, acc_g0[t]);
                    acc_g0[t] = __fmaf_rn(wg1, x0_hi, acc_g0[t]);
                    acc_g0[t] = __fmaf_rn(wg2, x1_lo, acc_g0[t]);
                    acc_g0[t] = __fmaf_rn(wg3, x1_hi, acc_g0[t]);
                    acc_g0[t] = __fmaf_rn(wg4, x2_lo, acc_g0[t]);
                    acc_g0[t] = __fmaf_rn(wg5, x2_hi, acc_g0[t]);
                    acc_g0[t] = __fmaf_rn(wg6, x3_lo, acc_g0[t]);
                    acc_g0[t] = __fmaf_rn(wg7, x3_hi, acc_g0[t]);

                    acc_u0[t] = __fmaf_rn(wu0, x0_lo, acc_u0[t]);
                    acc_u0[t] = __fmaf_rn(wu1, x0_hi, acc_u0[t]);
                    acc_u0[t] = __fmaf_rn(wu2, x1_lo, acc_u0[t]);
                    acc_u0[t] = __fmaf_rn(wu3, x1_hi, acc_u0[t]);
                    acc_u0[t] = __fmaf_rn(wu4, x2_lo, acc_u0[t]);
                    acc_u0[t] = __fmaf_rn(wu5, x2_hi, acc_u0[t]);
                    acc_u0[t] = __fmaf_rn(wu6, x3_lo, acc_u0[t]);
                    acc_u0[t] = __fmaf_rn(wu7, x3_hi, acc_u0[t]);
                }
            }
        }

        // --- Column 1: Gate & Up Weights ---
        if (col1 < ne1) {
            const unsigned char* blk_g = gate1_col_w + (size_t)b * 144u;
            Q4K_Header hg;
            unpack_q4k_header(blk_g, &hg);
            const unsigned char* qs_g = blk_g + 16;

            const unsigned char* blk_u = up1_col_w + (size_t)b * 144u;
            Q4K_Header hu;
            unpack_q4k_header(blk_u, &hu);
            const unsigned char* qs_u = blk_u + 16;

            const unsigned char q0_g = qs_g[0 * 32 + lane];
            const unsigned char q1_g = qs_g[1 * 32 + lane];
            const unsigned char q2_g = qs_g[2 * 32 + lane];
            const unsigned char q3_g = qs_g[3 * 32 + lane];

            const unsigned char q0_u = qs_u[0 * 32 + lane];
            const unsigned char q1_u = qs_u[1 * 32 + lane];
            const unsigned char q2_u = qs_u[2 * 32 + lane];
            const unsigned char q3_u = qs_u[3 * 32 + lane];

            const float wg0 = __fmaf_rn(hg.d1[0], (float)(q0_g & 0x0Fu), hg.neg_m1[0]);
            const float wg1 = __fmaf_rn(hg.d2[0], (float)(q0_g >> 4),   hg.neg_m2[0]);
            const float wg2 = __fmaf_rn(hg.d1[1], (float)(q1_g & 0x0Fu), hg.neg_m1[1]);
            const float wg3 = __fmaf_rn(hg.d2[1], (float)(q1_g >> 4),   hg.neg_m2[1]);
            const float wg4 = __fmaf_rn(hg.d1[2], (float)(q2_g & 0x0Fu), hg.neg_m1[2]);
            const float wg5 = __fmaf_rn(hg.d2[2], (float)(q2_g >> 4),   hg.neg_m2[2]);
            const float wg6 = __fmaf_rn(hg.d1[3], (float)(q3_g & 0x0Fu), hg.neg_m1[3]);
            const float wg7 = __fmaf_rn(hg.d2[3], (float)(q3_g >> 4),   hg.neg_m2[3]);

            const float wu0 = __fmaf_rn(hu.d1[0], (float)(q0_u & 0x0Fu), hu.neg_m1[0]);
            const float wu1 = __fmaf_rn(hu.d2[0], (float)(q0_u >> 4),   hu.neg_m2[0]);
            const float wu2 = __fmaf_rn(hu.d1[1], (float)(q1_u & 0x0Fu), hu.neg_m1[1]);
            const float wu3 = __fmaf_rn(hu.d2[1], (float)(q1_u >> 4),   hu.neg_m2[1]);
            const float wu4 = __fmaf_rn(hu.d1[2], (float)(q2_u & 0x0Fu), hu.neg_m1[2]);
            const float wu5 = __fmaf_rn(hu.d2[2], (float)(q2_u >> 4),   hu.neg_m2[2]);
            const float wu6 = __fmaf_rn(hu.d1[3], (float)(q3_u & 0x0Fu), hu.neg_m1[3]);
            const float wu7 = __fmaf_rn(hu.d2[3], (float)(q3_u >> 4),   hu.neg_m2[3]);

            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                if (xs[t] != nullptr) {
                    const float* x_blk = xs[t] + (size_t)b * 256u;
                    const float x0_lo = x_blk[0 * 64 + lane];
                    const float x0_hi = x_blk[0 * 64 + 32 + lane];
                    const float x1_lo = x_blk[1 * 64 + lane];
                    const float x1_hi = x_blk[1 * 64 + 32 + lane];
                    const float x2_lo = x_blk[2 * 64 + lane];
                    const float x2_hi = x_blk[2 * 64 + 32 + lane];
                    const float x3_lo = x_blk[3 * 64 + lane];
                    const float x3_hi = x_blk[3 * 64 + 32 + lane];

                    acc_g1[t] = __fmaf_rn(wg0, x0_lo, acc_g1[t]);
                    acc_g1[t] = __fmaf_rn(wg1, x0_hi, acc_g1[t]);
                    acc_g1[t] = __fmaf_rn(wg2, x1_lo, acc_g1[t]);
                    acc_g1[t] = __fmaf_rn(wg3, x1_hi, acc_g1[t]);
                    acc_g1[t] = __fmaf_rn(wg4, x2_lo, acc_g1[t]);
                    acc_g1[t] = __fmaf_rn(wg5, x2_hi, acc_g1[t]);
                    acc_g1[t] = __fmaf_rn(wg6, x3_lo, acc_g1[t]);
                    acc_g1[t] = __fmaf_rn(wg7, x3_hi, acc_g1[t]);

                    acc_u1[t] = __fmaf_rn(wu0, x0_lo, acc_u1[t]);
                    acc_u1[t] = __fmaf_rn(wu1, x0_hi, acc_u1[t]);
                    acc_u1[t] = __fmaf_rn(wu2, x1_lo, acc_u1[t]);
                    acc_u1[t] = __fmaf_rn(wu3, x1_hi, acc_u1[t]);
                    acc_u1[t] = __fmaf_rn(wu4, x2_lo, acc_u1[t]);
                    acc_u1[t] = __fmaf_rn(wu5, x2_hi, acc_u1[t]);
                    acc_u1[t] = __fmaf_rn(wu6, x3_lo, acc_u1[t]);
                    acc_u1[t] = __fmaf_rn(wu7, x3_hi, acc_u1[t]);
                }
            }
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            acc_g0[t] += __shfl_down_sync(0xffffffff, acc_g0[t], mask);
            acc_u0[t] += __shfl_down_sync(0xffffffff, acc_u0[t], mask);
            acc_g1[t] += __shfl_down_sync(0xffffffff, acc_g1[t], mask);
            acc_u1[t] += __shfl_down_sync(0xffffffff, acc_u1[t], mask);
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            int b_idx = batch_base + t;
            if (b_idx < batch_size) {
                float* out_row = out + (size_t)b_idx * (size_t)ne1;
                out_row[col0] = silu_f(acc_g0[t]) * acc_u0[t];
                if (col1 < ne1) {
                    out_row[col1] = silu_f(acc_g1[t]) * acc_u1[t];
                }
            }
        }
    }
}

extern "C" __global__ void gemm_q6k_batched_kernel(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ out,
    int ne0,
    int ne1,
    int batch_size,
    const float* __restrict__ residual)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int col0 = (blockIdx.x * (blockDim.x / 32) + warp_id) * 2;
    const int col1 = col0 + 1;
    const int batch_base = blockIdx.y * 8;
    if (batch_base >= batch_size) return;
    if (col0 >= ne1) return;

    const float* xs[8];
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        int b_idx = batch_base + t;
        xs[t] = (b_idx < batch_size) ? (x + (size_t)b_idx * (size_t)ne0) : nullptr;
    }

    const int n_blocks = ne0 / 256;
    const unsigned char* col0_w = weights + (size_t)col0 * (size_t)n_blocks * 210u;
    const unsigned char* col1_w = (col1 < ne1) ? (weights + (size_t)col1 * (size_t)n_blocks * 210u) : nullptr;

    float acc0[8] = {0.0f};
    float acc1[8] = {0.0f};
    const int is = lane >> 4;

    for (int b = 0; b < n_blocks; ++b) {
        // Col 0
        float w0_0 = 0.0f, w1_0 = 0.0f, w2_0 = 0.0f, w3_0 = 0.0f;
        float w4_0 = 0.0f, w5_0 = 0.0f, w6_0 = 0.0f, w7_0 = 0.0f;
        {
            const unsigned char* blk = col0_w + (size_t)b * 210u;
            Q6K_Header h;
            unpack_q6k_header(blk, is, &h);
            const unsigned char* ql = blk;
            const unsigned char* qh = blk + 128;

            const unsigned char ql0 = ql[lane];
            const unsigned char ql1 = ql[lane + 32];
            const unsigned char ql2 = ql[64 + lane];
            const unsigned char ql3 = ql[96 + lane];
            const unsigned char qh0 = qh[lane];
            const unsigned char qh1 = qh[32 + lane];

            const int q0 = (int)(ql0 & 0x0Fu) | (((int)(qh0 >> 0) & 3) << 4);
            const int q1 = (int)(ql1 & 0x0Fu) | (((int)(qh0 >> 2) & 3) << 4);
            const int q2 = (int)(ql0 >> 4)    | (((int)(qh0 >> 4) & 3) << 4);
            const int q3 = (int)(ql1 >> 4)    | (((int)(qh0 >> 6) & 3) << 4);
            const int q4 = (int)(ql2 & 0x0Fu) | (((int)(qh1 >> 0) & 3) << 4);
            const int q5 = (int)(ql3 & 0x0Fu) | (((int)(qh1 >> 2) & 3) << 4);
            const int q6 = (int)(ql2 >> 4)    | (((int)(qh1 >> 4) & 3) << 4);
            const int q7 = (int)(ql3 >> 4)    | (((int)(qh1 >> 6) & 3) << 4);

            w0_0 = h.ds[0] * (float)(q0 - 32);
            w1_0 = h.ds[1] * (float)(q1 - 32);
            w2_0 = h.ds[2] * (float)(q2 - 32);
            w3_0 = h.ds[3] * (float)(q3 - 32);
            w4_0 = h.ds[4] * (float)(q4 - 32);
            w5_0 = h.ds[5] * (float)(q5 - 32);
            w6_0 = h.ds[6] * (float)(q6 - 32);
            w7_0 = h.ds[7] * (float)(q7 - 32);
        }

        // Col 1
        float w0_1 = 0.0f, w1_1 = 0.0f, w2_1 = 0.0f, w3_1 = 0.0f;
        float w4_1 = 0.0f, w5_1 = 0.0f, w6_1 = 0.0f, w7_1 = 0.0f;
        if (col1_w != nullptr) {
            const unsigned char* blk = col1_w + (size_t)b * 210u;
            Q6K_Header h;
            unpack_q6k_header(blk, is, &h);
            const unsigned char* ql = blk;
            const unsigned char* qh = blk + 128;

            const unsigned char ql0 = ql[lane];
            const unsigned char ql1 = ql[lane + 32];
            const unsigned char ql2 = ql[64 + lane];
            const unsigned char ql3 = ql[96 + lane];
            const unsigned char qh0 = qh[lane];
            const unsigned char qh1 = qh[32 + lane];

            const int q0 = (int)(ql0 & 0x0Fu) | (((int)(qh0 >> 0) & 3) << 4);
            const int q1 = (int)(ql1 & 0x0Fu) | (((int)(qh0 >> 2) & 3) << 4);
            const int q2 = (int)(ql0 >> 4)    | (((int)(qh0 >> 4) & 3) << 4);
            const int q3 = (int)(ql1 >> 4)    | (((int)(qh0 >> 6) & 3) << 4);
            const int q4 = (int)(ql2 & 0x0Fu) | (((int)(qh1 >> 0) & 3) << 4);
            const int q5 = (int)(ql3 & 0x0Fu) | (((int)(qh1 >> 2) & 3) << 4);
            const int q6 = (int)(ql2 >> 4)    | (((int)(qh1 >> 4) & 3) << 4);
            const int q7 = (int)(ql3 >> 4)    | (((int)(qh1 >> 6) & 3) << 4);

            w0_1 = h.ds[0] * (float)(q0 - 32);
            w1_1 = h.ds[1] * (float)(q1 - 32);
            w2_1 = h.ds[2] * (float)(q2 - 32);
            w3_1 = h.ds[3] * (float)(q3 - 32);
            w4_1 = h.ds[4] * (float)(q4 - 32);
            w5_1 = h.ds[5] * (float)(q5 - 32);
            w6_1 = h.ds[6] * (float)(q6 - 32);
            w7_1 = h.ds[7] * (float)(q7 - 32);
        }

        // Stream x per token directly into accumulators (Zero register spill!)
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (xs[t] != nullptr) {
                const float* x_blk = xs[t] + (size_t)b * 256u;
                const float x0 = x_blk[0 * 32 + lane];
                const float x1 = x_blk[1 * 32 + lane];
                const float x2 = x_blk[2 * 32 + lane];
                const float x3 = x_blk[3 * 32 + lane];
                const float x4 = x_blk[4 * 32 + lane];
                const float x5 = x_blk[5 * 32 + lane];
                const float x6 = x_blk[6 * 32 + lane];
                const float x7 = x_blk[7 * 32 + lane];

                acc0[t] = __fmaf_rn(w0_0, x0, acc0[t]);
                acc0[t] = __fmaf_rn(w1_0, x1, acc0[t]);
                acc0[t] = __fmaf_rn(w2_0, x2, acc0[t]);
                acc0[t] = __fmaf_rn(w3_0, x3, acc0[t]);
                acc0[t] = __fmaf_rn(w4_0, x4, acc0[t]);
                acc0[t] = __fmaf_rn(w5_0, x5, acc0[t]);
                acc0[t] = __fmaf_rn(w6_0, x6, acc0[t]);
                acc0[t] = __fmaf_rn(w7_0, x7, acc0[t]);

                if (col1_w != nullptr) {
                    acc1[t] = __fmaf_rn(w0_1, x0, acc1[t]);
                    acc1[t] = __fmaf_rn(w1_1, x1, acc1[t]);
                    acc1[t] = __fmaf_rn(w2_1, x2, acc1[t]);
                    acc1[t] = __fmaf_rn(w3_1, x3, acc1[t]);
                    acc1[t] = __fmaf_rn(w4_1, x4, acc1[t]);
                    acc1[t] = __fmaf_rn(w5_1, x5, acc1[t]);
                    acc1[t] = __fmaf_rn(w6_1, x6, acc1[t]);
                    acc1[t] = __fmaf_rn(w7_1, x7, acc1[t]);
                }
            }
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            acc0[t] += __shfl_down_sync(0xffffffff, acc0[t], mask);
            acc1[t] += __shfl_down_sync(0xffffffff, acc1[t], mask);
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            int b_idx = batch_base + t;
            if (b_idx < batch_size) {
                float* out_row = out + (size_t)b_idx * (size_t)ne1;
                float a0 = acc0[t];
                float a1 = acc1[t];
                if (residual != nullptr) {
                    const float* res_row = residual + (size_t)b_idx * (size_t)ne1;
                    a0 += res_row[col0];
                    if (col1 < ne1) a1 += res_row[col1];
                }
                out_row[col0] = a0;
                if (col1 < ne1) out_row[col1] = a1;
            }
        }
    }
}

extern "C" __global__ void gemm_fused_qkv_batched_kernel(
    const unsigned char* __restrict__ wq,
    const unsigned char* __restrict__ wk,
    const unsigned char* __restrict__ wv,
    const float* __restrict__ x,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    int ne0,
    int qdim,
    int kvd,
    int batch_size,
    const float* __restrict__ qb,
    const float* __restrict__ kb,
    const float* __restrict__ vb)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int col0 = (blockIdx.x * (blockDim.x / 32) + warp_id) * 2;
    const int col1 = col0 + 1;
    const int batch_base = blockIdx.y * 8;
    if (batch_base >= batch_size) return;

    const int total_cols = qdim + kvd + kvd;
    if (col0 >= total_cols) return;

    const float* xs[8];
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        int b_idx = batch_base + t;
        xs[t] = (b_idx < batch_size) ? (x + (size_t)b_idx * (size_t)ne0) : nullptr;
    }

    const int n_blocks = ne0 / 256;

    if (col0 < qdim) {
        // Q projection (Q4_K)
        const unsigned char* col0_w = wq + (size_t)col0 * (size_t)n_blocks * 144u;
        const unsigned char* col1_w = (col1 < qdim) ? (wq + (size_t)col1 * (size_t)n_blocks * 144u) : nullptr;

        float acc0[8] = {0.0f};
        float acc1[8] = {0.0f};

        for (int b = 0; b < n_blocks; ++b) {
            // Col 0
            {
                const unsigned char* blk = col0_w + (size_t)b * 144u;
                Q4K_Header h;
                unpack_q4k_header(blk, &h);
                const unsigned char* qs = blk + 16;

                const unsigned char q0 = qs[0 * 32 + lane];
                const unsigned char q1 = qs[1 * 32 + lane];
                const unsigned char q2 = qs[2 * 32 + lane];
                const unsigned char q3 = qs[3 * 32 + lane];

                const float w0 = __fmaf_rn(h.d1[0], (float)(q0 & 0x0Fu), h.neg_m1[0]);
                const float w1 = __fmaf_rn(h.d2[0], (float)(q0 >> 4),   h.neg_m2[0]);
                const float w2 = __fmaf_rn(h.d1[1], (float)(q1 & 0x0Fu), h.neg_m1[1]);
                const float w3 = __fmaf_rn(h.d2[1], (float)(q1 >> 4),   h.neg_m2[1]);
                const float w4 = __fmaf_rn(h.d1[2], (float)(q2 & 0x0Fu), h.neg_m1[2]);
                const float w5 = __fmaf_rn(h.d2[2], (float)(q2 >> 4),   h.neg_m2[2]);
                const float w6 = __fmaf_rn(h.d1[3], (float)(q3 & 0x0Fu), h.neg_m1[3]);
                const float w7 = __fmaf_rn(h.d2[3], (float)(q3 >> 4),   h.neg_m2[3]);

                #pragma unroll
                for (int t = 0; t < 8; ++t) {
                    if (xs[t] != nullptr) {
                        const float* x_blk = xs[t] + (size_t)b * 256u;
                        acc0[t] = __fmaf_rn(w0, x_blk[0 * 64 + lane], acc0[t]);
                        acc0[t] = __fmaf_rn(w1, x_blk[0 * 64 + 32 + lane], acc0[t]);
                        acc0[t] = __fmaf_rn(w2, x_blk[1 * 64 + lane], acc0[t]);
                        acc0[t] = __fmaf_rn(w3, x_blk[1 * 64 + 32 + lane], acc0[t]);
                        acc0[t] = __fmaf_rn(w4, x_blk[2 * 64 + lane], acc0[t]);
                        acc0[t] = __fmaf_rn(w5, x_blk[2 * 64 + 32 + lane], acc0[t]);
                        acc0[t] = __fmaf_rn(w6, x_blk[3 * 64 + lane], acc0[t]);
                        acc0[t] = __fmaf_rn(w7, x_blk[3 * 64 + 32 + lane], acc0[t]);
                    }
                }
            }

            // Col 1
            if (col1_w != nullptr) {
                const unsigned char* blk = col1_w + (size_t)b * 144u;
                Q4K_Header h;
                unpack_q4k_header(blk, &h);
                const unsigned char* qs = blk + 16;

                const unsigned char q0 = qs[0 * 32 + lane];
                const unsigned char q1 = qs[1 * 32 + lane];
                const unsigned char q2 = qs[2 * 32 + lane];
                const unsigned char q3 = qs[3 * 32 + lane];

                const float w0 = __fmaf_rn(h.d1[0], (float)(q0 & 0x0Fu), h.neg_m1[0]);
                const float w1 = __fmaf_rn(h.d2[0], (float)(q0 >> 4),   h.neg_m2[0]);
                const float w2 = __fmaf_rn(h.d1[1], (float)(q1 & 0x0Fu), h.neg_m1[1]);
                const float w3 = __fmaf_rn(h.d2[1], (float)(q1 >> 4),   h.neg_m2[1]);
                const float w4 = __fmaf_rn(h.d1[2], (float)(q2 & 0x0Fu), h.neg_m1[2]);
                const float w5 = __fmaf_rn(h.d2[2], (float)(q2 >> 4),   h.neg_m2[2]);
                const float w6 = __fmaf_rn(h.d1[3], (float)(q3 & 0x0Fu), h.neg_m1[3]);
                const float w7 = __fmaf_rn(h.d2[3], (float)(q3 >> 4),   h.neg_m2[3]);

                #pragma unroll
                for (int t = 0; t < 8; ++t) {
                    if (xs[t] != nullptr) {
                        const float* x_blk = xs[t] + (size_t)b * 256u;
                        acc1[t] = __fmaf_rn(w0, x_blk[0 * 64 + lane], acc1[t]);
                        acc1[t] = __fmaf_rn(w1, x_blk[0 * 64 + 32 + lane], acc1[t]);
                        acc1[t] = __fmaf_rn(w2, x_blk[1 * 64 + lane], acc1[t]);
                        acc1[t] = __fmaf_rn(w3, x_blk[1 * 64 + 32 + lane], acc1[t]);
                        acc1[t] = __fmaf_rn(w4, x_blk[2 * 64 + lane], acc1[t]);
                        acc1[t] = __fmaf_rn(w5, x_blk[2 * 64 + 32 + lane], acc1[t]);
                        acc1[t] = __fmaf_rn(w6, x_blk[3 * 64 + lane], acc1[t]);
                        acc1[t] = __fmaf_rn(w7, x_blk[3 * 64 + 32 + lane], acc1[t]);
                    }
                }
            }
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                acc0[t] += __shfl_down_sync(0xffffffff, acc0[t], mask);
                acc1[t] += __shfl_down_sync(0xffffffff, acc1[t], mask);
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                int b_idx = batch_base + t;
                if (b_idx < batch_size) {
                    float* out_row = q_out + (size_t)b_idx * (size_t)qdim;
                    float a0 = acc0[t];
                    float a1 = acc1[t];
                    if (qb != nullptr) {
                        a0 += qb[col0];
                        if (col1 < qdim) a1 += qb[col1];
                    }
                    out_row[col0] = a0;
                    if (col1 < qdim) out_row[col1] = a1;
                }
            }
        }
    } else if (col0 < qdim + kvd) {
        // K projection (Q4_K)
        const int k_col0 = col0 - qdim;
        const int k_col1 = col1 - qdim;
        const unsigned char* col0_w = wk + (size_t)k_col0 * (size_t)n_blocks * 144u;
        const unsigned char* col1_w = (k_col1 < kvd) ? (wk + (size_t)k_col1 * (size_t)n_blocks * 144u) : nullptr;

        float acc0[8] = {0.0f};
        float acc1[8] = {0.0f};

        for (int b = 0; b < n_blocks; ++b) {
            // Col 0
            {
                const unsigned char* blk = col0_w + (size_t)b * 144u;
                Q4K_Header h;
                unpack_q4k_header(blk, &h);
                const unsigned char* qs = blk + 16;

                const unsigned char q0 = qs[0 * 32 + lane];
                const unsigned char q1 = qs[1 * 32 + lane];
                const unsigned char q2 = qs[2 * 32 + lane];
                const unsigned char q3 = qs[3 * 32 + lane];

                const float w0 = __fmaf_rn(h.d1[0], (float)(q0 & 0x0Fu), h.neg_m1[0]);
                const float w1 = __fmaf_rn(h.d2[0], (float)(q0 >> 4),   h.neg_m2[0]);
                const float w2 = __fmaf_rn(h.d1[1], (float)(q1 & 0x0Fu), h.neg_m1[1]);
                const float w3 = __fmaf_rn(h.d2[1], (float)(q1 >> 4),   h.neg_m2[1]);
                const float w4 = __fmaf_rn(h.d1[2], (float)(q2 & 0x0Fu), h.neg_m1[2]);
                const float w5 = __fmaf_rn(h.d2[2], (float)(q2 >> 4),   h.neg_m2[2]);
                const float w6 = __fmaf_rn(h.d1[3], (float)(q3 & 0x0Fu), h.neg_m1[3]);
                const float w7 = __fmaf_rn(h.d2[3], (float)(q3 >> 4),   h.neg_m2[3]);

                #pragma unroll
                for (int t = 0; t < 8; ++t) {
                    if (xs[t] != nullptr) {
                        const float* x_blk = xs[t] + (size_t)b * 256u;
                        acc0[t] = __fmaf_rn(w0, x_blk[0 * 64 + lane], acc0[t]);
                        acc0[t] = __fmaf_rn(w1, x_blk[0 * 64 + 32 + lane], acc0[t]);
                        acc0[t] = __fmaf_rn(w2, x_blk[1 * 64 + lane], acc0[t]);
                        acc0[t] = __fmaf_rn(w3, x_blk[1 * 64 + 32 + lane], acc0[t]);
                        acc0[t] = __fmaf_rn(w4, x_blk[2 * 64 + lane], acc0[t]);
                        acc0[t] = __fmaf_rn(w5, x_blk[2 * 64 + 32 + lane], acc0[t]);
                        acc0[t] = __fmaf_rn(w6, x_blk[3 * 64 + lane], acc0[t]);
                        acc0[t] = __fmaf_rn(w7, x_blk[3 * 64 + 32 + lane], acc0[t]);
                    }
                }
            }

            // Col 1
            if (col1_w != nullptr) {
                const unsigned char* blk = col1_w + (size_t)b * 144u;
                Q4K_Header h;
                unpack_q4k_header(blk, &h);
                const unsigned char* qs = blk + 16;

                const unsigned char q0 = qs[0 * 32 + lane];
                const unsigned char q1 = qs[1 * 32 + lane];
                const unsigned char q2 = qs[2 * 32 + lane];
                const unsigned char q3 = qs[3 * 32 + lane];

                const float w0 = __fmaf_rn(h.d1[0], (float)(q0 & 0x0Fu), h.neg_m1[0]);
                const float w1 = __fmaf_rn(h.d2[0], (float)(q0 >> 4),   h.neg_m2[0]);
                const float w2 = __fmaf_rn(h.d1[1], (float)(q1 & 0x0Fu), h.neg_m1[1]);
                const float w3 = __fmaf_rn(h.d2[1], (float)(q1 >> 4),   h.neg_m2[1]);
                const float w4 = __fmaf_rn(h.d1[2], (float)(q2 & 0x0Fu), h.neg_m1[2]);
                const float w5 = __fmaf_rn(h.d2[2], (float)(q2 >> 4),   h.neg_m2[2]);
                const float w6 = __fmaf_rn(h.d1[3], (float)(q3 & 0x0Fu), h.neg_m1[3]);
                const float w7 = __fmaf_rn(h.d2[3], (float)(q3 >> 4),   h.neg_m2[3]);

                #pragma unroll
                for (int t = 0; t < 8; ++t) {
                    if (xs[t] != nullptr) {
                        const float* x_blk = xs[t] + (size_t)b * 256u;
                        acc1[t] = __fmaf_rn(w0, x_blk[0 * 64 + lane], acc1[t]);
                        acc1[t] = __fmaf_rn(w1, x_blk[0 * 64 + 32 + lane], acc1[t]);
                        acc1[t] = __fmaf_rn(w2, x_blk[1 * 64 + lane], acc1[t]);
                        acc1[t] = __fmaf_rn(w3, x_blk[1 * 64 + 32 + lane], acc1[t]);
                        acc1[t] = __fmaf_rn(w4, x_blk[2 * 64 + lane], acc1[t]);
                        acc1[t] = __fmaf_rn(w5, x_blk[2 * 64 + 32 + lane], acc1[t]);
                        acc1[t] = __fmaf_rn(w6, x_blk[3 * 64 + lane], acc1[t]);
                        acc1[t] = __fmaf_rn(w7, x_blk[3 * 64 + 32 + lane], acc1[t]);
                    }
                }
            }
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                acc0[t] += __shfl_down_sync(0xffffffff, acc0[t], mask);
                acc1[t] += __shfl_down_sync(0xffffffff, acc1[t], mask);
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                int b_idx = batch_base + t;
                if (b_idx < batch_size) {
                    float* out_row = k_out + (size_t)b_idx * (size_t)kvd;
                    float a0 = acc0[t];
                    float a1 = acc1[t];
                    if (kb != nullptr) {
                        a0 += kb[k_col0];
                        if (k_col1 < kvd) a1 += kb[k_col1];
                    }
                    out_row[k_col0] = a0;
                    if (k_col1 < kvd) out_row[k_col1] = a1;
                }
            }
        }
    } else {
        // V projection (Q6_K)
        const int v_col0 = col0 - qdim - kvd;
        const int v_col1 = col1 - qdim - kvd;
        const unsigned char* col0_w = wv + (size_t)v_col0 * (size_t)n_blocks * 210u;
        const unsigned char* col1_w = (v_col1 < kvd) ? (wv + (size_t)v_col1 * (size_t)n_blocks * 210u) : nullptr;

        float acc0[8] = {0.0f};
        float acc1[8] = {0.0f};
        const int is = lane >> 4;

        for (int b = 0; b < n_blocks; ++b) {
            // Col 0
            {
                const unsigned char* blk = col0_w + (size_t)b * 210u;
                Q6K_Header h;
                unpack_q6k_header(blk, is, &h);
                const unsigned char* ql = blk;
                const unsigned char* qh = blk + 128;

                const unsigned char ql0 = ql[lane];
                const unsigned char ql1 = ql[lane + 32];
                const unsigned char ql2 = ql[64 + lane];
                const unsigned char ql3 = ql[96 + lane];
                const unsigned char qh0 = qh[lane];
                const unsigned char qh1 = qh[32 + lane];

                const int q0 = (int)(ql0 & 0x0Fu) | (((int)(qh0 >> 0) & 3) << 4);
                const int q1 = (int)(ql1 & 0x0Fu) | (((int)(qh0 >> 2) & 3) << 4);
                const int q2 = (int)(ql0 >> 4)    | (((int)(qh0 >> 4) & 3) << 4);
                const int q3 = (int)(ql1 >> 4)    | (((int)(qh0 >> 6) & 3) << 4);
                const int q4 = (int)(ql2 & 0x0Fu) | (((int)(qh1 >> 0) & 3) << 4);
                const int q5 = (int)(ql3 & 0x0Fu) | (((int)(qh1 >> 2) & 3) << 4);
                const int q6 = (int)(ql2 >> 4)    | (((int)(qh1 >> 4) & 3) << 4);
                const int q7 = (int)(ql3 >> 4)    | (((int)(qh1 >> 6) & 3) << 4);

                const float w0 = h.ds[0] * (float)(q0 - 32);
                const float w1 = h.ds[1] * (float)(q1 - 32);
                const float w2 = h.ds[2] * (float)(q2 - 32);
                const float w3 = h.ds[3] * (float)(q3 - 32);
                const float w4 = h.ds[4] * (float)(q4 - 32);
                const float w5 = h.ds[5] * (float)(q5 - 32);
                const float w6 = h.ds[6] * (float)(q6 - 32);
                const float w7 = h.ds[7] * (float)(q7 - 32);

                #pragma unroll
                for (int t = 0; t < 8; ++t) {
                    if (xs[t] != nullptr) {
                        const float* x_blk = xs[t] + (size_t)b * 256u;
                        acc0[t] = __fmaf_rn(w0, x_blk[0 * 32 + lane], acc0[t]);
                        acc0[t] = __fmaf_rn(w1, x_blk[1 * 32 + lane], acc0[t]);
                        acc0[t] = __fmaf_rn(w2, x_blk[2 * 32 + lane], acc0[t]);
                        acc0[t] = __fmaf_rn(w3, x_blk[3 * 32 + lane], acc0[t]);
                        acc0[t] = __fmaf_rn(w4, x_blk[4 * 32 + lane], acc0[t]);
                        acc0[t] = __fmaf_rn(w5, x_blk[5 * 32 + lane], acc0[t]);
                        acc0[t] = __fmaf_rn(w6, x_blk[6 * 32 + lane], acc0[t]);
                        acc0[t] = __fmaf_rn(w7, x_blk[7 * 32 + lane], acc0[t]);
                    }
                }
            }

            // Col 1
            if (col1_w != nullptr) {
                const unsigned char* blk = col1_w + (size_t)b * 210u;
                Q6K_Header h;
                unpack_q6k_header(blk, is, &h);
                const unsigned char* ql = blk;
                const unsigned char* qh = blk + 128;

                const unsigned char ql0 = ql[lane];
                const unsigned char ql1 = ql[lane + 32];
                const unsigned char ql2 = ql[64 + lane];
                const unsigned char ql3 = ql[96 + lane];
                const unsigned char qh0 = qh[lane];
                const unsigned char qh1 = qh[32 + lane];

                const int q0 = (int)(ql0 & 0x0Fu) | (((int)(qh0 >> 0) & 3) << 4);
                const int q1 = (int)(ql1 & 0x0Fu) | (((int)(qh0 >> 2) & 3) << 4);
                const int q2 = (int)(ql0 >> 4)    | (((int)(qh0 >> 4) & 3) << 4);
                const int q3 = (int)(ql1 >> 4)    | (((int)(qh0 >> 6) & 3) << 4);
                const int q4 = (int)(ql2 & 0x0Fu) | (((int)(qh1 >> 0) & 3) << 4);
                const int q5 = (int)(ql3 & 0x0Fu) | (((int)(qh1 >> 2) & 3) << 4);
                const int q6 = (int)(ql2 >> 4)    | (((int)(qh1 >> 4) & 3) << 4);
                const int q7 = (int)(ql3 >> 4)    | (((int)(qh1 >> 6) & 3) << 4);

                const float w0 = h.ds[0] * (float)(q0 - 32);
                const float w1 = h.ds[1] * (float)(q1 - 32);
                const float w2 = h.ds[2] * (float)(q2 - 32);
                const float w3 = h.ds[3] * (float)(q3 - 32);
                const float w4 = h.ds[4] * (float)(q4 - 32);
                const float w5 = h.ds[5] * (float)(q5 - 32);
                const float w6 = h.ds[6] * (float)(q6 - 32);
                const float w7 = h.ds[7] * (float)(q7 - 32);

                #pragma unroll
                for (int t = 0; t < 8; ++t) {
                    if (xs[t] != nullptr) {
                        const float* x_blk = xs[t] + (size_t)b * 256u;
                        acc1[t] = __fmaf_rn(w0, x_blk[0 * 32 + lane], acc1[t]);
                        acc1[t] = __fmaf_rn(w1, x_blk[1 * 32 + lane], acc1[t]);
                        acc1[t] = __fmaf_rn(w2, x_blk[2 * 32 + lane], acc1[t]);
                        acc1[t] = __fmaf_rn(w3, x_blk[3 * 32 + lane], acc1[t]);
                        acc1[t] = __fmaf_rn(w4, x_blk[4 * 32 + lane], acc1[t]);
                        acc1[t] = __fmaf_rn(w5, x_blk[5 * 32 + lane], acc1[t]);
                        acc1[t] = __fmaf_rn(w6, x_blk[6 * 32 + lane], acc1[t]);
                        acc1[t] = __fmaf_rn(w7, x_blk[7 * 32 + lane], acc1[t]);
                    }
                }
            }
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                acc0[t] += __shfl_down_sync(0xffffffff, acc0[t], mask);
                acc1[t] += __shfl_down_sync(0xffffffff, acc1[t], mask);
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                int b_idx = batch_base + t;
                if (b_idx < batch_size) {
                    float* out_row = v_out + (size_t)b_idx * (size_t)kvd;
                    float a0 = acc0[t];
                    float a1 = acc1[t];
                    if (vb != nullptr) {
                        a0 += vb[v_col0];
                        if (v_col1 < kvd) a1 += vb[v_col1];
                    }
                    out_row[v_col0] = a0;
                    if (v_col1 < kvd) out_row[v_col1] = a1;
                }
            }
        }
    }
}

extern "C" __global__ void gemm_q4k_batched_kernel(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ out,
    int ne0,
    int ne1,
    int batch_size,
    const float* __restrict__ residual)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int col0 = (blockIdx.x * (blockDim.x / 32) + warp_id) * 2;
    const int col1 = col0 + 1;
    const int batch_base = blockIdx.y * 8;
    if (batch_base >= batch_size) return;
    if (col0 >= ne1) return;

    const float* xs[8];
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        int b_idx = batch_base + t;
        xs[t] = (b_idx < batch_size) ? (x + (size_t)b_idx * (size_t)ne0) : nullptr;
    }

    const int n_blocks = ne0 / 256;
    const unsigned char* col0_w = weights + (size_t)col0 * (size_t)n_blocks * 144u;
    const unsigned char* col1_w = (col1 < ne1) ? (weights + (size_t)col1 * (size_t)n_blocks * 144u) : nullptr;

    float acc0[8] = {0.0f};
    float acc1[8] = {0.0f};

    for (int b = 0; b < n_blocks; ++b) {
        // Col 0
        {
            const unsigned char* blk = col0_w + (size_t)b * 144u;
            Q4K_Header h;
            unpack_q4k_header(blk, &h);
            const unsigned char* qs = blk + 16;

            const unsigned char q0 = qs[0 * 32 + lane];
            const unsigned char q1 = qs[1 * 32 + lane];
            const unsigned char q2 = qs[2 * 32 + lane];
            const unsigned char q3 = qs[3 * 32 + lane];

            const float w0 = __fmaf_rn(h.d1[0], (float)(q0 & 0x0Fu), h.neg_m1[0]);
            const float w1 = __fmaf_rn(h.d2[0], (float)(q0 >> 4),   h.neg_m2[0]);
            const float w2 = __fmaf_rn(h.d1[1], (float)(q1 & 0x0Fu), h.neg_m1[1]);
            const float w3 = __fmaf_rn(h.d2[1], (float)(q1 >> 4),   h.neg_m2[1]);
            const float w4 = __fmaf_rn(h.d1[2], (float)(q2 & 0x0Fu), h.neg_m1[2]);
            const float w5 = __fmaf_rn(h.d2[2], (float)(q2 >> 4),   h.neg_m2[2]);
            const float w6 = __fmaf_rn(h.d1[3], (float)(q3 & 0x0Fu), h.neg_m1[3]);
            const float w7 = __fmaf_rn(h.d2[3], (float)(q3 >> 4),   h.neg_m2[3]);

            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                if (xs[t] != nullptr) {
                    const float* x_blk = xs[t] + (size_t)b * 256u;
                    acc0[t] = __fmaf_rn(w0, x_blk[0 * 64 + lane], acc0[t]);
                    acc0[t] = __fmaf_rn(w1, x_blk[0 * 64 + 32 + lane], acc0[t]);
                    acc0[t] = __fmaf_rn(w2, x_blk[1 * 64 + lane], acc0[t]);
                    acc0[t] = __fmaf_rn(w3, x_blk[1 * 64 + 32 + lane], acc0[t]);
                    acc0[t] = __fmaf_rn(w4, x_blk[2 * 64 + lane], acc0[t]);
                    acc0[t] = __fmaf_rn(w5, x_blk[2 * 64 + 32 + lane], acc0[t]);
                    acc0[t] = __fmaf_rn(w6, x_blk[3 * 64 + lane], acc0[t]);
                    acc0[t] = __fmaf_rn(w7, x_blk[3 * 64 + 32 + lane], acc0[t]);
                }
            }
        }

        // Col 1
        if (col1_w != nullptr) {
            const unsigned char* blk = col1_w + (size_t)b * 144u;
            Q4K_Header h;
            unpack_q4k_header(blk, &h);
            const unsigned char* qs = blk + 16;

            const unsigned char q0 = qs[0 * 32 + lane];
            const unsigned char q1 = qs[1 * 32 + lane];
            const unsigned char q2 = qs[2 * 32 + lane];
            const unsigned char q3 = qs[3 * 32 + lane];

            const float w0 = __fmaf_rn(h.d1[0], (float)(q0 & 0x0Fu), h.neg_m1[0]);
            const float w1 = __fmaf_rn(h.d2[0], (float)(q0 >> 4),   h.neg_m2[0]);
            const float w2 = __fmaf_rn(h.d1[1], (float)(q1 & 0x0Fu), h.neg_m1[1]);
            const float w3 = __fmaf_rn(h.d2[1], (float)(q1 >> 4),   h.neg_m2[1]);
            const float w4 = __fmaf_rn(h.d1[2], (float)(q2 & 0x0Fu), h.neg_m1[2]);
            const float w5 = __fmaf_rn(h.d2[2], (float)(q2 >> 4),   h.neg_m2[2]);
            const float w6 = __fmaf_rn(h.d1[3], (float)(q3 & 0x0Fu), h.neg_m1[3]);
            const float w7 = __fmaf_rn(h.d2[3], (float)(q3 >> 4),   h.neg_m2[3]);

            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                if (xs[t] != nullptr) {
                    const float* x_blk = xs[t] + (size_t)b * 256u;
                    acc1[t] = __fmaf_rn(w0, x_blk[0 * 64 + lane], acc1[t]);
                    acc1[t] = __fmaf_rn(w1, x_blk[0 * 64 + 32 + lane], acc1[t]);
                    acc1[t] = __fmaf_rn(w2, x_blk[1 * 64 + lane], acc1[t]);
                    acc1[t] = __fmaf_rn(w3, x_blk[1 * 64 + 32 + lane], acc1[t]);
                    acc1[t] = __fmaf_rn(w4, x_blk[2 * 64 + lane], acc1[t]);
                    acc1[t] = __fmaf_rn(w5, x_blk[2 * 64 + 32 + lane], acc1[t]);
                    acc1[t] = __fmaf_rn(w6, x_blk[3 * 64 + lane], acc1[t]);
                    acc1[t] = __fmaf_rn(w7, x_blk[3 * 64 + 32 + lane], acc1[t]);
                }
            }
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            acc0[t] += __shfl_down_sync(0xffffffff, acc0[t], mask);
            acc1[t] += __shfl_down_sync(0xffffffff, acc1[t], mask);
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            int b_idx = batch_base + t;
            if (b_idx < batch_size) {
                float* out_row = out + (size_t)b_idx * (size_t)ne1;
                float a0 = acc0[t];
                float a1 = acc1[t];
                if (residual != nullptr) {
                    const float* res_row = residual + (size_t)b_idx * (size_t)ne1;
                    a0 += res_row[col0];
                    if (col1 < ne1) a1 += res_row[col1];
                }
                out_row[col0] = a0;
                if (col1 < ne1) out_row[col1] = a1;
            }
        }
    }
}


