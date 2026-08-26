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

__device__ __forceinline__ float f16_to_f32(unsigned bits) {
    const unsigned sign = (bits >> 15) & 1u;
    const unsigned exp  = (bits >> 10) & 0x1Fu;
    const unsigned mant = bits & 0x3FFu;
    if (exp == 0u) {
        if (mant == 0u) return sign ? -0.0f : 0.0f;
        const float val = (float)mant * exp2f(-24.0f);
        return sign ? -val : val;
    } else if (exp == 31u) {
        return sign ? -__int_as_float(0x7f800000u) : __int_as_float(0x7f800000u);
    }
    const float val = (1.0f + (float)mant / 1024.0f) * exp2f((float)(int)exp - 15.0f);
    return sign ? -val : val;
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

extern "C" __global__ void gemm_q4k_kernel(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ out,
    int ne0,
    int ne1,
    int batch_size)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y;
    if (col >= ne1 || batch_idx >= batch_size) return;

    const float* x_row = x + (size_t)batch_idx * (size_t)ne0;
    float* out_row = out + (size_t)batch_idx * (size_t)ne1;

    const int n_blocks = ne0 / 256;
    const unsigned char* col_weights = weights + (size_t)col * (size_t)n_blocks * 144u;

    float acc = 0.0f;

    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* blk = col_weights + (size_t)b * 144u;
        const float* x_blk = x_row + (size_t)b * 256u;

        const unsigned d_bits    = (unsigned)blk[0] | ((unsigned)blk[1] << 8);
        const unsigned dmin_bits = (unsigned)blk[2] | ((unsigned)blk[3] << 8);
        const float d    = f16_to_f32(d_bits);
        const float dmin = f16_to_f32(dmin_bits);

        const unsigned char* scales = blk + 4;
        const unsigned char* qs = blk + 16;

        for (int g = 0; g < 4; ++g) {
            const int is_idx = g * 2;
            unsigned sc0, m0, sc1, m1s;
            get_scale_min(is_idx, scales, &sc0, &m0);
            get_scale_min(is_idx + 1, scales, &sc1, &m1s);

            const float d1 = __fmul_rn(d, (float)sc0);
            const float m1 = __fmul_rn(dmin, (float)m0);
            const float d2 = __fmul_rn(d, (float)sc1);
            const float m2 = __fmul_rn(dmin, (float)m1s);

            const int qbase = g * 32;
            const float* x_lo = x_blk + g * 64;
            const float* x_hi = x_blk + g * 64 + 32;

            #pragma unroll
            for (int l = 0; l < 32; ++l) {
                const unsigned char q = qs[qbase + l];
                const float w_lo = __fsub_rn(__fmul_rn(d1, (float)(q & 0x0Fu)), m1);
                acc = __fadd_rn(acc, __fmul_rn(w_lo, x_lo[l]));
            }

            #pragma unroll
            for (int l = 0; l < 32; ++l) {
                const unsigned char q = qs[qbase + l];
                const float w_hi = __fsub_rn(__fmul_rn(d2, (float)(q >> 4)), m2);
                acc = __fadd_rn(acc, __fmul_rn(w_hi, x_hi[l]));
            }
        }
    }

    out_row[col] = acc;
}

extern "C" __global__ void gemm_q6k_kernel(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ out,
    int ne0,
    int ne1,
    int batch_size)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y;
    if (col >= ne1 || batch_idx >= batch_size) return;

    const float* x_row = x + (size_t)batch_idx * (size_t)ne0;
    float* out_row = out + (size_t)batch_idx * (size_t)ne1;

    const int n_blocks = ne0 / 256;
    const unsigned char* col_weights = weights + (size_t)col * (size_t)n_blocks * 210u;

    float acc = 0.0f;

    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* blk = col_weights + (size_t)b * 210u;
        const float* x_blk = x_row + (size_t)b * 256u;

        const unsigned d_bits = (unsigned)blk[208] | ((unsigned)blk[209] << 8);
        const float d = f16_to_f32(d_bits);

        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;
        const signed char* scales = (const signed char*)(blk + 192);

        // Process 4 pairs of chunks (8 chunks of 32 weights = 256 weights)
        for (int c = 0; c < 8; ++c) {
            const int half = c >> 2;
            const int pos  = c & 3;

            const int ql_base = (half == 0) ? 0 : 64;
            const int qh_base = (half == 0) ? 0 : 32;
            const int sc_base = (half == 0) ? 0 : 8;

            const float* x_chunk = x_blk + c * 32;

            for (int l = 0; l < 32; ++l) {
                const int is = l >> 4;
                int q = 0;
                int sc_idx = 0;

                if (pos == 0) {
                    const int ql_idx = ql_base + l;
                    const int qh_idx = qh_base + l;
                    q = (int)(ql[ql_idx] & 0x0Fu) | (((int)(qh[qh_idx] >> 0) & 3) << 4);
                    sc_idx = sc_base + is;
                } else if (pos == 1) {
                    const int ql_idx = ql_base + l + 32;
                    const int qh_idx = qh_base + l;
                    q = (int)(ql[ql_idx] & 0x0Fu) | (((int)(qh[qh_idx] >> 2) & 3) << 4);
                    sc_idx = sc_base + 2 + is;
                } else if (pos == 2) {
                    const int ql_idx = ql_base + l;
                    const int qh_idx = qh_base + l;
                    q = (int)(ql[ql_idx] >> 4) | (((int)(qh[qh_idx] >> 4) & 3) << 4);
                    sc_idx = sc_base + 4 + is;
                } else {
                    const int ql_idx = ql_base + l + 32;
                    const int qh_idx = qh_base + l;
                    q = (int)(ql[ql_idx] >> 4) | (((int)(qh[qh_idx] >> 6) & 3) << 4);
                    sc_idx = sc_base + 6 + is;
                }

                q -= 32;
                const float w = d * (float)scales[sc_idx] * (float)q;
                acc = __fadd_rn(acc, __fmul_rn(w, x_chunk[l]));
            }
        }
    }

    out_row[col] = acc;
}

extern "C" __global__ void gemm_q8_kernel(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ out,
    int ne0,
    int ne1,
    int batch_size)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
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

        #pragma unroll
        for (int l = 0; l < 32; ++l) {
            const float w = __fmul_rn(d, (float)qs[l]);
            acc = __fadd_rn(acc, __fmul_rn(w, x_blk[l]));
        }
    }

    out_row[col] = acc;
}

extern "C" __global__ void gemm_f16_kernel(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ out,
    int ne0,
    int ne1,
    int batch_size)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y;
    if (col >= ne1 || batch_idx >= batch_size) return;

    const float* x_row = x + (size_t)batch_idx * (size_t)ne0;
    float* out_row = out + (size_t)batch_idx * (size_t)ne1;

    const unsigned short* col_w = (const unsigned short*)weights + (size_t)col * (size_t)ne0;

    float acc = 0.0f;
    for (int i = 0; i < ne0; ++i) {
        const float w = f16_to_f32((unsigned)col_w[i]);
        acc = __fadd_rn(acc, __fmul_rn(w, x_row[i]));
    }

    out_row[col] = acc;
}

extern "C" __global__ void gemm_f32_kernel(
    const float* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ out,
    int ne0,
    int ne1,
    int batch_size)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y;
    if (col >= ne1 || batch_idx >= batch_size) return;

    const float* x_row = x + (size_t)batch_idx * (size_t)ne0;
    float* out_row = out + (size_t)batch_idx * (size_t)ne1;

    const float* col_w = weights + (size_t)col * (size_t)ne0;

    float acc = 0.0f;
    for (int i = 0; i < ne0; ++i) {
        acc = __fadd_rn(acc, __fmul_rn(col_w[i], x_row[i]));
    }

    out_row[col] = acc;
}
