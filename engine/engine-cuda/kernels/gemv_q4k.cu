// Port of llama.cpp vcdotq.cuh::vec_dot_q4_K_q8_K (ggml/src/ggml-cuda/vecdotq.cuh @ cb1adf8)
//
// Multi-format GEMV kernel: Q4_K_M, Q8_0, and F16 column-major reduction kernels.
// Each thread handles one output column (dot product over ne0 elements).
//
// Layouts:
//   Q4_K: 256 weights per 144-byte super-block
//   Q8_0: 32 weights per 34-byte block ([fp16 d][int8 qs x32])
//   F16:  ne0 * 2 bytes (fp16 LE)

__device__ __forceinline__ float f16_to_f32(unsigned bits) {
    const unsigned sign = (bits >> 15) & 1u;
    const unsigned exp  = (bits >> 10) & 0x1Fu;
    const unsigned mant = bits & 0x3FFu;
    if (exp == 0u) {
        if (mant == 0u) return sign ? -0.0f : 0.0f;
        return (float)mant * exp2f(-24.0f);
    } else if (exp == 31u) {
        return __int_as_float(0x7f800000u);  // +inf (NVRTC has no INFINITY macro)
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

extern "C" __global__ void gemv_q4k_kernel(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ out,
    int ne0,
    int ne1)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ne1) return;

    const int n_blocks = ne0 / 256;
    const unsigned char* col_weights = weights + (size_t)col * (size_t)n_blocks * 144u;

    float acc = 0.0f;

    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* blk = col_weights + (size_t)b * 144u;
        const float* x_blk = x + (size_t)b * 256u;

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

    out[col] = acc;
}

extern "C" __global__ void gemv_q8_kernel(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ out,
    int ne0,
    int ne1)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ne1) return;

    const int n_blocks = ne0 / 32;
    const unsigned char* col_weights = weights + (size_t)col * (size_t)n_blocks * 34u;

    float acc = 0.0f;

    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* blk = col_weights + (size_t)b * 34u;
        const float* x_blk = x + (size_t)b * 32u;

        const unsigned d_bits = (unsigned)blk[0] | ((unsigned)blk[1] << 8);
        const float d = f16_to_f32(d_bits);
        const signed char* qs = (const signed char*)(blk + 2);

        #pragma unroll
        for (int l = 0; l < 32; ++l) {
            const float w = __fmul_rn((float)qs[l], d);
            acc = __fadd_rn(acc, __fmul_rn(w, x_blk[l]));
        }
    }

    out[col] = acc;
}

extern "C" __global__ void gemv_f16_kernel(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ out,
    int ne0,
    int ne1)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ne1) return;

    const unsigned char* col_weights = weights + (size_t)col * (size_t)ne0 * 2u;

    float acc = 0.0f;

    #pragma unroll 4
    for (int i = 0; i < ne0; ++i) {
        const unsigned bits = (unsigned)col_weights[2 * i] | ((unsigned)col_weights[2 * i + 1] << 8);
        const float w = f16_to_f32(bits);
        acc = __fadd_rn(acc, __fmul_rn(w, x[i]));
    }

    out[col] = acc;
}
