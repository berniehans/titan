// dequant_q6k_kernel - Q6_K GPU dequantization (p8-gpu-mixed-gemv, task 1.1).
//
// Q6_K super-block layout (210 bytes -> 256 weights):
//   bytes[0..128]    ql[128]      256 weights, low 4 bits (2 per byte)
//   bytes[128..192]  qh[64]       256 weights, high 2 bits (4 per byte)
//   bytes[192..208]  scales[16]   16 signed int8 block scales
//   bytes[208..210]  fp16 LE d    super scale factor
//
// Dequant (must match engine-core::dequant::dequant_q6k_cpu exactly):
//   q = ((ql nibble) | ((qh bits & 3) << 4)) - 32
//   y = d * scales[k] * q
//
// Indexing: 8 threads per super-block (each thread processes one 32-float chunk).

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

extern "C" __global__ void dequant_q6k_kernel(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int n_blocks)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = n_blocks * 8;
    if (idx >= total_threads) return;

    const int block_id = idx >> 3;     // idx / 8
    const int c        = idx & 7;      // chunk index 0..7 (32 floats each)

    const unsigned char* blk = src + (size_t)block_id * 210u;

    const unsigned d_bits = (unsigned)blk[208] | ((unsigned)blk[209] << 8);
    const float d = f16_to_f32(d_bits);

    const unsigned char* ql = blk;
    const unsigned char* qh = blk + 128;
    const signed char* scales = (const signed char*)(blk + 192);

    const int half = c >> 2;          // 0 or 1
    const int pos  = c & 3;           // 0, 1, 2, 3

    const int ql_base = (half == 0) ? 0 : 64;
    const int qh_base = (half == 0) ? 0 : 32;
    const int sc_base = (half == 0) ? 0 : 8;

    float* out_ptr = dst + (size_t)block_id * 256u + (size_t)c * 32u;

    for (int l = 0; l < 32; ++l) {
        const int is = l >> 4;        // l / 16 (0 or 1)
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
        } else { // pos == 3
            const int ql_idx = ql_base + l + 32;
            const int qh_idx = qh_base + l;
            q = (int)(ql[ql_idx] >> 4) | (((int)(qh[qh_idx] >> 6) & 3) << 4);
            sc_idx = sc_base + 6 + is;
        }

        q -= 32;
        out_ptr[l] = d * (float)scales[sc_idx] * (float)q;
    }
}
