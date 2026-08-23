// dequant_q4k_kernel - Q4_K_M GPU dequantization (f3-gpu-dequant, task 2.2).
//
// Q4_K super-block layout (144 bytes -> 256 weights):
//   bytes[0..2]   fp16 LE d     super scale for the 6-bit sub-block scales
//   bytes[2..4]   fp16 LE dmin  super scale for the 6-bit sub-block mins
//   bytes[4..16]  scales[12]    8x 6-bit sub-block scales + 8x 6-bit mins
//   bytes[16..144] qs[128]      256 weights packed 4-bit, 2 per byte
//
// Dequant (must be bit-identical to engine-core::dequant::dequant_q4k_cpu):
//   per sub-block sb in 0..7:
//     d, dmin from fp16 header
//     (sc, min) from get_scale_min(sb)
//     d1 = d * sc ; m1 = dmin * min
//     weights = qs[(sb/2)*32 .. +32], low nibble if even sb, high nibble if odd
//     out = d1 * weight - m1
//
// Indexing: one thread per 32-weight sub-block (8 threads per super-block).
// No shared memory, no __syncthreads. Each thread writes a contiguous 32-float tile.

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

extern "C" __global__ void dequant_q4k_kernel(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int n_blocks)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = n_blocks * 8;
    if (idx >= total_threads) return;

    const int block_id = idx >> 3;     // idx / 8
    const int sb       = idx & 7;      // idx % 8 -> sub-block index 0..7

    const unsigned char* blk = src + (size_t)block_id * 144u;

    const unsigned d_bits    = (unsigned)blk[0] | ((unsigned)blk[1] << 8);
    const unsigned dmin_bits = (unsigned)blk[2] | ((unsigned)blk[3] << 8);
    const float d    = f16_to_f32(d_bits);
    const float dmin = f16_to_f32(dmin_bits);

    const unsigned char* scales = blk + 4;
    unsigned s, m;
    if (sb < 4) {
        s = (unsigned)(scales[sb]     & 63u);
        m = (unsigned)(scales[sb + 4] & 63u);
    } else {
        s = (unsigned)(scales[sb + 4] & 0xFu) | (((unsigned)scales[sb - 4] >> 6) << 4);
        m = (unsigned)(scales[sb + 4] >> 4)   | (((unsigned)scales[sb]     >> 6) << 4);
    }

    const float d1 = d * (float)s;
    const float m1 = dmin * (float)m;

    const int qbase = (sb >> 1) * 32;   // 0,0,32,32,64,64,96,96
    const unsigned char* qs = blk + 16;
    const bool low = ((sb & 1) == 0);    // even sb -> low nibble
    float* out = dst + (size_t)block_id * 256u + (size_t)sb * 32u;

    #pragma unroll
    for (int l = 0; l < 32; ++l) {
        const unsigned char q = qs[qbase + l];
        const float w = low ? (float)(q & 0xFu) : (float)(q >> 4);
        out[l] = d1 * w - m1;
    }
}
