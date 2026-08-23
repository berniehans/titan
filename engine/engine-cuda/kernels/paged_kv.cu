// paged_kv.cu - GPU paged KV-cache kernels (NVRTC raw driver).
//
// Layout of the flat device pool (bit-identical to CPU reference):
//   physical block `b`  -> float offset `b * floats_per_block`
//   token slot `s`      -> `+ s * floats_per_token`
//   key row             -> `+ 0 .. row_len`
//   value row           -> `+ row_len .. 2 * row_len`
//
// where:
//   floats_per_token = 2 * row_len
//   floats_per_block = block_tokens * floats_per_token

extern "C" __global__ void paged_append_kv_kernel(
    const float* __restrict__ keys,
    const float* __restrict__ values,
    const unsigned* __restrict__ block_table,
    float* __restrict__ pool,
    int n_tokens,
    int start_token,
    int row_len,
    int block_tokens)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = n_tokens * row_len;
    if (idx >= total_threads) return;

    const int ti = idx / row_len;
    const int h = idx % row_len;
    const int g = start_token + ti;
    const int block_idx = g / block_tokens;
    const int slot = g % block_tokens;
    const unsigned phys = block_table[block_idx];

    const int floats_per_token = 2 * row_len;
    const int floats_per_block = block_tokens * floats_per_token;

    const size_t pool_base = (size_t)phys * (size_t)floats_per_block + (size_t)slot * (size_t)floats_per_token;
    pool[pool_base + (size_t)h] = keys[idx];
    pool[pool_base + (size_t)row_len + (size_t)h] = values[idx];
}

extern "C" __global__ void paged_gather_kernel(
    const float* __restrict__ pool,
    const unsigned* __restrict__ block_table,
    float* __restrict__ out,
    int n_tokens,
    int start_token,
    int row_len,
    int block_tokens,
    int is_value)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = n_tokens * row_len;
    if (idx >= total_threads) return;

    const int ti = idx / row_len;
    const int h = idx % row_len;
    const int g = start_token + ti;
    const int block_idx = g / block_tokens;
    const int slot = g % block_tokens;
    const unsigned phys = block_table[block_idx];

    const int floats_per_token = 2 * row_len;
    const int floats_per_block = block_tokens * floats_per_token;

    const size_t offset = (size_t)phys * (size_t)floats_per_block + (size_t)slot * (size_t)floats_per_token + (is_value ? (size_t)row_len : 0u) + (size_t)h;
    out[idx] = pool[offset];
}
