// Port of high-performance GPU logit bitmasking for grammar-constrained decoding (Titan Agent Runtime).
//
// In-place bitmask suppression directly in GPU VRAM:
// Sets logits of disallowed tokens (bit == 0) to -INFINITY (-1e30f) before softmax / sampling.

extern "C" __global__ void apply_logit_mask_kernel(
    float* __restrict__ logits,              // [vocab_size] FP32 logits
    const unsigned int* __restrict__ mask,   // [vocab_size / 32] packed bitmask
    int vocab_size                           // Total vocabulary size (e.g. 128256 or 151936)
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= vocab_size) return;

    const int word_idx = idx >> 5;           // idx / 32
    const int bit_idx  = idx & 31;           // idx % 32
    const unsigned int word = mask[word_idx];

    // Check if bit is 0 (disallowed token)
    if (!((word >> bit_idx) & 1u)) {
        logits[idx] = -1e30f; // -INFINITY mask
    }
}