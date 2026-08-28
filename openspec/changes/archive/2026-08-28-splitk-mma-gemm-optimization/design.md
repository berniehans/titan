## Context

Titan currently uses single-warp GEMV kernels (`gemm_q4k_mma_kernel` and `gemm_q4k_kernel`) where 1 warp of 32 threads processes the entire $K$ reduction dimension of a weight column $N$. On models with large hidden dimensions ($K \ge 2048$, such as Llama 3.2 3B with $d_{\text{hidden}} = 3072$), the GPU has too few active threadblocks to saturate the 30 Ampere SMs of the RTX 3060 Laptop GPU, leaving hardware compute units idle.

## Goals / Non-Goals

**Goals:**
- Implement Split-K grid decomposition for Q4_K matrix-vector multiplication where the reduction loop across $K / 256$ super-blocks is split into $S_K$ parallel chunks.
- Provide a fast parallel reduction kernel (`reduce_splitk_kernel`) that merges intermediate sums into the output buffer with residual addition.
- Preserve zero-overhead single-pass execution when $K < 2048$.
- Maintain 100% numerical parity against standard dequantization references.

**Non-Goals:**
- Split-K for prefill batched GEMM (Prefill uses chunked GEMM with $M > 1$, which already saturates SMs).
- Changes to tokenization, KV cache layout, or CPU fallback pipelines.

## Decisions

1. **Grid Layout for Split-K:**
   - Grid dimensions: `grid.x = (ne1 + (blockDim.x / 32) - 1) / (blockDim.x / 32)`, `grid.y = batch_size`, `grid.z = split_k`.
   - Each threadblock handles a sub-slice `[k_start, k_end)` of the $K / 256$ super-blocks.
2. **Intermediate Workspace Storage:**
   - Pre-allocate a device buffer `splitk_scratch_dev` of size `batch_size * ne1 * split_k * sizeof(float)` once during `ForwardDriver` initialization.
3. **Reduction Aggregation Pass:**
   - Single-pass vectorized reduction kernel `reduce_splitk_kernel` launched on the same stream, computing `out[col] = sum_{s=0..split_k-1} scratch[s, col] + residual[col]`.

## Risks / Trade-offs

- **[Risk]** Launch latency overhead of 2 kernel launches (Split-K GEMV + Reduction) on small models $\to$ **Mitigation:** Only route to Split-K when $K \ge 2048$ and $ne1 \le 4096$; keep standard 1-kernel GEMV for 0.6B / 1.5B.
- **[Risk]** Scratch buffer memory footprint $\to$ **Mitigation:** Scratch buffer for $S_K = 4$ on 3B models is $4 \times 14336 \times 4\text{ bytes} \approx 229\text{ KB}$, negligible against 6 GB VRAM budget.
