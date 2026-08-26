# Proposal: Phase 11 — Chunked Prefill & FlashAttention-2 GPU Kernel

## 1. Summary

Implement a batched, chunked prefill engine with a native CUDA **FlashAttention-2 causal kernel** and **batched quantized GEMM (Q4_K, Q6_K, Q8_0)** to accelerate prompt evaluation (Time To First Token — TTFT) by $10\times$ to $50\times$ over serial token-by-token prefill.

---

## 2. Motivation

- **The TTFT Bottleneck:** Currently, `ForwardDriver::prefill` evaluates prompt sequences sequentially (one token per pass). While decode is memory-bandwidth bound ($M=1$), prompt prefill is compute-bound ($M=S$) and can be parallelized across GPU Streaming Multiprocessors (SMs). Serial prefill on a 512-token prompt incurs 512 separate kernel dispatches and memory round-trips.
- **O(N) VRAM Footprint with FlashAttention-2:** A naive batched prefill attention materializes an intermediate $S \times S$ attention matrix in VRAM, which for long contexts (2k–8k tokens) explodes in memory consumption. FlashAttention-2 uses tiled shared-memory blocks ($B_r \times B_c$) with online softmax renormalization, maintaining an $O(S)$ working memory footprint strictly bounded within Titan's 5.2 GB usable budget.
- **Chunked Prefill for Infinite Context & Continuous Batching:** By partitioning long prompts into bounded chunks ($S \le \text{CHUNK\_SIZE} = 128\text{ or }256$), prefill computes causal attention within the chunk and cross-attention against preceding paged KV blocks, interleaving seamlessly with decode steps.

---

## 3. Scope & Sub-Changes

1. **Sub-change 11.1 — Batched Quantized Matrix Multiplication Kernels (`engine-cuda`):**
   - Implement `gemm_q4k_kernel`, `gemm_q6k_kernel`, and `gemm_q80_kernel` supporting batch size $M \in [1, 512]$.
   - Use 2D tiled shared-memory matrix multiplication with on-the-fly register dequantization.
   - Parity gate: Bit-exact / numerical parity ($< 10^{-4}$ error, $\text{cos-sim} \ge 0.9999$) against CPU reference batched GEMM across batch sizes $M \in \{16, 32, 64, 128, 256\}$.

2. **Sub-change 11.2 — FlashAttention-2 Causal GPU Kernel (`engine-cuda`):**
   - Implement `flash_attention_2_kernel` in `engine-cuda/kernels/flash_attention_2.cu`.
   - Tiled computation ($B_r = 64, B_c = 64$) with online softmax running statistics ($m, l$).
   - Dynamic causal masking and paged KV cache integration (reading historical context from `PagedKvCache` pool).
   - TDD parity gate against CPU reference multi-head attention.

3. **Sub-change 11.3 — ForwardDriver Batched & Chunked Prefill Integration (`engine-core`):**
   - Implement `ForwardDriver::prefill_chunked(&mut self, prompt_tokens: &[u32], chunk_size: usize)`.
   - Fused batched RMSNorm + RoPE for sequence slices.
   - Batched KV append writing $S$ tokens into resident paged KV blocks in a single kernel dispatch.
   - Multi-prompt golden parity gate against llama.cpp logits (`cos-sim >= 0.997`).

4. **Sub-change 11.4 — TTFT Benchmarks & Phase 11 Seal (`engine-server` / `docs`):**
   - Benchmark TTFT across prompt lengths ($S \in \{16, 64, 128, 256, 512, 1024\}$).
   - Verify $>10\times$ speedup in TTFT over serial prefill.
   - Record measured latencies in `docs/BENCHMARKS.md`, sync delta spec, and archive change.
