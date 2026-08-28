# Proposal: FlashDecoding & Split-KV PagedAttention for Long-Context Inference

## Motivation

In Autonomous Agent workloads (e.g. multi-step reasoning, Hermes Agent tool-calling traces, code analysis), context windows quickly expand from hundreds to thousands of tokens (2,048 – 8,192 tokens).

Currently, Titan uses single-warp sequential PagedAttention (`paged_attention_decode_kernel`), which assigns 1 thread block per query head and loops sequentially over all KV tokens in the sequence. For short contexts (<256 tokens), this takes <0.3 ms. However, at $N = 4,096$ tokens, sequential warp execution becomes a severe bottleneck (taking >3.5 ms per token), dropping overall engine throughput by more than 40%.

By implementing **FlashDecoding (Split-KV PagedAttention)**, Titan parallelizes the sequence dimension across all 30 Streaming Multiprocessors (SMs) on the NVIDIA RTX 3060 Mobile. Each block processes a chunk of 256 tokens in parallel, producing partial log-sum-exp $(\mathbf{m}_s, \mathbf{l}_s, \mathbf{acc}_s)$ that are merged in a single fused reduction kernel. This delivers up to **5x faster attention decode** at long contexts, maintaining consistent >120 tok/s throughput across the entire context window.

## Performance Goals & Target Metrics

1. **Attention Latency at Long Contexts (RTX 3060 Mobile):**
   * Context length $N = 2,048$: Attention compute time reduced from 1.6 ms $\to$ **<0.4 ms** (4.0x speedup).
   * Context length $N = 4,096$: Attention compute time reduced from 3.2 ms $\to$ **<0.7 ms** (4.5x speedup).
   * Context length $N = 8,192$: Attention compute time reduced from 6.5 ms $\to$ **<1.2 ms** (5.4x speedup).
2. **End-to-End Decode Throughput:**
   * Maintain **>120 tok/s** on Qwen 2.5 1.5B and **>150 tok/s** on Llama 3.2 1B across 4,096-token agent conversations.
3. **Exact Mathematical Equivalence:**
   * 100% numerical bit-parity with standard Softmax SDPA within float error tolerance ($< 10^{-4}$).
4. **Autonomous CUDA Graph Compatibility:**
   * All intermediate partial buffer allocations (`max_splits = 32`) must be pre-allocated statically in VRAM to allow zero-overhead CUDA Graph capture.

## Explicit Non-Goals

1. Changing the physical block allocation layout of `engine-kvcache`. FlashDecoding operates directly over the existing `BlockTable` and `DeviceBuffer` pools.
2. Modifying non-attention layers.
