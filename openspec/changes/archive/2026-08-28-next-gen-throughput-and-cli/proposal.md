# Proposal: Next-Gen Throughput Optimization, Multi-Model Speculative Decoding, and Interactive CLI

## Problem Statement
Titan has achieved 96% decode throughput parity with `llama.cpp` on 1.5B models (135.3 tok/s vs 140.7 tok/s) and 163 tok/s on Llama 3.2 1B using autonomous CUDA Graphs and fused Q8_1 DP4A GEMV kernels. However:
1. **Medium models (3B, 7B)** currently decode at 70.2 tok/s (vs 93.2 tok/s in llama.cpp) due to higher register pressure and sub-optimal threadblock partitioning on wider hidden layers ($d_{\text{model}} = 3072$, $\text{FFN} = 8192$).
2. **Speculative decoding** has not yet been integrated across distinct GGUF models in GPU memory (e.g. using Llama 3.2 1B as a Draft model to accelerate Llama 3.2 3B or 7B by 2x-3x).
3. **Chunked Prefill & FlashAttention** are needed to maintain high throughput during large prompt ingestion (>2k, 4k, 8k tokens) without memory allocation spikes.
4. **Interactive CLI (`titan run`) & Production Server (`titan serve`)**: Users need a single-command REPL experience with live token streaming and telemetry in the terminal, plus an OpenAI-compatible HTTP daemon.

## Proposed Solution
1. **Adaptive Split-K / Multi-Warp GEMV Tuning for 3B/7B**:
   - Implement adaptive column and split-K grid sizing based on matrix inner dimension ($K \ge 3072$) to maximize SM occupancy on Ampere/Ada architectures.
2. **End-to-End GPU Speculative Decoding Pipeline**:
   - Execute Draft model generation (e.g. $K=3..5$ tokens) directly in VRAM and verify speculative tokens in a single batched verification step on the Target model.
3. **Chunked Prefill & FlashAttention Engine**:
   - Stream prefill tokens in 512-token chunks with fused FlashAttention-2 kernels to saturate GPU memory bandwidth while keeping KV cache allocations linear and bounded.
4. **Unified CLI Suite (`titan run` / `titan serve`)**:
   - Deliver `titan run <model.gguf>` with animated markdown streaming, interactive REPL, and real-time tok/s & VRAM HUD.
   - Deliver `titan serve <model.gguf> --port 8000` with production-grade Axum HTTP server.

## Impact & Acceptance Criteria
- **Llama 3.2 3B Throughput**: Reach $\ge 90$ tok/s on RTX 3060 Laptop (closing the gap with llama.cpp).
- **Speculative Speedup**: Reach $\ge 140$ tok/s on 3B models with 1B draft speculative generation (2x speedup).
- **Interactive CLI**: Instant chat startup in terminal with `<1` second initial load.
