# Design: Next-Gen Throughput Optimization, Multi-Model Speculative Decoding, and Interactive CLI

## Architectural Overview

```
                          +---------------------------------------------------+
                          |                  TITAN CLI                        |
                          |  [titan run (REPL)]   |   [titan serve (HTTP SSE)]|
                          +---------------------------------------------------+
                                                   |
                                                   v
                          +---------------------------------------------------+
                          |            SPECULATIVE ORCHESTRATOR               |
                          |   Draft Model (1B)  --->  Target Model (3B/7B)    |
                          |   (Emits K tokens)        (Batched Verify Step)   |
                          +---------------------------------------------------+
                                                   |
                                                   v
                          +---------------------------------------------------+
                          |            FORWARD DRIVER & CUDA GRAPH            |
                          |   Chunked Prefill    |    Autonomous Decode Graph |
                          |  (FlashAttention-2)  |   (Fused Q8_1 DP4A GEMV)   |
                          +---------------------------------------------------+
                                                   |
                                                   v
                          +---------------------------------------------------+
                          |                 HARDWARE GPU (VRAM)               |
                          |   NVIDIA RTX 3060 Laptop (CUDA 12 / NVRTC JIT)   |
                          +---------------------------------------------------+
```

## Key Technical Components

### 1. Adaptive GEMV Kernel Tuning for 3B/7B Architectures
- **Problem**: For 1.5B ($K=1536$), 1 column per warp with `BLOCK_X = 256` gives 8 warps = 8 columns per threadblock. For 3B ($K=3072$) and FFN down-projection ($K=8192$), each warp must iterate over $2\times$ to $5\times$ more blocks, causing thread register spilling and instruction cache stalls.
- **Solution**:
  - Implement a specialized kernel variant `gemm_q4k_wide_kernel` that unrolls the DP4A accumulator over multiple sub-accumulators and dynamically tunes grid dimension according to $K$.
  - Adjust shared memory staging for $K \ge 3072$ to keep activation reuse strictly inside L1 cache.

### 2. Multi-Model Speculative Engine in VRAM
- **Dual Model Loading**: Load both Draft model ($M_1$, ~800 MB VRAM) and Target model ($M_2$, ~2.0 GB VRAM) into GPU VRAM (Total = ~2.8 GB, fitting within the 6 GB capacity of RTX 3060).
- **GPU-Native Candidate Generation**:
  - Step 1: Draft model generates $K=3..5$ speculative tokens using its captured CUDA Graph.
  - Step 2: Target model runs a single batched forward pass ($B=K$) over candidate tokens using `driver.prefill()` or batched decode.
  - Step 3: Fast GPU greedy verification compares target argmax vs draft tokens and advances KV-cache pointers by the number of accepted tokens $\alpha \ge 1$.

### 3. Chunked Prefill with FlashAttention-2
- **Chunk Sizing**: Chunks of $C=512$ tokens are processed in sequence.
- **Paged Attention Integration**: Each chunk computes self-attention and writes key/value projections into paged KV blocks without ever materializing an $N \times N$ attention matrix in VRAM.

### 4. Interactive CLI (`titan run` & `titan serve`)
- Implement `titan-cli` entrypoint with `clap` supporting:
  - `titan run <path> [--draft <draft_path>] [--temp <f32>]`: Interactive terminal session with stream rendering.
  - `titan serve <path> [--port <u16>] [--host <ip>]`: Production HTTP server with OpenAI `/v1/chat/completions` endpoint.
