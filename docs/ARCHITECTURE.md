# 🏗️ Titan — Deep Technical Architecture

> **Canonical Specifications:** [`openspec/specs/`](../openspec/specs/)  
> **Constitutional Invariants:** [`openspec/constitution.md`](../openspec/constitution.md)

**Titan** is a 100% Pure Rust, high-throughput LLM inference engine engineered specifically for consumer and datacenter NVIDIA GPUs. It implements both **GPU-Resident** execution (with Autonomous CUDA Graphs) for models that fit in VRAM, and **Layer-Streaming** execution (with double-buffered DMA) for models that exceed VRAM capacity.

---

## 1. System Architecture Diagram

```
+----------------------------------------------------------------------------------------------------+
|                                      TITAN USER INTERFACE LAYER                                    |
|   [titan run (Interactive REPL Chat)]       |       [titan serve (OpenAI SSE API Daemon)]          |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                    ENGINE ORCHESTRATION LAYER                                      |
|  ┌──────────────────────────────────────────────┐  ┌────────────────────────────────────────────┐  |
|  │        Speculative Multi-Model Engine        │  │          Continuous Batching Engine        │  |
|  │   (Draft 1B Generator -> Target 3B Verifier) │  │          (Dynamic Request Scheduler)       │  |
|  └──────────────────────────────────────────────┘  └────────────────────────────────────────────┘  |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                  RESIDENT FORWARD DRIVER (GPU VRAM)                                 |
|                                                                                                    |
|  ┌──────────────────────────────────────────────────────────────────────────────────────────────┐  |
|  │                           AUTONOMOUS CUDA GRAPH EXECUTION STREAM                             │  |
|  │                                                                                              │  |
|  │   [Embedded Token]                                                                           │  |
|  │          │                                                                                   │  |
|  │          ▼ (28 Transformer Layers Executed Entirely in VRAM)                                 │  |
|  │   ┌───────────────────────────────────────────────────────────────────────────────────────┐  │  |
|  │   │ 1. Dynamic Q8_1 Act Quantization + RMSNorm Fused Kernel (__dp4a)                      │  │  |
|  │   │ 2. Fused QKV GEMV Projection (uint4 coalesced loads)                                  │  │  |
|  │   │ 3. Per-Head RoPE + Paged KV-Cache Append (uint32 pos_dev)                             │  │  |
|  │   │ 4. PagedAttention / FlashAttention-2 Causal Kernel                                    │  │  |
|  │   │ 5. Out Projection (Wo) GEMV + In-Place Residual 1 Addition                            │  │  |
|  │   │ 6. SwiGLU Fused Gate/Up Projections (silu(Gate) * Up) + FFN RMSNorm                   │  │  |
|  │   │ 7. Down Projection (Wdown) GEMV + In-Place Residual 2 Addition                        │  │  |
|  │   └───────────────────────────────────────────────────────────────────────────────────────┘  │  |
|  │          │                                                                                   │  |
|  │          ▼                                                                                   │  |
|  │   [Final RMSNorm] -> [LM Head GEMV] -> [GPU Argmax Reduction] -> [Next Token in VRAM]       │  |
|  └──────────────────────────────────────────────────────────────────────────────────────────────┘  |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                    NVIDIA HARDWARE RUNTIME (CUDA)                                  |
|   • Direct NVRTC JIT Compilation via Driver API (nvcuda.dll / libcuda.so)                          |
|   • Zero MSVC (cl.exe), Zero CMake, Zero Python, Zero C++ External Dependencies                    |
|   • Hardware Target: Compute Capability 8.6+ (Ampere / Ada Lovelace / Hopper / Blackwell)          |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Core Architectural Pillars

### 2.1 Autonomous CUDA Graphs in GPU VRAM
* **The Problem:** In conventional inference engines (e.g. standard PyTorch or naive Rust implementations), generating a single token requires issuing 150–200 individual CUDA kernel launches from the CPU host across the PCIe bus. At 150+ tok/s, CPU-to-GPU dispatch latency dominates execution time.
* **Titan's Solution:** Titan captures the entire forward pass into a resident `CudaGraphExec`. 
* **Zero Host Round-Trips:** The token embedding lookup, layer loops, norm operations, RoPE, attention, SwiGLU, LM Head projection, and greedy argmax reduction are chained in device memory. The GPU loops autonomously without ever returning control to the CPU host between token generations.

```
Conventional:   [CPU Dispatch] -> [Kernel 1] -> [CPU Dispatch] -> [Kernel 2] ... (~150 launches/tok)
Titan Engine:   [Launch Autonomous Graph (1 call)] ===> [GPU executes 28 layers + argmax] (0 CPU stalls)
```

---

### 2.2 Vectorized DP4A SIMD GEMV (`__dp4a`) & Fused QKV / Fused Q8_1 Quantization
Titan implements custom hand-optimized CUDA kernels that utilize NVIDIA hardware **DP4A** (4-way 8-bit integer dot product and 32-bit accumulation SIMD instruction):
1. **Dynamic Activation Quantization:** Float activations are quantized on-the-fly into `Q8_1` blocks (32 signed 8-bit integers + fp32 scale + fp32 sum) in shared memory with 128-bit (`int4`) memory coalescing.
2. **Fused QKV Projection (`gemm_fused_qkv_q4k` / `gemm_fused_qkv_q6k`):** Instead of launching 3 separate matrix-vector multiplications for Query, Key, and Value projections ($3 \times 28 = 84$ launches per token), Titan collapses them into a single unified kernel launch ($1 \times 28 = 28$ launches), eliminating 56 kernel launch bubbles per token.
3. **Matrix-Vector Kernel:** Weights stored in `Q4_K` / `Q6_K` superblocks (144 / 210 bytes) are unpacked directly in registers.
4. **RoPE Dimension Specialization:** Dedicated execution path for $d_{\text{head}} = 64$ (Llama 3.2 1B) performing full 32-pair parallel rotation across warp threads in 1 step without branch divergence or memory out-of-bounds.
5. **Pre-Multiplied Scales & Unrolled Loops:** `d_sc` scales and `s_qd` activation scales are hoisted into register files, enabling maximum instruction-level parallelism (ILP) and saturating GPU memory bandwidth (336 GB/s on RTX 3060).

---

### 2.3 Paged KV-Cache, Virtual Block Table & Attention Sinks (StreamingLLM)
* **Non-Contiguous Memory Allocation:** Memory for Key and Value vectors is allocated in fixed-size blocks (e.g. 16 tokens per block) managed by `engine-kvcache`.
* **Attention Sinks & Infinite Context:** Preserves initial sink tokens ($K=4$) while pruning intermediate evicted blocks, guaranteeing numerical stability and infinite continuous generation under strict VRAM caps.
* **Block-Table Indirection:** A GPU device buffer `bt_dev` maps logical sequence tokens to physical memory pool pages, eliminating VRAM fragmentation and enabling instantaneous sequence rollback during speculative decoding.
* **Chunked Prefill:** Long input sequences ($N \ge 2048$) are evaluated in discrete chunks of $C \le 512$ tokens via `prefill_chunked()`, keeping KV allocation bounded and achieving $>1000$ tok/s prompt ingestion.

---

### 2.4 Multi-Model GPU Speculative Decoding
Titan supports simultaneous loading of two distinct GGUF models into GPU VRAM:
* **Draft Model ($M_1$):** Lightweight model (e.g. *Llama 3.2 1B*, ~800 MB VRAM) running at **168 tok/s**.
* **Target Model ($M_2$):** Higher capacity model (e.g. *Llama 3.2 3B*, ~2.0 GB VRAM).
* **Parallel GPU Verification:**
  1. Draft model generates $K=3..5$ candidate tokens using its captured CUDA Graph.
  2. Target model evaluates all candidate tokens in a single parallel verification pass.
  3. Speculative verifier checks logits and commits accepted tokens, rolling back or advancing the virtual KV-cache pointers with zero data re-copying.

---

### 2.5 Grammar-Constrained JSON Decoding & Tool Calling
* **RFC 8259 Deterministic State Machine:** Direct token-by-token grammar validation with state transitions for objects, arrays, keys, string values, booleans (`true`/`false`), and `null`.
* **Space Anti-Looping Rules:** Prevents greedy tokenizers from degenerating into endless whitespace cycles after JSON delimiters.
* **Overlapped GPU Logit Bitmasking:** Integrates directly with the GPU sampling pipeline for <0.04 ms overhead per token.

---

### 2.6 Layer Streaming & Double-Buffered DMA (Out-of-Core Execution)
For massive models that exceed total GPU VRAM (e.g. 14B or 32B models on a 6GB card):
* **Single NVMe Pass:** Tensors are loaded into non-pageable pinned host RAM (`cuMemAllocHost`) once at startup.
* **Ping-Pong Slots:** Two layer buffers `slot[0]` and `slot[1]` reside in VRAM.
* **Asynchronous Overlap:** While the compute stream executes Layer $N$ on `slot[0]`, the transfer stream asynchronously copies Layer $N+1$ into `slot[1]` via PCIe DMA, synchronized purely through device-side CUDA events (`cuStreamWaitEvent`).

---

## 3. Crate Dependency Graph & Boundaries

```
engine/
├── engine-api/          # Interface boundaries, public traits, telemetry structs
├── engine-io/           # Single-pass GGUF v3 parser, zero-copy loader, model config
├── engine-cuda/         # NVRTC JIT compilation, DP4A GEMV kernels, PagedAttention, CudaStream/Event
├── engine-kvcache/      # Virtual block table and paged cache allocation
├── engine-core/         # ForwardDriver, Speculative Engine, Autonomous Graph, Sampler
└── engine-server/       # Axum HTTP API daemon, SSE streaming, CLI terminal REPL
```

* **No Circular Dependencies:** Every crate has a strict acyclic dependency hierarchy verified by CI.
* **Zero C++ Dependencies:** CUDA kernels in `engine-cuda/kernels/` are embedded as string literals and JIT-compiled at runtime using the system's `nvcuda.dll` / `libcuda.so`.

---

## 4. VRAM Footprint & Budgeting (RTX 3060 6GB Example)

| Memory Region | Allocation Size (1.5B / 3B) | Lifecycle | Description |
| :--- | :--- | :--- | :--- |
| **Model Weights (Resident)** | ~930 MB (1.5B) / ~2.02 GB (3B) | Static Lifetime | Quantized GGUF tensor data in GPU device memory. |
| **Paged KV Cache** | ~256 MB – 512 MB | Dynamic Pool | Virtual memory block pages for Keys and Values. |
| **Activation & Intermediate Buffers** | ~64 MB | Static Preallocated | Fused QKV, SwiGLU, RMSNorm scratchpads. |
| **Autonomous CUDA Graph State** | ~12 MB | Captured Executable | Executable graph nodes and device parameter bindings. |
| **Free Headroom** | **>3.0 GB Available** | Free VRAM | Available for Speculative Draft models or batching. |
