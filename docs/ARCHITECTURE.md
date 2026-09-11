# 🏗️ Titan — Deep Technical Architecture

> **Canonical Specifications:** [`openspec/specs/`](../openspec/specs/)
> **Constitutional Invariants:** [`openspec/constitution.md`](../openspec/constitution.md)
> **Stable Functional MVP Contract:** [`docs/MVP.md`](MVP.md)
> **Workspace State & Evidence:** [`docs/WORKSPACE_STATE.md`](WORKSPACE_STATE.md)

---

## 🛑 Current Status & Architectural Boundary

| Dimension | Current Value | Authority / Artifact |
| :--- | :--- | :--- |
| **Functional MVP Status** | **`mvp_blocked_evidence`** (NOT accepted) | [`local-artifacts/reviews/mvp-decision-20260911T162326Z.json`](MVP.md) |
| **Strict Production Release** | **`not_accepted`** / **`rejected`** | `local-artifacts/reviews/p5-release-gate-20260912T130000Z.json` |
| **Release Approval** | **`not_granted`** | [`docs/RELEASE_ENVELOPE_V1.json`](RELEASE_ENVELOPE_V1.json) |
| **Production Promotion** | **`promotion_authorized = false`** | No candidate promoted to default |
| **Default Runtime Dispatch** | **`production_dispatch_changed = false`** | Unchanged F32 resident baseline |
| **Release Candidate Generation** | **`rc_generated = false`** | No release candidate generated |
| **MVP Gate Unit Suite** | **`25 passed, 0 failed`** (100%) | `uv run --project tools --no-sync pytest tools/test_mvp_gate.py` |
| **Full Python Tooling Suite** | **`88 passed, 0 failed; 8 subtests passed`** (100%) | `uv run --project tools --no-sync pytest tools` |
| **OpenSpec Live State** | **`40 passed, 0 failed`** (40 items) | `openspec validate --all` |

> [!IMPORTANT]
> **Functional MVP vs. Strict Production Release:**
> * The $\ge 0.95\times$ comparison against `llama.cpp` is **NOT** an acceptance criterion for the Stable Functional MVP; it remains a separate strict production-performance diagnostic.
> * In the functional MVP evaluation, throughput ratios (aggregate median `0.5286115177544821`) and baseline regressions are advisory warnings (`ratio_advisory: advisory_warning`, `regression_advisory: failed`).
> * **Blocking Gates:** Generation correctness (F8.R1.P verified for 5 models), operational E2E execution (Resident, Streaming SSE with N-gram, grammar JSON parsed as `{"city": "Tokyo"}`), and clean benchmark execution safety remain strictly blocking.
>
> **Exact MVP Blockers:**
> 1. **`build_binding_unverified`:** The F8 correctness execution does not record the Titan binary/source identity matching the clean benchmark's `engine_identity.titan` build hash (`sha256:bc92211356bb0eb87ddd59e252d9d4497f8669d7257ac0133459040ce4eb131c`).
> 2. **`model_hash_missing`:** The final clean benchmark artifact (`local-artifacts/benchmarks/p5-final-clean-20260911T151844Z.json`, 30 rows) records model names but lacks per-result GGUF weight file hashes required for exact binding to F8.
>
> **Architectural Capability vs. Envelope Certification:**
> The deep technical architecture below details the complete capability surface of the engine (e.g., Autonomous CUDA Graphs, layer streaming, speculative decoding, continuous batching, attention sinks). However, **implementation presence does not mean all capabilities are production-integrated or certified under the release envelope**. At the current checkpoint:
> * The verified production route is strictly non-Graph F32 batch 1 resident-KV decode (`CUDA Graphs = false` observed in clean baseline runs).
> * Q8 activation quantization and alternative kernels remain experimental / opt-in.
> * Local directories (`local-artifacts/`, `.hermes/`, `models/`, `target/`) are local runtime assets and not publishable repository content.

---

## 0. Current Implementation & Release Boundary

At the current checkpoint:

- **F32 Baseline:** F32 remains the production correctness default; Q8 is benchmark-only/experimental because its strict full-driver contract is not accepted.
- **Production Dispatch:** The validated default dispatch paths and public APIs remain strictly unchanged (`production_dispatch_changed = false`).
- **Rejected Kernels:** Fused Gate/Up, Q6_K single-row, and QKV multi-row batch 1 kernels remain rejected for production default.
- **Grouped FFN Attribution:** `ffn_gate_up` and `ffn_down` timing is verified; `ffn_sync` remains explicitly `not_available/missing_real_frontier` as no independent stream dependency frontier exists.
- **Hardware Profiling:** Nsight profiling remains blocked by Windows permission `ERR_NVGPUCTRPERM`.
- **Opt-in FFN Candidate:** Multi-shape Gate/Up candidate `q4k_multi_shape_single_row` remains accepted opt-in only (`TITAN_F32_FFN_GATE_UP_VARIANT=q4k_multi_shape_single_row`).
- **QKV Projection Attribution & RCA:** Projection attribution verified QKV as the largest non-FFN projection group. The QKV multi-row batch 1 candidate passed 5/5 parity and dispatch smoke, but was decisively rejected as a regression in the five-model benchmark (`local-artifacts/reviews/phase18-qkv-multi-row-batch1-decision-20260908.json`). The subsequent paired diagnostic exited `101` with sequence divergence on the first fixture (`local-artifacts/benchmarks/real-f32-qkv-multi-row-batch1-paired-diagnostic-1788956770585-6596-0.json`).
- **Attention Bias Parity:** The default F32 path incorporates verified Q/K/V attention-bias handling for bias-bearing Qwen2 and DeepSeek fixtures, verified by the five-model F32 correctness matrix.

See [`docs/MVP.md`](MVP.md) and [`docs/WORKSPACE_STATE.md`](WORKSPACE_STATE.md) for live state details.

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
* **Chunked Prefill:** Long input sequences ($N \ge 2048$) are evaluated in discrete chunks of $C \le 512$ tokens via `prefill_chunked()`, keeping KV allocation bounded. Throughput depends on model, prompt, and current kernel path; see [`BENCHMARKS.md`](./BENCHMARKS.md) for reproduced figures.

---

### 2.4 Multi-Model GPU Speculative Decoding
Titan supports simultaneous loading of two distinct GGUF models into GPU VRAM:
* **Draft Model ($M_1$):** Lightweight model (e.g. *Llama 3.2 1B*, ~800 MB VRAM). Throughput is benchmark-dependent and is not stated here without a current reproduced run.
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
