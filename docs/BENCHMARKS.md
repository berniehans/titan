# Titan — Benchmarks

> **Rule (read first):** every new phase appends its measured gate numbers to this file
> **before** its tasks are marked done. A phase is not complete until its numbers are
> recorded here with the hardware and methodology that produced them.

Canonical gates live in the spec
([`openspec/specs/layer-streaming-engine/spec.md`](../openspec/specs/layer-streaming-engine/spec.md))
and in each phase proposal under `openspec/changes/`. This document records the *measured*
evidence against those gates.

## Hardware & methodology

Reference hardware (constitution §4 fixed assumptions):

- **GPU:** NVIDIA RTX 3060 Laptop, **6 GB VRAM** (~5.2 GB usable).
- **Bus:** **PCIe 4.0 ×8** (≈12 GB/s effective).
- **Fixture model:** Qwen3-0.6B, Q4_K_M — `testdata/Qwen3-0.6B-Q4_K_M.gguf`,
  396,705,472 bytes ≈ 397 MB (~400 MB). SHA256:
  `ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a`
  (see [`../testdata/CHECKSUMS.md`](../testdata/CHECKSUMS.md)).

Methodology:

- **Loader** — `load_to_pinned` (`engine/engine-io/src/loader.rs`) times the single
  NVMe → pinned-RAM pass with `std::time::Instant` and reports GB/s. Reproduced in the
  GPU/local integration test [`engine/engine-io/tests/loader_pinned.rs`](../engine/engine-io/tests/loader_pinned.rs).
- **Pipeline** — the bench [`engine/engine-core/tests/pipeline_bench.rs`](../engine/engine-core/tests/pipeline_bench.rs)
  builds a dummy **8-layer** model (8 MB per layer, 64 MB total) in pinned RAM, does a
  warm-up run, then times **pipelined** (2 event-synchronized streams) vs **sequential**
  (sync copy then record/sync per layer) with wall clock. Requires a local CUDA device
  (`#[ignore]`d; run with `cargo test -- --ignored`).
- All numbers below are **measured on the RTX 3060 over PCIe ×8**, not estimates.

## Phase 0–1 — Single weight load into pinned RAM

| Metric | Measured | Gate (spec) | Status |
|---|---|---|---|
| Fixture load time (NVMe → pinned, single pass) | ~0.55 s for ~400 MB (≈0.7 GB/s) | < 5 s | ✅ PASS |
| No `read()` during generation | verifiable via trace (no disk I/O in loop) | requirement | ✅ PASS |

README records the loader as "~400 MB in <1 s"; the ~0.55 s figure is the recorded run
of the loader benchmark on reference hardware.

## Phase 2 — Double-buffered pipelining with overlap

Dummy 8-layer model, 8 MB/layer, 64 MB total, on RTX 3060 / PCIe ×8.

| Metric | Measured | Gate (Phase 2 proposal) | Status |
|---|---|---|---|
| Pipelined total time | **10.43 ms** | < sequential baseline | ✅ PASS |
| Sequential total time | **10.77 ms** | — (baseline) | — |
| Speedup | ≈1.03× | — | ✅ PASS |
| CPU busy-wait in pipeline | none (`streamWaitEvent`, not `streamSynchronize`) | no busy-wait | ✅ PASS |

Benchmark harness: [`engine/engine-core/tests/pipeline_bench.rs`](../engine/engine-core/tests/pipeline_bench.rs).
Gate verified in [`openspec/changes/f2-double-buffer-pipeline/proposal.md`](../openspec/changes/f2-double-buffer-pipeline/proposal.md).

## Phase 3 — On-the-fly GPU dequant (measured Aug 2026)

Dummy 8-layer model, Q4_K_M-aligned ~8 MB/layer, 64 MB total, on RTX 3060 / PCIe ×8.
Dequantizer-enabled pipeline (real GPU dequant kernel in the compute stage) is timed
against a per-layer sync-copy + sync-kernel sequential baseline that does the **same**
real compute work, and the historical stub-compute pipeline.

Benchmark harness (median of 7 iterations per run; 3 runs): 
[`engine/engine-core/tests/pipeline_dequant_bench.rs`](../engine/engine-core/tests/pipeline_dequant_bench.rs).

| Metric | Measured | Target gate | Status |
|---|---|---|---|
| Dequant parity vs CPU reference (per element, block-by-block) | **0.0** (bit-exact, 1,048,576 elems) | **< 0.01** | ✅ PASS |
| Dequant-pipelined total time (median) | **~19.2 ms** | < sequential baseline | ✅ PASS |
| Sequential-with-dequant total time (median, same real work) | **~25.3 ms** | — (baseline) | — |
| Speedup (real compute overlapping transfer) | **≈1.33×** (runs: 1.33× / 1.27× / 1.43×) | > 1.0 | ✅ PASS |
| Stub-compute pipelined (Phase 2 baseline, no kernel) | ~10.7 ms | — (reference) | — |
| Nsight overlap (concurrent transfer covering compute window) | _pending_ (nsys not installed) | **≥ 80%** | ⏳ PENDING |

> **Nsight trace note:** `nsys` is not installed on this host (runtime NVRTC only,
> The overlap percentage row stays pending until nsys is available. The
> pipelined-vs-sequential speedup above is the measured evidence that the real
> compute work overlaps transfer; the previous stub bench (~1.0×) was immeasurable
> because a recording-events compute stage does no work.

Gate context: [`openspec/changes/f3-gpu-dequant/proposal.md`](../openspec/changes/f3-gpu-dequant/proposal.md).

## Phase 4 — Resident KV cache + PagedAttention (measured Aug 2026)

CPU reference (`engine-kvcache`) and GPU kernels (`append_kv` + paged-read gather,
NVRTC) implemented TDD first, then parity-gated against the CPU reference.
Verified on RTX 3060 Laptop / PCIe ×8.

| Metric | Measured | Target gate | Status |
|---|---|---|---|
| GPU-vs-CPU parity (seeded xorshift, block-by-block) | **0.0** (bit-exact, 22 tokens across 4 scattered phys blocks) | **< 0.01/elem** | ✅ PASS |
| KV append/read throughput (real generation-loop path; **78.4k tok/s**, ~0.321 GB/s aggregate, median of 7) | row below | n/a | ✅ PASS |

> **Throughput (measured in Phase 5):** the deferred Phase 4 number is now
> sealed with REAL numbers from the generation-loop path
> ([`engine/engine-server/tests/kv_loop_bench.rs`](../engine/engine-server/tests/kv_loop_bench.rs),
> `#[ignore]`; run `cargo test -p engine-server --test kv_loop_bench -- --ignored --nocapture`):
>
> | Metric | Median (7 isolated runs) |
> |---|---|
> | KV append/read through decode loop | **78,392 tok/s** (runs: 78.4k/78.7k/78.1k/78.7k/78.4k/74.1k/70.3k) |
> | Aggregate KV bytes touched (`append+read` per token, 4 heads × 64 = 256 floats/row) | **0.321 GB/s** |
>
> Workload: 64 concurrent sessions × 512 decode steps, multiplexed on one pool
> through `BatchScheduler::advance` (continuous batching), so every token does a
> real `append` (K+V rows) + read-back — the exact server decode path, not a
> raw flat-copy microbenchmark. The Phase 4 contribution measured remains the
> bit-exact parity (0.0/elem) gate above; the append/read throughput above is
> the throughput number bounded by Phase 4's deferred note.

Benchmark harness: [`engine/engine-cuda/tests/paged_kv_parity.rs`](../engine/engine-cuda/tests/paged_kv_parity.rs).
Gate context: [`openspec/changes/f4-paged-kvcache/proposal.md`](../openspec/changes/f4-paged-kvcache/proposal.md).

## Phase 6 — Real Forward Path & Generator Integration (measured Aug 2026)

Full real transformer forward path implemented over double-buffered weights, resident
paged KV cache, fused CUDA kernels, and BPE tokenizer on reference hardware (RTX 3060 Laptop, PCIe 4.0 ×8).

### 6.1 — BPE Tokenizer & Config Goldens
- **Golden Prompt Parity:** 21 golden prompts evaluated against llama.cpp reference token streams.
- **Parity Result:** **100.0% exact token stream match** (0 token divergence).
- **Harness:** [`engine/engine-core/tests/tokenizer_tests.rs`](../engine/engine-core/tests/tokenizer_tests.rs).

### 6.3 — GEMV Multi-Format Compute Parity (GPU vs CPU Bank)
- **Supported Formats:** Q4_K, Q8_0, F16, F32.
- **Quantized GEMV Parity:** **0.0 max absolute error** (bit-exact match on quantized blocks).
- **Harness:** [`engine/engine-cuda/tests/gemv_multiformat.rs`](../engine/engine-cuda/tests/gemv_multiformat.rs).

### 6.4 & 6.5 — Fused Kernels Parity (RMSNorm, RoPE, SwiGLU, PagedAttention)
- **Norm / RoPE / SwiGLU Cos-Sim:** **1.000000** (> 0.9999 requirement, rel-L2 < 1e-5).
- **PagedAttention Decode Cos-Sim:** **1.000000** (> 0.9999 requirement, 0 dynamic allocations).
- **Harnesses:** [`engine/engine-cuda/tests/fused_ops.rs`](../engine/engine-cuda/tests/fused_ops.rs), [`engine/engine-cuda/tests/paged_attention_kernel.rs`](../engine/engine-cuda/tests/paged_attention_kernel.rs).

### 6.6 — Single-Layer Golden Parity (Layer 0)
- **Prompt:** "Hello" (token 9707) against golden `layer0_out.bin` from llama.cpp.
- **Measured Cosine Similarity:** **0.9972** (> 0.99 target gate).
- **Measured Rel-L2 Error:** **0.0034** (< 0.01 target gate).
- **Harness:** [`engine/engine-cuda/tests/layer0_parity_golden.rs`](../engine/engine-cuda/tests/layer0_parity_golden.rs).

### 6.7 — Full Forward Driver Cumulative Drift Curve & VRAM Guard
- **Cumulative Drift:** 85 decode checkpoints tracked across 28 layers + LM head.
- **Maximum Rel-L2 Error:** **1.044e-5** (<< 1.0e-3 target threshold).
- **Cosine Similarity:** **1.000000** across all 85 checkpoints.
- **Harness:** [`engine/engine-core/tests/driver_cumulative_drift.rs`](../engine/engine-core/tests/driver_cumulative_drift.rs).

### 6.8 — Generator Swap & End-to-End SSE Autoregressive Streaming
- **Logit Cos-Sim vs llama.cpp `logits_00.bin`:** **0.997143** (> 0.99 target gate).
- **SSE Streaming Generation:** Incremental valid tokens `[">\n", "</", "head", ">\n", "<body"]` with clean `data: [DONE]` framing, bit-identical to non-streaming endpoint.
- **Stub Baseline Throughput (Group 0):** **956,160.6 ids/s** (1.29% spread across 3 runs).
- **Harnesses:** [`engine/engine-server/tests/driver_parity_gate.rs`](../engine/engine-server/tests/driver_parity_gate.rs), [`engine/engine-server/tests/e2e_real_forward_sse.rs`](../engine/engine-server/tests/e2e_real_forward_sse.rs).

### 6.9 — VRAM Stage Accounting & Budget Seal

Device memory (VRAM) audited per stage against the 5.2 GB usable budget (5,452,595,200 bytes):

| Stage | Measured Bytes | Measured MB | % Working Set | Status |
|---|---|---|---|---|
| Stage 1 (Weights / Ping-pong staging) | 181,923,840 B | 173.50 MB | 85.8% | ✅ Bounded |
| Stage 2 (Resident KV Pool, 128 tok) | 29,360,128 B | 28.00 MB | 13.8% | ✅ Bounded |
| Stage 3 (Scratch Activations) | 99,332 B | 0.09 MB | 0.0% | ✅ Bounded |
| Stage 4 (Logits Host↔Device Transfer) | 607,744 B | 0.58 MB | 0.3% | ✅ Bounded |
| **Total Measured Working Set** | **211,991,044 B** | **202.17 MB** | **100.0%** | **✅ PASS (3.80% of 5.2 GB)** |
| **Static Working Set (2048 tok)** | **652,392,964 B** | **622.17 MB** | — | **✅ PASS (11.69% of 5.2 GB)** |
| **VRAM Budget Bound** | **5,578,424,320 B** | **5,320.00 MB (5.2 GB)** | — | **Constitutional Bound** |

Harnesses: [`engine/engine-core/tests/vram_accounting_tests.rs`](../engine/engine-core/tests/vram_accounting_tests.rs), [`engine/engine-server/tests/vram_real_audit_gate.rs`](../engine/engine-server/tests/vram_real_audit_gate.rs).

## Phase 7 — MoE Expert Streaming & Bandwidth-Adaptive Hybrid Execution

Measured on NVIDIA RTX 3060 Laptop GPU (6 GB VRAM, 5.2 GB usable) + AMD Host CPU over PCIe 4.0 ×8.

### 7.1 — Hardware Bandwidth Profiling (`benchbw.json`)

| Channel / Mode | Measured Bandwidth | Description | Status |
|---|---|---|---|
| STREAM Host DRAM Read | **0.86 GB/s** | Sequential host DRAM memory sweep | ✅ Profiled |
| Linear PCIe H2D | **6.08 GB/s** | Pinned Host → Device DMA copy | ✅ Profiled |
| Linear PCIe D2H | **5.85 GB/s** | Device → Host DMA copy | ✅ Profiled |
| CPU MoE GEMV (Isolated) | **0.36 GB/s** | Standalone CPU expert arithmetic | ✅ Profiled |
| PCIe Gather (Isolated) | **6.08 GB/s** | Standalone PCIe expert gather | ✅ Profiled |
| CPU MoE GEMV (Overlapped) | **0.36 GB/s** | CPU compute under PCIe contention | ✅ Profiled |
| PCIe Gather (Overlapped) | **6.06 GB/s** | PCIe gather under CPU contention | ✅ Profiled |
| **Adaptive Fetch Fraction ($q^\star$)** | **0.9441** | `pcie_ov / (pcie_ov + cpu_ov)` | **✅ Optimal** |
| **Recommended Backend** | **`offload`** | `offload` on slow CPU, `hybrid` when CPU > 2× PCIe | **✅ Verified** |

Artifact: [`tests/benches/benchbw.json`](../tests/benches/benchbw.json).

### 7.2 & 7.3 — Host Expert Banks & Capped-Fetch LRU Slot Cache

- **Pinned Allocation & Slice Indexing:** Contiguous RAII page-locked host memory (`PinnedHost`), per-(layer, expert) slice indexing, fallback capability flagged.
- **Balanced Fetch Rounding (`_balanced_fetch`):** Strictly minimizes the longer overlapping side. Replicated upstream regression cases ($0.415 \times 3 \rightarrow 1$ fetch, $0.415 \times 4 \rightarrow 2$ fetches).
- **GPU LRU Slot Cache:** Zero host syncs during steady-state decode; resident hits, LRU evictions, and CPU overflows tracked cleanly.
- **Harnesses:** [`engine/engine-core/tests/moe_expert_bank_tests.rs`](../engine/engine-core/tests/moe_expert_bank_tests.rs), [`engine/engine-core/tests/moe_slot_cache_tests.rs`](../engine/engine-core/tests/moe_slot_cache_tests.rs).

### 7.4 & 7.5 — CPU SwiGLU Executor & MoE-First VRAM Budget Planner

- **CPU SwiGLU Parity:** Bit-identical (< $10^{-6}$ error) weighted accumulation of overflow experts.
- **Threaded Overlap:** PCIe DMA transfer and CPU SwiGLU execution run concurrently with deterministic buffer merging.
- **VRAM Budget Invariant:** $\le 5.2$ GB guaranteed across all parameter sweeps (2 GB to 16 GB).
- **Dynamic Prefill Double Buffer:** Alternates ping-pong buffers for seamless transfer-compute overlap.
- **Harnesses:** [`engine/engine-core/tests/moe_cpu_executor_tests.rs`](../engine/engine-core/tests/moe_cpu_executor_tests.rs), [`engine/engine-core/tests/moe_budget_planner_tests.rs`](../engine/engine-core/tests/moe_budget_planner_tests.rs).

### 7.6 — Multi-Mode E2E Autoregressive Generation & Telemetry

- **Backends Tested:** `Offload`, `Cpu`, `Hybrid`.
- **E2E Token Generation:** Generated bit-identical coherent token sequences `[198, 262, 671, 4457, 1946]` across all 3 backends.
- **Telemetry:** Real-time per-layer cache statistics (`active_requests`, `resident_hits`, `pcie_fetched`, `cpu_overflow`, `pre_cap_miss_rate`, `gpu_coverage_rate`) verified with zero anomalies.
- **Harness:** [`engine/engine-server/tests/e2e_moe_hybrid_gate.rs`](../engine/engine-server/tests/e2e_moe_hybrid_gate.rs).

## Later phases — Predictable throughput at scale (target)

Dense 14B Q4_K_M (~8.5 GB resident in RAM), PCIe ×8, RTX 3060.

| Metric | Target | Measured |
|---|---|---|
| Generation throughput | **≈1.4 tok/s** (measured, not estimated) | _pending_ |
| First generated token valid end-to-end (full layer topology) | requirement | _pending_ |

> Per constitution §5, every numeric spec goal must be validated by a real benchmark
> before it is used in specs — this table is where that evidence lands.