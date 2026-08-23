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

## Later phases — Predictable throughput at scale (target)

Dense 14B Q4_K_M (~8.5 GB resident in RAM), PCIe ×8, RTX 3060.

| Metric | Target | Measured |
|---|---|---|
| Generation throughput | **≈1.4 tok/s** (measured, not estimated) | _pending_ |
| First generated token valid end-to-end (full layer topology) | requirement | _pending_ |

> Per constitution §5, every numeric spec goal must be validated by a real benchmark
> before it is used in specs — this table is where that evidence lands.