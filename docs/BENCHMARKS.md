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

## Phase 3 — On-the-fly GPU dequant (pending)

Placeholder rows — to be filled from the Phase 3 parity + overlap gate before Phase 3
tasks are marked done.

| Metric | Target gate | Measured |
|---|---|---|
| Nsight overlap (concurrent transfer covering compute window) | **≥ 80%** | _pending_ |
| Dequant parity vs CPU reference (per element, block-by-block) | **< 0.01** | _pending_ |
| Pipeline benchmark with real compute work (vs sequential) | total per-layer < sequential | _pending_ |

Gate context: [`openspec/changes/f3-gpu-dequant/proposal.md`](../openspec/changes/f3-gpu-dequant/proposal.md).

## Later phases — Predictable throughput at scale (target)

Dense 14B Q4_K_M (~8.5 GB resident in RAM), PCIe ×8, RTX 3060.

| Metric | Target | Measured |
|---|---|---|
| Generation throughput | **≈1.4 tok/s** (measured, not estimated) | _pending_ |
| First generated token valid end-to-end (full layer topology) | requirement | _pending_ |

> Per constitution §5, every numeric spec goal must be validated by a real benchmark
> before it is used in specs — this table is where that evidence lands.