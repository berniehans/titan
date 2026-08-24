# Change: VRAM audit + benchmarks seal (Phase 6.9)

## Why
Phase 6 delivers a real forward path over the streamed, resident-KV architecture. Before calling integration "done", the VRAM usage per stage must be accounted against the 5.2 GB budget, and the deferred Phase 4 benchmark row plus the new Phase 6 rows in `docs/BENCHMARKS.md` must carry real numbers. This is the accounting + sealing change that ends the phase's correctness story.

## What Changes
- **VRAM accounting** produced from the static allocation map + runtime traces, per stage:
  - ping-pong slot buffers (double-buffered layer staging),
  - KV pool growth per token (resident, paged blocks),
  - activations/cliff buffers per kernel,
  - logits transfer buffers (host↔device),
  - totals aggregated per stage and across the pipeline.
- **Budget assertion:** test that aggregated working set ≤ 5.2 GB under the fixture generation workload.
- **Benchmarks seal:** fill `docs/BENCHMARKS.md` Phase 4 deferred row (resident KV + paged attention) and Phase 6 rows (6.1 goldens, 6.3 GEMV parity, 6.4-6.5 parity, 6.6 single-layer, 6.7 cumulative drift, 6.8 throughput) with REAL measured numbers recorded during the phase.

## Gate
Total ≤ 5.2 GB asserted by test; BENCHMARKS updated with real numbers.

## Non-goals
- No new kernel work (audit + accounting only).
- No throughput tuning of the architecture; only measurement.

## Impact
- **Affected code:** `docs/BENCHMARKS.md`, VRAM accounting module + guard test
- **Gate:** working set ≤ 5.2 GB asserted; BENCHMARKS Phase 4 row + Phase 6 rows filled

## Tasks (summary — details in tasks.md)
1. Per-stage accounting map + assert working set ≤ 5.2 GB
2. Record real stage numbers from the generation workload
3. Fill `docs/BENCHMARKS.md` Phase 4 deferred + Phase 6 rows
4. Gate

## Environment notes
- NVRTC `%LOCALAPPDATA%/Temp` PATH trick; GPU tests `#[ignore]`. Accounting via runtime traces (no Nsight dependency).