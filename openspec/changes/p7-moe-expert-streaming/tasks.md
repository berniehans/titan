# Tasks: p7-moe-expert-streaming (master)

> Execute via bot coder. Strict TDD. One commit per task group. GPU tests `#[ignore]` + `%LOCALAPPDATA%/Temp` PATH trick for NVRTC. Policy reference: FreeToken local clone `%LOCALAPPDATA%/hermes/workspace/_ref/FreeToken`; detailed notes `docs/freetoken-reference.md`. NO code changes until each sub-change has its own proposal/tasks approved.

## 1. Bandwidth profiler (sub-change 7.1)
- [x] 1.1 Schema + writer for `benchbw.json` (GPU-name keyed, versioned) in `engine-core/src/moe/profile.rs`.
- [x] 1.2 STREAM host-read measurement (`measure_stream_host_dram_read`: 0.86 GB/s).
- [x] 1.3 Linear pinned↔device copy measurement (`measure_linear_pcie_h2d`: 6.08 GB/s, `measure_linear_pcie_d2h`: 5.85 GB/s).
- [x] 1.4 Overlapped CPU-GEMV + PCIe-gather measurement (contended DRAM: CPU MoE ov = 0.36 GB/s, PCIe gather ov = 6.06 GB/s).
- [x] 1.5 fetch_fraction resolver: `pcie_ov / (pcie_ov + cpu_ov)` = 0.9441, clamp [0,1], backend recommendation (`offload` on slow CPU, `hybrid` when CPU > 2x PCIe).
- Gate PASS: repeat-run variance within tolerance, CI-safe graceful skip without CUDA, profile saved to `tests/benches/benchbw.json`.

## 2. Host expert banks (sub-change 7.2)
> NOTE (coder advisory): 7.2 banks and 7.3 GPU slot cache must draw from shared bank descriptors — single allocation owner to avoid double-allocating against F2's existing pinned prefill buffers.
- [x] 2.1 Pinned allocator RAII (`PinnedHost`) + fallback-to-pageable flag (`is_pinned`) in `engine-core/src/moe/expert_bank.rs`.
- [x] 2.2 Per-(layer,expert) slice view over bank memory (`expert_slice`, `expert_slice_mut`).
- [x] 2.3 Stream GGUF expert tensors into banks (`write_expert_tensor`, `get_expert_tensor`).
- Gate PASS: copy round-trip bit-identical across multi-tensor slices, leak-free RAII drop, fallback capability flagged.

## 3. GPU slot cache + capped-fetch LRU (sub-change 7.3)
- [x] 3.1 RED: routing rewrite model test on synthetic access patterns (`engine-core/tests/moe_slot_cache_tests.rs`).
- [x] 3.2 Slot assignment & ID rewrite (resident hit -> slot, fetched -> slot, overflow -> -1), recency tracking in `engine-core/src/moe/slot_cache.rs`.
- [x] 3.3 Stat accumulators (`active_requests`, `resident_hits`, `pcie_fetched`, `cpu_overflow`, `pre_cap_miss_rate`, `gpu_coverage_rate`).
- [x] 3.4 Balanced-rounding fetch count (`balanced_fetch`): floor-or-ceil minimizing the longer overlapping side; verified upstream regression cases (0.415 × 3 misses → fetch 1, 0.415 × 4 misses → fetch 2).
- Gate PASS: zero host-syncs in decode loop asserted, parity vs CPU model verified across access patterns.

## 4. CPU executor + overlap (sub-change 7.4)
- [ ] 4.1 CPU GEMV over host banks reusing 6.2/6.3 arithmetic
- [ ] 4.2 Threaded overlap: PCIe fetch ∥ CPU compute, merge buffers
- Gate: merged == sequential bitwise on synthetic routing; timing proves overlap

## 5. Budget planner + prefill double buffer (sub-change 7.5)
- [ ] 5.1 MoE-first planner (KV reserve → greedy experts → floors → hard assert ≤ budget), integrated with F4 pool enforcement
- [ ] 5.2 Two-buffer prefill pipeline; auto-disable when slots < 2×experts_per_layer
- Gate: planner rejects impossible budgets arithmetically; E2E VRAM ≤ 5.2 GB asserted

## 6. Hybrid scheduler + E2E (sub-change 7.6)
- [ ] 6.1 Config surface: moe_backend ∈ {offload, cpu, hybrid}; startup choice rule from 7.1 profile (>2× ⇒ hybrid)
- [ ] 6.2 Wire modes into forward driver beside 6.7/6.8 stack (additive)
- [ ] 6.3 SSE telemetry: miss-rate, fetch/cpu split per layer
- [ ] 6.4 E2E generation on largest fitting MoE GGUF fixture
- Gate: 3 modes E2E green; coherent SSE text; telemetry consistent; PLUS hard per-layer miss-rate upper bound on the fixture (not just mean) so trivial top-k sparsity can't pass silently

## 7. Benchmarks seal (sub-change 7.7)
- [ ] 7.1 BENCHMARKS.md: per-mode tok/s, miss-rate vs cache-size sweep, fetch-fraction sensitivity, VRAM per stage
- Gate: real measured numbers only; regression baselines recorded
