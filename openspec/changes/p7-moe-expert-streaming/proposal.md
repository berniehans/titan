# Change: Phase 7 — MoE expert-streaming under VRAM constraint (master plan)

## Why
Titan's dense layer-streaming is bandwidth-bound and non-interactive on PCIe 4.0 x8 (~12 GB/s effective) for models ≥14B. The high-value path for Bernie's RTX 3060 Laptop (~5.2 GB usable VRAM) is **MoE expert-streaming**: models like Qwen3-30B-A3B / gpt-oss-20b activate ~3B params/token, so only the routed experts need to move. This phase absorbs the proven scheduling patterns from **FreeToken** (FlashML, Apache 2.0, arXiv:2608.16157) as *design references* — no upstream code is ported; all logic is bandwidth arithmetic and policy, portable to Rust + NVRTC.

Reference notes: `docs/freetoken-reference.md` (detailed extraction with file/line pointers into the local clone at `%LOCALAPPDATA%/hermes/workspace/_ref/FreeToken`).

## Design decisions absorbed from FreeToken
1. **Hybrid decode mode (default target)**: GPU slot cache + CPU executor. Per decode step, fetch `balanced_round(fetch_fraction × misses)` missing experts over PCIe (floor-or-ceil chosen to minimize the longer overlapping side — NOT plain ceil, which upstream proved over-fetches; computed on GPU together with hits), the overflow misses compute on CPU overlapped, then merge. On this hardware (DRAM ~50–70 GB/s vs PCIe ~12 GB/s), hybrid beats pure offload.
2. **Adaptive q★ fetch fraction**: offline micro-benchmark measures real CPU MoE GEMV bandwidth AND real PCIe gather bandwidth **overlapped** (contending DRAM); `fetch_fraction = pcie_ov / (pcie_ov + cpu_ov)`, cached as JSON keyed by GPU name. No hardcoded guesses.
3. **Backend choice rule**: recommend hybrid when measured CPU kernel bandwidth > 2× PCIe gather bandwidth, else pure offload. Decision is data-driven per dtype/model, decided at engine startup.
4. **Capped-fetch LRU slot cache, host-sync-free**: kernel rewrites routing ids in place to slot ids (hit → slot, fetched → slot, overflow → `-1` route-to-CPU). Fixed shapes, device-side stats → CUDA-graph/NVRTC-friendly, no host sync in the hot loop.
5. **MoE-first static budget planner**: net pool = free VRAM − weights − fixed overhead; reserve KV pages first, experts fill remainder clamped `[floor, min(total_experts, max_slots)]`; floor = 2×experts_per_layer when prefill overlap feasible else 1×; hard arithmetic assert that plan ≤ budget (fail at planning, not OOM at runtime). Prefill double-buffer borrows two full expert-layer buffers.
6. **Prefill streaming with double buffering**: load expert layer N+1 while computing layer N; requires pinned host banks; auto-disabled if slot cache < 2×experts_per_layer.
7. **Telemetry**: device-side per-layer counters (pre-cap misses vs PCIe-fetched vs active) → live miss-rate and split stats surfaced by the SSE server.

Deferred (noted, not scheduled): runtime VRAM rebalancing between expert cache and KV without restart; semantic anchor checkpoints for agentic context edits (candidate F8+ server feature); FTW-style streaming weight format (we keep GGUF + llama.cpp cb1adf8 pin).

## Porting policy (inherits master plan)
Same as Phase 6: kernels are ports of proven sources with traceability comments and pinned commits. For Phase 7 the *policy* references are FreeToken files (`python/freetoken/moe/offload_cache.py`, `benchbw.py`, `bench_profile.py`, `engine/cache_budget.py`) — cite them in proposals; the *kernels* still port from llama.cpp/vLLM where arithmetic is explicit. CPU references written from formulas, never transliterated back from our own CUDA.

## Sub-changes
### 7.1 — Bandwidth profiler (`ft-bw` equivalent)
Rust bin/bench measuring: STREAM-style host DRAM read; linear pinned↔device copy; real CPU MoE GEMV (our 6.3 GEMV arithmetic on CPU); real PCIe gather of expert-sized blocks — each alone AND overlapped. Emits `benchbw.json` keyed by GPU name.
**Gate:** reproducible numbers within noise on repeat runs; JSON schema versioned; CI-safe (skips gracefully without CUDA).

### 7.2 — Host expert banks + pinned allocator
Pinned host memory bank layout for expert weights streamed from GGUF (per-(layer,expert) slices), RAII pinned allocator, page-lock fallback to pageable with capability flag.
**Gate:** round-trip copy correctness; no leaks (miri/ASAN-clean where applicable); pageable fallback flagged.

### 7.3 — GPU expert slot cache + capped-fetch LRU
Device slot cache sized by budget planner; NVRTC kernel assigning slots and rewriting routing ids (hit/fetched/overflow→-1), global per-expert recency, reset per sequence; device-side stat accumulators.
**Gate:** parity of rewritten routing vs CPU model across adversarial access patterns; zero host syncs in decode loop (asserted by test).

### 7.4 — CPU executor path
CPU GEMV over host banks for overflow misses (reuses 6.2 bank arithmetic), overlapped with PCIe fetch via threads; merge into GPU output buffers.
**Gate:** bit-exactness of merged output vs sequential execution on synthetic routing; overlap actually overlaps (timing assertion).

### 7.5 — Budget planner + prefill double buffer
Port of the MoE-first split (reserve KV, greedy experts, floors, hard asserts) integrated with Titan's existing pool-budget enforcement from F4; two-buffer prefill pipeline with auto-disable below 2×experts slots.
**Gate:** planner rejects impossible budgets in arithmetic; VRAM ≤ 5.2 GB asserted end-to-end; overlap disabled correctly at small budgets.

### 7.6 — Hybrid scheduler integration + E2E
Wire offload/cpu/hybrid modes behind config; startup backend choice rule using 7.1 profile; SSE server surfaces miss-rate/fetch-split telemetry; E2E generation on Qwen3-30B-A3B-class fixture (or largest MoE GGUF that fits the streaming constraint).
**Gate:** all three modes green E2E; coherent autoregressive text via SSE; throughput targets declared vs measured; telemetry numbers consistent with mode.

### 7.7 — Benchmarks seal
BENCHMARKS.md rows: per-mode tokens/s, miss-rate vs cache size sweep, fetch fraction sensitivity, VRAM accounting per stage.
**Gate:** docs updated with real measured numbers; regression baselines recorded.

## Dependency order
7.1 ∥ 7.2 → 7.3 → 7.4 → 7.5 → 7.6 → 7.7

## Top risks & mitigations
1. **PCIe contention mis-modeled** — mitigated by measuring overlapped, not isolated (FreeToken's core lesson).
2. **Host-sync stalls in decode loop** — mitigated by fixed-shape device-side slot assignment (7.3 gate asserts zero syncs).
3. **Budget overrun with three consumers (slots + KV + double buffer)** — mitigated by single planner owning the whole pool with hard asserts (7.5).
4. **CPU executor quality** — reuses validated 6.2/6.3 arithmetic; never invents new numerics. Note: MoE GEMV is routed/grouped, not dense — 7.4 must add an explicit parity test for the per-(layer,expert) slice view path (GGUF stores experts as per-expert tensors, so banks slice naturally).
5. **CPU-executor host memory pressure** — the CPU path itself needs the host banks resident while PCIe fetch also streams from them (pinned lock pressure). Mitigation: bank descriptors shared between 7.2/7.3/7.4 (single allocation owner), and a fallback that reduces concurrent fetch width under pin failure.
