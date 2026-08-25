# FreeToken (FlashML) — Reference Notes for Titan

Source: https://github.com/FlashML-org/FreeToken (Apache 2.0), paper arXiv:2608.16157.
Local clone for consultation: `workspace/_ref/FreeToken` (Python/Triton/CUDA, ~72k LOC).
Purpose: absorb design ideas (not code) for Titan Phase 6+ (MoE expert-streaming under VRAM constraint). All logic below is bandwidth arithmetic and scheduling policy — fully portable to Rust + NVRTC.

## 1. The three MoE offload backends (`python/freetoken/moe/__init__.py`)

All three serve experts from pinned host banks through one shared `OffloadMoeCache`; they differ only in how **decode** gets missing experts:

| Backend | Decode path |
|---|---|
| `offload` | Stream all missing experts over PCIe into a GPU slot cache; run GEMM on GPU. |
| `cpu` | Ship activations to CPU, compute the experts there (high RAM bandwidth), ship results back. |
| `hybrid` | Keep GPU slot cache AND CPU executor. Fetch at most K missing experts/layer over PCIe (GPU computes those + hits); remaining misses computed on CPU, overlapped; merge results. |

Titan relevance: on Bernie's RTX 3060 Laptop (PCIe 4.0 x8 ≈ 12 GB/s effective vs ~50–70 GB/s dual-channel DRAM), `hybrid` is likely the winning mode, not pure offload.

## 2. q★ / hybrid fetch fraction (the core idea)

Files: `moe/benchbw.py` (~measurement) and `moe/bench_profile.py` (~policy resolution).

**Offline profiling** (`ft bench bw`), per (workload, expert dtype):
1. Ceilings: STREAM-style host DRAM read; linear pinned↔device copy.
2. Real kernels: actual CPU MoE GEMV (`CpuMoeExecutor`) AND actual PCIe gather (`OffloadMoeCache.copy_missing` / fast_index_copy).
3. Crucially, both measured **overlapped** (running concurrently, contending for host DRAM).

**Policy resolution** (`load_hybrid_fetch_fraction`, bench_profile.py:104):
```
fetch_fraction = pcie_overlapped_GBps / (pcie_overlapped_GBps + cpu_overlapped_GBps)
```
Fallback without overlap data: `pcie/cpu` (full-DRAM-contention assumption). Clamped to [0,1], cached as JSON (`$XDG_CACHE_HOME/freetoken/benchbw.json`), keyed by GPU name.

**Backend choice rule**: recommend `hybrid` when CPU kernel bandwidth > 2× PCIe gather bandwidth (threshold configurable), else `offload`. Dtype-dominated decision → default is a per-dtype tuning bench against a canonical geometry.

## 3. Capped-fetch decode step (`offload_cache.py:800-900`)

Per decode step, per layer:
- Kernel rewrites `expert_ids` in place to slot ids: resident-hit → slot, fetched-this-step → slot, overflow miss → `-1` (route to CPU).
- Fetch count = balanced rounding of `fetch_fraction × misses` (`_balanced_fetch`, offload_cache.py:19-24): pick floor or ceil, whichever minimizes the LONGER overlapping side — NOT plain ceil (upstream test documents ceil over-fetching: fraction 0.415 × 3 misses → fetch 1, not 2, else PCIe runs ~1.6× slower than balance). When an explicit fixed cap is set it overrides; in auto mode the fraction is the sole mechanism.
- All device-side, fixed-shape → CUDA-graph compatible. In Titan terms: NVRTC-kernel-friendly, no host sync in the hot loop.
- Stats accumulated device-side: pre-cap misses vs PCIe-fetched vs active per layer → live miss-rate and split telemetry.

## 4. Memory budget planning (`engine/cache_budget.py`)

MoE-first greedy split of the net pool (budget = free VRAM − weights − fixed overhead):
- Reserve KV first (`kv_reserve_pages × cache_per_page`).
- Experts greedily fill remainder, clamped `[floor, min(total_experts, max_slots)]`; floor = 2×experts_per_layer if prefill overlap feasible else 1×.
- Prefill overlap borrows two full expert-layer double buffers → needs ≥ 2×experts slots, else disabled automatically.
- Hard assert that plan ≤ budget (fail in arithmetic, not OOM later).
- `(1 − memory_ratio)` remainder reserved for activations/graph headroom.

Titan note: this is *static* planning. Their runtime VRAM reallocation between expert cache and KV (no restart) is a separate advertised feature — worth studying later.

## 5. Prefill double-buffered streaming

- GPU holds only a two-buffer window of expert layers during prefill; load layer N+1 while computing layer N.
- Requires pinned layers; non-pinned (pageable/locked) layers take a whole-layer pageable branch in `copy_missing` and are incompatible with overlap.
- Budget coupling: overlap enabled only if slot cache ≥ 2×experts_per_layer (see §4).

## 6. Other transferable details

- **Global LRU expert cache**: per-(layer,expert) last-active decode step tracked device-side (`expert_recency`), reset per new sequence.
- **Semantic anchor checkpoints**: persisted KV/recurrent-state anchors so agentic context edits (tool calls, thinking blocks) avoid recomputing the unchanged prefix — radix cache + persistence. Relevant to Titan's SSE server if targeting agent workloads.
- **FTW fast weight format**: their streaming-friendly weight layout (analogous role to GGUF for llama.cpp; we already pinned llama.cpp cb1adf8 as reference).
- **Decode frequency histogram** (`decode_freq[layer]`) collected during normal routing — cheap input for future admission policies beyond LRU.

## 7. Suggested Titan follow-ups

1. Port the overlapped CPU-vs-PCIe micro-benchmark as a Rust `cargo bench`/bin (cuBLAS-free: our own GEMV kernels + cudaMemcpyDtoD-style gather).
2. Add `fetch_fraction` adaptive split to the Phase 6 scheduler design (hybrid mode).
3. Adopt the MoE-first budget planner with hard arithmetic asserts (fits our existing pool-budget enforcement).
4. Consider semantic anchor checkpoints as an F7+ server feature.

Paper: Yang et al., "FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution", arXiv:2608.16157 (2026).
