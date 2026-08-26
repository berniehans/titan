# Change: Resident KV cache + PagedAttention (Phase 4)

## Why
Generation needs attention over past tokens. With ~6 GB VRAM (5.2 GB usable) and layer weights streaming through only ~0.9 GB of ping-pong buffers, the KV cache must be a *resident* VRAM structure with a strict budget (~3 GB), allocated once and paged to avoid fragmentation as sequences grow and batch later arrives (Phase 5).

## What Changes
- `engine-kvcache`: paged KV-cache design in the spirit of vLLM's PagedAttention:
  - Fixed-size **blocks** (e.g. 16 tokens x n_kv_heads x head_dim per K and V) allocated from a device-side block pool at startup; no per-token cudaMalloc ever.
  - **Block table** indirection: sequence -> list of block indices; logical contiguous positions map to scattered physical blocks.
  - Free-block allocator with O(1) alloc/free, plus accounting against the VRAM budget.
- CPU reference implementation first (TDD): pool + block table + append/read semantics, unit-tested without GPU.
- CUDA kernels for paged attention primitives: `append_kv` (write new K/V into physical block given block table) and a gather-style read kernel that materializes a contiguous [n_tokens, head_dim] view for the (future) attention kernel — parity-gated vs CPU reference (<0.01/elem).
- Budget guard: requesting beyond budget returns a typed error (`KvCacheFull`-style); test eviction-free behavior is out of scope (no eviction in Phase 4).

## Non-goals
- No real attention/flash kernel yet (needs matmul infra, Phase 5+).
- No eviction/swapping to host RAM.
- No batching/multiple sequences concurrency beyond the block-table abstraction supporting it.

## Impact
- **Affected code:** `engine-kvcache/src/*` (CPU reference + config), new GPU kernels via NVRTC pattern from engine-cuda
- **Gate:** CPU unit tests green; GPU parity vs CPU < 0.01/elem; budget enforcement test; all suites green

## Tasks (summary — details in tasks.md)
1. Config + CPU reference: block pool, block table, append/read — TDD
2. GPU kernels (NVRTC, same pattern as dequant_q4k.cu): append_kv + paged read — TDD with #[ignore] tests
3. Parity gate GPU vs CPU on randomized deterministic data
4. Budget enforcement tests (alloc until full → typed error)
5. Gate: full suite green + docs/BENCHMARKS.md updated if measurable numbers exist
