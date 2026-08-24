# Change: Fused norm/rope/swiglu kernel (`norm_rope.cu`) (Phase 6.4)

## Why
RNSNorm+residual, RoPE, and SwiGLU are the cheapest fused ops per layer but run for every token and every layer. On the layer-streaming stack (only ~0.9 GB ping-pong buffers, 5.2 GB VRAM budget) they must be fused into one kernel launch to avoid round-tripping intermediate buffers through device memory. This kernel ports the proven arithmetic from llama.cpp — not new math.

## What Changes
- `norm_rope.cu` (new `engine-cuda` module): fused RMSNorm + residual add + in-place RoPE (Qwen3 NeoX-style partial rotary) + SwiGLU gating in a single launch.
- RMSNorm epsilon placement per ggml.c; RoPE convention from llama.cpp rope.cu.
- Declared worst-case VRAM for this kernel (registers/shared + io buffers) stated in the proposal and guarded by a test: total footprint ≤ declared bound and within the 5.2 GB budget.
- Parity vs CPU twin: cos-sim ≥ 0.9999.

## Traceability gate (port of llama.cpp)
- Top-of-source comment REQUIRED in `norm_rope.cu`:
  `// Port of llama.cpp ggml_compute_forward_rms_norm + rope (ggml/src/ggml.c, ggml/src/ggml-cuda/rope.cu @ cb1adf8)`

## VRAM worst-case (declared)
- Declared here for the guard test: input+residual+output buffers on the ping-pong slot (streamed, reused) plus registers; no persistent allocations. The guard enforces: `alloc_norm_rope_total <= declared_worst_case` and `declared_worst_case + resident_kv + pingpong <= 5.2 GB`.
- (Final global accounting is sealed in 6.9.)

## Gate
Parity vs CPU twins cos-sim ≥ 0.9999 for RMSNorm, RoPE, SwiGLU individually and fused; declared VRAM worst-case asserted by test.

## Non-goals
- No GEMV/attention here (6.3 / 6.5 own those).
- No multi-layer correctness aggregation (6.6).

## Impact
- **Affected code:** `engine-cuda/src/`, `norm_rope.cu`, `engine-cuda/tests/`
- **Gate:** cos-sim ≥ 0.9999 vs CPU; VRAM worst-case declared + guarded

## Tasks (summary — details in tasks.md)
1. CPU twins (from 6.2 bank) + RED per op
2. Implement fused `norm_rope.cu` + RAII launcher
3. Parity tests + VRAM worst-case guard
4. Gate

## Environment notes
- NVRTC via `%LOCALAPPDATA%/Temp` PATH trick; GPU tests `#[ignore]`. Reference pinned `cb1adf8`.