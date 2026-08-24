# Change: Single-layer parity gate (Phase 6.6)

## Why
At this point every kernel is individually parity-gated (6.3 GEMV, 6.4 norm/rope/swiglu, 6.5 paged attention) and the CPU forward bank (6.2) is cross-validated against golden L0 (6.1). The next correct move is to wire ONE complete transformer block and debug compound drift THERE, where it is cheap — not after the full forward path exists (6.7).

## What Changes
- Wire ONE complete transformer block over the streamed pipeline: norm → QKV GEMV + bias → RoPE → paged KV append + attention → out GEMV → residual → norm → SwiGLU → down GEMV → residual.
- The block reads the real fixture weight tensors through `engine-io` and reuses kernels: `MultiFormatGEMV` (6.3), `norm_rope.cu` (6.4), PagedAttention (6.5).
- Reference for parity: llama.cpp golden layer-0 activation (from 6.1) — NOT a self-referential CPU twin.
- Debugging harness: per-op stage outputs (norm/QKV/RoPE/attn/out/SwiGLU/down) dumped for bisection when the compound gate fails.

## Gate
Cos-sim > 0.999 AND relative-L2 < 1e-3 vs golden L0 activation.

## Non-goals
- No full multi-layer forward yet (6.7 owns multi-layer).
- No throughput/VRAM budget work here (6.7/6.9 own those).

## Impact
- **Affected code:** `engine-core/src/`, `engine-cuda/`, wired layer path `Titan::run_block`
- **Gate:** cos-sim > 0.999, rel-L2 < 1e-3 vs golden L0

## Tasks (summary — details in tasks.md)
1. RED: single-layer wiring test vs golden L0
2. GREEN: wire the block (GEMV→RoPE→paged→out→residual→SwiGLU→down→residual)
3. Bisect-compound-drift via stage traces, then parity
4. Gate: golden L0 cos-sim + rel-L2

## Environment notes
- NVRTC `%LOCALAPPDATA%/Temp` PATH trick; GPU tests `#[ignore]`. Goldens from 6.1 (no llama.cpp at test time).