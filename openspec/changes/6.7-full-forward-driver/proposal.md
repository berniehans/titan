# Change: Full forward driver (additive beside stub) (Phase 6.7)

## Why
With single-layer parity proven (6.6), the forward pass must run the FULL stack over the streamed pipeline — but WITHOUT touching `stub_next_token` yet. This change adds prefill and single-token decode as separate entry points beside the stub, proving end-to-end correctness and VRAM guardrails before the swap (6.8). Additive = the stub path stays byte-identical and all existing suites stay green.

## What Changes
- Prefill entry point: run all layers over the streamed pipeline on the full prompt; single-token decode entry point: run one layer-topology step given the resident KV cache.
- Full stack: env streamed per layer (ping-pong buffers), MultiFormatGEMV (6.3), fused norm/rope/swiglu (6.4), PagedAttention + paged KV append (6.5), multi-layer residual + logits out.
- ADDITIVE: `stub_next_token` untouched — swap happens only in 6.8.
- Hard per-kernel VRAM guards: each kernel launch checks its declared worst case ≤ budget; total resident (ping-pong + KV pool + activations + logits buffer) stays ≤ 5.2 GB.
- Cumulative drift checkpoint: teacher-forced ≥10 tokens; cumulative logits drift vs goldens stays within tolerance at every step.

## Gate
All existing suites green (stub untouched); cumulative logits drift within tolerance vs goldens over ≥10 teacher-forced tokens; VRAM ≤ budget at every step (per-kernel assertions).

## Non-goals
- NO stub replacement (6.8 owns the swap).
- No SSE/generation serving yet (6.8).
- No final VRAM accounting tables (6.9).

## Impact
- **Affected code:** `engine-core/src/forward_driver.rs`, `engine-cuda/` kernel wiring, `Titan::run_prefill` / `Titan::run_decode`
- **Gate:** suites green + cumulative-drift + per-kernel VRAM guards

## Tasks (summary — details in tasks.md)
1. Prefill entry point over streamed pipeline
2. Single-token decode entry point + logits
3. Cumulative drift checkpoint (≥10 teacher-forced tokens)
4. Per-kernel VRAM guards + budget assertion, additive-safety tests
5. Gate

## Environment notes
- NVRTC `%LOCALAPPDATA%/Temp` PATH trick; GPU tests `#[ignore]`. Goldens from 6.1.