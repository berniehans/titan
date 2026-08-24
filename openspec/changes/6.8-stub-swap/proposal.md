# Change: Swap the stub (three sub-gates) (Phase 6.8)

## Why
The forward driver (6.7) is proven additive and drift-checked. This change promotes it to be THE generator: swap `stub_next_token` for the real streamed forward pass, behind a pre-measured throughput baseline, in three independently-verifiable sub-gates. The swap is gated on teacher-forced logit agreement with llama.cpp goldens — NOT a naive top-k token match, which a single borderline flip can collapse.

## What Changes
- Replace the deterministic placeholder path `stub_next_token` with the real driver (`engine-core/src/forward_driver.rs` from 6.7) as the generation source.
- Baseline artifact: throughput of the stub path (ids/s) measured BEFORE the swap, committed as the pre-defined baseline for sub-gate 3.
- Three sub-gates (each independently verified):
  1. **Deterministic driver parity** — for the fixed prompt, the driver's teacher-forced logits match goldens.
  2. **Autoregressive generation via SSE** — coherent text over the streamed pipeline end-to-end.
  3. **Throughput vs baseline** — ids/s on the real path ≥ declared target relative to the pre-measured stub baseline; regression caught by the baseline artifact.

## Gate
- Teacher-forced logit cos-sim > 0.999 vs llama.cpp goldens (goldens from 6.1; NOT raw top-k match).
- SSE E2E green (generated text coherent, streamed pipeline).
- Throughput within declared target vs the pre-measured baseline artifact.

## Non-goals
- No architectural rewrite of the streamed pipeline (identity is retained).
- No multi-sequence batching new work beyond SSE.
- No final VRAM accounting (6.9).

## Impact
- **Affected code:** `engine-core` generator entry, SSE handler swap, benchmark harness for the baseline artifact
- **Gate:** sub-gates 1+2+3 all pass; stub path removed/aliased after gate

## Tasks (summary — details in tasks.md)
1. Compute baseline: stub-path ids/s BEFORE swap committed
2. Sub-gate 1: deterministic driver parity (logit cos-sim vs goldens)
3. Sub-gate 2: SSE E2E autoregressive generation
4. Sub-gate 3: throughput vs baseline artifact
5. Gate: cos-sim + SSE + throughput

## Environment notes
- NVRTC `%LOCALAPPDATA%/Temp` PATH trick; GPU tests `#[ignore]`.
- Goldens from 6.1 (pinned llama.cpp `cb1adf8`); binaries only needed to regenerate, not at test time.