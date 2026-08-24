# Tasks: 6.7-full-forward-driver

> Execute via bot coder. Strict TDD. One commit per task group. GPU tests: `#[ignore]` + NVRTC PATH trick.

## 1. Prefill entry point (engine-core/src/forward_driver.rs)
- [ ] 1.1 Failing test: prefill over streamed pipeline produces per-layer outputs matching expected (first-layer state from 6.6)
- [ ] 1.2 Implement `run_prefill` (all layers on prompt, KV appended into resident cache)
- [ ] 1.3 Failing test: full-prompt prefill logits vs golden (tolerance)
- [ ] 1.4 Verify PASS

## 2. Single-token decode entry point
- [ ] 2.1 Failing test: `run_decode` (one token) reuses resident KV and emits next-token logits
- [ ] 2.2 Implement decode: single-topology step (GEMV+attn+norm+swiglu+down+logits)
- [ ] 2.3 Failing test: ≥10 teacher-forced tokens cumulative logits drift within tolerance vs golden (checkpoint at each token)
- [ ] 2.4 Record drift curve; verify PASS

## 3. VRAM guards + additive safety
- [ ] 3.1 Failing test: per-kernel declared worst-case asserted ≤ budget (resident pingpong + kv_pool + activations + logits ≤ 5.2 GB)
- [ ] 3.2 Failing test: `stub_next_token` path byte-identical result (additive safety — stub untouched)
- [ ] 3.3 Verify PASS (budget trace logged)

## 4. Gate
- [ ] 4.1 Full suite green (existing + new; stub untouched)
- [ ] 4.2 Gate sealed: cumulative drift ≤ tolerance over ≥10 tokens, VRAM ≤ 5.2 GB every step