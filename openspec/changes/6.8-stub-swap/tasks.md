# Tasks: 6.8-stub-swap

> Execute via bot coder. Strict TDD. One commit per task group. GPU tests: `#[ignore]` + NVRTC PATH trick.

## 0. Pre-measured baseline (REQUIRED BEFORE swap)
- [ ] 0.1 Measure stub-path throughput (ids/s) on the fixed prompt set via the benchmark harness
- [ ] 0.2 Commit the measured baseline as an artifact (e.g. `tests/benches/stub_throughput_baseline.json`) with prompt, n_tokens, ids/s
- [ ] 0.3 Assert baseline recorded > 0 and stable across 3 runs (spread < 5%)

## 1. Sub-gate 1 — deterministic driver parity
- [ ] 1.1 Failing test: driver teacher-forced logits vs golden (fixed prompt) cos-sim > 0.999
- [ ] 1.2 Implement swap hook: generator reads from real driver when `real` op, `stub` retained as compat fallback during rollout
- [ ] 1.3 Verify PASS — logit cos-sim > 0.999 vs goldens (NOT raw top-k; borderline-flip-tolerant)

## 2. Sub-gate 2 — SSE E2E autoregressive generation
- [ ] 2.1 Failing test: SSE request → coherent autoregressive text from the streamed pipeline (golden-anchored prompt)
- [ ] 2.2 Wire real generator into SSE handler
- [ ] 2.3 E2E green: valid tokens, stream closes cleanly, n>1 tokens

## 3. Sub-gate 3 — throughput vs baseline
- [ ] 3.1 Benchmark real path ids/s on same prompt set as baseline artifact
- [ ] 3.2 Failing test: real ids/s ≥ declared target (relative to pre-measured baseline)
- [ ] 3.3 Verify PASS; if below, optimize before swapping

## 4. Stub removal + Gate
- [ ] 4.1 Replace `stub_next_token` entirely (real driver is the generator; stub removed or aliased)
- [ ] 4.2 Full suite green (no reference to removed stub)
- [ ] 4.3 Gate sealed: logit cos-sim > 0.999 + SSE E2E green + throughput within target vs baseline