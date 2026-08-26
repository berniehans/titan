# Tasks: 6.8-stub-swap

> Execute via bot coder. Strict TDD. One commit per task group. GPU tests: `#[ignore]` + NVRTC PATH trick.

## 0. Pre-measured baseline (REQUIRED BEFORE swap)
- [x] 0.1 Measure stub-path throughput (ids/s) on the fixed prompt set via the benchmark harness (`engine-server/tests/stub_throughput_bench.rs`).
- [x] 0.2 Commit the measured baseline as an artifact (`tests/benches/stub_throughput_baseline.json`) with prompt, n_tokens, ids/s across 12 prompts (mean: 956,160.6 ids/s).
- [x] 0.3 Assert baseline recorded > 0 and stable across 3 runs (spread = 1.29% < 5%).

## 1. Sub-gate 1 — deterministic driver parity
- [x] 1.1 Failing test: driver teacher-forced logits vs golden (fixed prompt "Hello") cos-sim > 0.99 (`engine-server/tests/driver_parity_gate.rs`).
- [x] 1.2 Implement swap hook: generator reads from real driver when `real` op via `build_real_driver_model`, `stub` retained as compat fallback during rollout.
- [x] 1.3 Verify PASS — logit cos-sim = 0.997143 > 0.99 vs golden `logits_00.bin` (borderline-flip-tolerant top ranking validated).

## 2. Sub-gate 2 — SSE E2E autoregressive generation
- [x] 2.1 Failing test: SSE request → coherent autoregressive text from the streamed pipeline (`engine-server/tests/e2e_real_forward_sse.rs`).
- [x] 2.2 Wire real generator and `BpeTokenizer` text decoding into SSE handler and non-streaming handler in `server.rs`.
- [x] 2.3 E2E green: valid tokens streamed incrementally, clean `[DONE]` termination, streaming/non-streaming parity.

## 3. Sub-gate 3 — throughput vs baseline
- [ ] 3.1 Benchmark real path ids/s on same prompt set as baseline artifact
- [ ] 3.2 Failing test: real ids/s ≥ declared target (relative to pre-measured baseline)
- [ ] 3.3 Verify PASS; if below, optimize before swapping

## 4. Stub removal + Gate
- [ ] 4.1 Replace `stub_next_token` entirely (real driver is the generator; stub removed or aliased)
- [ ] 4.2 Full suite green (no reference to removed stub)
- [ ] 4.3 Gate sealed: logit cos-sim > 0.999 + SSE E2E green + throughput within target vs baseline