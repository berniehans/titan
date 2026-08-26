# Implementation Tasks: Phase 12 — Speculative Decoding Engine (Draft Verification & Multi-Token Speculation)

## 1. Speculative Verification & Rejection Sampling (sub-change 12.1)
- [x] 1.1 Implement `SpeculativeVerifier` and `rejection_sample_multi_token` in `engine-core/src/speculative.rs`.
- [x] 1.2 Implement deterministic greedy verification matching argmax prefixes with bonus token sampling.
- [x] 1.3 Create unit test `engine-core/tests/speculative_sampling_test.rs` verifying mathematical distribution equivalence and exact greedy acceptance logic.
- Gate PASS: `cargo test -p engine-core --test speculative_sampling_test` PASS.

## 2. Context N-Gram Draft Proposer (sub-change 12.2)
- [ ] 2.1 Implement `NgramDraftProposer` in `engine-core/src/ngram_draft.rs` with dynamic context indexing and prefix matching ($n \in [2, 5]$).
- [ ] 2.2 Create unit test `engine-core/tests/ngram_draft_test.rs` asserting correct candidate proposals on repetitive prompt patterns and code syntax.
- Gate: `cargo test -p engine-core --test ngram_draft_test` PASS.

## 3. Batched Candidate Verification in ForwardDriver (sub-change 12.3)
- [ ] 3.1 Implement `ForwardDriver::verify_speculative_candidates(&mut self, candidates: &[u32]) -> Result<Vec<u32>, EngineError>` using batched GEMM and FlashAttention-2.
- [ ] 3.2 Implement KV cache rollback committing only accepted tokens.
- [ ] 3.3 Create parity test `engine-core/tests/speculative_driver_parity.rs` asserting 100% token sequence identity against serial autoregressive generation.
- Gate: `cargo test -p engine-core --test speculative_driver_parity` PASS.

## 4. Speculative Streaming Integration & Phase 12 Seal (sub-change 12.4)
- [ ] 4.1 Integrate speculative decoding into `POST /v1/chat/completions` and CLI `titan chat`.
- [ ] 4.2 Benchmark generation speedup (tok/s) and acceptance rate in `engine-server/tests/speculative_benchmark_gate.rs`.
- [ ] 4.3 Record measured speedups and acceptance rates in `docs/BENCHMARKS.md`.
- [ ] 4.4 Verify full workspace test suite `cargo test --workspace` with 0 regressions.
- [ ] 4.5 Sync delta spec to main spec and archive change.
- Gate: Speculative decoding verified, speedup recorded, tests green, Phase 12 sealed.
