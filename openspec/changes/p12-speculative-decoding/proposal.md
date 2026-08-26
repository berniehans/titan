# Proposal: Phase 12 — Speculative Decoding Engine (Draft Verification & Multi-Token Speculation)

## 1. Summary

Implement a **Speculative Decoding Engine** that proposes $K \in [2, 5]$ candidate tokens via an ultra-low-latency n-gram / draft mechanism and verifies them simultaneously in a single batched GPU forward pass ($M = K$) through `ForwardDriver` using Batched GEMM and FlashAttention-2. Delivers a **$2\times$ to $3\times$ acceleration in generation throughput (tok/s)** with **100% mathematical output distribution equivalence**.

---

## 2. Motivation

- **Arithmetic Underutilization in Single-Token Decode:** During single-token autoregressive decoding ($M = 1$), GPU memory bandwidth limits throughput. The compute cores are underutilized because matrix weights are streamed to multiply a single token vector.
- **Batched Verification Synergy with Phase 11:** Phase 11 established Batched Quantized GEMM and FlashAttention-2, enabling the model to evaluate $M = K$ tokens in almost the exact same execution time as $M = 1$.
- **Exact Parity & Zero Quality Degradation:** Unlike approximate methods, speculative decoding uses exact target model verification with rejection sampling, guaranteeing that the generated text is bit-for-bit identical to standard greedy/sampled autoregression.

---

## 3. Scope & Sub-Changes

1. **Sub-change 12.1 — Speculative Verification & Rejection Sampling Algorithm (`engine-core`):**
   - Implement `rejection_sample_multi_token` supporting greedy prefix matching and temperature-scaled probabilistic rejection sampling.
   - TDD unit tests verifying exact token acceptance rules and multi-token bonus emission.

2. **Sub-change 12.2 — Context N-Gram & Self-Draft Proposer (`engine-core`):**
   - Implement `NgramDraftProposer` that indexes sequence history to propose recurring syntactic patterns, keywords, and code structures with $< 0.1\text{ ms}$ overhead and zero additional VRAM footprint.

3. **Sub-change 12.3 — Batched Candidate Verification in ForwardDriver (`engine-core` / `engine-cuda`):**
   - Implement `ForwardDriver::verify_speculative_candidates(&mut self, candidates: &[u32])`.
   - Executes single forward pass over $K$ candidate tokens, commits accepted KV cache blocks to `PagedKvCache`, and discards rejected tokens.
   - Parity gate: Assert 100% token sequence identity against standard autoregressive decode.

4. **Sub-change 12.4 — Speculative Streaming Integration & Phase 12 Seal (`engine-server` / `docs`):**
   - Hook speculative engine into `POST /v1/chat/completions` and CLI `titan chat`.
   - Benchmark generation speedup (tok/s) and acceptance rate ($\alpha \ge 60\%$) across domains (Code, Structured Data, QA).
   - Record metrics in `docs/BENCHMARKS.md`, sync delta spec, and archive change.
