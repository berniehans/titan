# Change: SSE inference server + continuous batching (Phase 5)

## Why
Phases 1-4 built the machinery (loader, double-buffered pipeline, GPU dequant, paged KV cache) but nothing serves requests yet. Phase 5 exposes the engine as an OpenAI-compatible HTTP API with streaming (SSE) and introduces a decode loop with continuous batching over the paged KV cache. It is also the first end-to-end flow: HTTP request in → tokens out, exercising every layer underneath.

**Honesty constraint:** the model "forward pass" between embedding and output is still not real attention/matmul (that lands later); the server wires a *generation loop* whose per-layer work is: stream layer bytes → dequant kernel → (stub) combine into logits via a deterministic placeholder that depends on dequantized weights + input tokens. This keeps E2E tests deterministic and meaningful (same input → same output) while making no claims about model quality.

## What Changes
- New crate `engine-server`:
  - Axum/Tokio HTTP server, `POST /v1/chat/completions` and `POST /v1/completions`, OpenAI-compatible JSON; SSE streaming (`stream: true`) and non-streaming responses.
  - `GenerationSession`: holds a sequence in the paged KV cache, appends token KV per step.
  - Decode loop with **continuous batching**: multiple sessions multiplexed on the same pipeline; a session appends KV and computes its next-token stub independently; finished sessions leave the batch without stalling others.
  - Graceful shutdown, request cancellation (drop of SSE stream stops the session).
- Real KV throughput benchmark now possible: bench harness drives N sessions through append/read cycles (fills the deferred Phase 4 number).
- End-to-end integration test: full flow GGUF fixture → loader → pipeline → dequant → KV append/read → SSE chunks received by a real HTTP client, asserting deterministic output ordering and correct `data:` framing / `[DONE]` terminator.
- CI note: server unit tests run CPU-only in CI; E2E with fixture skips when fixture absent; GPU paths remain local-only.

## Non-goals
- No real attention kernel or trained-model quality (placeholder logits are documented as such).
- No TLS/auth beyond localhost binding.
- No tokenizer training; reuse metadata from fixture (vocab stub mapping token ids deterministically).

## Impact
- **Affected code:** new `engine/` crate `engine-server`; small additions to engine-kvcache if session API needs it
- **Gate:** E2E test green locally (fixture present): ≥2 concurrent SSE streams complete with interleaved-but-correct per-session outputs; KV throughput number recorded in docs/BENCHMARKS.md; all suites green

## Tasks (summary — details in tasks.md)
1. engine-server skeleton: axum routes, OpenAI-compatible request/response types — TDD (CPU)
2. GenerationSession + decode loop with continuous batching over PagedKvCache — TDD (CPU reference path)
3. SSE streaming framing — TDD (unit: chunk format, [DONE], cancellation)
4. E2E integration test (#[ignore] where GPU needed): full stack, 2+ concurrent streams, deterministic outputs
5. KV throughput benchmark (real numbers) + gate: full suites green, docs updated
