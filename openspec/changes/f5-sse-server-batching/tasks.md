# Tasks: f5-sse-server-batching

> Execute via bot coder. Strict TDD. One commit per task group. ENVIRONMENT: NVRTC via %LOCALAPPDATA%/Temp on PATH; GPU/bench tests isolated.

## 1. Server skeleton (engine-server, CPU)
- [x] 1.1 Failing tests: POST /v1/completions returns OpenAI-compatible JSON shape (id, choices[0].text, usage); unknown route 404; malformed body 400
- [x] 1.2 Implement axum server bound to 127.0.0.1:0 (ephemeral port for tests), typed request/response models
- [x] 1.3 Gates green

## 2. GenerationSession + batching (CPU-first)
- [x] 2.1 Failing tests: single session produces deterministic next-token stubs from (dequantized layer digest, token id) using PagedKvCache append/read each step; multi-session batch advances all sessions without head-of-line blocking; finished session exits cleanly
- [x] 2.2 Implement GenerationSession + BatchScheduler (continuous batching semantics)
- [x] 2.3 Gates green

## 3. SSE streaming
- [x] 3.1 Failing tests: chunk framing `data: {json}\n\n`, first chunk role/content_shape, terminal `data: [DONE]\n\n`; client drop cancels session (KV blocks freed)
- [x] 3.2 Implement SSE via axum; wire to scheduler
- [x] 3.3 Gates green

## 4. E2E integration (full flow)
- [x] 4.1 #[ignore] test (fixture-dependent): download-fixture → load_to_pinned → Pipeline::with_dequantizer → server; two concurrent SSE clients receive interleaved but per-session-correct deterministic outputs; non-streaming request also correct
- [x] 4.2 Verify PASS locally (GPU + PATH trick)
- [x] 4.3 CI-safe variant: same E2E against synthetic in-memory layout (no fixture/GPU) runs in CI

## 5. Bench + gate
- [x] 5.1 KV append/read throughput measured through the real generation-loop path (median of isolated runs) → docs/BENCHMARKS.md Phase 4 row filled with REAL numbers
- [x] 5.2 Full gates green: fmt, clippy -D warnings, cargo test --workspace, GPU --ignored
- [x] 5.3 README status row + tasks.md checkboxes closed only after verification
