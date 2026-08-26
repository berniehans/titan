# Implementation Tasks: Phase 10 — OpenAI Chat Completions Server, Streaming SSE & Interactive CLI

## 1. OpenAI Chat Models & Wire Protocol (sub-change 10.1)
- [x] 1.1 Define OpenAI Chat wire models in `engine-server/src/models.rs` (`ChatMessage`, `ChatCompletionRequest`, `ChatCompletionResponse`, `ChatCompletionChunk`, `ChatChoice`, `DeltaMessage`).
- [x] 1.2 Implement ChatML template formatting for Qwen chat messages with `<|im_start|>` and `<|im_end|>` delimiters.
- [x] 1.3 Create unit test verifying ChatML prompt formatting and JSON serialization.
- Gate PASS: `cargo test -p engine-server --lib models` PASS.

## 2. Advanced Production Sampler & Stop Handling (sub-change 10.2)
- [ ] 2.1 Implement `Sampler` in `engine-core` supporting greedy argmax, temperature scaling, top-$k$, top-$p$ (nucleus), repetition penalty, and stop token detection.
- [ ] 2.2 Create unit tests verifying deterministic greedy fallback when temperature=0, top-p filtering, and stop sequence trimming.
- Gate: `cargo test -p engine-core --lib sampler` PASS.

## 3. Server Real-Engine Integration & SSE Streaming (sub-change 10.3)
- [ ] 3.1 Implement background GPU `EngineService` actor running `ForwardDriver` with CUDA Graph decode and continuous token generation.
- [ ] 3.2 Implement `POST /v1/chat/completions` route supporting both non-streaming JSON and streaming Server-Sent Events (`text/event-stream`).
- [ ] 3.3 Implement `GET /v1/models` endpoint.
- [ ] 3.4 Create integration test `engine-server/tests/e2e_chat_completions.rs` testing HTTP requests (streaming & non-streaming) on real fixture.
- Gate: `cargo test -p engine-server --test e2e_chat_completions` PASS.

## 4. Interactive Terminal CLI & Phase 10 Seal (sub-change 10.4)
- [ ] 4.1 Implement `titan chat` interactive REPL in `engine-server` with multi-turn conversation memory and live stdout token streaming.
- [ ] 4.2 Implement `titan serve` CLI command parsing `--model`, `--port`, and `--capacity`.
- [ ] 4.3 Verify full workspace test suite `cargo test --workspace` with 0 regressions.
- [ ] 4.4 Record measured throughput and latencies in `docs/BENCHMARKS.md`, sync delta spec, and archive change.
- Gate: Full workspace test suite clean, interactive chat verified, Phase 10 sealed.
