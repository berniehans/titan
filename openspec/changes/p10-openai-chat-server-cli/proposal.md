# Proposal: Phase 10 — OpenAI Chat Completions Server, Streaming SSE & Interactive CLI

## 1. Summary

Connect Titan's 100% GPU `ForwardDriver` (with CUDA Graphs and PagedAttention) to an OpenAI-compatible HTTP server and interactive terminal CLI.

The resulting server will expose standard endpoints (`POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`) supporting Server-Sent Events (SSE) token streaming, ChatML prompt templating, and advanced sampling (`temperature`, `top_p`, `top_k`, `repetition_penalty`, `stop` sequences). An interactive CLI (`titan chat`) will provide direct, low-latency multi-turn conversations in the terminal.

---

## 2. Motivation

- **Immediate Usability:** With Phase 8 and Phase 9 complete, Titan executes complete transformer decode passes 100% on GPU via single-launch CUDA Graphs with bit-exact parity. However, users currently interact with the engine via unit/integration test harnesses rather than a live HTTP server or CLI.
- **Client Ecosystem Compatibility:** Supporting the standard `/v1/chat/completions` wire contract allows any OpenAI-compatible client (Cursor, Open-WebUI, LibreChat, LiteLLM, LangChain) to connect directly to Titan on `http://localhost:8000`.
- **Production Sampling Controls:** Real-world generation requires flexible sampling parameters (greedy, temperature, nucleus top-$p$, top-$k$, stop token trimming) to prevent repetitive loops and control creativity.

---

## 3. Scope & Sub-Changes

1. **Sub-change 10.1 — ChatML Templating & OpenAI Wire Types (`engine-server`):**
   - Define `ChatMessage`, `ChatCompletionRequest`, `ChatCompletionResponse`, `ChatCompletionChunk`, `ChatChoice`, `DeltaMessage`.
   - Implement Qwen ChatML formatter: `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`.
   - Parse sampling parameters: `temperature`, `top_p`, `top_k`, `repetition_penalty`, `stop`.

2. **Sub-change 10.2 — Production Sampler & Real Engine Service Actor (`engine-core` / `engine-server`):**
   - Implement `Sampler`: temperature scaling, softmax, top-$k$ filtering, top-$p$ nucleus cumulative mass filtering, repetition penalty, and deterministic fallback (temperature = 0).
   - Implement `EngineService` actor managing GPU worker thread, prefill queue, and continuous single-launch CUDA graph decode loops.
   - Implement SSE streaming channel and non-streaming HTTP handler in Axum.

3. **Sub-change 10.3 — Interactive Terminal CLI (`titan chat` & `titan serve`):**
   - Provide CLI commands:
     - `titan serve [--model <path>] [--port <port>] [--capacity <tokens>]`
     - `titan chat [--model <path>]` (interactive terminal REPL with multi-turn history and live token-by-token streaming).

4. **Sub-change 10.4 — End-to-End Verification & Benchmarks Seal:**
   - Integration tests:
     - `tests/e2e_chat_completions.rs`: HTTP test for `/v1/chat/completions` (streaming SSE & non-streaming JSON).
     - `tests/sampler_parity_test.rs`: verification of greedy, temperature, top-$p$, and stop sequence triggers.
   - Record measured latencies and tok/s in `docs/BENCHMARKS.md`.
   - Sync main spec and archive change.
