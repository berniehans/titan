## 1. StreamingLLM Attention Sinks & H2O Eviction Policy

- [ ] 1.1 Implement `StreamingKvPolicy` in `engine-kvcache/src/streaming.rs`.
- [ ] 1.2 Implement rolling block eviction in `PagedKvCache`.
- [ ] 1.3 Add long-context streaming verification test in `engine-server/tests/streaming_llm_infinite_context_test.rs`.
- [ ] 1.4 Benchmark memory consumption and decode speed under bounded VRAM budget.
