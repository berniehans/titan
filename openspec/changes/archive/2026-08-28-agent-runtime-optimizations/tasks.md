## 1. Radix Prefix Cache & Zero-Copy Branching (`engine-kvcache`)

- [x] 1.1 Implement `RadixNode` and `RadixTree` with Longest Common Prefix (LCP) matching and block assignment in `engine-kvcache/src/radix.rs`.
- [x] 1.2 Implement pinned node protection (`is_pinned = true`) and LRU eviction policy in `RadixTree::evict_lru()`.
- [x] 1.3 Implement `CowBlockTable` with atomic reference counting (`Arc<SharedBlock>`) and GPU copy-on-write page cloning in `engine-kvcache/src/cow.rs`.
- [x] 1.4 Add unit tests and parity benchmarks in `engine-kvcache/tests/radix_cache_test.rs` and `engine-kvcache/tests/cow_fork_test.rs`.

## 2. CUDA Logit Bitmasking & Attention Sinks (`engine-cuda`)

- [x] 2.1 Implement `apply_logit_mask_kernel` in `engine-cuda/kernels/logit_mask.cu` with 128-bit memory coalescing and $-\infty$ logit suppression.
- [x] 2.2 Add RAII wrapper `LogitMaskGpu` with asynchronous DMA streaming in `engine-cuda/src/logit_mask.rs`.
- [x] 2.3 Modify `paged_attention.cu` to retain initial 4 Attention Sink tokens and evaluate rolling sliding windows.
- [x] 2.4 Add kernel unit tests in `engine-cuda/tests/logit_mask_parity.rs` and `engine-cuda/tests/attention_sinks_test.rs`.

## 3. Asynchronous Grammar Parsing & Orchestration (`engine-core`)

- [x] 3.1 Implement `GrammarParser` trait and JSON Schema bitmask generator in `engine-core/src/grammar.rs`.
- [x] 3.2 Implement overlapped host-device worker pipeline in `ForwardDriver` executing CPU grammar evaluation concurrently with GPU forward passes.
- [x] 3.3 Integrate `RadixTree` prefix lookup into `ForwardDriver::prefill_with_prefix()` to bypass prefill on cached prefixes.
- [x] 3.4 Add integration tests in `engine-core/tests/json_constrained_decoding.rs` and `engine-core/tests/prefix_cache_speedup.rs`.

## 4. Agent API & Tool-Calling E2E Server (`engine-server`)

- [x] 4.1 Update `/v1/chat/completions` schema to parse `tools`, `tool_choice`, and `response_format` JSON schema constraints.
- [x] 4.2 Integrate session-persistent `RadixTree` prefix cache into `RealModel` server runtime.
- [x] 4.3 Write end-to-end integration benchmark `engine-server/tests/agent_tool_loop_bench.rs` measuring TTFT reduction and 100% JSON syntactic validity over a 10-turn tool loop.
