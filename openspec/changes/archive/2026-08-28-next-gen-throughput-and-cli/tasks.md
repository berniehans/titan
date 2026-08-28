## 1. GEMV Tuning for 3B/7B Models

- [x] 1.1 Implement adaptive accumulator unrolling and multi-warp tiling in `gemm_quant.cu` for wide inner dimensions ($K \ge 3072$).
- [x] 1.2 Tune down-projection and SwiGLU FFN kernels for 3B architectures ($K=8192$) in `batched_gemm.rs` and `forward_driver.rs`.
- [x] 1.3 Validate Llama 3.2 3B decode throughput reaches $\ge 90$ tok/s in head-to-head benchmark.

## 2. Multi-Model Speculative Decoding Engine

- [x] 2.1 Implement `SpeculativeDriver` in `engine-core/src/speculative_driver.rs` managing concurrent Draft ($M_1$) and Target ($M_2$) GPU resident models.
- [x] 2.2 Implement GPU batched candidate verification and KV cache rollback/advance logic.
- [x] 2.3 Write integration benchmark `speculative_speedup_bench.rs` validating 2x speedup on Llama 3.2 3B.

## 3. Chunked Prefill with FlashAttention

- [x] 3.1 Implement chunked prompt prefill ($C=512$) in `ForwardDriver::prefill_chunked()` in `forward_driver.rs`.
- [x] 3.2 Verify prompt ingestion on long sequences ($N \ge 2048$) with bounded VRAM usage and $>1000$ tok/s throughput.

## 4. Interactive CLI (`titan run` / `titan serve`)

- [x] 4.1 Update `titan/src/main.rs` with `titan run <model.gguf>` REPL chat command supporting real-time streaming and performance HUD.
- [x] 4.2 Update `titan serve <model.gguf>` command with production OpenAI API server options.
- [x] 4.3 Verify end-to-end user experience with live prompt generation.
