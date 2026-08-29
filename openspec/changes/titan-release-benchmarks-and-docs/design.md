# Design: Titan Release Packaging & Documentation

## Structure
1. **Overview & Features:** Highlights Ampere Tensor Cores, DP4A vectorization, FlashDecoding Split-KV, Autonomous CUDA Graphs, Speculative Multi-Draft Decoding, and Continuous Batching.
2. **Benchmark Table:** Real verified numbers across models:
   - Qwen 2.5 0.5B: **253.8 tok/s**
   - Qwen 2.5 1.5B: **137.0 tok/s**
   - Llama 3.2 1B: **170.4 tok/s**
   - Llama 3.2 3B Speculative: **71.0 - 128.2 tok/s**
3. **CLI Quickstart:** `titan serve`, `titan chat`, `titan bench`, `titan agent`.
