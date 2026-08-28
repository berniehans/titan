## 1. Multi-Model Driver Allocation & Context Sizing

- [x] 1.1 Support dual-model concurrent residency in ForwardDriver without VRAM fragmentation.
- [x] 1.2 Implement batched speculative candidate verification in ForwardDriver::verify_speculative.

## 2. Fast Speculative Sampling & KV Synchronization

- [x] 2.1 Implement fast speculative verification CUDA graph and rejection verification.
- [x] 2.2 Implement virtual BlockTable and sequence position synchronization upon rejection / acceptance.

## 3. End-to-End Benchmark & Throughput Validation

- [x] 3.1 Verify speculative output token identity against standalone 3B target in speculative_speedup_bench.
- [x] 3.2 Fused multi-row DP4A vectorized kernels for verification speedup (verification down to ~32ms / 8.0ms per token).

