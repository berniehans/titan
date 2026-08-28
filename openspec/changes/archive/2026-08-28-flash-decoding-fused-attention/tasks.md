## 1. FlashDecoding CUDA Kernels (`engine-cuda`)

- [x] 1.1 Implement `flash_decoding_split_kernel` and `flash_decoding_reduce_kernel` in `engine-cuda/kernels/paged_attention.cu`.
- [x] 1.2 Add RAII launcher methods `launch_flash_decoding` in `engine-cuda/src/paged_attention.rs`.
- [x] 1.3 Allocate static partial buffers in `PagedAttentionGpu` (`partial_m`, `partial_l`, `partial_acc`).
- [x] 1.4 Add unit test and numerical parity validation in `engine-cuda/tests/flash_decoding_test.rs`.

## 2. Forward Driver Integration & CUDA Graphs (`engine-core`)

- [x] 2.1 Integrate FlashDecoding into `ForwardDriver::record_decode_pass()`.
- [x] 2.2 Re-capture Autonomous CUDA Graph with FlashDecoding enabled.
- [x] 2.3 Verify golden parity with local inference quality suite.

## 3. Long-Context Benchmarks & Verification (`engine-server`)

- [x] 3.1 Benchmark long-context attention latency at $N = 2,048$, $N = 4,096$, and $N = 8,192$ tokens.
- [x] 3.2 Update `docs/BENCHMARKS.md` with FlashDecoding speedups.
