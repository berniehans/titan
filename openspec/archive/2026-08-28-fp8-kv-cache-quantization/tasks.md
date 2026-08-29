## 1. CUDA FP8 KV-Cache Kernels

- [x] 1.1 Add FP8 quantization / dequantization conversion helpers in `engine-cuda/kernels/`.
- [x] 1.2 Update `flash_decoding_splitk` and `paged_attention` to support FP8 KV cache loads.

## 2. KV-Cache Pool Integration & Benchmarks

- [x] 2.1 Update `PagedKvGpu` and `PagedKvLayout` to support `KvDataType::FP8`.
- [x] 2.2 Validate numerical parity and measure attention bandwidth speedup on long contexts.
