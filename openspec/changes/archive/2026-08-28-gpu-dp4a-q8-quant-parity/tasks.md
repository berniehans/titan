## 1. On-the-Fly Q8_1 Quantization

- [x] 1.1 Implement `struct block_q8_1` and `quantize_row_q8_1` device functions in `gemm_quant.cu`, computing 8-bit quantized values with FP16 scale and block sum per 32 elements.
- [x] 1.2 Integrate Q8_1 quantization into shared memory activation loading in `gemm_quant.cu`, storing quantized int8 activations directly in shared memory.

## 2. Integer SIMD GEMV Kernels with __dp4a

- [x] 2.1 Rewrite `gemm_q4k_kernel` in `gemm_quant.cu` using `__dp4a` for 4-way 8-bit integer dot products with 128-bit vectorized weight loading (`uint4`).
- [x] 2.2 Rewrite `gemm_fused_qkv_kernel` in `gemm_quant.cu` using `__dp4a` for fused Q and K projections.
- [x] 2.3 Rewrite `gemm_q4k_fused_gate_up_swiglu_kernel` in `gemm_quant.cu` using `__dp4a` for fused Gate and Up SwiGLU projections.
- [x] 2.4 Rewrite `gemm_q4k_splitk_kernel` in `gemm_quant.cu` using `__dp4a` for Down-projection.

## 3. Verification and Parity Benchmark

- [x] 3.1 Verify NVRTC JIT compilation and graph instantiation across all transformer layers with zero compilation errors.
- [x] 3.2 Run the head-to-head benchmark (`test_titan_vs_llama_cpp_benchmark`) to validate throughput reaches 135+ tok/s and matches or exceeds llama.cpp.
