## Why

Titan's pure-Rust GPU-resident inference engine currently achieves ~103 tok/s (9.78 ms/tok) on Qwen 2.5 1.5B Q4_K_M (NVIDIA RTX 3060 Laptop GPU), while official llama.cpp reaches ~140 tok/s (7.16 ms/tok). 

The root cause of this remaining 2.6 ms gap is that Titan performs FP32 floating-point dequantization and arithmetic per thread, reading activations in 32-bit floats and issuing 1-byte scalar memory loads. In contrast, llama.cpp quantizes activations on-the-fly to `Q8_1` in shared memory and executes hardware integer dot products via NVIDIA's `__dp4a` SIMD instruction (4 integer multiply-accumulates in a single clock cycle) with 128-bit vectorized weight loads. This proposal defines the architecture and tasks to achieve full physical wire bandwidth saturation and parity (140+ tok/s).

## What Changes

- **On-the-fly `Q8_1` Activation Quantization**: Implement a fast CUDA shared-memory / warp-level quantization routine that converts FP32 activation rows into `Q8_1` blocks (int8 values with FP16 scale and block sum) in <0.005 ms.
- **Hardware Integer SIMD GEMV Kernels (`__dp4a`)**: Rewrite `gemm_q4k_kernel`, `gemm_fused_qkv_kernel`, `gemm_q4k_fused_gate_up_swiglu_kernel`, and `gemm_q4k_splitk_kernel` to compute dot-products between `Q4_K` weights and `Q8_1` activations using `__dp4a`.
- **Vectorized 128-bit Weight Transactions (`uint4`)**: Upgrade memory read transactions in GEMV kernels to 128-bit vectorized loads, eliminating serialized 1-byte read barriers.
- **Pure-Rust Driver Integration**: Wire the `Q8_1` quantizer and `__dp4a` GEMV kernels into `engine-cuda` and `ForwardDriver` without external C++ or Python dependencies.

## Capabilities

### New Capabilities
- `gpu-gemv-dp4a`: On-the-fly Q8_1 activation quantization and hardware `__dp4a` SIMD quantized matrix-vector multiplication for Q4_K and Q6_K weights on NVIDIA GPUs.

### Modified Capabilities
<!-- None: No existing spec requirement changes -->

## Impact

- **Affected Code**: `engine/engine-cuda/kernels/gemm_quant.cu`, `engine/engine-cuda/src/batched_gemm.rs`, `engine/engine-core/src/forward_driver.rs`, `engine/engine-server/tests/llama_cpp_comparison_bench.rs`.
- **APIs**: Internal kernel launch interfaces in `BatchedGEMM`.
- **Performance**: Decode throughput improves from ~103 tok/s to 140+ tok/s on NVIDIA RTX 3060 Laptop GPU. Zero regressions in model output accuracy.
