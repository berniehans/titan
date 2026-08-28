## 1. Ampere Tensor Core Kernels (`engine-cuda`)

- [x] 1.1 Implement `gemm_q4k_mma_kernel` using PTX inline `mma.sync.aligned.m16n8k32` in `engine-cuda/kernels/gemm_q4k_mma.cu`.
- [x] 1.2 Implement `gemm_q4k_fused_gate_up_swiglu_mma_kernel` in `engine-cuda/kernels/gemm_q4k_mma.cu`.
- [x] 1.3 Add Rust launcher methods `gemm_q4k_mma` and `gemm_q4k_fused_gate_up_swiglu_mma` in `engine-cuda/src/batched_gemm.rs`.
- [x] 1.4 Add numerical parity and throughput tests in `engine-cuda/tests/mma_parity_test.rs`.

## 2. Engine Core Integration & Autonomous Graph (`engine-core`)

- [x] 2.1 Integrate Tensor Core GEMV launchers into `ForwardDriver::record_decode_pass()`.
- [x] 2.2 Re-capture Autonomous CUDA Graph in GPU VRAM with MMA kernels.
- [x] 2.3 Validate numerical agreement with golden CPU reference in `engine-server/tests/inference_quality_demo.rs`.

## 3. Real-Hardware Benchmark & Verification (`engine-server`)

- [x] 3.1 Run head-to-head empirical benchmark against `llama.cpp` measuring throughput on Qwen 2.5 1.5B, DeepSeek-R1-Distill 1.5B, Llama 3.2 1B, and Llama 3.2 3B.
- [x] 3.2 Update `docs/BENCHMARKS.md` with new Tensor Core acceleration data.
