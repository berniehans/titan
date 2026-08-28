## 1. CUDA Kernel Enhancement

- [x] 1.1 Add `gemm_q4k_splitk_kernel` and `reduce_splitk_kernel` in `engine-cuda/kernels/gemm_quant.cu` and `engine-cuda/kernels/gemm_q4k_mma.cu` and verify compilation with NVRTC.
- [x] 1.2 Implement Split-K dispatch and workspace management in `engine-cuda/src/batched_gemm.rs` with automatic $S_K$ selection based on $K$.

## 2. Engine Core Integration & Forward Routing

- [x] 2.1 Integrate Split-K scratch buffer allocation in `ForwardDriver` and route large-$K$ layer projections ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}, W_{\text{qkv}}, W_o, W_{\text{lm_head}}$) to Split-K GEMV.
- [x] 2.2 Verify golden numerical parity of Split-K execution against CPU reference (`cos-sim >= 0.997`).

## 3. Benchmarking & Verification

- [x] 3.1 Run `cargo test --release -p engine-server --test multi_model_comparison_bench` across Llama 3.2 3B and verify throughput improvement.
- [x] 3.2 Verify all workspace tests pass with `cargo test --workspace`.
