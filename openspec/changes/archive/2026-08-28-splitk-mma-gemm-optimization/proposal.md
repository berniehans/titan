## Why

When scaling local models to larger parameter counts ($\ge 3\text{B}$, such as Llama 3.2 3B Instruct), single-warp GEMV kernels suffer from GPU Streaming Multiprocessor (SM) underutilization because the matrix column dimension $N$ is small while the reduction dimension $K$ ($d_{\text{hidden}} = 3072$) is large. `llama.cpp` addresses this by partitioning the reduction dimension $K$ across multiple threadblocks using a Split-K reduction scheme with atomic or staged workspace aggregation. Introducing Split-K quantized MMA/DP4A matrix multiplication into Titan closes the performance gap on 3B models, lifting decoding throughput from ~45.9 tok/s towards 85-95+ tok/s on the RTX 3060.

## What Changes

- Add Split-K multi-block reduction support to `gemm_q4k_mma_kernel` and `gemm_q4k_kernel` in `engine-cuda/kernels/`.
- Introduce a high-efficiency partial accumulator reduction pass in `engine-cuda/src/batched_gemm.rs`.
- Specialize Split-K factor $S_K \in \{2, 4, 8\}$ based on hidden dimension $K$ and model architecture size.
- Integrate Split-K dispatch seamlessly into `ForwardDriver` layer passes ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}, W_{\text{qkv}}, W_o, W_{\text{lm_head}}$) without impacting $B=1$ sub-1B low-latency paths.
- Update `specs/tensor-core-gemv/spec.md` with Split-K partitioning requirements and benchmark gates.

## Capabilities

### Modified Capabilities
- `tensor-core-gemv`: Adds Split-K reduction dimension partitioning to `gemm_q4k_mma_kernel` and staged reduction accumulation for large hidden dimensions ($K \ge 2048$).

## Impact

- `engine-cuda/kernels/gemm_q4k_mma.cu`: Split-K kernel variants with block-indexed partial sums.
- `engine-cuda/src/batched_gemm.rs`: Split-K launcher and reduction buffer allocation.
- `engine-core/src/forward_driver.rs`: Dynamic routing to Split-K GEMV for layers with large $K$.
