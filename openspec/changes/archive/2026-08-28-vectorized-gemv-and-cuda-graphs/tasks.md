## 1. 128-bit Vectorized Coalesced Quantized Loads

- [x] 1.1 Implement 128-bit uint4 vector quantized loads in gemm_q4k_mma_kernel in engine-cuda/kernels/gemm_q4k_mma.cu and verify via cargo test -p engine-cuda.
- [x] 1.2 Implement 128-bit uint4 vector quantized loads in gemm_q4k_fused_gate_up_swiglu_mma_kernel in engine-cuda/kernels/gemm_q4k_mma.cu.
- [x] 1.3 Implement 128-bit uint4 vector quantized loads in gemm_fused_qkv_q4k_kernel in engine-cuda/kernels/gemm_quant.cu.

## 2. End-to-End Fused FFN Pipeline

- [x] 2.1 Implement integrated SwiGLU + Down staging in ForwardDriver to avoid intermediate global memory writeback of gate_dev.
- [x] 2.2 Validate parity against driver_parity_gate ensuring cosine similarity $\ge 0.99$.

## 3. CUDA Graphs Autoregressive Decode Execution

- [x] 3.1 Implement CUDA Graph bindings (CudaGraph, CudaGraphExec) wrapping cuStreamBeginCapture, cuStreamEndCapture, cuGraphInstantiate, and cuGraphLaunch in engine-cuda.
- [x] 3.2 Integrate graph capture and replay in ForwardDriver::stream_decode for token generation steps $\ge 1$.
- [x] 3.3 Benchmark end-to-end throughput with multi_model_comparison_bench and verify Llama 3.2 3B reaches $\ge 90 - 95\text{ tok/s}$.
