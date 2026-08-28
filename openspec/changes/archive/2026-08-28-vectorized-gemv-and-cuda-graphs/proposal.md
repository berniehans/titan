## Why

While Titan achieves 90%-98% parity with llama.cpp on sub-2B models (e.g. 204.2 tok/s on Qwen3 0.6B), larger models such as Llama 3.2 3B experience a 36% throughput deficit (64.4 tok/s vs 99.8 tok/s) due to DRAM memory transaction uncoalescing on large weights (1.81 GB/token), un-fused intermediate activations in the FFN stage, and ~1.14 ms of Windows WDDM driver launch overhead across 284 individual kernel dispatches per token.

## What Changes

- **128-bit Vectorized Coalesced Weight Loads (uint4)**: Reorganize Q4_K and Q6_K quant blocks to be loaded into registers via 128-bit uint4 memory transactions, saturating the 192-bit DRAM memory bus across all 30 SMs.
- **Full FFN Fusion Pipeline (RMSNorm + Gate + Up + SwiGLU + Down)**: Eliminate intermediate DRAM / global memory roundtrips and separate quantization launches between Gate/Up and Down projections by keeping intermediate SwiGLU activations in shared memory and registers.
- **Decode CUDA Graph Capture (cuGraphCreate / cuGraphLaunch)**: Capture the complete 28-layer static decode execution sequence into an instantiable CUDA Graph, reducing CPU-to-GPU dispatch latency from 1.14 ms (284 cuLaunchKernel calls) to ~1 μs per token.

## Capabilities

### New Capabilities
- cuda-graph-execution: Captures and launches static autoregressive decode execution graphs to eliminate host driver submission overhead on Windows and Linux.

### Modified Capabilities
- 	ensor-core-gemv: Updates memory load requirements to mandate 128-bit vector coalescing (uint4) and fused FFN multi-stage projections.

## Impact

- **engine-cuda**: Adds CUDA graph capture abstractions (CudaGraph, CudaGraphExec), modifies gemm_q4k_mma.cu and gemm_quant.cu with 128-bit vector loaders and fused FFN pipeline.
- **engine-core**: Updates ForwardDriver to record decode passes into reusable graph execution structures.
- **engine-server**: Leverages zero-overhead graph replays during continuous token generation.
