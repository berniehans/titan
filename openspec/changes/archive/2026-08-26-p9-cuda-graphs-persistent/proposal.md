# Proposal: Phase 9 — CUDA Graphs & Persistent Decode Kernel

## Why
In Phase 8, we achieved 100% GPU-native execution across all mixed-quantized layers (`Q4_K`, `Q6_K`, `Q8_0`, `F16`), eliminating all intermediate CPU tensor fallbacks. However, each single token decode step still executes ~280 individual CUDA kernel launches across 28 layers (`RMSNorm`, `gemv_q4k`, `gemv_q6k`, `PagedKv`, `PagedAttention`, `SwiGLU`) driven from the host CPU.

Host-side kernel dispatch overhead (CPU $\rightarrow$ driver $\rightarrow$ GPU command queue) accounts for the majority of per-token latency when individual kernels finish in microseconds. By capturing the entire 28-layer forward decode pass into a **CUDA Graph** (`cudaStreamBeginCapture` / `cudaStreamEndCapture` / `cudaGraphInstantiate` / `cudaGraphLaunch`), the entire multi-layer pass is launched in a single driver call, drastically increasing autoregressive decoding throughput.

## What Changes
1. **CUDA Graph Capture Module (`engine-cuda`):**
   - Implement `CudaGraph` and `CudaGraphExec` wrappers over CUDA Driver APIs (`cuStreamBeginCapture`, `cuStreamEndCapture`, `cuGraphInstantiate`, `cuGraphLaunch`).
   - Implement dynamic node parameter updates for mutable per-token parameters (e.g. sequence position `p`).
2. **ForwardDriver Graph Execution Mode (`engine-core`):**
   - Add graph capture and graph replay methods to `ForwardDriver`.
   - On the first decode step, capture the entire 28-layer GPU execution sequence into a static graph.
   - On subsequent decode steps, replay the instantiated graph with zero per-kernel host dispatch overhead.
3. **Throughput Benchmark & Parity Verification:**
   - Verify that CUDA Graph execution produces identical logits (`cos-sim = 1.000000`, `diff < 1e-5`) compared to manual stream dispatch.
   - Benchmark generation throughput in `real_throughput_gate.rs` and record sustained tok/s in `docs/BENCHMARKS.md`.

## Capabilities Affected
- `layer-streaming-engine`: Adds graph-captured decode execution mode to `ForwardDriver` and `MultiFormatGEMV`.
