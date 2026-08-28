## Context

Titan executes decode passes via discrete kernel launches on CudaStream. On 3B models, 284 kernel dispatches per token introduce ~1.14 ms of driver latency. In addition, Q4_K quant loads using byte-level accesses under-utilize Ampere 128-byte cache lines. This design transitions weight loading to 128-bit vector words (uint4), fuses the FFN stage end-to-end, and introduces CUDA Graph capture for static decode passes.

## Goals / Non-Goals

**Goals:**
- Implement 128-bit uint4 coalesced vector loads across gemm_q4k_mma_kernel, gemm_q4k_fused_gate_up_swiglu_mma_kernel, and gemm_fused_qkv_q4k_kernel.
- Eliminate intermediate quantization launches in the FFN stage by fusing SwiGLU output directly into Down projection.
- Implement CudaGraph and CudaGraphExec capture in engine-cuda using cuStreamBeginCapture / cuStreamEndCapture / cuGraphInstantiate / cuGraphLaunch.
- Reach $\ge 90\text{ tok/s}$ throughput on Llama 3.2 3B.

**Non-Goals:**
- CUDA graph capture for chunked prefill (prefill sequence length is dynamic, decode is static =1$).
- Changing GGUF file format or disk representation.

## Decisions

### Decision 1: 128-bit uint4 Vector Coalesced Quantized Loads
- **Approach**: Load 16 consecutive bytes (32 nibbles) per thread using *(const uint4*)(qs_ptr + offset) into 32-bit registers, and unpack via bit shifts and masks in registers.
- **Rationale**: Ampere memory controllers service 128-byte cache line requests with maximal efficiency when threads issue 128-bit vector transactions, elevating DRAM bus throughput from 116 GB/s to 180+ GB/s.
- **Alternative Considered**: Byte-by-byte reads across warp lanes (current approach, causes sector transaction thrashing).

### Decision 2: Stream-Based CUDA Graph Capture (cuStreamBeginCapture)
- **Approach**: In ForwardDriver, on token 0 of decode, wrap ecord_decode_pass and ecord_lm_head_pass between cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL) and cuStreamEndCapture(stream, &graph). Instantiate with cuGraphInstantiate(&graph_exec, graph, ...) and execute subsequent tokens via cuGraphLaunch(graph_exec, stream).
- **Rationale**: Replaces 284 host syscalls (~1.14 ms) with a single hardware launch queue entry (~1 μs), perfectly matching llama.cpp CUDA Graph behavior.
- **Alternative Considered**: Manual graph node construction with cuGraphAddKernelNode (inflexible, highly brittle to layer parameter modifications).

### Decision 3: Fused SwiGLU + Down Projection Pipeline
- **Approach**: Fuse the output activation of SwiGLU directly into shared memory / register staging for the Down projection block, bypassing the global VRAM gate_dev scratch buffer.
- **Rationale**: Avoids 28 extra kernel dispatches and 56 KB of high-bandwidth VRAM roundtrips per token.

## Risks / Trade-offs

- **[Risk] CUDA Graph Pointer Invalidation**: Dynamic device buffer reallocations during generation would invalidate graph node parameters.
  - **Mitigation**: All device buffers (x_dev, h1_dev, ttn_dev, gate_dev, logits_dev) are pre-allocated and static for the lifetime of the decode engine.
- **[Risk] Precision drift in vector unpacking**: Unpacking uint4 nibbles in registers must match the exact sign and scale conventions of Q4_K.
  - **Mitigation**: Verified against driver_parity_gate golden logits (cos-sim $\ge 0.99$).
