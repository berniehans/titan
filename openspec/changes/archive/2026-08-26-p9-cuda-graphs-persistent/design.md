# Design Document: Phase 9 — CUDA Graphs & Persistent Decode Kernel

## Architecture Overview

### 1. CUDA Graph Driver Wrapper (`engine-cuda::graphs`)
CUDA Driver API exposed via `cudarc::driver::sys`:
- `cuStreamBeginCapture(stream, CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)`
- `cuStreamEndCapture(stream, &mut cu_graph)`
- `cuGraphInstantiate(&mut cu_graph_exec, cu_graph, ...)`
- `cuGraphLaunch(cu_graph_exec, stream)`
- `cuGraphDestroy(cu_graph)` / `cuGraphExecDestroy(cu_graph_exec)`

### 2. Parameter Updating for Mutable State
During single-token decoding, the buffer addresses of all weight matrices and activations are 100% constant.
The only state that increments per token is:
- Sequence position `p` (used for RoPE rotary embeddings and PagedAttention decoding).
- Input token embedding $x$ (copied into `x_dev`).

Two approaches for position `p`:
1. **Device-side position scalar:** Store `p` in a 4-byte `DeviceBuffer` (`pos_dev`) and pass pointer `pos_dev.device_ptr()` into RoPE/PagedAttention kernels. An atomic increment or host scalar copy updates `pos_dev` before graph launch.
2. **Graph node update (`cuGraphExecKernelNodeSetParams`):** Alternatively, update the kernel arguments directly in the instantiated graph.

### 3. ForwardDriver Integration
- `ForwardDriver::capture_decode_graph(&mut self)`: Runs one complete decode pass within stream capture mode.
- `ForwardDriver::decode_graph(&mut self, token: u32)`:
  1. Copies new token embedding into `x_dev`.
  2. Updates position in `pos_dev`.
  3. Launches the pre-instantiated `CudaGraphExec`.
  4. Downloads the final output hidden state / computes logits.

## Performance Impact
- Eliminates ~280 kernel launch driver roundtrips per token step.
- Reduces host latency from ~5-6 ms/tok CPU overhead down to sub-millisecond GPU execution.
