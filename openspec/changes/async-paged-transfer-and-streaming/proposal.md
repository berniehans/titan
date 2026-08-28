# Proposal: Asynchronous Paged Transfer & Double-Buffered Streaming Pipeline

## Problem & Motivation
When executing out-of-core models (e.g. 14B/32B dense models or MoE expert banks) that exceed physical VRAM, transferring weights synchronously over the PCIe bus stalls GPU compute pipelines.

By introducing high-priority asynchronous transfer streams (cuStreamCreateWithPriority), locked host memory pools (cuMemAllocHost), and ping-pong double buffering in GPU VRAM synchronized purely via device-side CUDA events (cuStreamWaitEvent), Titan can overlap Layer +1$ weight DMA transfers with Layer $ kernel computations with zero CPU blocking.

## Proposed Changes
1. **Priority Stream Management:** Add CudaStream::new_with_priority in engine-cuda.
2. **Double-Buffered Ping-Pong Slots:** Allocate dual layer slots in VRAM (slot[0] and slot[1]).
3. **Event-Driven Overlapping:** Connect transfer stream and compute stream via cuEventRecord and cuStreamWaitEvent.
4. **MoE Hot-Swap Cache:** Integrate dynamic LRU slot caching for MoE expert weights.

## Verification
- Automated tests streaming_driver_parity.rs and streaming_pipeline_sync_test.rs validating cosine similarity $\ge 0.99$ and zero CPU-side dispatch bubbles.
