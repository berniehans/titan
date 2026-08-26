# Change: Double-buffered CUDA pipeline (Phase 2)

## Why
Phase 0-1 delivered the I/O substrate (GGUF parser, pinned RAM, single-pass loader). Without overlapped transfer the engine is just a file reader: every layer would pay sequential copy + compute. This change turns the loader into an engine primitive — the double-buffered H2D pipeline that hides transfer time behind kernel execution.

## What Changes
- Create two `CudaStream`s per device in `engine-cuda` (transfer stream + compute stream) via RAII wrappers.
- Per-layer device buffers: two VRAM slots (`DeviceSlot` RAII with `cuMemAlloc`/`cuMemFree`) sized to the largest layer.
- Event synchronization: `copy_done[N]` recorded on the transfer stream; compute waits on it before launching kernels on slot N mod 2. No CPU busy-waiting (`streamWaitEvent`, not `streamSynchronize`).
- Pipeline driver in `engine-core`: `Pipeline::run(layers)` that for each layer enqueues async H2D copy into the next free slot and "computes" layer N on the other slot (compute = no-op stub until Phase 3 kernels).
- Benchmark harness: dummy 8-layer model measuring wall time per layer vs sequential (t_copy + t_kernel).

## Non-goals
- No real dequant/attention kernels (Phase 3) — compute stage is a timed stub.
- No KV-cache, no batching, no HTTP server.
- No GPUDirect Storage: source stays pinned RAM.

## Impact
- **Affected specs:** layer-streaming-engine ("Double-buffered pipelining with overlap" requirement)
- **Affected code:** `engine-cuda/src/{streams.rs,device_buffer.rs}` (new), `engine-core/src/pipeline.rs` (new)
- **Gate:** Nsight trace shows ≥80% of compute window covered by concurrent transfer AND total per-layer time < sequential t_copy+t_kernel AND all existing tests stay green

## Tasks (summary — details in tasks.md)
1. Stream RAII wrappers + tests
2. Device buffer (VRAM slot) RAII + tests
3. Event record/wait primitives + tests
4. Pipeline driver with overlap + integration test (GPU, #[ignore])
5. Benchmark vs sequential + gate verification
