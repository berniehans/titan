# Tasks: f2-double-buffer-pipeline

> Execute with harness (agy gemini-3.6-flash-high via coder profile). Strict TDD. One commit per task.

## 1. Streams (engine-cuda)
- [x] 1.1 Failing test (#[ignore]): create CudaStream, enqueue trivial memset, synchronize, verify output
- [x] 1.2 Implement RAII `CudaStream` wrapper (cuStreamCreate/cuStreamDestroy) with // SAFETY:
- [x] 1.3 Verify test PASS on local GPU

## 2. Device buffers (engine-cuda)
- [x] 2.1 Failing test (#[ignore]): alloc DeviceBuffer of 64 MB, write pattern H2D, read back D2H equal
- [x] 2.2 Implement RAII `DeviceBuffer` (cuMemAlloc/cuMemFree) with // SAFETY:
- [x] 2.3 Verify test PASS on local GPU

## 3. Events (engine-cuda)
- [x] 3.1 Failing test (#[ignore]): record event after memset on stream A; stream B waits on it before its own write; final read sees B's value (ordering proof)
- [x] 3.2 Implement RAII `CudaEvent` (cuEventCreate/cuEventDestroy) with record/wait API
- [x] 3.3 Verify test PASS on local GPU

## 4. Pipeline driver (engine-core)
- [x] 4.1 Failing GPU test (#[ignore]): run 8-layer dummy through Pipeline::run; assert every layer's compute waited on its copy_done event (event query timestamps monotonic per slot)
- [x] 4.2 Implement Pipeline: ping-pong slots, async copies on transfer stream, stub compute on compute stream gated by events; no streamSynchronize inside the loop
- [x] 4.3 Verify: cargo test -p engine-core green (CPU suite unaffected)

## 5. Gate
- [x] 5.1 Benchmark: 8-layer dummy, measure per-layer wall time pipelined vs sequential sum(t_copy + t_kernel); log both
- [ ] 5.2 Nsight trace capture showing ≥80% compute window covered by concurrent transfer (PENDING: requires manual Nsight Systems capture)
- [x] 5.3 Gate F2: pipelined < sequential on RTX 3060, clippy -D warnings clean, full test suite green
