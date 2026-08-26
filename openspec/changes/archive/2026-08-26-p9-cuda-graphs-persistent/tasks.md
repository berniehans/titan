# Implementation Tasks: Phase 9 — CUDA Graphs & Persistent Decode Kernel

## 1. CUDA Graph RAII Wrappers (sub-change 9.1)
- [x] 1.1 Implement `CudaGraph` and `CudaGraphExec` in `engine-cuda/src/graphs.rs` wrapping CUDA driver capture and launch APIs.
- [x] 1.2 Add capture helpers `CudaStream::begin_capture()` and `CudaStream::end_capture()`.
- [x] 1.3 Create TDD test `engine-cuda/tests/cuda_graphs_test.rs` capturing a multi-kernel sequence (Norm + GEMV) and verifying execution parity against stream execution.
- Gate PASS: `cuda_graphs_test` max difference = `0.000000e0` (bit-exact) vs stream execution.

## 2. Dynamic Position & Device-Side Parameter Updating (sub-change 9.2)
- [x] 2.1 Update `NormRope` kernel in `engine-cuda/kernels/norm_rope.cu` to optionally accept device-pointer position `pos_dev` (or device scalar).
- [x] 2.2 Update `PagedAttention` and `PagedKvGpu` kernels to read sequence length and slot dynamically from `pos_dev`.
- [x] 2.3 Create TDD test `engine-cuda/tests/graph_dynamic_params_test.rs` verifying sequential positions with graph replay.
- Gate PASS: `graph_dynamic_params_test` max difference = `0.000000e0` (bit-exact) across 8 sequential positions.

## 3. ForwardDriver Graph Capture & Execution (sub-change 9.3)
- [x] 3.1 Implement `ForwardDriver::capture_decode_graph()` in `engine-core/src/forward_driver.rs`.
- [x] 3.2 Implement `ForwardDriver::decode_graph()` executing single-token decode via `CudaGraphExec::launch()`.
- [x] 3.3 Create parity test `engine-core/tests/driver_graph_parity.rs` asserting bit-identical logits (`cos-sim >= 0.9999`) between graph decode and standard decode.
- Gate PASS: `driver_graph_parity` PASS on multi-prompt sequence with zero NaNs and bit-exact tokens.

## 4. End-to-End Speedup & Benchmarks Seal (sub-change 9.4)
- [x] 4.1 Benchmark sustained decode throughput in `engine-server/tests/real_throughput_gate.rs` with CUDA graphs enabled.
- [x] 4.2 Record measured speedup and tok/s in `docs/BENCHMARKS.md`.
- [x] 4.3 Verify full workspace test suite `cargo test --workspace` with 0 regressions.
- Gate PASS: All 4 quality domains verified, 100% GPU execution, Phase 9 sealed.
