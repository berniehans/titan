# Implementation Tasks: Phase 9 — CUDA Graphs & Persistent Decode Kernel

## 1. CUDA Graph RAII Wrappers (sub-change 9.1)
- [ ] 1.1 Implement `CudaGraph` and `CudaGraphExec` in `engine-cuda/src/graphs.rs` wrapping CUDA driver capture and launch APIs.
- [ ] 1.2 Add capture helpers `CudaStream::begin_capture()` and `CudaStream::end_capture()`.
- [ ] 1.3 Create TDD test `engine-cuda/tests/cuda_graphs_test.rs` capturing a multi-kernel sequence (Norm + GEMV) and verifying execution parity against stream execution.
- Gate: `cargo test -p engine-cuda --test cuda_graphs_test` PASS.

## 2. Dynamic Position & Device-Side Parameter Updating (sub-change 9.2)
- [ ] 2.1 Update `NormRope` kernel in `engine-cuda/kernels/norm_rope.cu` to optionally accept device-pointer position `pos_dev` (or device scalar).
- [ ] 2.2 Update `PagedAttention` kernel in `engine-cuda/kernels/paged_attention.cu` to read sequence length from device memory or support dynamic graph node parameter update.
- [ ] 2.3 Create TDD test `engine-cuda/tests/graph_dynamic_params_test.rs` verifying sequential positions with graph replay.
- Gate: `cargo test -p engine-cuda --test graph_dynamic_params_test` PASS.

## 3. ForwardDriver Graph Capture & Execution (sub-change 9.3)
- [ ] 3.1 Implement `ForwardDriver::capture_decode_graph()` in `engine-core/src/forward_driver.rs`.
- [ ] 3.2 Implement `ForwardDriver::decode_graph()` executing single-token decode via `CudaGraphExec::launch()`.
- [ ] 3.3 Create parity test `engine-core/tests/driver_graph_parity.rs` asserting bit-identical logits (`cos-sim >= 0.9999`) between graph decode and standard decode.
- Gate: `cargo test -p engine-core --test driver_graph_parity` PASS.

## 4. End-to-End Speedup & Benchmarks Seal (sub-change 9.4)
- [ ] 4.1 Benchmark sustained decode throughput in `engine-server/tests/real_throughput_gate.rs` with CUDA graphs enabled.
- [ ] 4.2 Record measured speedup and tok/s in `docs/BENCHMARKS.md`.
- [ ] 4.3 Verify full workspace test suite `cargo test --workspace` with 0 regressions.
- Gate: Throughput speedup verified, all tests green, Phase 9 sealed.
