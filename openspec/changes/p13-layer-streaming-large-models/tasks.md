# Implementation Tasks: Phase 13 — Large Model Scaling (>6 GB VRAM: 14B & 32B via Layer Streaming Pipeline)

## 1. Double-Buffered Layer Weight Ring (sub-change 13.1)
- [x] 1.1 Implement `LayerDoubleBuffer` in `engine-core/src/layer_double_buffer.rs` allocating two ping-pong VRAM slots sized for maximum layer weight size.
- [x] 1.2 Implement synchronous and asynchronous host-to-device slot population methods.
- [x] 1.3 Create unit test `engine-core/tests/layer_double_buffer_test.rs` asserting zero reallocation across repeated layer ping-pong swaps.
- Gate PASS: `cargo test -p engine-core --test layer_double_buffer_test` PASS.

## 2. StreamingForwardDriver Dual-Stream Pipeline (sub-change 13.2)
- [x] 2.1 Implement `StreamingForwardDriver` in `engine-core/src/streaming_forward_driver.rs` with dedicated `compute_stream` and `transfer_stream`.
- [x] 2.2 Implement asynchronous event barrier overlapping layer $L+1$ transfer with layer $L$ execution.
- [x] 2.3 Create unit test `engine-core/tests/streaming_pipeline_sync_test.rs` validating event recording, stream waits, and double-buffer ping-pong transitions.
- Gate PASS: `cargo test -p engine-core --test streaming_pipeline_sync_test` PASS.

## 3. Large Model Topology & Golden Parity Gate (sub-change 13.3)
- [x] 3.1 Validate arbitrary layer scaling ($N \ge 48$) and dimension scaling ($H \ge 5120$) in model configuration parsing.
- [x] 3.2 Create parity test `engine-core/tests/streaming_driver_parity.rs` comparing `StreamingForwardDriver` output against `ForwardDriver` across all layers.
- [x] Gate PASS: `cargo test -p engine-core --test streaming_driver_parity` PASS (`cos-sim = 1.000000`).

## 4. End-to-End Large Model Verification & Phase 13 Seal (sub-change 13.4)
- [ ] 4.1 Verify VRAM peak working set audit in `engine-server/tests/large_model_vram_audit_gate.rs` asserting $\le 2.0\text{ GB}$ total consumption.
- [ ] 4.2 Record throughput and VRAM footprint metrics in `docs/BENCHMARKS.md`.
- [ ] 4.3 Verify full workspace test suite `cargo test --workspace` with 0 regressions.
- [ ] 4.4 Sync delta spec to main spec and archive change.
- Gate: Large model streaming verified, VRAM budget bounded $\le 2.0\text{ GB}$, tests green, Phase 13 sealed.
