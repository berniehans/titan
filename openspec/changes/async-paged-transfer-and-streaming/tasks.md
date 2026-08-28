## 1. High-Priority Stream & Pinned Memory Infrastructure

- [ ] 1.1 Add CudaStream::new_with_priority in engine-cuda/src/streams.rs.
- [ ] 1.2 Implement PinnedHostBuffer pooling in engine-cuda for non-pageable weight host residency.

## 2. Double-Buffered Streaming Execution in ForwardDriver

- [ ] 2.1 Implement dual-slot ping-pong execution loop in StreamingDriver with cuStreamWaitEvent.
- [ ] 2.2 Validate stream overlap with streaming_pipeline_sync_test.

## 3. Parity & Throughput Verification

- [ ] 3.1 Verify numerical parity against reference in streaming_driver_parity.
- [ ] 3.2 Measure PCIe transfer overlap efficiency on large layer payloads.
