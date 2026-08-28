## 1. High-Priority Stream & Pinned Memory Infrastructure

- [x] 1.1 Add CudaStream::new_with_priority in engine-cuda/src/streams.rs.
- [x] 1.2 Implement PinnedHost buffer pooling in engine-cuda for non-pageable weight host residency.

## 2. Double-Buffered Streaming Execution in ForwardDriver

- [x] 2.1 Implement dual-slot ping-pong execution loop in StreamingDriver with cuStreamWaitEvent.
- [x] 2.2 Validate stream overlap with streaming_pipeline_sync_test.

## 3. Parity & Throughput Verification

- [x] 3.1 Verify numerical parity against reference in streaming_driver_parity.
- [x] 3.2 Measure PCIe transfer overlap efficiency on large layer payloads.

