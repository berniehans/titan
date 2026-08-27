//! Streaming Pipeline Synchronization Test (Phase 13, Sub-change 13.2).
//!
//! Validates:
//! 1. Multi-stream event synchronization (`compute_stream`, `transfer_stream`).
//! 2. Event barrier ordering: transfer does not overwrite active compute slot, compute does not read un-transferred slot.
//! 3. Deterministic execution over dual streams.

use engine_cuda::{CudaDevice, CudaEvent, CudaStream, DeviceBuffer};
use std::sync::Arc;

type DynError = Box<dyn std::error::Error + Send + Sync>;

#[test]
fn test_dual_stream_event_synchronization_barrier() -> Result<(), DynError> {
    let device = Arc::new(CudaDevice::new(0)?);
    let compute_stream = CudaStream::new(Arc::clone(&device))?;
    let transfer_stream = CudaStream::new(Arc::clone(&device))?;
    let event_transfer_done = CudaEvent::new(Arc::clone(&device))?;
    let event_compute_done = CudaEvent::new(Arc::clone(&device))?;

    let slot_a = DeviceBuffer::alloc(Arc::clone(&device), 1024)?;
    let slot_b = DeviceBuffer::alloc(Arc::clone(&device), 1024)?;

    let num_layers = 16;
    for l in 0..num_layers {
        let curr_is_a = l % 2 == 0;
        let host_data = vec![l as u8 + 1; 1024];

        // 1. Transfer layer L on transfer_stream
        if curr_is_a {
            slot_a.copy_from_host(&transfer_stream, &host_data)?;
        } else {
            slot_b.copy_from_host(&transfer_stream, &host_data)?;
        }
        event_transfer_done.record(&transfer_stream)?;

        // 2. Compute stream waits for transfer to complete before reading
        event_transfer_done.stream_wait(&compute_stream)?;

        // 3. Compute readback verification on compute_stream
        let mut readback = vec![0u8; 1024];
        if curr_is_a {
            slot_a.copy_to_host(&compute_stream, &mut readback)?;
        } else {
            slot_b.copy_to_host(&compute_stream, &mut readback)?;
        }
        event_compute_done.record(&compute_stream)?;

        // 4. Transfer stream waits for compute before next iteration can overwrite
        event_compute_done.stream_wait(&transfer_stream)?;

        compute_stream.sync()?;
        assert_eq!(
            readback, host_data,
            "Layer {l} data mismatch in stream synchronization"
        );
    }

    transfer_stream.sync()?;
    compute_stream.sync()?;
    Ok(())
}
