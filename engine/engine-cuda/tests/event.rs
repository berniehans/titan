use cudarc::driver::CudaDevice;
use engine_cuda::{CudaError, CudaEvent, CudaStream, DeviceBuffer};
use std::sync::Arc;

#[test]
#[ignore]
fn test_event_ordering_proof() -> Result<(), CudaError> {
    let device = CudaDevice::new(0)?;
    let stream_a = CudaStream::new(Arc::clone(&device))?;
    let stream_b = CudaStream::new(Arc::clone(&device))?;

    const SIZE: usize = 4 * 1024 * 1024; // 4 MiB
    let dev_buf = DeviceBuffer::alloc(Arc::clone(&device), SIZE)?;

    // Define patterns P1 and P2
    let p1 = vec![0xAAu8; SIZE];
    let p2 = vec![0x55u8; SIZE];

    let event_e = CudaEvent::new(Arc::clone(&device))?;
    let event_after_b = CudaEvent::new(Arc::clone(&device))?;

    // Stream A writes pattern P1 into DeviceBuffer
    dev_buf.copy_from_host(&stream_a, &p1)?;

    // Record event E on Stream A after write P1 completes on A
    event_e.record(&stream_a)?;

    // Stream B waits on event E before writing pattern P2
    event_e.stream_wait(&stream_b)?;

    // Stream B overwrites DeviceBuffer with pattern P2
    dev_buf.copy_from_host(&stream_b, &p2)?;

    // Record event_after_b on Stream B
    event_after_b.record(&stream_b)?;

    // Sync Stream B to guarantee all stream B ops (and preceding dependencies) finish
    stream_b.sync()?;

    // Verify both events query as completed
    assert!(
        event_e.query(),
        "event_e should query as completed after stream_b sync"
    );
    assert!(
        event_after_b.query(),
        "event_after_b should query as completed after stream_b sync"
    );

    // Copy D2H using Stream B and verify data matches P2
    let mut readback = vec![0u8; SIZE];
    dev_buf.copy_to_host(&stream_b, &mut readback)?;
    assert_eq!(readback, p2, "Readback buffer must yield pattern P2");

    // Timing check: elapsed_ms between event_e and event_after_b must be > 0.0
    let elapsed = event_after_b.elapsed_ms(&event_e)?;
    assert!(
        elapsed > 0.0,
        "elapsed_ms between event_e and event_after_b must be > 0, got {elapsed} ms"
    );

    Ok(())
}
