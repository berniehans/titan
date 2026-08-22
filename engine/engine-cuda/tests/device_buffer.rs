use cudarc::driver::CudaDevice;
use engine_cuda::{CudaError, CudaStream, DeviceBuffer};
use std::sync::Arc;

#[test]
#[ignore]
fn test_device_buffer_roundtrip() -> Result<(), CudaError> {
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;

    const MIB: usize = 1024 * 1024;
    const SIZE: usize = 64 * MIB;

    // Alloc 64 MB device buffer
    let dev_buf = DeviceBuffer::alloc(Arc::clone(&device), SIZE)?;
    assert_eq!(dev_buf.size(), SIZE);
    assert_ne!(dev_buf.device_ptr(), 0);

    // Write a deterministic pattern host-side
    let mut src_host = vec![0u8; SIZE];
    for (i, byte) in src_host.iter_mut().enumerate() {
        *byte = ((i.wrapping_mul(13)).wrapping_add(37) & 0xFF) as u8;
    }

    // Copy H2D on stream
    dev_buf.copy_from_host(&stream, &src_host)?;

    // Copy D2H back into a separate buffer
    let mut dst_host = vec![0u8; SIZE];
    dev_buf.copy_to_host(&stream, &mut dst_host)?;

    // Assert equality
    assert_eq!(
        src_host, dst_host,
        "Host and device roundtrip data mismatch"
    );

    Ok(())
}
