//! GPU dequantization test — Task 2.x of change f3-gpu-dequant.
//!
//! TDD RED phase: this file intentionally does NOT compile until Task 2.2
//! lands and `engine_cuda::Q4KDequantizer` is implemented. Runs only on a
//! CUDA-capable machine (`#[ignore]` keeps the normal suite green).

use cudarc::driver::CudaDevice;
use engine_cuda::{CudaError, CudaStream, DeviceBuffer, Q4KDequantizer};
use std::sync::Arc;

// Canonical Q4_K known-good bytes + expected dequantized floats, copied
// verbatim from the CPU reference fixture in engine-core/src/dequant.rs
// (module-level `BLOCK_BYTES` const at lines ~86-95 and `EXPECTED` const at
// lines ~97-118 of that file). Source of truth: engine-core/src/dequant.rs.
const BLOCK_BYTES: [u8; 144] = [
    0, 60, 0, 56, 10, 131, 31, 192, 2, 8, 65, 63, 17, 12, 69, 95, 48, 65, 82, 99, 116, 133, 150,
    167, 184, 201, 218, 235, 252, 13, 30, 47, 48, 65, 82, 99, 116, 133, 150, 167, 184, 201, 218,
    235, 252, 13, 30, 47, 116, 133, 150, 167, 184, 201, 218, 235, 252, 13, 30, 47, 48, 65, 82, 99,
    116, 133, 150, 167, 184, 201, 218, 235, 252, 13, 30, 47, 48, 65, 82, 99, 184, 201, 218, 235,
    252, 13, 30, 47, 48, 65, 82, 99, 116, 133, 150, 167, 184, 201, 218, 235, 252, 13, 30, 47, 48,
    65, 82, 99, 116, 133, 150, 167, 252, 13, 30, 47, 48, 65, 82, 99, 116, 133, 150, 167, 184, 201,
    218, 235, 252, 13, 30, 47, 48, 65, 82, 99, 116, 133, 150, 167, 184, 201, 218, 235,
];
const EXPECTED: [f32; 256] = [
    -1.0, 9.0, 19.0, 29.0, 39.0, 49.0, 59.0, 69.0, 79.0, 89.0, 99.0, 109.0, 119.0, 129.0, 139.0,
    149.0, -1.0, 9.0, 19.0, 29.0, 39.0, 49.0, 59.0, 69.0, 79.0, 89.0, 99.0, 109.0, 119.0, 129.0,
    139.0, 149.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0, 32.0, 35.0, 38.0, 41.0, -4.0,
    -1.0, 2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0, 32.0, 35.0, 38.0, 41.0, -4.0,
    -1.0, 2.0, 123.5, 154.5, 185.5, 216.5, 247.5, 278.5, 309.5, 340.5, 371.5, 402.5, 433.5, 464.5,
    -0.5, 30.5, 61.5, 92.5, 123.5, 154.5, 185.5, 216.5, 247.5, 278.5, 309.5, 340.5, 371.5, 402.5,
    433.5, 464.5, -0.5, 30.5, 61.5, 92.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5,
    -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5,
    -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, 7.5, 8.5, 9.5,
    10.5, 11.5, 12.5, 13.5, 14.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5,
    11.5, 12.5, 13.5, 14.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 484.0, 528.0, 572.0, 616.0,
    660.0, 0.0, 44.0, 88.0, 132.0, 176.0, 220.0, 264.0, 308.0, 352.0, 396.0, 440.0, 484.0, 528.0,
    572.0, 616.0, 660.0, 0.0, 44.0, 88.0, 132.0, 176.0, 220.0, 264.0, 308.0, 352.0, 396.0, 440.0,
    50.0, 55.0, 60.0, 65.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0,
    50.0, 55.0, 60.0, 65.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0,
    942.5, -2.5, 60.5, 123.5, 186.5, 249.5, 312.5, 375.5, 438.5, 501.5, 564.5, 627.5, 690.5, 753.5,
    816.5, 879.5, 942.5, -2.5, 60.5, 123.5, 186.5, 249.5, 312.5, 375.5, 438.5, 501.5, 564.5, 627.5,
    690.5, 753.5, 816.5, 879.5,
];
const N_BLOCKS: usize = 3;

#[test]
#[ignore]
fn dequant_q4k_gpu_matches_cpu_reference() -> Result<(), CudaError> {
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;

    let mut host_src: Vec<u8> = Vec::with_capacity(N_BLOCKS * BLOCK_BYTES.len());
    for _ in 0..N_BLOCKS {
        host_src.extend_from_slice(&BLOCK_BYTES);
    }
    let n_bytes_dst = N_BLOCKS * EXPECTED.len() * std::mem::size_of::<f32>();

    let src_dev = DeviceBuffer::alloc(Arc::clone(&device), host_src.len())?;
    let dst_dev = DeviceBuffer::alloc(Arc::clone(&device), n_bytes_dst)?;
    src_dev.copy_from_host(&stream, &host_src)?;

    let dequantizer = Q4KDequantizer::new(Arc::clone(&device))?;
    dequantizer.launch(&stream, &src_dev, &dst_dev)?;

    let mut dst_bytes = vec![0u8; n_bytes_dst];
    dst_dev.copy_to_host(&stream, &mut dst_bytes)?; // syncs the stream internally

    let got: Vec<f32> = dst_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut expected_all = Vec::with_capacity(N_BLOCKS * EXPECTED.len());
    for _ in 0..N_BLOCKS {
        expected_all.extend_from_slice(&EXPECTED);
    }

    assert_eq!(got.len(), expected_all.len());
    for (i, (g, e)) in got.iter().zip(expected_all.iter()).enumerate() {
        assert!((g - e).abs() < 1e-5, "index {i}: GPU {g} != reference {e}");
    }
    Ok(())
}
