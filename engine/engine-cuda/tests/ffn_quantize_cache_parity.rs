//! Regression gate for the FFN-only cached Q8_1 quantization path.

mod common;

use cudarc::driver::CudaDevice;
use engine_cuda::{BatchedGEMM, CudaStream, DeviceBuffer};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bytes_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[test]
#[ignore]
fn cached_ffn_quantize_matches_reference_q8_1() -> Result<(), Box<dyn std::error::Error>> {
    common::initialize_cuda();
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(device.clone())?;
    let gemm = BatchedGEMM::new(device.clone())?;

    let ne0 = 256;
    let input: Vec<f32> = (0..ne0)
        .map(|index| ((index as f32) * 0.03125).sin() * 2.0)
        .collect();
    let norm: Vec<f32> = (0..ne0)
        .map(|index| 0.75 + (index % 7) as f32 * 0.03125)
        .collect();
    let eps = 1e-5;

    let x = DeviceBuffer::alloc(device.clone(), ne0 * 4)?;
    let norm_dev = DeviceBuffer::alloc(device.clone(), ne0 * 4)?;
    let qx = DeviceBuffer::alloc(device.clone(), ne0)?;
    let qd = DeviceBuffer::alloc(device.clone(), ne0 / 32 * 4)?;
    let qs = DeviceBuffer::alloc(device.clone(), ne0 / 32 * 4)?;
    x.copy_from_host(&stream, &f32_bytes(&input))?;
    norm_dev.copy_from_host(&stream, &f32_bytes(&norm))?;

    gemm.quantize_q8_1_cached(&stream, &x, &norm_dev, &qx, &qd, &qs, ne0, eps)?;

    let mut qx_bytes = vec![0u8; ne0];
    let mut qd_bytes = vec![0u8; ne0 / 32 * 4];
    let mut qs_bytes = vec![0u8; ne0 / 32 * 4];
    qx.copy_to_host(&stream, &mut qx_bytes)?;
    qd.copy_to_host(&stream, &mut qd_bytes)?;
    qs.copy_to_host(&stream, &mut qs_bytes)?;

    let sum_sq: f32 = input.iter().map(|value| value * value).sum();
    let scale = (sum_sq / ne0 as f32 + eps).sqrt().recip();
    for block in 0..ne0 / 32 {
        let values: Vec<f32> = input[block * 32..(block + 1) * 32]
            .iter()
            .zip(&norm[block * 32..(block + 1) * 32])
            .map(|(value, weight)| value * scale * weight)
            .collect();
        let amax = values
            .iter()
            .fold(0.0f32, |max, value| max.max(value.abs()));
        let d = amax / 127.0;
        let expected_q: Vec<i8> = values
            .iter()
            .map(|value| {
                (value * if d > 0.0 { d.recip() } else { 0.0 })
                    .round()
                    .clamp(-128.0, 127.0) as i8
            })
            .collect();
        assert_eq!(&qx_bytes[block * 32..(block + 1) * 32], unsafe {
            std::slice::from_raw_parts(expected_q.as_ptr() as *const u8, expected_q.len())
        });
        assert!((bytes_f32(&qd_bytes)[block] - d).abs() < 1e-6);
        assert!((bytes_f32(&qs_bytes)[block] - values.iter().sum::<f32>()).abs() < 1e-4);
    }

    Ok(())
}
