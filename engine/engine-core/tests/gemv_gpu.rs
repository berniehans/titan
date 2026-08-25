//! GPU multi-format GEMV parity tests — tasks 1.2/1.3/1.4 of change 6.3.
//!
//! TDD RED phase: `engine_cuda::MultiFormatGEMV` does NOT exist yet, so this
//! file intentionally does not compile until tasks 2.1-2.3 land. The tests are
//! `#[ignore]`d so the normal CPU suite stays green; run on a CUDA machine with
//! `cargo test -- --ignored` (plus the `%LOCALAPPDATA%/Temp` NVRTC PATH trick).
//!
//! Each test builds a controlled weight tensor in one of the three supported
//! formats (Q4_K_M, Q8_0, F16), computes the CPU reference via the 6.2 forward
//! bank (`dequant_col` + scalar dot), runs the device kernel through the
//! `MultiFormatGEMV` wrapper, and asserts per-element parity.

use cudarc::driver::CudaDevice;
use engine_core::forward_cpu::{TensorType, dequant_col};
use engine_cuda::{CudaError, CudaStream, DeviceBuffer, GemvFormat, MultiFormatGEMV};
use std::sync::Arc;

// Canonical known-good Q4_K_M super-block bytes (source: engine-core/src/dequant.rs).
const Q4K_BLOCK: [u8; 144] = [
    0, 60, 0, 56, 10, 131, 31, 192, 2, 8, 65, 63, 17, 12, 69, 95, 48, 65, 82, 99, 116, 133, 150,
    167, 184, 201, 218, 235, 252, 13, 30, 47, 48, 65, 82, 99, 116, 133, 150, 167, 184, 201, 218,
    235, 252, 13, 30, 47, 116, 133, 150, 167, 184, 201, 218, 235, 252, 13, 30, 47, 48, 65, 82, 99,
    116, 133, 150, 167, 184, 201, 218, 235, 252, 13, 30, 47, 48, 65, 82, 99, 184, 201, 218, 235,
    252, 13, 30, 47, 48, 65, 82, 99, 116, 133, 150, 167, 184, 201, 218, 235, 252, 13, 30, 47, 48,
    65, 82, 99, 116, 133, 150, 167, 252, 13, 30, 47, 48, 65, 82, 99, 116, 133, 150, 167, 184, 201,
    218, 235, 252, 13, 30, 47, 48, 65, 82, 99, 116, 133, 150, 167, 184, 201, 218, 235,
];

/// Encode one Q8_0 block: `[fp16 d LE][int8 qs x32]` (34 bytes).
fn q8_block(d: f32, base: i32) -> Vec<u8> {
    let mut b = Vec::with_capacity(34);
    b.extend_from_slice(&f16_bits(d).to_le_bytes());
    for i in 0..32 {
        b.push(((base + i) & 127) as u8);
    }
    b
}

/// One F16 tensor column as raw fp16 LE bytes.
fn f16_col(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| f16_bits((i as f32) * 0.25 - 4.0).to_le_bytes())
        .flatten()
        .collect()
}

/// Approximate positive fp16 bit pattern (finite normal range).
fn f16_bits(v: f32) -> u16 {
    // Simple round-to-nearest encode for |v| in a normal fp16 range.
    let abs = v.abs();
    let sign = if v < 0.0 { 0x8000u16 } else { 0 };
    let (mut exp, mut mant) = (0u16, 0u16);
    let mut x = abs;
    if x == 0.0 {
        return sign;
    }
    // binary exponent
    let e = x.log2().floor() as i32;
    let mut exp_field = (e + 15) as u16;
    // normalize [1,2)
    while x >= 2.0 {
        x *= 0.5;
    }
    while x < 1.0 {
        x *= 2.0;
    }
    let m = ((x - 1.0) * 1024.0).round() as u16;
    if m >= 1024 {
        mant = 0;
        exp_field += 1;
    } else {
        mant = m;
    }
    sign | (exp_field << 10) | mant
}

/// Deterministic fp32 activation vector (shared across formats).
fn input_x(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 + 0.01 * (i as f32)).collect()
}

/// Scalar fp32 dot reference.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Runs one GEMV through the wrapper and returns the host `out` vector.
/// After Group 2 the wrapper signature is fixed by the kernel/wrapper contract.
fn run_gemv(
    gemv: &MultiFormatGEMV,
    stream: &CudaStream,
    format: GemvFormat,
    w_bytes: &[u8],
    x: &[f32],
    ne0: usize,
    ne1: usize,
) -> Result<Vec<f32>, CudaError> {
    let device = gemv.device();
    let x_dev = DeviceBuffer::alloc(Arc::clone(device), x.len() * 4)?;
    x_dev.copy_from_host(stream, &f32_bytes(x))?;
    let w_dev = DeviceBuffer::alloc(Arc::clone(device), w_bytes.len())?;
    w_dev.copy_from_host(stream, w_bytes)?;
    let out_dev = DeviceBuffer::alloc(Arc::clone(device), ne1 * 4)?;
    gemv.gemv(stream, format, &w_dev, &x_dev, &out_dev, ne0, ne1)?;
    let mut out = vec![0u8; ne1 * 4];
    out_dev.copy_to_host(stream, &mut out)?;
    Ok(bytes_f32(&out))
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}
fn bytes_f32(v: &[u8]) -> Vec<f32> {
    v.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Build a Q4_K weight matrix of `ne1` columns x 256 elements (one block each).
fn q4k_matrix(ne1: usize) -> Vec<u8> {
    let mut m = Vec::with_capacity(ne1 * 144);
    for _ in 0..ne1 {
        m.extend_from_slice(&Q4K_BLOCK);
    }
    m
}

#[test]
#[ignore]
fn gemv_q4k_gpu_matches_cpu_reference() -> Result<(), CudaError> {
    const NE0: usize = 256;
    const NE1: usize = 3;
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;
    let gemv = MultiFormatGEMV::new(Arc::clone(&device))?;

    let w = q4k_matrix(NE1);
    let x = input_x(NE0);
    let got = run_gemv(&gemv, &stream, GemvFormat::Q4K, &w, &x, NE0, NE1)?;

    // CPU reference: dequant each column's single block, then dot.
    let deq = dequant_col(TensorType::Q4K, &Q4K_BLOCK, NE0);
    for j in 0..NE1 {
        let expected = dot(&deq, &x);
        assert!(
            (got[j] - expected).abs() < 1e-3,
            "col {j}: GPU {} != CPU {}",
            got[j],
            expected
        );
    }
    Ok(())
}

#[test]
#[ignore]
fn gemv_q8_gpu_matches_cpu_reference() -> Result<(), CudaError> {
    const NE0: usize = 256;
    const NE1: usize = 3;
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;
    let gemv = MultiFormatGEMV::new(Arc::clone(&device))?;

    // 8 Q8_0 blocks per column (8*32=256 elements), 3 columns.
    let d = 2.0f32;
    let mut w = Vec::new();
    for j in 0..NE1 {
        for b in 0..8 {
            w.extend_from_slice(&q8_block(d, (j as i32) * 128 + b as i32 * 32));
        }
    }
    let x = input_x(NE0);
    let got = run_gemv(&gemv, &stream, GemvFormat::Q8, &w, &x, NE0, NE1)?;

    // CPU reference dequant via engine-core Q8 path.
    for j in 0..NE1 {
        let col = &w[j * 8 * 34..(j + 1) * 8 * 34];
        let deq = dequant_col(TensorType::Q8, col, NE0);
        let expected = dot(&deq, &x);
        assert!(
            (got[j] - expected).abs() < 1e-3,
            "col {j}: GPU {} != CPU {}",
            got[j],
            expected
        );
    }
    Ok(())
}

#[test]
#[ignore]
fn f16_gpu_matches_cpu_reference() -> Result<(), CudaError> {
    const NE0: usize = 256;
    const NE1: usize = 3;
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;
    let gemv = MultiFormatGEMV::new(Arc::clone(&device))?;

    let mut w = Vec::new();
    for j in 0..NE1 {
        w.extend_from_slice(&f16_col(NE0));
    }
    let x = input_x(NE0);
    let got = run_gemv(&gemv, &stream, GemvFormat::F16, &w, &x, NE0, NE1)?;

    for j in 0..NE1 {
        let col = &w[j * NE0 * 2..(j + 1) * NE0 * 2];
        let deq = dequant_col(TensorType::F16, col, NE0);
        let expected = dot(&deq, &x);
        assert!(
            (got[j] - expected).abs() < 1e-3,
            "col {j}: GPU {} != CPU {}",
            got[j],
            expected
        );
    }
    Ok(())
}
