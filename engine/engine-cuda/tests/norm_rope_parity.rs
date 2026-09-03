//! Per-op GPU parity tests for NormRope (Phase 6.4 group 3).
//!
//! Verifies individual operation modes of the fused kernel against independent
//! CPU reference implementations in fp64 arithmetic:
//! - [`MODE_NORM`]: RMSNorm + residual addition
//! - [`MODE_ROPE`]: Rotary position embedding (skip norm)
//! - [`MODE_SWIGLU`]: SwiGLU gating (skip norm & rope)
//!
//! Each op must achieve cosine similarity >= 0.9999 and relative L2 error < 0.02.
//! Runs only on CUDA-capable machines (`#[ignore]`).

mod common;

use cudarc::driver::CudaDevice;
use engine_cuda::{
    CudaError, CudaStream, DeviceBuffer, MODE_NORM, MODE_ROPE, MODE_SWIGLU, NormRope,
};
use std::sync::Arc;

/// Deterministic pseudo-random generator: xorshift32, pure and reproducible.
struct Xorshift(u32);

impl Xorshift {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Deterministic float in [-1, 1) derived from 23 mantissa bits.
    fn next_f32(&mut self) -> f32 {
        let bits = self.next_u32() & 0x00FF_FFFF;
        (bits as f32 / 0x0080_0000 as f32) - 1.0
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_f32(v: &[u8]) -> Vec<f32> {
    v.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Cosine similarity between two vectors in fp64.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        let x = x as f64;
        let y = y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Relative L2 error `||a - b||_2 / ||b||_2` in fp64.
fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut diff_norm: f64 = 0.0;
    let mut ref_norm: f64 = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        let diff = (x - y) as f64;
        let y = y as f64;
        diff_norm += diff * diff;
        ref_norm += y * y;
    }
    diff_norm.sqrt() / ref_norm.sqrt()
}

const N: usize = 256;
const N_DIMS: usize = 64;
const FREQ_BASE: f32 = 10000.0;
const POS: u32 = 3;
const EPS: f32 = 1e-5;

/// Generate seeded inputs with small magnitudes (~0.5..3.0).
fn seeded_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Xorshift(0x1234_5678);
    let mut x = vec![0.0f32; N];
    let mut residual = vec![0.0f32; N];
    let mut w = vec![0.0f32; N];
    let mut up = vec![0.0f32; N];
    for i in 0..N {
        x[i] = 1.0 + rng.next_f32() * 0.5; // ~0.5..1.5
        residual[i] = 0.8 + rng.next_f32() * 0.3; // ~0.5..1.1
        w[i] = 1.2 + rng.next_f32() * 0.4; // ~0.8..1.6
        up[i] = 2.0 + rng.next_f32() * 1.0; // ~1.0..3.0
    }
    (x, residual, w, up)
}

/// Helper: uploads 4 buffers, calls NormRope::launch, copies out.
fn run_mode(
    device: &Arc<CudaDevice>,
    stream: &CudaStream,
    mode: u8,
) -> Result<Vec<f32>, CudaError> {
    let (x, residual, w, up) = seeded_inputs();
    let norm_rope = NormRope::new(Arc::clone(device))?;
    let byte_len = N * 4;

    let d_x = DeviceBuffer::alloc(Arc::clone(device), byte_len)?;
    let d_res = DeviceBuffer::alloc(Arc::clone(device), byte_len)?;
    let d_w = DeviceBuffer::alloc(Arc::clone(device), byte_len)?;
    let d_up = DeviceBuffer::alloc(Arc::clone(device), byte_len)?;
    let d_out = DeviceBuffer::alloc(Arc::clone(device), byte_len)?;

    d_x.copy_from_host(stream, &f32_bytes(&x))?;
    d_res.copy_from_host(stream, &f32_bytes(&residual))?;
    d_w.copy_from_host(stream, &f32_bytes(&w))?;
    d_up.copy_from_host(stream, &f32_bytes(&up))?;

    norm_rope.launch(
        stream, &d_x, &d_res, &d_w, &d_up, &d_out, EPS, N, N_DIMS, FREQ_BASE, POS, mode,
    )?;

    let mut out_bytes = vec![0u8; byte_len];
    d_out.copy_to_host(stream, &mut out_bytes)?;
    Ok(bytes_f32(&out_bytes))
}

#[test]
#[ignore]
fn norm_mode_matches_reference() -> Result<(), CudaError> {
    common::initialize_cuda();
    let (x, residual, w, _up) = seeded_inputs();

    // Direct reference: t[i]=x[i]+resid[i]; sum_sq in f64; mean=(float)(sum_sq/n); scale=1/sqrt(mean+eps); y[i]=t[i]*scale*w[i]
    let mut sum_sq: f64 = 0.0;
    let mut t = vec![0.0f32; N];
    for i in 0..N {
        let ti = x[i] + residual[i];
        t[i] = ti;
        let p = ti * ti;
        sum_sq += p as f64;
    }
    let mean = (sum_sq / N as f64) as f32;
    let scale = 1.0f32 / (mean + EPS).sqrt();
    let mut expected = vec![0.0f32; N];
    for i in 0..N {
        expected[i] = t[i] * scale * w[i];
    }

    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;
    let got = run_mode(&device, &stream, MODE_NORM)?;

    let cos_sim = cosine(&expected, &got);
    let l2 = rel_l2(&expected, &got);

    println!("norm mode parity: cos_sim = {cos_sim:.6}, rel_l2 = {l2:.6e}");
    assert!(
        cos_sim >= 0.9999,
        "norm mode cosine similarity {cos_sim} < 0.9999"
    );
    assert!(l2 < 0.02, "norm mode relative L2 error {l2} >= 0.02");

    Ok(())
}

#[test]
#[ignore]
fn rope_mode_matches_reference() -> Result<(), CudaError> {
    common::initialize_cuda();
    let (x, residual, _w, _up) = seeded_inputs();

    // Direct reference: out[i]=x[i]+resid[i] then rotates pairs k, k+half for k<half
    let mut expected = vec![0.0f32; N];
    for i in 0..N {
        expected[i] = x[i] + residual[i];
    }
    let half = N_DIMS / 2;
    for k in 0..half {
        let theta = POS as f32 * FREQ_BASE.powf(-2.0f32 * k as f32 / N_DIMS as f32);
        let c = theta.cos();
        let s = theta.sin();
        let x0 = expected[k];
        let x1 = expected[k + half];
        expected[k] = x0 * c - x1 * s;
        expected[k + half] = x0 * s + x1 * c;
    }

    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;
    let got = run_mode(&device, &stream, MODE_ROPE)?;

    let cos_sim = cosine(&expected, &got);
    let l2 = rel_l2(&expected, &got);

    println!("rope mode parity: cos_sim = {cos_sim:.6}, rel_l2 = {l2:.6e}");
    assert!(
        cos_sim >= 0.9999,
        "rope mode cosine similarity {cos_sim} < 0.9999"
    );
    assert!(l2 < 0.02, "rope mode relative L2 error {l2} >= 0.02");

    Ok(())
}

#[test]
#[ignore]
fn swiglu_mode_matches_reference() -> Result<(), CudaError> {
    common::initialize_cuda();
    let (x, residual, _w, up) = seeded_inputs();

    // Direct reference: out[i]=x[i]+resid[i] then out[i]=silu(out[i])*up[i]
    let mut expected = vec![0.0f32; N];
    for i in 0..N {
        let t = x[i] + residual[i];
        let silu = t / (1.0f32 + (-t).exp());
        expected[i] = silu * up[i];
    }

    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;
    let got = run_mode(&device, &stream, MODE_SWIGLU)?;

    let cos_sim = cosine(&expected, &got);
    let l2 = rel_l2(&expected, &got);

    println!("swiglu mode parity: cos_sim = {cos_sim:.6}, rel_l2 = {l2:.6e}");
    assert!(
        cos_sim >= 0.9999,
        "swiglu mode cosine similarity {cos_sim} < 0.9999"
    );
    assert!(l2 < 0.02, "swiglu mode relative L2 error {l2} >= 0.02");

    Ok(())
}

#[test]
#[ignore]
fn swiglu_batched_rows_match_reference() -> Result<(), CudaError> {
    common::initialize_cuda();
    const ROWS: usize = 2;
    let (x_row, residual_row, _w, up_row) = seeded_inputs();
    let mut x = Vec::with_capacity(ROWS * N);
    let mut residual = Vec::with_capacity(ROWS * N);
    let mut up = Vec::with_capacity(ROWS * N);
    let mut expected = Vec::with_capacity(ROWS * N);
    for row in 0..ROWS {
        let row_bias = row as f32 * 0.125;
        for i in 0..N {
            let x_value = x_row[i] + row_bias;
            let residual_value = residual_row[i] - row_bias;
            let up_value = up_row[i] + row_bias;
            x.push(x_value);
            residual.push(residual_value);
            up.push(up_value);
            let t = x_value + residual_value;
            expected.push((t / (1.0f32 + (-t).exp())) * up_value);
        }
    }

    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;
    let norm_rope = NormRope::new(Arc::clone(&device))?;
    let byte_len = x.len() * 4;
    let d_x = DeviceBuffer::alloc(Arc::clone(&device), byte_len)?;
    let d_res = DeviceBuffer::alloc(Arc::clone(&device), byte_len)?;
    let d_w = DeviceBuffer::alloc(Arc::clone(&device), N * 4)?;
    let d_up = DeviceBuffer::alloc(Arc::clone(&device), byte_len)?;
    let d_out = DeviceBuffer::alloc(Arc::clone(&device), byte_len)?;
    d_x.copy_from_host(&stream, &f32_bytes(&x))?;
    d_res.copy_from_host(&stream, &f32_bytes(&residual))?;
    d_up.copy_from_host(&stream, &f32_bytes(&up))?;

    norm_rope.launch_batched_with_pos_ptr(
        &stream,
        &d_x,
        &d_res,
        &d_w,
        &d_up,
        &d_out,
        EPS,
        N,
        N_DIMS,
        FREQ_BASE,
        POS,
        MODE_SWIGLU,
        None,
        ROWS,
        1,
    )?;

    let mut out_bytes = vec![0u8; byte_len];
    d_out.copy_to_host(&stream, &mut out_bytes)?;
    let got = bytes_f32(&out_bytes);
    let l2 = rel_l2(&expected, &got);
    assert!(l2 < 0.02, "batched SwiGLU relative L2 error {l2} >= 0.02");
    Ok(())
}
