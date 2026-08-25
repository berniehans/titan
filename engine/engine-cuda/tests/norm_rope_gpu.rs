//! GPU fused RMSNorm / RoPE / SwiGLU parity gate (Phase 6.4).
//!
//! Deterministic seeded xorshift input data is processed through the GPU
//! `NormRope::launch` (fused mode) and compared against a hand-computed CPU
//! reference oracle. Cosine similarity must be >= 0.9999 and relative L2 error
//! must be < 0.02. Runs only on a CUDA machine (`#[ignore]`).

use cudarc::driver::CudaDevice;
use engine_cuda::{CudaError, CudaStream, DeviceBuffer, MODE_FUSED, NormRope};
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

#[test]
#[ignore]
fn test_fused_norm_rope_swiglu_parity() -> Result<(), CudaError> {
    const N: usize = 256;
    const N_DIMS: usize = 64;
    const FREQ_BASE: f32 = 10000.0;
    const POS: u32 = 5;
    const EPS: f32 = 1e-5;

    let mut rng = Xorshift(0x1234_5678);
    let mut x = vec![0.0f32; N];
    let mut residual = vec![0.0f32; N];
    let mut w = vec![0.0f32; N];
    let mut up = vec![0.0f32; N];

    for i in 0..N {
        x[i] = rng.next_f32() * 0.1;
        residual[i] = rng.next_f32() * 0.1;
        w[i] = 1.0 + rng.next_f32() * 0.1;
        up[i] = rng.next_f32() * 0.1;
    }

    // --- Host-side CPU fused oracle computed by hand ---
    // 1. RMSNorm + residual addition
    let mut sum_sq: f64 = 0.0;
    let mut tmp = vec![0.0f32; N];
    for i in 0..N {
        let t = x[i] + residual[i];
        tmp[i] = t;
        let p = t * t;
        sum_sq += p as f64;
    }
    let mean = (sum_sq / N as f64) as f32;
    let scale = 1.0f32 / (mean + EPS).sqrt();
    let mut y = vec![0.0f32; N];
    for i in 0..N {
        y[i] = tmp[i] * scale * w[i];
    }

    // 2. NeoX partial RoPE rotation in-place
    let half = N_DIMS / 2;
    for k in 0..half {
        let theta = POS as f32 * FREQ_BASE.powf(-2.0f32 * k as f32 / N_DIMS as f32);
        let c = theta.cos();
        let s = theta.sin();
        let x0 = y[k];
        let x1 = y[k + half];
        y[k] = x0 * c - x1 * s;
        y[k + half] = x0 * s + x1 * c;
    }

    // 3. SwiGLU gating: silu(y) * up
    let mut expected = vec![0.0f32; N];
    for i in 0..N {
        let silu = y[i] / (1.0f32 + (-y[i]).exp());
        expected[i] = silu * up[i];
    }

    // --- GPU execution (4 DeviceBuffers total) ---
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;
    let norm_rope = NormRope::new(Arc::clone(&device))?;

    let byte_len = N * 4;
    let d_x = DeviceBuffer::alloc(Arc::clone(&device), byte_len)?;
    let d_res = DeviceBuffer::alloc(Arc::clone(&device), byte_len)?;
    let d_w = DeviceBuffer::alloc(Arc::clone(&device), byte_len)?;
    let d_up = DeviceBuffer::alloc(Arc::clone(&device), byte_len)?;

    d_x.copy_from_host(&stream, &f32_bytes(&x))?;
    d_res.copy_from_host(&stream, &f32_bytes(&residual))?;
    d_w.copy_from_host(&stream, &f32_bytes(&w))?;
    d_up.copy_from_host(&stream, &f32_bytes(&up))?;

    norm_rope.launch(
        &stream, &d_x, &d_res, &d_w, &d_up, &d_x, EPS, N, N_DIMS, FREQ_BASE, POS, MODE_FUSED,
    )?;

    let mut out_bytes = vec![0u8; byte_len];
    d_x.copy_to_host(&stream, &mut out_bytes)?;
    let got = bytes_f32(&out_bytes);

    let cos_sim = cosine(&expected, &got);
    let l2 = rel_l2(&expected, &got);

    println!("fused norm/rope/swiglu parity: cos_sim = {cos_sim:.6}, rel_l2 = {l2:.6e}");
    assert!(cos_sim >= 0.9999, "cosine similarity {cos_sim} < 0.9999");
    assert!(l2 < 0.02, "relative L2 error {l2} >= 0.02");

    Ok(())
}
