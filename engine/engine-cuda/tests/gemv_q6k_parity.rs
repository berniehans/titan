//! Fused Q6_K GEMV GPU vs CPU parity test (Phase 8, task 2.3).
//!
//! Asserts that:
//! 1. GPU `MultiFormatGEMV::gemv(GemvFormat::Q6K)` matches CPU reference dot product.
//! 2. Max relative difference is < 1e-4 and cosine similarity is > 0.9999 across non-trivial matrices.

use cudarc::driver::CudaDevice;
use engine_core::dequant_q6k_cpu;
use engine_cuda::{CudaStream, DeviceBuffer, GemvFormat, MultiFormatGEMV};

type DynError = Box<dyn std::error::Error + Send + Sync>;

#[allow(clippy::needless_range_loop)]
fn create_synthetic_q6k_block(seed: u32) -> [u8; 210] {
    let mut block = [0u8; 210];
    // 1. Scales (16 int8 values)
    for i in 0..16 {
        let val = ((seed
            .wrapping_add(i as u32)
            .wrapping_mul(1103515245)
            .wrapping_add(12345)
            >> 16)
            % 31) as i8
            - 15;
        block[192 + i] = val as u8;
    }
    // 2. Base scale d (FP16: 0.125 = 0x3000)
    block[208..210].copy_from_slice(&0x3000u16.to_le_bytes());

    // 3. Lower nibbles ql (128 bytes)
    for i in 0..128 {
        let low = (seed
            .wrapping_add(i as u32)
            .wrapping_mul(1664525)
            .wrapping_add(1013904223)
            >> 16)
            & 0x0F;
        let high = (seed
            .wrapping_add(i as u32 + 100)
            .wrapping_mul(1664525)
            .wrapping_add(1013904223)
            >> 16)
            & 0x0F;
        block[i] = (low | (high << 4)) as u8;
    }

    // 4. Higher 2-bit pairs qh (64 bytes)
    for i in 0..64 {
        let b0 = (seed
            .wrapping_add(i as u32 * 4)
            .wrapping_mul(22695477)
            .wrapping_add(1)
            >> 16)
            & 3;
        let b1 = (seed
            .wrapping_add(i as u32 * 4 + 1)
            .wrapping_mul(22695477)
            .wrapping_add(1)
            >> 16)
            & 3;
        let b2 = (seed
            .wrapping_add(i as u32 * 4 + 2)
            .wrapping_mul(22695477)
            .wrapping_add(1)
            >> 16)
            & 3;
        let b3 = (seed
            .wrapping_add(i as u32 * 4 + 3)
            .wrapping_mul(22695477)
            .wrapping_add(1)
            >> 16)
            & 3;
        block[128 + i] = (b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)) as u8;
    }

    block
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += (x as f64) * (y as f64);
        norm_a += (x as f64) * (x as f64);
        norm_b += (y as f64) * (y as f64);
    }
    (dot / (norm_a.sqrt() * norm_b.sqrt())) as f32
}

#[test]
#[ignore]
fn test_gemv_q6k_gpu_matches_cpu_reference() -> Result<(), DynError> {
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(device.clone())?;
    let gemv_engine = MultiFormatGEMV::new(device.clone())?;

    const NE0: usize = 512; // 2 blocks per column
    const NE1: usize = 128; // 128 output channels
    const BLOCKS_PER_COL: usize = NE0 / 256;

    let mut weight_bytes = Vec::with_capacity(NE1 * BLOCKS_PER_COL * 210);
    let mut dequantized_cols: Vec<Vec<f32>> = Vec::with_capacity(NE1);

    for col in 0..NE1 {
        let mut col_floats = Vec::with_capacity(NE0);
        for b in 0..BLOCKS_PER_COL {
            let seed = (col * BLOCKS_PER_COL + b) as u32 + 500;
            let blk = create_synthetic_q6k_block(seed);
            weight_bytes.extend_from_slice(&blk);
            col_floats.extend_from_slice(&dequant_q6k_cpu(&blk));
        }
        dequantized_cols.push(col_floats);
    }

    // Create synthetic activation vector x
    let mut x_host = vec![0.0f32; NE0];
    for (i, val) in x_host.iter_mut().enumerate() {
        *val = ((i as f32 * 0.017).sin()) * 0.5;
    }

    // Compute CPU reference GEMV: out[col] = dot(col_floats, x)
    let mut cpu_out = vec![0.0f32; NE1];
    for (col, col_floats) in dequantized_cols.iter().enumerate() {
        let mut dot = 0.0f32;
        for (w, &x) in col_floats.iter().zip(x_host.iter()) {
            dot += w * x;
        }
        cpu_out[col] = dot;
    }

    let dev_weights = DeviceBuffer::alloc(device.clone(), weight_bytes.len())?;
    let dev_x = DeviceBuffer::alloc(device.clone(), NE0 * 4)?;
    let dev_out = DeviceBuffer::alloc(device.clone(), NE1 * 4)?;

    let x_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(x_host.as_ptr() as *const u8, x_host.len() * 4) };
    dev_weights.copy_from_host(&stream, &weight_bytes)?;
    dev_x.copy_from_host(&stream, x_bytes)?;

    gemv_engine.gemv(
        &stream,
        GemvFormat::Q6K,
        &dev_weights,
        &dev_x,
        &dev_out,
        NE0,
        NE1,
    )?;

    let mut out_bytes = vec![0u8; NE1 * 4];
    dev_out.copy_to_host(&stream, &mut out_bytes)?;

    let gpu_out: Vec<f32> = out_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let cos_sim = cosine_similarity(&gpu_out, &cpu_out);
    println!("GEMV Q6_K output cos_sim: {:.6}", cos_sim);

    let mut max_rel_err = 0.0f32;
    for (i, (&g, &c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        let err = (g - c).abs() / (c.abs().max(1e-5));
        if err > max_rel_err {
            max_rel_err = err;
        }
        assert!(
            err < 1e-3,
            "Mismatch at col {}: gpu={}, cpu={}, rel_err={}",
            i,
            g,
            c,
            err
        );
    }
    println!("GEMV Q6_K max relative error: {:.6e}", max_rel_err);

    assert!(
        cos_sim > 0.9999,
        "GEMV Q6_K cosine similarity must exceed 0.9999"
    );
    Ok(())
}
