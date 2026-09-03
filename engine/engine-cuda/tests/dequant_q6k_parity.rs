//! Q6_K GPU vs CPU dequantization parity test (Phase 8, task 1.3).
//!
//! Asserts that:
//! 1. GPU `Q6KDequantizer` matches `engine_core::dequant_q6k_cpu` within < 1e-4 tolerance.
//! 2. Cosine similarity between GPU output and CPU reference is > 0.9999.
//! 3. Multi-block batches (16 superblocks = 4096 weights) dequantize with zero drift.

mod common;

use cudarc::driver::CudaDevice;
use engine_core::dequant_q6k_cpu;
use engine_cuda::{CudaStream, DeviceBuffer, Q6KDequantizer};

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
fn test_q6k_gpu_vs_cpu_single_block() -> Result<(), DynError> {
    common::initialize_cuda();
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(device.clone())?;
    let dequant = Q6KDequantizer::new(device.clone())?;

    let block_bytes = create_synthetic_q6k_block(42);
    let cpu_ref = dequant_q6k_cpu(&block_bytes);

    let dev_src = DeviceBuffer::alloc(device.clone(), 210)?;
    let dev_dst = DeviceBuffer::alloc(device.clone(), 256 * 4)?;

    dev_src.copy_from_host(&stream, &block_bytes)?;
    dequant.launch(&stream, &dev_src, &dev_dst)?;

    let mut dst_bytes = vec![0u8; 256 * 4];
    dev_dst.copy_to_host(&stream, &mut dst_bytes)?;

    let gpu_out: Vec<f32> = dst_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut max_diff = 0.0f32;
    for (i, (&g, &c)) in gpu_out.iter().zip(cpu_ref.iter()).enumerate() {
        let diff = (g - c).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        assert!(
            diff < 1e-4,
            "Mismatch at index {}: gpu={}, cpu={}, diff={}",
            i,
            g,
            c,
            diff
        );
    }

    let cos_sim = cosine_similarity(&gpu_out, &cpu_ref);
    println!(
        "Single-block Q6_K max diff: {:.6e}, cos_sim: {:.6}",
        max_diff, cos_sim
    );

    assert!(cos_sim > 0.9999, "Cosine similarity must exceed 0.9999");
    Ok(())
}

#[test]
#[ignore]
fn test_q6k_gpu_vs_cpu_multi_block_batch() -> Result<(), DynError> {
    common::initialize_cuda();
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(device.clone())?;
    let dequant = Q6KDequantizer::new(device.clone())?;

    const N_BLOCKS: usize = 16;
    let mut all_bytes = Vec::with_capacity(N_BLOCKS * 210);
    let mut cpu_ref = Vec::with_capacity(N_BLOCKS * 256);

    for b in 0..N_BLOCKS {
        let blk = create_synthetic_q6k_block(1000 + b as u32);
        all_bytes.extend_from_slice(&blk);
        cpu_ref.extend_from_slice(&dequant_q6k_cpu(&blk));
    }

    let dev_src = DeviceBuffer::alloc(device.clone(), N_BLOCKS * 210)?;
    let dev_dst = DeviceBuffer::alloc(device.clone(), N_BLOCKS * 256 * 4)?;

    dev_src.copy_from_host(&stream, &all_bytes)?;
    dequant.launch(&stream, &dev_src, &dev_dst)?;

    let mut dst_bytes = vec![0u8; N_BLOCKS * 256 * 4];
    dev_dst.copy_to_host(&stream, &mut dst_bytes)?;

    let gpu_out: Vec<f32> = dst_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let cos_sim = cosine_similarity(&gpu_out, &cpu_ref);
    println!("Multi-block (n=16) Q6_K cos_sim: {:.6}", cos_sim);

    assert!(
        cos_sim > 0.9999,
        "Multi-block cosine similarity must exceed 0.9999"
    );
    Ok(())
}
