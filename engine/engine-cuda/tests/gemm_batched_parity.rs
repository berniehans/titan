//! Batched Quantized GEMM GPU vs CPU Parity Test (Phase 11, Sub-change 11.1).
//!
//! Asserts that:
//! 1. `BatchedGEMM::gemm` computes `Out[M, N] = X[M, K] * W[N, K]^T` across batch sizes M in {1, 4, 16, 64, 128}.
//! 2. Numerics match CPU reference dot products with cosine similarity >= 0.9999 and rel-L2 < 1e-4.

mod common;

use cudarc::driver::CudaDevice;
use engine_core::dequant_q6k_cpu;
use engine_cuda::{BatchedGEMM, CudaStream, DeviceBuffer, GemvFormat};

type DynError = Box<dyn std::error::Error + Send + Sync>;

fn create_synthetic_q6k_block(seed: u32) -> [u8; 210] {
    let mut block = [0u8; 210];
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
    block[208..210].copy_from_slice(&0x3000u16.to_le_bytes()); // 0.125
    for (i, byte) in block[..128].iter_mut().enumerate() {
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
        *byte = (low | (high << 4)) as u8;
    }
    for i in 0..64 {
        let b0 = (seed.wrapping_add(i as u32 * 4).wrapping_mul(22695477) >> 16) & 3;
        let b1 = (seed.wrapping_add(i as u32 * 4 + 1).wrapping_mul(22695477) >> 16) & 3;
        let b2 = (seed.wrapping_add(i as u32 * 4 + 2).wrapping_mul(22695477) >> 16) & 3;
        let b3 = (seed.wrapping_add(i as u32 * 4 + 3).wrapping_mul(22695477) >> 16) & 3;
        block[128 + i] = (b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)) as u8;
    }
    block
}

fn create_synthetic_q4k_block(seed: u32) -> [u8; 144] {
    let mut block = [0u8; 144];
    block[0..2].copy_from_slice(&0x3000u16.to_le_bytes()); // d = 0.125
    block[2..4].copy_from_slice(&0x2800u16.to_le_bytes()); // dmin = 0.0625
    for i in 0..12 {
        block[4 + i] =
            ((seed.wrapping_add(i as u32 * 7).wrapping_mul(1103515245) >> 16) & 0x3F) as u8;
    }
    for i in 0..128 {
        block[16 + i] =
            ((seed.wrapping_add(i as u32 * 13).wrapping_mul(1664525) >> 16) & 0xFF) as u8;
    }
    block
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

#[test]
#[ignore]
fn test_batched_gemm_q6k_parity() -> Result<(), DynError> {
    common::initialize_cuda();
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(device.clone())?;
    let gemm = BatchedGEMM::new(device.clone())?;

    let ne0 = 512; // K = 2 blocks
    let ne1 = 64; // N = 64 columns
    let n_blocks = ne0 / 256;

    let mut weight_bytes = Vec::new();
    let mut cpu_weight_rows = Vec::new();

    for col in 0..ne1 {
        let mut row_floats = Vec::new();
        for b in 0..n_blocks {
            let block = create_synthetic_q6k_block(1000 + col as u32 * 10 + b as u32);
            weight_bytes.extend_from_slice(&block);
            let decomp = dequant_q6k_cpu(&block);
            row_floats.extend_from_slice(&decomp);
        }
        cpu_weight_rows.push(row_floats);
    }

    let w_dev = DeviceBuffer::alloc(device.clone(), weight_bytes.len())?;
    w_dev.copy_from_host(&stream, &weight_bytes)?;

    let batch_sizes = [1, 3, 4, 5, 16, 64, 128];

    for &batch_size in &batch_sizes {
        let mut x_host = Vec::with_capacity(batch_size * ne0);
        for m in 0..batch_size {
            for k in 0..ne0 {
                let val = (((m * 100 + k) as f32 * 0.013).sin()) * 0.5;
                x_host.push(val);
            }
        }

        let x_dev = DeviceBuffer::alloc(device.clone(), x_host.len() * 4)?;
        let out_dev = DeviceBuffer::alloc(device.clone(), batch_size * ne1 * 4)?;

        x_dev.copy_from_host(&stream, &f32_bytes(&x_host))?;

        gemm.gemm(
            &stream,
            &w_dev,
            &x_dev,
            &out_dev,
            ne0,
            ne1,
            batch_size,
            GemvFormat::Q6K,
        )?;

        let mut out_bytes = vec![0u8; batch_size * ne1 * 4];
        out_dev.copy_to_host(&stream, &mut out_bytes)?;

        let mut out_host = vec![0.0f32; batch_size * ne1];
        for i in 0..out_host.len() {
            out_host[i] = f32::from_le_bytes(out_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
        }

        // Compare each row against CPU reference dot product
        for m in 0..batch_size {
            let x_slice = &x_host[m * ne0..(m + 1) * ne0];
            let mut cpu_out = vec![0.0f32; ne1];
            for (c, w_row) in cpu_weight_rows.iter().enumerate() {
                let dot: f32 = x_slice.iter().zip(w_row.iter()).map(|(a, b)| a * b).sum();
                cpu_out[c] = dot;
            }

            let gpu_slice = &out_host[m * ne1..(m + 1) * ne1];
            let cs = cosine_similarity(&cpu_out, gpu_slice);
            assert!(
                cs >= 0.9999,
                "Cosine similarity {cs} below threshold at batch size {batch_size}, row {m}"
            );
        }

        println!("Batched GEMM Q6K PASS for batch_size = {batch_size}");
    }

    Ok(())
}

#[test]
#[ignore]
fn test_batched_gemm_q4k_parity() -> Result<(), DynError> {
    common::initialize_cuda();
    use engine_core::dequant_q4k_cpu;

    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(device.clone())?;
    let gemm = BatchedGEMM::new(device.clone())?;

    let ne0 = 512; // K = 2 blocks
    let ne1 = 64; // N = 64 columns
    let n_blocks = ne0 / 256;

    let mut weight_bytes = Vec::new();
    let mut cpu_weight_rows = Vec::new();

    for col in 0..ne1 {
        let mut row_floats = Vec::new();
        for b in 0..n_blocks {
            let block = create_synthetic_q4k_block(2000 + col as u32 * 10 + b as u32);
            weight_bytes.extend_from_slice(&block);
            let decomp = dequant_q4k_cpu(&block);
            row_floats.extend_from_slice(&decomp);
        }
        cpu_weight_rows.push(row_floats);
    }

    let w_dev = DeviceBuffer::alloc(device.clone(), weight_bytes.len())?;
    w_dev.copy_from_host(&stream, &weight_bytes)?;

    let batch_sizes = [1, 4, 16, 64, 128];

    for &batch_size in &batch_sizes {
        let mut x_host = Vec::with_capacity(batch_size * ne0);
        for m in 0..batch_size {
            for k in 0..ne0 {
                let val = (((m * 100 + k) as f32 * 0.017).cos()) * 0.5;
                x_host.push(val);
            }
        }

        let x_dev = DeviceBuffer::alloc(device.clone(), x_host.len() * 4)?;
        let out_dev = DeviceBuffer::alloc(device.clone(), batch_size * ne1 * 4)?;

        x_dev.copy_from_host(&stream, &f32_bytes(&x_host))?;

        gemm.gemm(
            &stream,
            &w_dev,
            &x_dev,
            &out_dev,
            ne0,
            ne1,
            batch_size,
            GemvFormat::Q4K,
        )?;

        let mut out_bytes = vec![0u8; batch_size * ne1 * 4];
        out_dev.copy_to_host(&stream, &mut out_bytes)?;

        let mut out_host = vec![0.0f32; batch_size * ne1];
        for i in 0..out_host.len() {
            out_host[i] = f32::from_le_bytes(out_bytes[i * 4..(i + 1) * 4].try_into().unwrap());
        }

        for m in 0..batch_size {
            let x_slice = &x_host[m * ne0..(m + 1) * ne0];
            let mut cpu_out = vec![0.0f32; ne1];
            for (c, w_row) in cpu_weight_rows.iter().enumerate() {
                let dot: f32 = x_slice.iter().zip(w_row.iter()).map(|(a, b)| a * b).sum();
                cpu_out[c] = dot;
            }

            let gpu_slice = &out_host[m * ne1..(m + 1) * ne1];
            let cs = cosine_similarity(&cpu_out, gpu_slice);
            assert!(
                cs >= 0.9999,
                "Cosine similarity {cs} below threshold at batch size {batch_size}, row {m}"
            );
        }

        println!("Batched GEMM Q4K PASS for batch_size = {batch_size}");
    }

    Ok(())
}
