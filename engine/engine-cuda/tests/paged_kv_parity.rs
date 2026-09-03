//! Parity gate — Task 3.x of change f4-paged-kvcache.
//!
//! Deterministic seeded xorshift K/V data is appended to the CPU reference
//! (`engine_kvcache::PagedKvCache`) and, independently, written through the GPU
//! `PagedKvGpu::append_kv` + paged-read gather kernels. Logical token order is
//! checked block-by-block (per physical block of the block table); max absolute
//! error per element must stay `< 0.01`. Runs only on a CUDA machine
//! (`#[ignore]`).

mod common;

use cudarc::driver::CudaDevice;
use engine_cuda::paged_kv::{PagedKvGpu, PagedKvLayout};
use engine_cuda::{CudaError, CudaStream, DeviceBuffer};
use engine_kvcache::cache::{PagedKvCache, PagedKvCacheConfig};
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

fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_f32(v: &[u8]) -> Vec<f32> {
    v.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Generate `n_tokens` deterministic K/V rows from a seed.
fn generate_rows(seed: u32, n_tokens: usize, row_len: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Xorshift(seed);
    let mut keys = vec![0.0f32; n_tokens * row_len];
    let mut vals = vec![0.0f32; n_tokens * row_len];
    for t in 0..n_tokens {
        for h in 0..row_len {
            keys[t * row_len + h] = rng.next_f32();
            vals[t * row_len + h] = rng.next_f32();
        }
    }
    (keys, vals)
}

/// Parity gate: CPU reference vs GPU, block by block, on seeded xorshift data.
/// Two sequences are interleaved so the parity target is scattered across
/// physically non-adjacent blocks.
#[test]
#[ignore]
fn gpu_append_read_matches_cpu_block_by_block() -> Result<(), CudaError> {
    common::initialize_cuda();
    const HEAD_DIM: usize = 8;
    const HEADS: usize = 2;
    const BLOCK_TOKENS: usize = 6;
    const N_BLOCKS: usize = 9;

    let row_len = HEADS * HEAD_DIM; // 16 floats/row
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;
    let kv = PagedKvGpu::new(Arc::clone(&device))?;

    // --- CPU reference (source of truth for allocation + layout) ---
    let cfg = PagedKvCacheConfig {
        n_blocks: N_BLOCKS,
        block_tokens: BLOCK_TOKENS,
        heads: HEADS,
        head_dim: HEAD_DIM,
    };
    let mut cpu = PagedKvCache::new(cfg).expect("cpu pool");
    let seq_a = cpu.new_sequence();
    let seq_b = cpu.new_sequence();

    // seq_a steals 5 blocks first so seq_b's blocks become physically scattered.
    let n_a = 5 * BLOCK_TOKENS;
    let (ka, va) = generate_rows(0xDEAD_BEEF, n_a, row_len);
    for t in 0..n_a {
        cpu.append(
            seq_a,
            &ka[t * row_len..(t + 1) * row_len],
            &va[t * row_len..(t + 1) * row_len],
        )
        .expect("append seq_a");
    }

    // seq_b is the parity target: 3 full blocks + a partial to cross a boundary.
    let n_b = 3 * BLOCK_TOKENS + 4;
    let (keys_b, vals_b) = generate_rows(0x1234_5678, n_b, row_len);
    for t in 0..n_b {
        cpu.append(
            seq_b,
            &keys_b[t * row_len..(t + 1) * row_len],
            &vals_b[t * row_len..(t + 1) * row_len],
        )
        .expect("append seq_b");
    }

    let block_table_b: Vec<u32> = cpu
        .seq_block_table(seq_b)
        .expect("seq_b block table")
        .to_vec();
    assert!(
        block_table_b.len() > 1,
        "expected seq_b to span multiple blocks, got {:?}",
        block_table_b
    );

    // CPU reference contiguous gather.
    let keys_ref = cpu.read_keys(seq_b).expect("cpu read keys");
    let vals_ref = cpu.read_values(seq_b).expect("cpu read values");
    assert_eq!(keys_ref.len(), n_b * row_len);
    assert_eq!(vals_ref.len(), n_b * row_len);

    // --- GPU path: same data, same layout, same block table ---
    let layout = PagedKvLayout {
        n_blocks: N_BLOCKS,
        block_tokens: BLOCK_TOKENS,
        row_len,
        data_type: engine_cuda::KvDataType::F32,
    };
    let pool_bytes = layout.floats_total() * std::mem::size_of::<f32>();
    let pool = DeviceBuffer::alloc(Arc::clone(&device), pool_bytes)?;

    let kv_dev = DeviceBuffer::alloc(Arc::clone(&device), keys_b.len() * 4)?;
    let val_dev = DeviceBuffer::alloc(Arc::clone(&device), vals_b.len() * 4)?;
    kv_dev.copy_from_host(&stream, &f32_bytes(&keys_b))?;
    val_dev.copy_from_host(&stream, &f32_bytes(&vals_b))?;

    let block_dev = {
        let b = u32_bytes(&block_table_b);
        let d = DeviceBuffer::alloc(Arc::clone(&device), b.len())?;
        d.copy_from_host(&stream, &b)?;
        d
    };

    kv.append_kv(
        &stream, &layout, &pool, &kv_dev, &val_dev, &block_dev, 0, n_b,
    )?;

    let out_bytes = n_b * row_len * 4;
    let out_k = DeviceBuffer::alloc(Arc::clone(&device), out_bytes)?;
    let out_v = DeviceBuffer::alloc(Arc::clone(&device), out_bytes)?;
    kv.read_keys(&stream, &layout, &pool, &block_dev, &out_k, 0, n_b)?;
    kv.read_values(&stream, &layout, &pool, &block_dev, &out_v, 0, n_b)?;

    let mut k_host = vec![0u8; out_bytes];
    let mut v_host = vec![0u8; out_bytes];
    out_k.copy_to_host(&stream, &mut k_host)?;
    out_v.copy_to_host(&stream, &mut v_host)?;
    let got_k = bytes_f32(&k_host);
    let got_v = bytes_f32(&v_host);

    // --- Block-by-block parity: GPU vs CPU, max abs err < 0.01/elem ---
    assert_eq!(got_k.len(), keys_ref.len());
    assert_eq!(got_v.len(), vals_ref.len());

    // Compare the GPU gather output (logical token order) against CPU, and
    // separately confirm progress is per-block for reporting.
    let mut max_err_k = 0.0f32;
    let mut max_err_v = 0.0f32;
    let mut start = 0usize;
    for &phys in &block_table_b {
        let n_tok_in_block = (n_b - start).min(BLOCK_TOKENS);
        for _t in 0..n_tok_in_block {
            let base = start * row_len;
            for h in 0..row_len {
                let ek = (got_k[base + h] - keys_ref[base + h]).abs();
                let ev = (got_v[base + h] - vals_ref[base + h]).abs();
                max_err_k = max_err_k.max(ek);
                max_err_v = max_err_v.max(ev);
                assert!(
                    ek < 0.01,
                    "key err in phys block {phys} at logickal token {} (h{}): got {} cpu {}",
                    start,
                    h,
                    got_k[base + h],
                    keys_ref[base + h]
                );
                assert!(
                    ev < 0.01,
                    "value err in phys block {phys} at logical token {} (h{}): got {} cpu {}",
                    start,
                    h,
                    got_v[base + h],
                    vals_ref[base + h]
                );
            }
        }
        start += n_tok_in_block;
    }

    // Redundant cross-check: whole-buffer equality (upper bound signal).
    for i in 0..keys_ref.len() {
        assert!((got_k[i] - keys_ref[i]).abs() < 0.01, "key {i} out of tol");
        assert!(
            (got_v[i] - vals_ref[i]).abs() < 0.01,
            "value {i} out of tol"
        );
    }

    println!(
        "parity: {n_b} seq_b tokens across {} phys blocks; max abs err keys={max_err_k:.3e}, values={max_err_v:.3e} (limit 0.01/elem)",
        block_table_b.len()
    );
    Ok(())
}
