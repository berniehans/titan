//! GPU paged KV-cache test — Task 2.x of change f4-paged-kvcache.
//!
//! TDD RED phase: this file intentionally does NOT compile until the
//! `engine_cuda::paged_kv::{PagedKvGpu, PagedKvLayout}` wrapper is implemented.
//! Runs only on a CUDA-capable machine (`#[ignore]` keeps the normal suite
//! green).
//!
//! The device block pool is laid out bit-identically to the CPU flat buffer in
//! `engine-kvcache`:
//!   physical block `b`  -> float offset `b * floats_per_block`
//!   token slot `s`      -> `+ s * floats_per_token`  (key row then value row)
//!   key row             -> `+ 0 .. row_len`
//!   value row           -> `+ row_len .. 2 * row_len`
//!   `row_len = heads * head_dim` ; `floats_per_token = 2 * row_len` ;
//!   `floats_per_block = block_tokens * floats_per_token`.

mod common;

use cudarc::driver::CudaDevice;
use engine_cuda::paged_kv::{PagedKvGpu, PagedKvLayout};
use engine_cuda::{CudaError, CudaStream, DeviceBuffer};
use std::sync::Arc;

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

/// RED test for `append_kv` + paged-read gather. 12 K/V rows and a
/// physically-scattered block table are uploaded; `append_kv` writes them into
/// a device-side pool laid out like the CPU flat buffer; a gather kernel then
/// materializes a contiguous [n_tokens, row_len] view back to host. Asserts
/// each row exactly.
#[test]
#[ignore]
fn append_and_gather_paged_kv_gpu_roundtrip() -> Result<(), CudaError> {
    common::initialize_cuda();
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;

    // 2 heads * 3 dims = 6 floats/row. 4 tokens/block, 3 blocks total.
    let layout = PagedKvLayout {
        n_blocks: 3,
        block_tokens: 4,
        row_len: 6,
        data_type: engine_cuda::KvDataType::F32,
    };
    let kv = PagedKvGpu::new(Arc::clone(&device))?;

    let pool_bytes = layout.floats_total() * std::mem::size_of::<f32>();
    let pool = DeviceBuffer::alloc(Arc::clone(&device), pool_bytes)?;

    // Sequence 0: 12 logically-contiguous tokens scattered across physical
    // blocks [1, 2, 0] to prove the block-table indirection.
    let seq0_blocks = [1u32, 2, 0];
    let n_tokens = 12usize;

    // K/V rows: token t -> key row "t + h*0.125", value row "-(t) - h*0.5".
    let mut keys = vec![0.0f32; n_tokens * layout.row_len];
    let mut values = vec![0.0f32; n_tokens * layout.row_len];
    for t in 0..n_tokens {
        for h in 0..layout.row_len {
            keys[t * layout.row_len + h] = t as f32 + h as f32 * 0.125;
            values[t * layout.row_len + h] = -(t as f32) - h as f32 * 0.5;
        }
    }

    let kv_host_len = keys.len();
    let kv_dev = DeviceBuffer::alloc(Arc::clone(&device), kv_host_len * 4)?;
    let val_dev = DeviceBuffer::alloc(Arc::clone(&device), kv_host_len * 4)?;
    kv_dev.copy_from_host(&stream, &f32_bytes(&keys))?;
    val_dev.copy_from_host(&stream, &f32_bytes(&values))?;

    let block_dev = {
        let b = u32_bytes(&seq0_blocks);
        let d = DeviceBuffer::alloc(Arc::clone(&device), b.len())?;
        d.copy_from_host(&stream, &b)?;
        d
    };

    // Append K/V rows for seq 0 tokens 0..12 at logical start 0.
    kv.append_kv(
        &stream, &layout, &pool, &kv_dev, &val_dev, &block_dev, 0, n_tokens,
    )?;

    // Paged-read: materialize contiguous [n_tokens, row_len] keys and values.
    let out_bytes = n_tokens * layout.row_len * 4;
    let out_key_dev = DeviceBuffer::alloc(Arc::clone(&device), out_bytes)?;
    let out_val_dev = DeviceBuffer::alloc(Arc::clone(&device), out_bytes)?;
    kv.read_keys(
        &stream,
        &layout,
        &pool,
        &block_dev,
        &out_key_dev,
        0,
        n_tokens,
    )?;
    kv.read_values(
        &stream,
        &layout,
        &pool,
        &block_dev,
        &out_val_dev,
        0,
        n_tokens,
    )?;

    let mut key_bytes = vec![0u8; out_bytes];
    let mut val_bytes_out = vec![0u8; out_bytes];
    out_key_dev.copy_to_host(&stream, &mut key_bytes)?;
    out_val_dev.copy_to_host(&stream, &mut val_bytes_out)?;

    let got_keys = bytes_f32(&key_bytes);
    let got_values = bytes_f32(&val_bytes_out);

    assert_eq!(got_keys.len(), kv_host_len);
    assert_eq!(got_values.len(), kv_host_len);
    for i in 0..got_keys.len() {
        assert!(
            (got_keys[i] - keys[i]).abs() < 1e-5,
            "key {i}: got {} want {}",
            got_keys[i],
            keys[i]
        );
        assert!(
            (got_values[i] - values[i]).abs() < 1e-5,
            "value {i}: got {} want {}",
            got_values[i],
            values[i]
        );
    }
    Ok(())
}
