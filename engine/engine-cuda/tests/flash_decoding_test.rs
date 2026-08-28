use cudarc::driver::CudaDevice;
use engine_cuda::paged_attention::PagedAttention;
use engine_cuda::streams::CudaStream;
use engine_cuda::{ensure_cuda_dll_paths, DeviceBuffer};
use std::sync::Arc;

#[test]
#[ignore]
fn test_flash_decoding_compilation_and_execution() -> Result<(), Box<dyn std::error::Error>> {
    ensure_cuda_dll_paths();
    let dev = CudaDevice::new(0)?;
    let stream = CudaStream::new(dev.clone())?;
    let pa = PagedAttention::new(dev.clone())?;

    let n_head = 32;
    let n_head_kv = 8;
    let head_dim = 128;
    let block_tokens = 16;
    let seq_tokens = 1024; // 1024 tokens = 4 splits of 256 tokens

    let q = DeviceBuffer::alloc(dev.clone(), n_head * head_dim * 4)?;
    let floats_per_block = block_tokens * 2 * n_head_kv * head_dim;
    let num_blocks = (seq_tokens + block_tokens - 1) / block_tokens;
    let pool = DeviceBuffer::alloc(dev.clone(), num_blocks * floats_per_block * 4)?;
    let block_table = DeviceBuffer::alloc(dev.clone(), num_blocks * 4)?;
    let out = DeviceBuffer::alloc(dev.clone(), n_head * head_dim * 4)?;

    // Fill block table with identity mapping [0, 1, 2, ...]
    let bt_host: Vec<u32> = (0..num_blocks as u32).collect();
    let bt_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(bt_host.as_ptr() as *const u8, bt_host.len() * 4)
    };
    block_table.copy_from_host(&stream, bt_bytes)?;

    pa.launch_flash_decoding(
        &stream,
        &q,
        &pool,
        &block_table,
        &out,
        n_head,
        n_head_kv,
        head_dim,
        block_tokens,
        seq_tokens,
        seq_tokens - 1,
        true,
        None,
    )?;

    stream.sync()?;
    println!("Successfully compiled and executed FlashDecoding Split-KV Attention (1024 tokens) on GPU!");

    Ok(())
}