//! Phase 6.5 group-3 GPU-vs-CPU scattered parity (1..2048 tokens) and zero-runtime-cudaMalloc tests.
//!
//! Validates the GPU `PagedAttention` decode kernel against the CPU `sdpa_decode` reference
//! across sequence lengths from 1 to 2048 tokens with non-trivial GQA head configurations
//! over scattered physical blocks, and asserts that the launch path executes with zero runtime
//! `cudaMalloc` allocations. All tests are `#[ignore]` and run only on a CUDA machine.

mod common;

use common::{
    Xorshift, bytes_f32, cosine, f32_bytes, fill_pool_paged, floats_per_block_of, u32_bytes,
};

use cudarc::driver::CudaDevice;
use engine_core::forward_cpu::sdpa_decode;
use engine_cuda::{CudaError, CudaStream, DeviceBuffer, PagedAttention};
use std::sync::Arc;

#[test]
#[ignore]
fn scattered_parity_1_to_2048() -> Result<(), CudaError> {
    common::initialize_cuda();
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;
    let pa = PagedAttention::new(Arc::clone(&device))?;

    let num_blocks = 4;
    let block_table = [3u32, 0u32, 2u32, 1u32];

    let run_sweep = |device: &Arc<CudaDevice>,
                     stream: &CudaStream,
                     pa: &PagedAttention,
                     block_tokens: usize,
                     n_head: usize,
                     n_head_kv: usize,
                     head_dim: usize,
                     base_seed: u32|
     -> Result<(usize, f64), CudaError> {
        let capacity = num_blocks * block_tokens;
        let floats_per_block = floats_per_block_of(block_tokens, n_head_kv, head_dim);
        let mut pool_host = vec![0.0f32; num_blocks * floats_per_block];

        let mut q_rng = Xorshift::new(base_seed ^ 0xC0FFEE);
        let query: Vec<f32> = (0..n_head * head_dim).map(|_| q_rng.next_f32()).collect();

        let q_bytes = f32_bytes(&query);
        let bt_bytes = u32_bytes(&block_table);
        let out_byte_len = n_head * head_dim * 4;

        let d_q = DeviceBuffer::alloc(Arc::clone(device), q_bytes.len())?;
        let d_pool = DeviceBuffer::alloc(Arc::clone(device), pool_host.len() * 4)?;
        let d_bt = DeviceBuffer::alloc(Arc::clone(device), bt_bytes.len())?;
        let d_out = DeviceBuffer::alloc(Arc::clone(device), out_byte_len)?;

        d_q.copy_from_host(stream, &q_bytes)?;
        d_bt.copy_from_host(stream, &bt_bytes)?;

        let allocs_before = DeviceBuffer::live_allocations();

        let mut seqs: Vec<usize> = (1..=64).collect();
        seqs.extend([128, 192, 256]);
        seqs.extend((256..=2048).step_by(64));
        seqs.extend([511, 512, 513, 1024, 2047, 2048]);
        seqs.retain(|&s| s <= capacity);
        seqs.sort_unstable();
        seqs.dedup();

        let mut min_cos = 1.0f64;
        let mut max_seq = 0usize;
        let mut out_bytes = vec![0u8; out_byte_len];

        for seq in seqs {
            pool_host.fill(0.0f32);
            fill_pool_paged(
                &mut pool_host,
                &block_table,
                block_tokens,
                n_head_kv,
                head_dim,
                base_seed,
                seq,
            );
            d_pool.copy_from_host(stream, &f32_bytes(&pool_host))?;

            let cpu = sdpa_decode(
                &pool_host,
                &block_table,
                block_tokens,
                seq,
                &query,
                n_head,
                n_head_kv,
                head_dim,
                false,
                seq,
            );

            pa.launch(
                stream,
                &d_q,
                &d_pool,
                &d_bt,
                &d_out,
                n_head,
                n_head_kv,
                head_dim,
                block_tokens,
                seq,
                seq,
                false,
            )?;

            d_out.copy_to_host(stream, &mut out_bytes)?;
            let gpu = bytes_f32(&out_bytes);

            let cos = cosine(&gpu, &cpu);
            assert!(
                cos >= 0.9999,
                "cos {cos} < 0.9999 at seq={seq} (block_tokens={block_tokens}, n_head={n_head}, n_head_kv={n_head_kv}, head_dim={head_dim})"
            );
            min_cos = min_cos.min(cos);
            max_seq = max_seq.max(seq);
        }

        let allocs_after = DeviceBuffer::live_allocations();
        assert_eq!(
            allocs_after, allocs_before,
            "live allocations changed: before={allocs_before}, after={allocs_after}"
        );

        Ok((max_seq, min_cos))
    };

    let (max_seq_a, min_cos_a) = run_sweep(&device, &stream, &pa, 512, 8, 4, 16, 0x5EED)?;
    assert_eq!(max_seq_a, 2048, "Config A max_seq must be 2048");
    assert!(max_seq_a >= 2048, "Config A max_seq must be >= 2048");
    assert!(min_cos_a >= 0.9999, "Config A min_cos {min_cos_a} < 0.9999");

    let (max_seq_b, min_cos_b) = run_sweep(&device, &stream, &pa, 256, 4, 2, 32, 0xAAA5)?;
    assert_eq!(max_seq_b, 1024, "Config B max_seq must be 1024");
    assert!(min_cos_b >= 0.9999, "Config B min_cos {min_cos_b} < 0.9999");

    println!(
        "scattered parity: configA({min_cos_a}, {max_seq_a}) configB({min_cos_b}, {max_seq_b})"
    );

    Ok(())
}

#[test]
#[ignore]
fn launch_path_zero_runtime_cudamalloc() -> Result<(), CudaError> {
    common::initialize_cuda();
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;
    let pa = PagedAttention::new(Arc::clone(&device))?;

    let block_tokens = 64;
    let n_head = 2;
    let n_head_kv = 1;
    let head_dim = 8;
    let seq_tokens = 200;
    let num_blocks = 4;
    let block_table = [3u32, 0u32, 2u32, 1u32];
    let seed = 0xBEEF;

    let floats_per_block = floats_per_block_of(block_tokens, n_head_kv, head_dim);
    let mut pool = vec![0.0f32; num_blocks * floats_per_block];
    fill_pool_paged(
        &mut pool,
        &block_table,
        block_tokens,
        n_head_kv,
        head_dim,
        seed,
        seq_tokens,
    );

    let mut q_rng = Xorshift::new(seed ^ 0xC0FFEE);
    let query: Vec<f32> = (0..n_head * head_dim).map(|_| q_rng.next_f32()).collect();

    let q_bytes = f32_bytes(&query);
    let pool_bytes = f32_bytes(&pool);
    let bt_bytes = u32_bytes(&block_table);
    let out_byte_len = n_head * head_dim * 4;

    let d_q = DeviceBuffer::alloc(Arc::clone(&device), q_bytes.len())?;
    let d_pool = DeviceBuffer::alloc(Arc::clone(&device), pool_bytes.len())?;
    let d_bt = DeviceBuffer::alloc(Arc::clone(&device), bt_bytes.len())?;
    let d_out = DeviceBuffer::alloc(Arc::clone(&device), out_byte_len)?;

    d_q.copy_from_host(&stream, &q_bytes)?;
    d_pool.copy_from_host(&stream, &pool_bytes)?;
    d_bt.copy_from_host(&stream, &bt_bytes)?;

    let before = DeviceBuffer::live_allocations();
    pa.launch(
        &stream,
        &d_q,
        &d_pool,
        &d_bt,
        &d_out,
        n_head,
        n_head_kv,
        head_dim,
        block_tokens,
        seq_tokens,
        seq_tokens,
        false,
    )?;
    stream.sync()?;
    let after = DeviceBuffer::live_allocations();
    assert_eq!(
        after, before,
        "launch path allocated memory: before={before} after={after}"
    );

    let mut out_bytes = vec![0u8; out_byte_len];
    d_out.copy_to_host(&stream, &mut out_bytes)?;
    let gpu = bytes_f32(&out_bytes);
    let cpu = sdpa_decode(
        &pool,
        &block_table,
        block_tokens,
        seq_tokens,
        &query,
        n_head,
        n_head_kv,
        head_dim,
        false,
        seq_tokens,
    );

    let cos = cosine(&gpu, &cpu);
    assert!(cos >= 0.9999, "cosine {cos} < 0.9999");

    println!("zero-cudaMalloc: before={before} after={after}");
    Ok(())
}
