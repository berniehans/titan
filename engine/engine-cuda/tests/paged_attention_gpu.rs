//! Phase 6.5 group-2 GPU decode correctness tests (single-token identity, GQA multi-config, causal prefill).
//!
//! Validates the GPU `PagedAttention` decode kernel against the CPU `sdpa_decode` reference
//! across single-token decode, grouped-query attention (GQA) multi-configurations, and causal
//! prefill masking over scattered physical blocks. All tests are `#[ignore]` and run only on a CUDA machine.

mod common;

use common::{
    Xorshift, bytes_f32, cosine, f32_bytes, fill_pool_paged, floats_per_block_of, rel_l2, u32_bytes,
};
use cudarc::driver::CudaDevice;
use engine_core::forward_cpu::sdpa_decode;
use engine_cuda::{CudaError, CudaStream, DeviceBuffer, PagedAttention};
use std::sync::Arc;

#[allow(clippy::too_many_arguments)]
fn build_case(
    device: &Arc<CudaDevice>,
    stream: &CudaStream,
    pa: &PagedAttention,
    block_tokens: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    n_tokens: usize,
    seed: u32,
    causal: bool,
    query_pos: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), CudaError> {
    let num_blocks = 4;
    assert!(
        n_tokens <= num_blocks * block_tokens,
        "n_tokens ({n_tokens}) exceeds capacity ({} * {})",
        num_blocks,
        block_tokens
    );
    let floats_per_block = floats_per_block_of(block_tokens, n_head_kv, head_dim);
    let mut pool = vec![0.0f32; num_blocks * floats_per_block];
    let block_table = [3u32, 0u32, 2u32, 1u32];

    fill_pool_paged(
        &mut pool,
        &block_table,
        block_tokens,
        n_head_kv,
        head_dim,
        seed,
        n_tokens,
    );

    let mut q_rng = Xorshift::new(seed ^ 0xF00D);
    let query: Vec<f32> = (0..n_head * head_dim).map(|_| q_rng.next_f32()).collect();

    let query_pos_sdpa = if causal { query_pos } else { n_tokens };
    let cpu = sdpa_decode(
        &pool,
        &block_table,
        block_tokens,
        n_tokens,
        &query,
        n_head,
        n_head_kv,
        head_dim,
        causal,
        query_pos_sdpa,
    );

    let q_bytes = f32_bytes(&query);
    let pool_bytes = f32_bytes(&pool);
    let bt_bytes = u32_bytes(&block_table);
    let out_byte_len = n_head * head_dim * 4;

    let d_q = DeviceBuffer::alloc(Arc::clone(device), q_bytes.len())?;
    let d_pool = DeviceBuffer::alloc(Arc::clone(device), pool_bytes.len())?;
    let d_bt = DeviceBuffer::alloc(Arc::clone(device), bt_bytes.len())?;
    let d_out = DeviceBuffer::alloc(Arc::clone(device), out_byte_len)?;

    d_q.copy_from_host(stream, &q_bytes)?;
    d_pool.copy_from_host(stream, &pool_bytes)?;
    d_bt.copy_from_host(stream, &bt_bytes)?;

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
        n_tokens,
        query_pos_sdpa,
        causal,
    )?;

    let mut out_bytes = vec![0u8; out_byte_len];
    d_out.copy_to_host(stream, &mut out_bytes)?;
    let gpu = bytes_f32(&out_bytes);

    Ok((gpu, cpu, query))
}

#[test]
#[ignore]
fn decode_single_token_returns_value_row() -> Result<(), CudaError> {
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;
    let pa = PagedAttention::new(Arc::clone(&device))?;

    let block_tokens = 4;
    let n_head = 2;
    let n_head_kv = 1;
    let head_dim = 4;
    let n_tokens = 1;
    let causal = false;
    let query_pos = 1;

    let (gpu, cpu, _query) = build_case(
        &device,
        &stream,
        &pa,
        block_tokens,
        n_head,
        n_head_kv,
        head_dim,
        n_tokens,
        0x1234,
        causal,
        query_pos,
    )?;

    let cos_sim = cosine(&gpu, &cpu);
    let l2 = rel_l2(&gpu, &cpu);
    assert!(cos_sim >= 0.9999, "cosine similarity {cos_sim} < 0.9999");
    assert!(l2 < 1e-5, "relative L2 error {l2} >= 1e-5");

    Ok(())
}

#[test]
#[ignore]
fn gqa_multiconfig_decode_finite_and_matches_cpu() -> Result<(), CudaError> {
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;
    let pa = PagedAttention::new(Arc::clone(&device))?;

    let configs = [(4, 1, 64, 16), (8, 2, 32, 16), (4, 4, 16, 8), (2, 1, 8, 4)];

    for &(n_head, n_head_kv, head_dim, block_tokens) in &configs {
        for n_tokens in [1, block_tokens * 3 + 3] {
            let (gpu, cpu, _query) = build_case(
                &device,
                &stream,
                &pa,
                block_tokens,
                n_head,
                n_head_kv,
                head_dim,
                n_tokens,
                0xABCD,
                false,
                n_tokens,
            )?;

            assert!(
                gpu.iter().all(|x| x.is_finite()),
                "config ({n_head},{n_head_kv},{head_dim},{block_tokens}) n_tokens={n_tokens}: GPU output contains non-finite values"
            );
            let cos_sim = cosine(&gpu, &cpu);
            assert!(
                cos_sim >= 0.9999,
                "config ({n_head},{n_head_kv},{head_dim},{block_tokens}) n_tokens={n_tokens}: cos_sim {cos_sim} < 0.9999"
            );
        }
    }

    Ok(())
}

#[test]
#[ignore]
fn causal_prefill_matches_cpu_reference() -> Result<(), CudaError> {
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;
    let pa = PagedAttention::new(Arc::clone(&device))?;

    let block_tokens = 4;
    let n_head = 4;
    let n_head_kv = 2;
    let head_dim = 16;
    let n_tokens = 3 * block_tokens + 2;
    let causal = true;
    let query_pos = 2 * block_tokens + 1;

    let (gpu, cpu, _query) = build_case(
        &device,
        &stream,
        &pa,
        block_tokens,
        n_head,
        n_head_kv,
        head_dim,
        n_tokens,
        0xCAFE,
        causal,
        query_pos,
    )?;

    let cos_sim = cosine(&gpu, &cpu);
    let l2 = rel_l2(&gpu, &cpu);
    assert!(cos_sim >= 0.9999, "cosine similarity {cos_sim} < 0.9999");
    assert!(l2 < 1e-4, "relative L2 error {l2} >= 1e-4");

    Ok(())
}
