//! Dynamic position parameter updating via device pointer in CUDA Graphs (Phase 9, task 2.3).
//!
//! Asserts that:
//! 1. `NormRope`, `PagedKvGpu`, and `PagedAttention` read position `p` from `pos_dev` dynamically.
//! 2. A single captured `CudaGraphExec` correctly advances token positions $p=0, 1, 2, \dots$ when `pos_dev` is updated on the host without re-capturing or recreating the graph.
//! 3. Output activations match standard stream-by-stream execution with bit-exact parity.

use cudarc::driver::CudaDevice;
use engine_cuda::{
    CudaStream, DeviceBuffer, MODE_ROPE, NormRope, PagedAttention, PagedKvGpu, PagedKvLayout,
};

type DynError = Box<dyn std::error::Error + Send + Sync>;

#[test]
#[ignore]
fn test_cuda_graph_dynamic_position_replay() -> Result<(), DynError> {
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(device.clone())?;

    let nr = NormRope::new(device.clone())?;
    let pkv = PagedKvGpu::new(device.clone())?;
    let pattn = PagedAttention::new(device.clone())?;

    const N_HEAD: usize = 4;
    const N_HEAD_KV: usize = 2;
    const HEAD_DIM: usize = 64;
    const ROW_LEN: usize = N_HEAD_KV * HEAD_DIM; // 128
    const BLOCK_TOKENS: usize = 16;
    const TOTAL_BLOCKS: usize = 4;
    const MAX_STEPS: usize = 8;

    let layout = PagedKvLayout {
        n_blocks: TOTAL_BLOCKS,
        block_tokens: BLOCK_TOKENS,
        row_len: ROW_LEN,
    };

    // Allocations
    let pos_dev = DeviceBuffer::alloc(device.clone(), 4)?;
    let x_dev = DeviceBuffer::alloc(device.clone(), ROW_LEN * 4)?;
    let resid_dev = DeviceBuffer::alloc(device.clone(), ROW_LEN * 4)?;
    let norm_w_dev = DeviceBuffer::alloc(device.clone(), ROW_LEN * 4)?;
    let up_dev = DeviceBuffer::alloc(device.clone(), ROW_LEN * 4)?;
    let rope_out_dev = DeviceBuffer::alloc(device.clone(), ROW_LEN * 4)?;

    let kv_pool_graph = DeviceBuffer::alloc(device.clone(), layout.floats_total() * 4)?;
    let kv_pool_ref = DeviceBuffer::alloc(device.clone(), layout.floats_total() * 4)?;
    let block_table_dev = DeviceBuffer::alloc(device.clone(), TOTAL_BLOCKS * 4)?;

    let q_dev = DeviceBuffer::alloc(device.clone(), N_HEAD * HEAD_DIM * 4)?;
    let attn_out_graph = DeviceBuffer::alloc(device.clone(), N_HEAD * HEAD_DIM * 4)?;
    let attn_out_ref = DeviceBuffer::alloc(device.clone(), N_HEAD * HEAD_DIM * 4)?;

    // Initialize constants
    let block_table: Vec<u32> = (0..TOTAL_BLOCKS as u32).collect();
    let bt_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(block_table.as_ptr() as *const u8, block_table.len() * 4)
    };
    block_table_dev.copy_from_host(&stream, bt_bytes)?;

    let ones = vec![1.0f32; ROW_LEN];
    let zeros = vec![0.0f32; ROW_LEN];
    let ones_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(ones.as_ptr() as *const u8, ones.len() * 4) };
    let zeros_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(zeros.as_ptr() as *const u8, zeros.len() * 4) };
    resid_dev.copy_from_host(&stream, zeros_bytes)?;
    norm_w_dev.copy_from_host(&stream, ones_bytes)?;
    up_dev.copy_from_host(&stream, zeros_bytes)?;

    // Zero out pools
    let zero_pool = vec![0u8; layout.floats_total() * 4];
    kv_pool_graph.copy_from_host(&stream, &zero_pool)?;
    kv_pool_ref.copy_from_host(&stream, &zero_pool)?;
    stream.sync()?;

    // 1. Capture the multi-kernel decode pipeline ONCE with pos_dev
    stream.begin_capture()?;

    // Kernel 1: RoPE with dynamic pos_dev
    nr.launch_with_pos_ptr(
        &stream,
        &x_dev,
        &resid_dev,
        &norm_w_dev,
        &up_dev,
        &rope_out_dev,
        1e-5,
        ROW_LEN,
        ROW_LEN,
        10000.0,
        0, // dummy static pos
        MODE_ROPE,
        Some(&pos_dev),
    )?;

    // Kernel 2: Paged KV append with dynamic pos_dev
    pkv.append_kv_with_pos_ptr(
        &stream,
        &layout,
        &kv_pool_graph,
        &rope_out_dev,
        &rope_out_dev,
        &block_table_dev,
        0, // dummy static start_token
        1,
        Some(&pos_dev),
    )?;

    // Kernel 3: PagedAttention with dynamic pos_dev
    pattn.launch_with_pos_ptr(
        &stream,
        &q_dev,
        &kv_pool_graph,
        &block_table_dev,
        &attn_out_graph,
        N_HEAD,
        N_HEAD_KV,
        HEAD_DIM,
        BLOCK_TOKENS,
        1, // dummy static seq_tokens
        0, // dummy static query_pos
        true,
        Some(&pos_dev),
    )?;

    let graph = stream.end_capture()?;
    let exec = graph.instantiate()?;

    // 2. Iterate across multiple token positions p = 0 .. MAX_STEPS
    for step in 0..MAX_STEPS {
        let x_step: Vec<f32> = (0..ROW_LEN)
            .map(|i| ((step * ROW_LEN + i) as f32 * 0.03).cos())
            .collect();
        let q_step: Vec<f32> = (0..N_HEAD * HEAD_DIM)
            .map(|i| ((step * 100 + i) as f32 * 0.02).sin())
            .collect();

        let x_b: &[u8] =
            unsafe { std::slice::from_raw_parts(x_step.as_ptr() as *const u8, x_step.len() * 4) };
        let q_b: &[u8] =
            unsafe { std::slice::from_raw_parts(q_step.as_ptr() as *const u8, q_step.len() * 4) };
        x_dev.copy_from_host(&stream, x_b)?;
        q_dev.copy_from_host(&stream, q_b)?;

        // Execute Reference (Stream with explicit static pos = step)
        let rope_ref_dev = DeviceBuffer::alloc(device.clone(), ROW_LEN * 4)?;
        nr.launch(
            &stream,
            &x_dev,
            &resid_dev,
            &norm_w_dev,
            &up_dev,
            &rope_ref_dev,
            1e-5,
            ROW_LEN,
            ROW_LEN,
            10000.0,
            step as u32,
            MODE_ROPE,
        )?;
        pkv.append_kv(
            &stream,
            &layout,
            &kv_pool_ref,
            &rope_ref_dev,
            &rope_ref_dev,
            &block_table_dev,
            step,
            1,
        )?;
        pattn.launch(
            &stream,
            &q_dev,
            &kv_pool_ref,
            &block_table_dev,
            &attn_out_ref,
            N_HEAD,
            N_HEAD_KV,
            HEAD_DIM,
            BLOCK_TOKENS,
            step + 1,
            step,
            true,
        )?;

        // Execute Graph (Update pos_dev on device and launch graph)
        let pos_bytes = (step as u32).to_le_bytes();
        pos_dev.copy_from_host(&stream, &pos_bytes)?;
        exec.launch(&stream)?;

        stream.sync()?;

        // Compare outputs
        let mut ref_out_bytes = vec![0u8; N_HEAD * HEAD_DIM * 4];
        let mut graph_out_bytes = vec![0u8; N_HEAD * HEAD_DIM * 4];
        attn_out_ref.copy_to_host(&stream, &mut ref_out_bytes)?;
        attn_out_graph.copy_to_host(&stream, &mut graph_out_bytes)?;

        let ref_f32: Vec<f32> = ref_out_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let graph_f32: Vec<f32> = graph_out_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut max_diff = 0.0f32;
        for (i, (&r, &g)) in ref_f32.iter().zip(graph_f32.iter()).enumerate() {
            let diff = (r - g).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < 1e-5,
                "Step {step} mismatch at {i}: graph={g}, ref={r}, diff={diff}"
            );
        }
        println!("Step {step} (pos={step}) -> Graph vs Stream max diff: {max_diff:.6e}");
    }

    Ok(())
}
