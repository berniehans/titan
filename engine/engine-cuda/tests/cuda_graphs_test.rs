//! CUDA Graphs capture and launch test (Phase 9, task 1.3).
//!
//! Asserts that:
//! 1. `CudaStream::begin_capture()` and `CudaStream::end_capture()` capture multi-kernel pipelines.
//! 2. `CudaGraph::instantiate()` creates an executable graph `CudaGraphExec`.
//! 3. `CudaGraphExec::launch()` executes all captured kernels with 100% numerical parity against standard stream execution.

use cudarc::driver::CudaDevice;
use engine_cuda::{CudaStream, DeviceBuffer, GemvFormat, MODE_NORM, MultiFormatGEMV, NormRope};

type DynError = Box<dyn std::error::Error + Send + Sync>;

#[test]
#[ignore]
fn test_cuda_graph_capture_and_launch_parity() -> Result<(), DynError> {
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(device.clone())?;

    let nr = NormRope::new(device.clone())?;
    let gemv = MultiFormatGEMV::new(device.clone())?;

    const DIM: usize = 256;
    const OUT_DIM: usize = 128;
    const N_BLOCKS: usize = DIM / 256;

    // 1. Prepare synthetic input data
    let x_host: Vec<f32> = (0..DIM).map(|i| (i as f32 * 0.05).sin()).collect();
    let weight_bytes = vec![0x11u8; OUT_DIM * N_BLOCKS * 144]; // Synthetic Q4_K
    let norm_w_host = vec![1.0f32; DIM];
    let zero_host = vec![0.0f32; DIM];

    let x_dev = DeviceBuffer::alloc(device.clone(), DIM * 4)?;
    let norm_w_dev = DeviceBuffer::alloc(device.clone(), DIM * 4)?;
    let zero_dev = DeviceBuffer::alloc(device.clone(), DIM * 4)?;
    let norm_out_graph = DeviceBuffer::alloc(device.clone(), DIM * 4)?;
    let gemv_out_graph = DeviceBuffer::alloc(device.clone(), OUT_DIM * 4)?;
    let w_dev = DeviceBuffer::alloc(device.clone(), weight_bytes.len())?;

    let x_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(x_host.as_ptr() as *const u8, x_host.len() * 4) };
    let norm_w_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(norm_w_host.as_ptr() as *const u8, norm_w_host.len() * 4)
    };
    let zero_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(zero_host.as_ptr() as *const u8, zero_host.len() * 4) };

    x_dev.copy_from_host(&stream, x_bytes)?;
    norm_w_dev.copy_from_host(&stream, norm_w_bytes)?;
    zero_dev.copy_from_host(&stream, zero_bytes)?;
    w_dev.copy_from_host(&stream, &weight_bytes)?;
    stream.sync()?;

    // 2. Standard stream execution (reference)
    let norm_out_ref = DeviceBuffer::alloc(device.clone(), DIM * 4)?;
    let gemv_out_ref = DeviceBuffer::alloc(device.clone(), OUT_DIM * 4)?;

    nr.launch(
        &stream,
        &x_dev,
        &zero_dev,
        &norm_w_dev,
        &norm_out_ref,
        &norm_out_ref,
        1e-5,
        DIM,
        0,
        10000.0,
        0,
        MODE_NORM,
    )?;
    gemv.gemv(
        &stream,
        GemvFormat::Q4K,
        &w_dev,
        &norm_out_ref,
        &gemv_out_ref,
        DIM,
        OUT_DIM,
    )?;
    stream.sync()?;

    let mut ref_bytes = vec![0u8; OUT_DIM * 4];
    gemv_out_ref.copy_to_host(&stream, &mut ref_bytes)?;

    let ref_out: Vec<f32> = ref_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // 3. Graph capture of the exact same multi-kernel sequence
    stream.begin_capture()?;

    nr.launch(
        &stream,
        &x_dev,
        &zero_dev,
        &norm_w_dev,
        &norm_out_graph,
        &norm_out_graph,
        1e-5,
        DIM,
        0,
        10000.0,
        0,
        MODE_NORM,
    )?;
    gemv.gemv(
        &stream,
        GemvFormat::Q4K,
        &w_dev,
        &norm_out_graph,
        &gemv_out_graph,
        DIM,
        OUT_DIM,
    )?;

    let graph = stream.end_capture()?;
    let exec = graph.instantiate()?;

    // 4. Launch executable graph and verify output
    exec.launch(&stream)?;
    stream.sync()?;

    let mut graph_bytes = vec![0u8; OUT_DIM * 4];
    gemv_out_graph.copy_to_host(&stream, &mut graph_bytes)?;

    let graph_out: Vec<f32> = graph_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    assert_eq!(ref_out.len(), graph_out.len());
    let mut max_diff = 0.0f32;
    for (i, (&r, &g)) in ref_out.iter().zip(graph_out.iter()).enumerate() {
        let diff = (r - g).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        assert!(
            diff < 1e-6,
            "Graph vs Stream mismatch at {i}: graph={g}, ref={r}, diff={diff}"
        );
    }
    println!("CUDA Graph vs Stream max difference: {:.6e}", max_diff);
    assert!(max_diff < 1e-6, "CUDA Graph output must be bit-identical");

    // 5. Verify multiple graph launches (replay capability)
    for _ in 0..5 {
        exec.launch(&stream)?;
    }
    stream.sync()?;

    Ok(())
}
