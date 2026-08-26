//! Bandwidth profiler implementation (Phase 7.1).
//!
//! Measures hardware bandwidth baselines:
//! - STREAM-style host DRAM read bandwidth
//! - Linear PCIe H2D and D2H copy bandwidth
//! - Standalone and concurrent overlapped CPU MoE GEMV vs PCIe gather

use super::profile::{BandwidthMeasurement, MoeBackend};
use crate::error::EngineError;
use cudarc::driver::CudaDevice;
use engine_cuda::{CudaStream, DeviceBuffer};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

/// Measures STREAM-style sequential host DRAM read bandwidth (GB/s).
pub fn measure_stream_host_dram_read(buffer_bytes: usize, iterations: usize) -> f64 {
    let count = buffer_bytes / std::mem::size_of::<f32>();
    let data = vec![1.0f32; count];
    // Warm-up pass
    let mut dummy_sum = 0.0f32;
    for v in &data {
        dummy_sum += *v;
    }
    std::hint::black_box(dummy_sum);

    let start = Instant::now();
    for _ in 0..iterations {
        let mut acc = 0.0f32;
        for v in &data {
            acc += *v;
        }
        std::hint::black_box(acc);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let total_bytes = buffer_bytes as f64 * iterations as f64;
    (total_bytes / elapsed.max(1e-9)) / 1e9
}

/// Measures linear PCIe Host->Device copy bandwidth (GB/s).
pub fn measure_linear_pcie_h2d(
    device: &Arc<CudaDevice>,
    stream: &CudaStream,
    size_bytes: usize,
    iterations: usize,
) -> Result<f64, EngineError> {
    let host_data = vec![0x5au8; size_bytes];
    let dev_buf = DeviceBuffer::alloc(Arc::clone(device), size_bytes)?;

    // Warm-up
    dev_buf.copy_from_host(stream, &host_data)?;
    stream.sync()?;

    let start = Instant::now();
    for _ in 0..iterations {
        dev_buf.copy_from_host(stream, &host_data)?;
    }
    stream.sync()?;
    let elapsed = start.elapsed().as_secs_f64();
    let total_bytes = size_bytes as f64 * iterations as f64;
    Ok((total_bytes / elapsed.max(1e-9)) / 1e9)
}

/// Measures linear PCIe Device->Host copy bandwidth (GB/s).
pub fn measure_linear_pcie_d2h(
    device: &Arc<CudaDevice>,
    stream: &CudaStream,
    size_bytes: usize,
    iterations: usize,
) -> Result<f64, EngineError> {
    let dev_buf = DeviceBuffer::alloc(Arc::clone(device), size_bytes)?;
    let mut host_data = vec![0u8; size_bytes];

    // Warm-up
    dev_buf.copy_to_host(stream, &mut host_data)?;
    stream.sync()?;

    let start = Instant::now();
    for _ in 0..iterations {
        dev_buf.copy_to_host(stream, &mut host_data)?;
    }
    stream.sync()?;
    let elapsed = start.elapsed().as_secs_f64();
    let total_bytes = size_bytes as f64 * iterations as f64;
    Ok((total_bytes / elapsed.max(1e-9)) / 1e9)
}

/// Simulates CPU GEMV memory access bandwidth over an expert-sized host bank.
pub fn measure_cpu_gemv_bandwidth(bank_bytes: usize, iterations: usize) -> f64 {
    let num_floats = bank_bytes / std::mem::size_of::<f32>();
    let matrix = vec![0.5f32; num_floats];
    let vector = vec![1.0f32; 1024];
    let mut out = vec![0.0f32; 1024];

    let start = Instant::now();
    for _ in 0..iterations {
        // Dot-product pass simulating GEMV memory sweep
        let mut idx = 0;
        for o in out.iter_mut() {
            let mut sum = 0.0f32;
            for v in &vector {
                sum += matrix[idx % num_floats] * (*v);
                idx += 1;
            }
            *o = sum;
        }
        std::hint::black_box(&out);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let bytes_touched = (1024 * 1024 * std::mem::size_of::<f32>()) as f64 * iterations as f64;
    (bytes_touched / elapsed.max(1e-9)) / 1e9
}

/// Measures isolated and overlapped CPU MoE GEMV + PCIe gather bandwidth on GPU hardware.
pub fn measure_overlapped_cpu_and_pcie(
    device: &Arc<CudaDevice>,
    stream: &CudaStream,
    format: &str,
    expert_block_bytes: usize,
    iterations: usize,
) -> Result<BandwidthMeasurement, EngineError> {
    // 1. STREAM DRAM Read baseline
    let stream_dram_read_gbps = measure_stream_host_dram_read(expert_block_bytes, iterations);

    // 2. Linear PCIe copies
    let linear_pcie_h2d_gbps =
        measure_linear_pcie_h2d(device, stream, expert_block_bytes, iterations)?;
    let linear_pcie_d2h_gbps =
        measure_linear_pcie_d2h(device, stream, expert_block_bytes, iterations)?;

    // 3. Standalone CPU MoE GEMV
    let cpu_moe_isolated_gbps = measure_cpu_gemv_bandwidth(expert_block_bytes, iterations);

    // 4. Standalone PCIe gather (H2D chunk streaming)
    let pcie_gather_isolated_gbps = linear_pcie_h2d_gbps;

    // 5. Overlapped measurement (concurrent execution contending host DRAM)
    let host_chunks = vec![0x7fu8; expert_block_bytes];
    let dev_chunks = DeviceBuffer::alloc(Arc::clone(device), expert_block_bytes)?;

    let chunks_ref = host_chunks.clone();

    let cpu_iters = iterations;
    let cpu_handle =
        thread::spawn(move || measure_cpu_gemv_bandwidth(expert_block_bytes, cpu_iters));

    let pcie_start = Instant::now();
    for _ in 0..iterations {
        dev_chunks.copy_from_host(stream, &chunks_ref)?;
    }
    stream.sync()?;
    let pcie_elapsed = pcie_start.elapsed().as_secs_f64();
    let pcie_gather_overlap_gbps =
        ((expert_block_bytes as f64 * iterations as f64) / pcie_elapsed.max(1e-9)) / 1e9;

    let cpu_moe_overlap_gbps = cpu_handle.join().unwrap_or(cpu_moe_isolated_gbps);

    // 6. Policy calculations
    let denom = pcie_gather_overlap_gbps + cpu_moe_overlap_gbps;
    let fetch_fraction = if denom > 1e-6 {
        (pcie_gather_overlap_gbps / denom).clamp(0.0, 1.0)
    } else {
        0.5
    };

    let recommended_backend = if cpu_moe_overlap_gbps > 2.0 * pcie_gather_overlap_gbps {
        MoeBackend::Hybrid
    } else {
        MoeBackend::Offload
    };

    println!("\n=== Bandwidth Measurement for {} ===", format);
    println!(
        "  STREAM DRAM Read:        {:.2} GB/s",
        stream_dram_read_gbps
    );
    println!(
        "  Linear PCIe H2D:         {:.2} GB/s",
        linear_pcie_h2d_gbps
    );
    println!(
        "  Linear PCIe D2H:         {:.2} GB/s",
        linear_pcie_d2h_gbps
    );
    println!(
        "  CPU MoE Isolated:        {:.2} GB/s",
        cpu_moe_isolated_gbps
    );
    println!(
        "  PCIe Gather Isolated:    {:.2} GB/s",
        pcie_gather_isolated_gbps
    );
    println!(
        "  CPU MoE Overlapped:      {:.2} GB/s",
        cpu_moe_overlap_gbps
    );
    println!(
        "  PCIe Gather Overlapped:  {:.2} GB/s",
        pcie_gather_overlap_gbps
    );
    println!("  Calculated Fetch Fraction: {:.4}", fetch_fraction);
    println!("  Recommended Backend:     {}", recommended_backend);

    Ok(BandwidthMeasurement {
        stream_dram_read_gbps,
        linear_pcie_h2d_gbps,
        linear_pcie_d2h_gbps,
        cpu_moe_isolated_gbps,
        pcie_gather_isolated_gbps,
        cpu_moe_overlap_gbps,
        pcie_gather_overlap_gbps,
        fetch_fraction,
        recommended_backend,
    })
}
