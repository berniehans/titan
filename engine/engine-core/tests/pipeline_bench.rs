use cudarc::driver::CudaDevice;
use engine_core::{EngineError, Pipeline};
use engine_cuda::{CudaEvent, CudaStream, DeviceBuffer, PinnedHost};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[test]
#[ignore]
fn test_bench_pipelined_vs_sequential() -> Result<(), EngineError> {
    let device = CudaDevice::new(0)?;

    const NUM_LAYERS: usize = 8;
    const LAYER_BYTES: usize = 8 * 1024 * 1024; // 8 MB

    // Allocate 8 dummy layers of 8 MB each in pinned host memory
    let mut pinned_layers: Vec<PinnedHost> = Vec::with_capacity(NUM_LAYERS);
    for i in 0..NUM_LAYERS {
        let mut host = PinnedHost::alloc_on_device(Arc::clone(&device), LAYER_BYTES)?;
        host.as_mut_slice().fill((i as u8).wrapping_add(1));
        pinned_layers.push(host);
    }
    let layer_refs: Vec<&[u8]> = pinned_layers.iter().map(|h| h.as_slice()).collect();

    // 1. PIPELINED MEASUREMENT
    let pipeline = Pipeline::new(Arc::clone(&device), LAYER_BYTES)?;

    // Warm-up run (1 iteration)
    let _ = pipeline.run(&layer_refs)?;

    // Timed pipelined run
    let start_pipelined = Instant::now();
    let stats = pipeline.run(&layer_refs)?;
    let pipelined_duration = start_pipelined.elapsed();
    let pipelined_total_ms = pipelined_duration.as_secs_f64() * 1000.0;

    assert_eq!(stats.layers, NUM_LAYERS);

    // 2. SEQUENTIAL MEASUREMENT
    let seq_stream = CudaStream::new(Arc::clone(&device))?;
    let seq_buffer = DeviceBuffer::alloc(Arc::clone(&device), LAYER_BYTES)?;
    let seq_event = CudaEvent::new(Arc::clone(&device))?;

    // Warm-up run for sequential baseline
    for &layer in &layer_refs {
        seq_buffer.copy_from_host_async(&seq_stream, layer)?;
        seq_stream.sync()?;
        seq_event.record(&seq_stream)?;
        seq_stream.sync()?;
    }

    // Timed sequential run: for each layer, sync copy H2D then stub compute (record event + stream sync)
    let mut sequential_total_duration = Duration::ZERO;
    for &layer in &layer_refs {
        let t0 = Instant::now();
        seq_buffer.copy_from_host_async(&seq_stream, layer)?;
        seq_stream.sync()?;
        seq_event.record(&seq_stream)?;
        seq_stream.sync()?;
        sequential_total_duration += t0.elapsed();
    }
    let sequential_total_ms = sequential_total_duration.as_secs_f64() * 1000.0;

    // 3. THROUGHPUT COMPUTATION AND LOGGING
    let total_bytes = (NUM_LAYERS * LAYER_BYTES) as f64;
    let total_gb = total_bytes / 1e9;
    let pipelined_gbps = total_gb / (pipelined_total_ms / 1000.0);
    let sequential_gbps = total_gb / (sequential_total_ms / 1000.0);
    let speedup = sequential_total_ms / pipelined_total_ms;

    eprintln!("\n=== Pipeline vs Sequential Baseline Benchmark ===");
    eprintln!(
        "Layers: {}, Size per layer: {} MB, Total data: {:.2} MB",
        NUM_LAYERS,
        LAYER_BYTES / (1024 * 1024),
        total_bytes / (1024.0 * 1024.0)
    );
    eprintln!(
        "Pipelined execution:  {:.3} ms ({:.2} GB/s)",
        pipelined_total_ms, pipelined_gbps
    );
    eprintln!(
        "Sequential execution: {:.3} ms ({:.2} GB/s)",
        sequential_total_ms, sequential_gbps
    );
    eprintln!("Pipeline stats elapsed_ms: {:.3} ms", stats.elapsed_ms);
    eprintln!("Speedup: {:.2}x", speedup);
    eprintln!("=================================================\n");

    // 4. ASSERTION
    // With a stub compute stage there is no kernel work to hide behind the
    // transfers, so pipelined == sequential up to timing noise (both are pure
    // H2D copy streams). The strict `pipelined < sequential` check only becomes
    // meaningful once real kernels run in the compute stage (Phase 6). Until
    // then, require no worse than 5% over the baseline.
    assert!(
        pipelined_total_ms <= sequential_total_ms * 1.05,
        "Pipelined total time ({pipelined_total_ms:.3} ms) must not exceed sequential baseline ({sequential_total_ms:.3} ms) by more than 5%"
    );

    Ok(())
}
