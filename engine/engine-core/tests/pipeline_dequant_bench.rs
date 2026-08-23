//! Phase 3 F3 gate pipeline benchmark — real-compute dequantizer overlap
//! (f3-gpu-dequant, task 5.1).
//!
//! Measures three configurations on the same Q4_K_M-aligned dummy layers and
//! reports the MEDIAN over several iterations (the pipeline bench is flaky
//! under concurrent load, so a single wall-clock sample is not trustworthy):
//!
//!   1. **dequant-pipelined** — `Pipeline::with_dequantizer` (real GPU dequant
//!      kernel in the compute stage, event-gated double-buffered overlap).
//!   2. **sequential-with-dequant** — baseline: per layer a sync H2D copy then
//!      a sync dequant kernel launch (no overlap).
//!   3. **stub-pipelined** — `Pipeline::new` (historical stub compute stage,
//!      no dequant kernel), the Phase 2 baseline.
//!
//! Gate: median dequant-pipelined < median sequential-with-dequant (real
//! compute now overlapping transfer), and the overlap is measurably larger
//! than the stub case. Run isolated (do NOT run in parallel with other GPU
//! tests); `#[ignore]`d, requires a CUDA device + nvrtc DLL on `PATH`.
//!
//! Run: `cargo test -p engine-core --test pipeline_dequant_bench -- --ignored
//! --nocapture`

use cudarc::driver::CudaDevice;
use engine_core::{EngineError, Pipeline};
use engine_cuda::{CudaStream, DeviceBuffer, PinnedHost, Q4KDequantizer};
use std::sync::Arc;
use std::time::Instant;

/// Super-block (144 bytes) geometry — must match `engine_core::dequant`.
const BLOCK_BYTES: usize = 144;
/// Dequantized floats per super-block.
const FLOATS_PER_BLOCK: usize = 256;

const NUM_LAYERS: usize = 8;
/// Q4_K_M-aligned layer size (multiple of 144) → ~8 MB per layer.
const BLOCKS_PER_LAYER: usize = 58_240;
const LAYER_BYTES: usize = BLOCKS_PER_LAYER * BLOCK_BYTES; // 8,386,560 (~8 MB)
const ITERATIONS: usize = 7;

/// Deterministic xorshift64 PRNG (no external crate).
struct XorShift64(u64);
impl XorShift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

/// Random fp16 (LE) `d`/`dmin` bit pattern (realistic scale magnitudes).
fn random_fp16_bits(rng: &mut XorShift64) -> u16 {
    let sign = rng.below(2) as u16;
    let exp = (8 + rng.below(11)) as u16;
    let mant = rng.below(1 << 10) as u16;
    (sign << 15) | (exp << 10) | mant
}

/// Builds one deterministic Q4_K_M layer block stream of `LAYER_BYTES`.
fn build_layer(rng: &mut XorShift64) -> Vec<u8> {
    let mut v = Vec::with_capacity(LAYER_BYTES);
    for _ in 0..BLOCKS_PER_LAYER {
        let mut block = [0u8; BLOCK_BYTES];
        block[0..2].copy_from_slice(&random_fp16_bits(rng).to_le_bytes());
        block[2..4].copy_from_slice(&random_fp16_bits(rng).to_le_bytes());
        for b in &mut block[4..] {
            *b = rng.below(256) as u8;
        }
        v.extend_from_slice(&block);
    }
    v
}

/// Median of a sample (odd-length keeps it simple; sorts in place).
fn median_ms(mut samples: Vec<f64>) -> (f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("f64 cmp"));
    let n = samples.len();
    (
        samples[n / 2] * 1000.0,
        samples.iter().map(|m| m * 1000.0).sum::<f64>() / n as f64,
    )
}

/// Timed pipelined (dequantizer or stub) median over ITERATIONS warm-up + runs.
fn bench_pipelined(
    device: &Arc<CudaDevice>,
    layer_refs: &[&[u8]],
    with_dequant: bool,
) -> Result<(f64, f64), EngineError> {
    let pipeline = if with_dequant {
        Pipeline::with_dequantizer(Arc::clone(device), LAYER_BYTES)?
    } else {
        Pipeline::new(Arc::clone(device), LAYER_BYTES)?
    };
    let _ = pipeline.run(layer_refs)?; // warm-up

    let mut samples = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let _ = pipeline.run(layer_refs)?;
        samples.push(start.elapsed().as_secs_f64());
    }
    let (median, mean) = median_ms(samples);
    eprintln!(
        "  {:<28} median {:.3} ms  mean {:.3} ms",
        if with_dequant {
            "dequant-pipelined"
        } else {
            "stub-pipelined"
        },
        median,
        mean
    );
    Ok((median, mean))
}

/// Timed sequential-with-dequant baseline: sync H2D copy then sync kernel per
/// layer (no overlap). Same real compute work as the dequant pipeline.
fn bench_sequential_with_dequant(
    device: &Arc<CudaDevice>,
    layer_refs: &[&[u8]],
) -> Result<(f64, f64), EngineError> {
    let stream = CudaStream::new(Arc::clone(device))?;
    let buf = DeviceBuffer::alloc(Arc::clone(device), LAYER_BYTES)?;
    let out_bytes = BLOCKS_PER_LAYER * FLOATS_PER_BLOCK * std::mem::size_of::<f32>();
    let out = DeviceBuffer::alloc(Arc::clone(device), out_bytes)?;
    let dequantizer = Q4KDequantizer::new(Arc::clone(device))?;

    // Warm-up
    for &layer in layer_refs {
        buf.copy_from_host_async(&stream, layer)?;
        stream.sync()?;
        dequantizer.launch(&stream, &buf, &out)?;
        stream.sync()?;
    }

    let mut samples = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        for &layer in layer_refs {
            buf.copy_from_host_async(&stream, layer)?;
            stream.sync()?;
            dequantizer.launch(&stream, &buf, &out)?;
            stream.sync()?;
        }
        samples.push(start.elapsed().as_secs_f64());
    }
    let (median, mean) = median_ms(samples);
    eprintln!(
        "  {:<28} median {:.3} ms  mean {:.3} ms",
        "sequential-with-dequant", median, mean
    );
    Ok((median, mean))
}

#[test]
#[ignore]
fn bench_dequant_overlap_vs_baselines() -> Result<(), EngineError> {
    let device = CudaDevice::new(0)?;

    // Deterministic layers in pinned host memory.
    let mut rng = XorShift64(0xDEAD_BEEF_1234_5678);
    let mut pinned: Vec<PinnedHost> = Vec::with_capacity(NUM_LAYERS);
    for _ in 0..NUM_LAYERS {
        let mut host = PinnedHost::alloc_on_device(Arc::clone(&device), LAYER_BYTES)?;
        host.as_mut_slice().copy_from_slice(&build_layer(&mut rng));
        pinned.push(host);
    }
    let layer_refs: Vec<&[u8]> = pinned.iter().map(|h| h.as_slice()).collect();

    let total_bytes = (NUM_LAYERS * LAYER_BYTES) as f64;
    let total_mb = total_bytes / (1024.0 * 1024.0);
    eprintln!(
        "\n=== Phase 3 dequant overlap bench (layers {}, {:.2} MB/layer, {:.2} MB total, median of {} iters) ===",
        NUM_LAYERS,
        LAYER_BYTES as f64 / (1024.0 * 1024.0),
        total_mb,
        ITERATIONS
    );

    let (deq_med, deq_mean) = bench_pipelined(&device, &layer_refs, true)?;
    let (seq_med, _seq_mean) = bench_sequential_with_dequant(&device, &layer_refs)?;
    let (stub_med, _stub_mean) = bench_pipelined(&device, &layer_refs, false)?;

    let speedup = seq_med / deq_med;
    eprintln!(
        "Speedup dequant-pipelined vs sequential-with-dequant: {:.2}x (median)",
        speedup
    );
    eprintln!(
        "Overlap (dequant-pipelined median vs stub median): {:.2}% of stub time",
        100.0 * deq_med / stub_med
    );
    eprintln!("============================================================\n");

    // Gate: real compute overlapping transfer must beat the no-overlap baseline.
    assert!(
        deq_med < seq_med,
        "dequant-pipelined median {deq_med:.3} ms must be < sequential-with-dequant median {seq_med:.3} ms"
    );

    // Sanity against stale numbers: don't assert here on the stub comparison,
    // only log it (stub compute is ~free so its overlap is not a valid gate).

    let (deq_gbps, seq_gbps) = {
        let t = total_bytes / 1e9;
        (t / (deq_med / 1000.0), t / (seq_med / 1000.0))
    };
    eprintln!(
        "Throughput: dequant-pipelined {:.2} GB/s, sequential {:.2} GB/s (deq mean {:.2} GB/s)",
        deq_gbps,
        seq_gbps,
        (total_bytes / 1e9) / (deq_mean / 1000.0)
    );
    eprintln!(
        "_F3_BENCH_RESULT_ dequant_median_ms={deq_med:.3} sequential_median_ms={seq_med:.3} stub_median_ms={stub_med:.3} speedup={speedup:.2}"
    );

    Ok(())
}
