//! Pipeline compute-stage dequantization test (f3-gpu-dequant, task 4).
//!
//! TDD RED phase: this test intentionally does NOT compile until `Pipeline`
//! exposes a dequantizer-enabled constructor and `dequant_out_slot`, and the
//! compute stage launches `Q4KDequantizer` after the `copy_done` wait. Runs
//! only on a CUDA-capable machine (`#[ignore]`).

use cudarc::driver::CudaDevice;
use engine_core::dequant::dequant_q4k_cpu;
use engine_core::{EngineError, Pipeline};
use engine_cuda::DeviceBuffer;
use std::sync::Arc;

/// Layers each carry this many Q4_K_M super-blocks (kept small for speed).
const BLOCKS_PER_LAYER: usize = 64;
/// Number of dequantized floats per super-block.
const FLOATS_PER_BLOCK: usize = 256;
const NUM_LAYERS: usize = 4;

/// Deterministic xorshift64 PRNG (no external crate).
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        XorShift64 { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

/// One random fp16 (LE) `d`/`dmin` bit pattern: random sign, exponent 8..=18.
fn random_fp16_bits(rng: &mut XorShift64) -> u16 {
    let sign = rng.below(2) as u16;
    let exp = (8 + rng.below(11)) as u16;
    let mant = rng.below(1 << 10) as u16;
    (sign << 15) | (exp << 10) | mant
}

/// One deterministic Q4_K_M super-block (144 bytes).
fn random_block(rng: &mut XorShift64) -> [u8; 144] {
    let mut block = [0u8; 144];
    block[0..2].copy_from_slice(&random_fp16_bits(rng).to_le_bytes());
    block[2..4].copy_from_slice(&random_fp16_bits(rng).to_le_bytes());
    for b in &mut block[4..144] {
        *b = rng.below(256) as u8;
    }
    block
}

/// Raw Q4_K_M bytes for `layer` (deterministic per-layer input data).
fn layer_bytes(layer: usize) -> Vec<u8> {
    let mut rng = XorShift64::new(0x5EED_0001 ^ (layer as u64));
    let mut v = Vec::with_capacity(BLOCKS_PER_LAYER * 144);
    for _ in 0..BLOCKS_PER_LAYER {
        v.extend_from_slice(&random_block(&mut rng));
    }
    v
}

/// CPU reference dequantization of `layer_bytes(layer)`.
fn cpu_reference(layer: usize) -> Vec<f32> {
    let bytes = layer_bytes(layer);
    let mut out = Vec::with_capacity(BLOCKS_PER_LAYER * FLOATS_PER_BLOCK);
    for blk in bytes.chunks_exact(144) {
        out.extend_from_slice(&dequant_q4k_cpu(blk));
    }
    out
}

/// Reads `n_floats` f32 values back from a device buffer.
fn read_f32s(
    buf: &DeviceBuffer,
    stream: &engine_cuda::CudaStream,
    n: usize,
) -> Result<Vec<f32>, EngineError> {
    let mut raw = vec![0u8; n * std::mem::size_of::<f32>()];
    buf.copy_to_host(stream, &mut raw)?;
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[test]
#[ignore]
fn pipeline_with_dequantizer_produces_float_outputs_per_layer() -> Result<(), EngineError> {
    let device = CudaDevice::new(0)?;
    let layer_size = layer_bytes(0).len();
    let max_layer_bytes = layer_size; // one layer's worth per slot

    let pipeline = Pipeline::with_dequantizer(Arc::clone(&device), max_layer_bytes)?;

    // Deterministic per-layer sand inputs.
    let layers: Vec<Vec<u8>> = (0..NUM_LAYERS).map(layer_bytes).collect();
    let layer_refs: Vec<&[u8]> = layers.iter().map(|l| l.as_slice()).collect();

    let stats = pipeline.run(&layer_refs)?;
    assert_eq!(stats.layers, NUM_LAYERS);
    assert!(
        pipeline.dequantizer_enabled(),
        "dequantizer should be enabled"
    );

    // After NUM_LAYERS (even) layers: slot 0 holds layer 2, slot 1 holds layer 3.
    let out0 = pipeline.dequant_out_slot(0).expect("slot 0 output");
    let out1 = pipeline.dequant_out_slot(1).expect("slot 1 output");
    let n_floats = BLOCKS_PER_LAYER * FLOATS_PER_BLOCK;

    let gpu_slot0 = read_f32s(out0, pipeline.transfer_stream(), n_floats)?;
    let gpu_slot1 = read_f32s(out1, pipeline.transfer_stream(), n_floats)?;

    let exp2 = cpu_reference(2);
    let exp3 = cpu_reference(3);

    let compare = |gpu: &[f32], exp: &[f32], label: &str| {
        assert_eq!(gpu.len(), exp.len(), "{label}: length mismatch");
        for (i, (g, e)) in gpu.iter().zip(exp.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-5,
                "{label} elem {i}: GPU {g} != expected {e}"
            );
        }
    };

    compare(&gpu_slot0, &exp2, "slot0 (dequant layer 2)");
    compare(&gpu_slot1, &exp3, "slot1 (dequant layer 3)");

    Ok(())
}
