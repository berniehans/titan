//! Q4_K_M parity gate — GPU kernel vs CPU reference (f3-gpu-dequant, task 3).
//!
//! Deterministic pseudo-random Q4_K_M super-blocks (seeded xorshift64, no
//! external crate) are dequantized by both the engine-core CPU reference
//! (`dequant_q4k_cpu`) and the engine-cuda GPU kernel (`Q4KDequantizer`),
//! then compared block-by-block. The parity gate asserts per-element max
//! absolute error < 0.01.
//!
//! Like all GPU tests it is `#[ignore]`d: run with `cargo test -- --ignored`
//! on a CUDA-capable machine (see f3-gpu-dequant PROPOSAL for the PATH trick).

use cudarc::driver::CudaDevice;
use engine_core::dequant::dequant_q4k_cpu;
use engine_cuda::{CudaError, CudaStream, DeviceBuffer, Q4KDequantizer};
use std::sync::Arc;

/// Number of Q4_K_M super-blocks generated for the parity gate.
const N_BLOCKS: usize = 4096;
/// Q4_K_M super-block size: 144 bytes of quantized data.
const BLOCK_BYTES: usize = 144;
/// Number of dequantized floats per super-block.
const FLOATS_PER_BLOCK: usize = 256;
/// Maximum per-element absolute error allowed vs the CPU reference.
const MAX_ABS_ERROR: f32 = 0.01;

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

    /// Random value in `0..bound` (bound must be non-zero).
    fn below(&mut self, bound: u64) -> u64 {
        debug_assert!(bound != 0);
        self.next_u64() % bound
    }
}

/// Builds an fp16 (LE) bit pattern in the finite normal range giving values on
/// the order of `[2^-8, 2^2)` with a random sign — realistic Q4_K_M `d`/`dmin`
/// super-scales. The borne magnitudes (d up to ~4, times scale 63 * weight 15)
/// keep dequantized products small enough that FP rounding stays well under the
/// 0.01 gate while still exercising realistic scale ranges.
fn random_fp16_bits(rng: &mut XorShift64) -> u16 {
    // sign bit ~50% of the time (yields negative scales, like real dmin)
    let sign = rng.below(2) as u16;
    // exponent field = 8..=18  => value in [2^-8, 2^4)
    let exp = (8 + rng.below(11)) as u16;
    let mant = rng.below(1 << 10) as u16;
    (sign << 15) | (exp << 10) | mant
}

/// Builds one random Q4_K_M super-block (144 bytes) using a deterministic seed.
fn random_block<const N: usize>(rng: &mut XorShift64) -> [u8; N] {
    let mut block = [0u8; N];
    // bytes[0..2] fp16 LE `d`
    block[0..2].copy_from_slice(&random_fp16_bits(rng).to_le_bytes());
    // bytes[2..4] fp16 LE `dmin`
    block[2..4].copy_from_slice(&random_fp16_bits(rng).to_le_bytes());
    // bytes[4..16] scales[12] (6-bit scales/mins), bytes[16..144] qs[128]
    for b in &mut block[4..144] {
        *b = rng.below(256) as u8;
    }
    block
}

#[test]
#[ignore]
fn q4k_parity_gpu_matches_cpu_reference() -> Result<(), CudaError> {
    let mut rng = XorShift64::new(0x5EED_1234_CAFE_BEEF);

    // 1. Build deterministic pseudo-random Q4_K_M data.
    let mut host_src = Vec::with_capacity(N_BLOCKS * BLOCK_BYTES);
    for _ in 0..N_BLOCKS {
        host_src.extend_from_slice(&random_block::<BLOCK_BYTES>(&mut rng));
    }

    // 2. CPU reference over each super-block.
    let mut cpu_out = Vec::with_capacity(N_BLOCKS * FLOATS_PER_BLOCK);
    for blk in host_src.chunks_exact(BLOCK_BYTES) {
        cpu_out.extend_from_slice(&dequant_q4k_cpu(blk));
    }
    assert_eq!(cpu_out.len(), N_BLOCKS * FLOATS_PER_BLOCK);

    // 3. Copy to device, launch kernel, read back.
    let device = CudaDevice::new(0)?;
    let stream = CudaStream::new(Arc::clone(&device))?;

    let src_dev = DeviceBuffer::alloc(Arc::clone(&device), host_src.len())?;
    let dst_bytes = cpu_out.len() * std::mem::size_of::<f32>();
    let dst_dev = DeviceBuffer::alloc(Arc::clone(&device), dst_bytes)?;

    src_dev.copy_from_host(&stream, &host_src)?;

    let dequantizer = Q4KDequantizer::new(Arc::clone(&device))?;
    dequantizer.launch(&stream, &src_dev, &dst_dev)?;

    let mut dst_raw = vec![0u8; dst_bytes];
    dst_dev.copy_to_host(&stream, &mut dst_raw)?; // syncs stream

    let gpu_out: Vec<f32> = dst_raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(gpu_out.len(), cpu_out.len());

    // 4. Block-by-block parity: max absolute error across all elements.
    let mut max_abs_error: f32 = 0.0;
    let mut worst_block = 0;
    let mut worst_elem = 0;
    let mut worst_gpu = 0.0f32;
    let mut worst_cpu = 0.0f32;

    for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
        let err = (g - c).abs();
        if err > max_abs_error {
            max_abs_error = err;
            worst_block = i / FLOATS_PER_BLOCK;
            worst_elem = i % FLOATS_PER_BLOCK;
            worst_gpu = *g;
            worst_cpu = *c;
        }
    }

    eprintln!(
        "Q4K parity gate: {} blocks, {} elements; max abs error = {max_abs_error} \
         (block {}, elem {}: GPU {} vs CPU {})",
        N_BLOCKS,
        gpu_out.len(),
        worst_block,
        worst_elem,
        worst_gpu,
        worst_cpu
    );

    assert!(
        max_abs_error < MAX_ABS_ERROR,
        "parity gate failed: max abs error {max_abs_error} >= {MAX_ABS_ERROR} \
         at block {worst_block}, elem {worst_elem} (GPU {worst_gpu} vs CPU {worst_cpu})"
    );

    Ok(())
}
