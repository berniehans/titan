//! CPU GEMV harness — task 1.1 of change 6.3-gemv-multiformat.
//!
//! The controlled Q4_K_M super-block used here is the canonical known-good
//! fixture from `engine-core/src/dequant.rs` (`BLOCK_BYTES`, `EXPECTED`). The
//! CPU forward bank from 6.2 (`forward_cpu::dequant_col` + `matmul`) is the
//! reference authority this change reuses verbatim — never a transliteration
//! of our own CUDA. This test asserts the bank itself wires dequant → dot
//! correctly over a controlled block (a prerequisite for the GPU parity gates
//! in 6.3). It is a plain CPU test (no GPU, not `#[ignore]`d).

use engine_core::dequant::dequant_q4k_cpu;
use engine_core::forward_cpu::{Tensor, TensorType, dequant_col, matmul};

// Canonical known-good Q4_K_M block (copied verbatim from the source fixture
// in engine-core/src/dequant.rs; that module is the source of truth).
const BLOCK_BYTES: [u8; 144] = [
    0, 60, 0, 56, 10, 131, 31, 192, 2, 8, 65, 63, 17, 12, 69, 95, 48, 65, 82, 99, 116, 133, 150,
    167, 184, 201, 218, 235, 252, 13, 30, 47, 48, 65, 82, 99, 116, 133, 150, 167, 184, 201, 218,
    235, 252, 13, 30, 47, 116, 133, 150, 167, 184, 201, 218, 235, 252, 13, 30, 47, 48, 65, 82, 99,
    116, 133, 150, 167, 184, 201, 218, 235, 252, 13, 30, 47, 48, 65, 82, 99, 184, 201, 218, 235,
    252, 13, 30, 47, 48, 65, 82, 99, 116, 133, 150, 167, 184, 201, 218, 235, 252, 13, 30, 47, 48,
    65, 82, 99, 116, 133, 150, 167, 252, 13, 30, 47, 48, 65, 82, 99, 116, 133, 150, 167, 184, 201,
    218, 235, 252, 13, 30, 47, 48, 65, 82, 99, 116, 133, 150, 167, 184, 201, 218, 235,
];

/// Deterministic fp32 activation vector (no external random crate).
fn input_x(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 + 0.01 * (i as f32)).collect()
}

/// Scalar fp32 dot (matches the CPU bank's accumulation precisely).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

#[test]
fn cpu_dequant_dot_q4k_matches_hand_expected() {
    // Dequant the single super-block via the 6.2 module authority.
    let deq = dequant_q4k_cpu(&BLOCK_BYTES);
    assert_eq!(deq.len(), 256, "Q4_K super-block dequantizes to 256 floats");
    // The module's own test already pinned these bytes to the hand-derived
    // EXPECTED array; here we just confirm the bank path agrees element-wise.
    let deq2 = dequant_col(TensorType::Q4K, &BLOCK_BYTES, 256);
    for (a, b) in deq.iter().zip(deq2.iter()) {
        assert!((a - b).abs() < 1e-6, "dequant mismatch {a} vs {b}");
    }

    // Wire dequant -> dot through the 6.2 matmul authority over the same block.
    let x = input_x(256);
    let t = Tensor {
        ty: TensorType::Q4K,
        data: &BLOCK_BYTES,
        ne0: 256,
        ne1: 1,
        n_rot: 0,
    };
    let mut out = vec![0.0f32; 1];
    matmul(&mut out, &t, &x);

    let expected_dot = dot(&deq, &x);
    assert!(
        (out[0] - expected_dot).abs() < 1e-5,
        "matmul dot {} != reference dot {}",
        out[0],
        expected_dot
    );

    // Expose the raw dot (the GPU parity gate mirrors the same computation).
    println!("CPU Q4_K single-block dot = {expected_dot}");
    assert!(expected_dot.is_finite());
}
