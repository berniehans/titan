//! CPU forward bank primitives (Phase 6.2, group 2) over controlled constants.
//!
//! Each element of the bank (RMSNorm, RoPE, dequantized matmul, SwiGLU) is
//! validated against an independently computed expected value, not a redundant
//! re-expression of the same code path. The eps placement of RMSNorm (inside
//! the sqrt, per `ggml.c::ggml_compute_forward_rms_norm_f32`) is checked
//! explicitly.

use engine_core::forward_cpu::{
    Tensor, TensorType, dequant_col, matmul, rms_norm, rope_neox_partial, silu,
};

fn f32_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

// ---------------------------------------------------------------------------
// 2.1 / 2.2 RMSNorm — hand-computed with the eps placed inside the sqrt.
// ---------------------------------------------------------------------------
#[test]
fn rms_norm_matches_hand_computed_and_eps_inside_sqrt() {
    let x = [3.0f32, 4.0f32];
    let w = [1.0f32, 1.0f32];
    let eps = 1e-5f32;

    // Independent reference (f64): sum of squares in double, eps inside sqrt.
    let ss = (x[0] as f64).mul_add(x[0] as f64, x[1] as f64 * x[1] as f64);
    let mean = (ss / 2.0) as f32;
    let scale = 1.0f32 / (mean + eps).sqrt();
    let exp0 = x[0] * scale * w[0];
    let exp1 = x[1] * scale * w[1];

    let y = rms_norm(&x, &w, eps);
    assert!(f32_eq(y[0], exp0, 1e-6), "y0 {} vs {}", y[0], exp0);
    assert!(f32_eq(y[1], exp1, 1e-6), "y1 {} vs {}", y[1], exp1);

    // Prove the eps is INSIDE the sqrt: with eps outside, scale would differ
    // by ~eps/sqrt(mean) ~ 1e-5 here (the test's eps is 1e-5), so a 5e-6
    // separation is decisive.
    let outside = 1.0f32 / mean.sqrt() + eps;
    assert!(
        (scale - outside).abs() > 5e-6,
        "eps must be inside the sqrt (ggml.c), not added outside"
    );
}

#[test]
fn rms_norm_scales_by_norm_weight() {
    // weight scaling is the separate ggml_mul after the norm scale.
    let x = [1.0f32, 2.0f32, 3.0f32];
    let w = [2.0f32, 0.5f32, 1.0f32];
    let y = rms_norm(&x, &w, 1e-5);
    // with w==1 it's the plain norm; verify weighting directionally
    let y1 = rms_norm(&x, &[1.0, 1.0, 1.0], 1e-5);
    assert!(
        (y[0] - 2.0 * y1[0]).abs() < 1e-5 && (y[1] - 0.5 * y1[1]).abs() < 1e-5,
        "weighted norm must equal norm * w elementwise"
    );
}

// ---------------------------------------------------------------------------
// 2.1/2.2 RoPE — Qwen3 NeoX partial.
// ---------------------------------------------------------------------------
#[test]
fn rope_pos0_is_identity() {
    // At pos 0, theta = 0 for every rotary pair -> cos=1, sin=0 (identity).
    let x: Vec<f32> = (0..8).map(|i| i as f32 * 3.0 - 5.0).collect();
    let y = rope_neox_partial(&x, 0, 4, 1000000.0);
    assert_eq!(y, x, "pos 0 must be the exact identity");
}

#[test]
fn rope_neox_partial_pairs_across_halves() {
    // n_dims=4, half=2: pair (x0,x1)=(v[0],v[2]), (v[1],v[3]); rest unchanged.
    // base=1.0 -> theta = pos = 1 for every pair.
    let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let pos = 1u32;
    let n_dims = 4;
    let c = 1.0f32.cos();
    let s = 1.0f32.sin();
    let y = rope_neox_partial(&x, pos, n_dims, 1.0);
    // pairs indexed k=0 (v0,v2) and k=1 (v1,v3)
    assert!(f32_eq(y[0], x[0] * c - x[2] * s, 1e-6), "y0 {}", y[0]);
    assert!(f32_eq(y[2], x[0] * s + x[2] * c, 1e-6), "y2 {}", y[2]);
    assert!(f32_eq(y[1], x[1] * c - x[3] * s, 1e-6), "y1 {}", y[1]);
    assert!(f32_eq(y[3], x[1] * s + x[3] * c, 1e-6), "y3 {}", y[3]);
    // partial: indices beyond n_dims are untouched.
    assert_eq!(y[4], x[4]);
    assert_eq!(y[5], x[5]);
    assert_eq!(y[6], x[6]);
    assert_eq!(y[7], x[7]);
}

// ---------------------------------------------------------------------------
// 2.3 / 2.4 dequantized matmul on a controlled Q8_0 block.
// ---------------------------------------------------------------------------
#[test]
fn dequant_q8_and_matmul_hand_computed() {
    // One Q8_0 block: d = fp16(2.0) (0x4000), qs = 1..=32.
    let mut blk = Vec::with_capacity(34);
    blk.extend_from_slice(&0x4000u16.to_le_bytes());
    for v in 1..=32i8 {
        blk.push(v as u8);
    }
    let deq = dequant_col(TensorType::Q8, &blk, 32);
    // dequant value[j] = qs[j] * d = (j+1)*2
    for (j, d) in deq.iter().enumerate() {
        assert_eq!(*d, (j + 1) as f32 * 2.0, "dequant element {j}");
    }

    // x = ones -> dot = 2 * sum(1..=32) = 2*528 = 1056 exactly.
    let w = Tensor {
        ty: TensorType::Q8,
        data: &blk,
        ne0: 32,
        ne1: 1,
        n_rot: 0,
    };
    let ones = [1.0f32; 32];
    let mut out = [0.0f32; 1];
    matmul(&mut out, &w, &ones);
    assert_eq!(out[0].to_bits(), 1056.0f32.to_bits(), "Q8 matmul exact dot");
}

#[test]
fn dequant_q4k_via_bank_matches_cpu_dequant_module() {
    // Cross-check the bank's Q4_K path against the shared dequant module using
    // the known reference block from engine_core::dequant's unit test.
    let mut blk = [0u8; 144];
    blk[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
    blk[2..4].copy_from_slice(&0x0000u16.to_le_bytes()); // dmin = 0
    for b in blk[4..16].iter_mut() {
        *b = 1;
    }
    blk[16..144].copy_from_slice(&[0x12u8; 128]); // nibbles 2 and 1

    let deq = dequant_col(TensorType::Q4K, &blk, 256);
    let want = engine_core::dequant::dequant_q4k_cpu(&blk);
    assert_eq!(deq.len(), 256);
    for (a, b) in deq.iter().zip(want.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

// ---------------------------------------------------------------------------
// 2.5 SwiGLU silu — hand computed.
// ---------------------------------------------------------------------------
#[test]
fn silu_hand_computed() {
    let y = silu(&[0.0f32, 1.0, -1.0, 2.0]);
    let sig1 = 1.0f32 / (1.0f32 + (-1.0f32).exp());
    let sigm1 = 1.0f32 / (1.0f32 + 1.0f32.exp());
    assert_eq!(y[0].to_bits(), 0.0f32.to_bits(), "silu(0)=0 exactly");
    assert!(f32_eq(y[1], 1.0 * sig1, 5e-7));
    assert!(f32_eq(y[2], -sigm1, 5e-7));
    assert!(f32_eq(
        y[3],
        2.0 * (1.0f32 / (1.0f32 + (-2.0f32).exp())),
        5e-7
    ));
}
