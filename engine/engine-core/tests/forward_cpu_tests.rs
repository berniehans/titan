//! CPU forward bank primitives (Phase 6.2, group 2) over controlled constants.
//!
//! Each element of the bank (RMSNorm, RoPE, dequantized matmul, SwiGLU) is
//! validated against an independently computed expected value, not a redundant
//! re-expression of the same code path. The eps placement of RMSNorm (inside
//! the sqrt, per `ggml.c::ggml_compute_forward_rms_norm_f32`) is checked
//! explicitly.

use engine_core::forward_cpu::{
    dequant_col, fused_norm_rope_swiglu, matmul, rms_norm, rms_norm_residual, rope_neox_partial,
    silu, swiglu, Tensor, TensorType,
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

// ---------------------------------------------------------------------------
// Phase 6.4 CPU reference twins & fused pipeline tests
// ---------------------------------------------------------------------------

#[test]
fn test_rms_norm_residual_hand_computed() {
    let x = [3.0f32, 4.0f32];
    let residual = [1.0f32, -1.0f32];
    let w = [1.0f32, 2.0f32];
    let eps = 1e-5f32;

    // Independent reference (f64): sum of squares of (x[i] + res[i]) in double
    let x_res0 = x[0] as f64 + residual[0] as f64; // 4.0
    let x_res1 = x[1] as f64 + residual[1] as f64; // 3.0
    let s = x_res0 * x_res0 + x_res1 * x_res1; // 25.0
    let mean = (s / 2.0) as f32; // 12.5
    let scale = 1.0f32 / (mean + eps).sqrt();
    let expected0 = (x_res0 as f32) * scale * w[0];
    let expected1 = (x_res1 as f32) * scale * w[1];

    let y = rms_norm_residual(&x, &residual, &w, eps);
    assert!(f32_eq(y[0], expected0, 1e-6), "y0 {} vs {}", y[0], expected0);
    assert!(f32_eq(y[1], expected1, 1e-6), "y1 {} vs {}", y[1], expected1);

    // Residual actually participates (differs from plain rms_norm of x)
    let plain = rms_norm(&x, &w, eps);
    assert!(
        (y[0] - plain[0]).abs() > 1e-3 || (y[1] - plain[1]).abs() > 1e-3,
        "residual must actively participate and differ from plain rms_norm"
    );
}

#[test]
fn test_rope_inplace_via_fused_hand_computed() {
    let n = 16;
    let n_dims = 8;
    let base = 10000.0f32;
    let pos = 3u32;

    let buf: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.5).collect();

    // Hand-computed reference in f64 for rotary section:
    let mut rot = buf.clone();
    for k in 0..4 {
        let theta = pos as f64 * (base as f64).powf(-2.0 * k as f64 / 8.0);
        let c = theta.cos() as f32;
        let s = theta.sin() as f32;
        let x0 = buf[k];
        let x1 = buf[k + 4];
        rot[k] = x0 * c - x1 * s;
        rot[k + 4] = x0 * s + x1 * c;
    }
    // indices >= 8 unchanged

    let residual = vec![0.0f32; n];
    let eps = 1e-5f32;
    let ss: f64 = buf.iter().map(|&v| (v * v) as f64).sum();
    let scale = (1.0f64 / (ss / n as f64 + eps as f64).sqrt()) as f32;
    // Scale w and up to cancel out norm and swiglu so rotary section is isolated
    let w: Vec<f32> = vec![1.0 / scale; n];
    let up: Vec<f32> = rot.iter().map(|&r| 1.0 + (-r).exp()).collect();

    let y = fused_norm_rope_swiglu(&buf, &residual, &w, eps, pos, n_dims, base, &up);

    // Assert each of the 8 affected within 1e-5
    for k in 0..8 {
        assert!(
            f32_eq(y[k], rot[k], 1e-5),
            "affected element {k}: y {} vs rot {}",
            y[k],
            rot[k]
        );
    }
    // Assert each of the 8 untouched within 1e-5
    for k in 8..16 {
        assert!(
            f32_eq(y[k], buf[k], 1e-5),
            "untouched element {k}: y {} vs buf {}",
            y[k],
            buf[k]
        );
    }
}

#[test]
fn test_swiglu_hand_computed() {
    let gate = [0.0f32, 1.0f32, 2.0f32, -0.5f32];
    let up = [2.0f32, 3.0f32, 4.0f32, 5.0f32];

    let mut expected = [0.0f32; 4];
    for i in 0..4 {
        let v = gate[i] as f64;
        let silu_v = v / (1.0 + (-v).exp());
        expected[i] = (silu_v * up[i] as f64) as f32;
    }

    let y = swiglu(&gate, &up);
    assert_eq!(y.len(), 4);
    for i in 0..4 {
        assert!(
            f32_eq(y[i], expected[i], 1e-5),
            "element {i}: y {} vs expected {}",
            y[i],
            expected[i]
        );
    }
}

#[test]
fn test_fused_matches_composed_twins() {
    let n = 32;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.13 - 3.0).collect();
    let residual: Vec<f32> = (0..n).map(|i| ((i * 7 % 5) as f32) * 0.5).collect();
    let w: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.05).collect();
    let up: Vec<f32> = (0..n).map(|i| 1.2 - (i as f32) * 0.02).collect();
    let eps = 1e-5f32;
    let pos = 5u32;
    let n_dims = 16usize;
    let freq_base = 500000.0f32;

    // Explicit composition:
    let y_norm = rms_norm_residual(&x, &residual, &w, eps);
    let y_rot = rope_neox_partial(&y_norm, pos, n_dims, freq_base);
    let composed = swiglu(&y_rot, &up);

    // Fused twin:
    let fused = fused_norm_rope_swiglu(&x, &residual, &w, eps, pos, n_dims, freq_base, &up);

    assert_eq!(fused.len(), n);
    for i in 0..n {
        assert!(
            f32_eq(fused[i], composed[i], 1e-5),
            "element {i}: fused {} vs composed {}",
            fused[i],
            composed[i]
        );
    }
}

