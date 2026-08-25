//! Single-layer parity gate — CPU reference (Phase 6.6, group 1 RED).
//!
//! Runs ONE transformer block over the real Qwen3 fixture for the "Hello"
//! token (`9707` — token-0 row of the `l_out-0{1024,2}` golden) through the CPU
//! forward-bank arithmetic class (fp32 dequant-dot), then asserts the 6.6 gate
//! (cos-sim > 0.999 AND rel-L2 < 1e-3) against the committed llama.cpp golden L0.
//!
//! ## Why this is RED
//! The golden was produced by llama.cpp cb1adf8, whose Q4_K/Q6_K GEMVs use
//! blockwise **i8-quantized integer dot products** (`vec_dot_q4_K_q8_K`,
//! activating `x` quantized to Q8_K). The landed kernels (6.3 `MultiFormatGEMV`,
//! 6.2 CPU bank) are fp32 dequantize-then-dot — a different arithmetic class.
//! A correctly wired fp32 block reproduces the golden to cos-sim ≈ 0.9998 but
//! rel-L2 ≈ 2e-2, so the `rel-L2 < 1e-3` leg fails. This file pins that failure
//! with real numbers so the 6.6 verdict is executable, not asserted into green.

use engine_core::forward_cpu::{
    Tensor, TensorType, embed_lookup, matmul, rms_norm, sdpa_decode, silu,
};
use engine_io::{GgmlType, GgufReader, LoadedPinned, ModelConfig, load_to_pinned};
use std::path::PathBuf;

/// Golden L0 commitment (committed `activations.json`), layer 0, token-0 row of
/// the `{1024,2}` prefill — the visibly-dumped first-3 + last-3 (4-decimal).
const GOLDEN_L0: [f32; 6] = [-0.0391, 0.2084, 0.0413, -0.2046, 0.1224, 0.1987];
/// Columns those map to in the 1024-wide token-0 vector.
const GOLD_IDX: [usize; 6] = [0, 1, 2, 1021, 1022, 1023];
/// "Hello" token id from `tokenize_reference.json`.
const TOKEN_ID: usize = 9707;

fn get_fixture_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ENGINE_TESTDATA") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let md = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for c in [
        md.join("../../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        md.join("../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
    ] {
        if c.exists() {
            return Some(c.canonicalize().unwrap_or(c));
        }
    }
    None
}

fn ggml_to_bank(t: GgmlType) -> Option<TensorType> {
    match t {
        GgmlType::F32 => Some(TensorType::F32),
        GgmlType::Q4_K => Some(TensorType::Q4K),
        GgmlType::Q6_K => Some(TensorType::Q6K),
        _ => None,
    }
}

fn bank_tensor<'a>(read: &GgufReader, pinned: &'a LoadedPinned, name: &str) -> Tensor<'a> {
    let info = read.get_tensor(name).unwrap_or_else(|| panic!("missing {name}"));
    assert_eq!(info.dims.len(), 2, "{name} not 2-D");
    let ty = ggml_to_bank(info.ggml_type)
        .unwrap_or_else(|| panic!("unsupported quant for {name}"));
    let data = pinned.tensor(name).unwrap_or_else(|| panic!("{name} bytes"));
    Tensor { ty, data, ne0: info.dims[0] as usize, ne1: info.dims[1] as usize, n_rot: 0 }
}

/// F32 1-D norm weights as a `Vec<f32>`.
fn f32_norm(pinned: &LoadedPinned, name: &str) -> Vec<f32> {
    let b = pinned.tensor(name).unwrap_or_else(|| panic!("{name} bytes"));
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Single-token (pos 0) layer-0 block over the fp32-dequant arithmetic class,
/// with correct GQA decode attention and Q6_K projections.
#[allow(clippy::too_many_arguments)]
fn block_cpu(
    x: &[f32],
    cfg: &ModelConfig,
    attn_norm: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    ffn_norm: &[f32],
    wq: &Tensor,
    wk: &Tensor,
    wv: &Tensor,
    wo: &Tensor,
    wgate: &Tensor,
    wup: &Tensor,
    wdown: &Tensor,
) -> Vec<f32> {
    let h = cfg.hidden_size as usize;
    let hd = cfg.head_dim as usize;
    let nh = cfg.n_head as usize;
    let nkv = cfg.n_head_kv as usize;
    let eps = cfg.rms_norm_eps;
    let ff = cfg.intermediate_size as usize;

    let qd = nh * hd;
    let kvd = nkv * hd;

    let normed = rms_norm(x, attn_norm, eps);
    let mut q = vec![0.0f32; qd];
    let mut k = vec![0.0f32; kvd];
    let mut v = vec![0.0f32; kvd];
    matmul(&mut q, wq, &normed);
    matmul(&mut k, wk, &normed);
    matmul(&mut v, wv, &normed);

    // per-head Q/K RMSNorm (Qwen3 attn_q_norm / attn_k_norm)
    for hh in 0..nh {
        let row = q[hh * hd..(hh + 1) * hd].to_vec();
        q[hh * hd..(hh + 1) * hd].copy_from_slice(&rms_norm(&row, q_norm, eps));
    }
    for hh in 0..nkv {
        let row = k[hh * hd..(hh + 1) * hd].to_vec();
        k[hh * hd..(hh + 1) * hd].copy_from_slice(&rms_norm(&row, k_norm, eps));
    }
    // RoPE at pos 0 is the identity — omitted.

    // Single-token paged decode attention: pool holds 1 token, [k rows | v rows].
    let pool: Vec<f32> = k.iter().chain(v.iter()).copied().collect();
    let attn = sdpa_decode(&pool, &[0u32], 1, 1, &q, nh, nkv, hd, true, 0);

    let mut out_proj = vec![0.0f32; h];
    matmul(&mut out_proj, wo, &attn);
    let mut h1 = vec![0.0f32; h];
    for i in 0..h {
        h1[i] = x[i] + out_proj[i];
    }

    let ffn_in = rms_norm(&h1, ffn_norm, eps);
    let mut gate = vec![0.0f32; ff];
    let mut up = vec![0.0f32; ff];
    let mut proj = vec![0.0f32; ff];
    matmul(&mut gate, wgate, &ffn_in);
    matmul(&mut up, wup, &ffn_in);
    let g = silu(&gate);
    for i in 0..ff {
        proj[i] = g[i] * up[i];
    }
    let mut down = vec![0.0f32; h];
    matmul(&mut down, wdown, &proj);
    let mut h2 = vec![0.0f32; h];
    for i in 0..h {
        h2[i] = h1[i] + down[i];
    }
    h2
}

fn cosim(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    dot / (na * nb).sqrt()
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x as f64 - *y as f64;
        num += d * d;
        den += *y as f64 * *y as f64;
    }
    (num / den).sqrt()
}

/// RED gate test. Runs the real block and asserts the 6.6 gate against the
/// golden. **Expected to fail on the rel-L2 leg** — pinned so a future i8-dot
/// engine (or a re-baselined bound) can flip it to green with evidence.
#[test]
fn single_layer_golden_gate_cpu_reference() {
    let Some(fixture) = get_fixture_path() else {
        eprintln!("SKIP: real fixture not present; cannot run layer-0 gate");
        return;
    };
    let reader = GgufReader::open(&fixture).expect("open gguf");
    let cfg = ModelConfig::from_reader(&reader).expect("model config");
    let pinned = load_to_pinned(&reader, &fixture).expect("load pinned");

    let wq = bank_tensor(&reader, &pinned, "blk.0.attn_q.weight");
    let wk = bank_tensor(&reader, &pinned, "blk.0.attn_k.weight");
    let wv = bank_tensor(&reader, &pinned, "blk.0.attn_v.weight");
    let wo = bank_tensor(&reader, &pinned, "blk.0.attn_output.weight");
    let wgate = bank_tensor(&reader, &pinned, "blk.0.ffn_gate.weight");
    let wup = bank_tensor(&reader, &pinned, "blk.0.ffn_up.weight");
    let wdown = bank_tensor(&reader, &pinned, "blk.0.ffn_down.weight");
    let emb = bank_tensor(&reader, &pinned, "token_embd.weight");

    let x = embed_lookup(&emb, TOKEN_ID);
    let an = f32_norm(&pinned, "blk.0.attn_norm.weight");
    let qn = f32_norm(&pinned, "blk.0.attn_q_norm.weight");
    let kn = f32_norm(&pinned, "blk.0.attn_k_norm.weight");
    let fn_ = f32_norm(&pinned, "blk.0.ffn_norm.weight");

    let h2 = block_cpu(&x, &cfg, &an, &qn, &kn, &fn_, &wq, &wk, &wv, &wo, &wgate, &wup, &wdown);
    assert_eq!(h2.len(), cfg.hidden_size as usize, "block output width");

    let got: Vec<f32> = GOLD_IDX.iter().map(|&i| h2[i]).collect();
    let cs = cosim(&got, &GOLDEN_L0);
    let rl = rel_l2(&got, &GOLDEN_L0);

    println!(
        "\n=== 6.6 layer-0 gate (CPU fp32-dequant reference) ===\n\
         engine token0 L0 @ {GOLD_IDX:?}: {got:?}\n\
         golden L0:                       {GOLDEN_L0:?}\n\
         cos_sim = {cs:.6} (gate > 0.999)\n\
         rel_L2  = {rl:.3e} (gate < 1e-3)"
    );

    assert!(
        cs > 0.999 && rl < 1e-3,
        "gate FAILED: cos_sim={cs:.6} / rel_L2={rl:.3e}. \n\
         The fp32 dequant-dot arithmetic class (MultiFormatGEMV/CPU bank) \
         reaches cos_sim>0.999 but rel_L2≈1e-2 vs the llama.cpp i8-dot golden — \
         this leg is structurally unreachable by wiring the landed kernels."
    );
}