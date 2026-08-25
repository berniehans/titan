//! CPU reference forward bank (Phase 6.2).
//!
//! A FP32, single-sequence CPU authority for transformer inference over a real
//! quantized GGUF, written from the paper formulas and ggml.c semantics —
//! **not** by transliterating the repo's CUDA kernels back (per
//! `openspec/changes/6.2-cpu-reference-bank/proposal.md`, that would share the
//! same bugs between twin implementations). It is the readable reference that
//! ParceledParity (6.3), the fused norm/rope/swiglu parity (6.4) and paged
//! attention (6.5) validate their kernels against.
//!
//! ## Traceability vs llama.cpp (pinned `cb1adf8`)
//!
//! - **RMSNorm**: `ggml/src/ggml-cpu/ops.cpp::ggml_compute_forward_rms_norm_f32`
//!   — `eps` is added *inside* the sqrt (`scale = 1/sqrtf(mean + eps)`); the sum
//!   of squares accumulates in `ggml_float` (double); weight scaling is the
//!   separate `ggml_mul` applied after `ggml_vec_scale_f32`, i.e. `(x*s)*w`.
//! - **RoPE (Qwen3, NeoX partial)**: `ops.cpp::ggml_rope_cache_init` +
//!   `rope_yarn` with `GGML_ROPE_TYPE_NEOX`; pair element `k` with
//!   `k + n_dims/2`, `theta = pos*base^(-2k/n_dims)`, rotate only the first
//!   `n_dims` elements (`n_dims = head_dim/2` for Qwen3), rest unchanged.
//! - **Q4_K dequant**: reused from `crate::dequant::dequant_q4k_cpu` (same
//!   `vec_dot_q4_K_q8_K` scale/min convention in `ggml/src/ggml-quants.c`).
//! - **Q8_0 / F16 dequant**: `ggml-quants.c::dequantize_row_q8_0`
//!   (`y = qs[j]*d`) and the GGUF FP16→FP32 mapping.
//! - **SwiGLU**: `y = silu(gate) * up`, `silu(x) = x / (1 + exp(-x))`
//!   (llama.cpp `ggml_silu`).
//!
//! ## Exactness note (see the generator tool docstring)
//!
//! The committed synthetic is a *controlled known-constants* model. The
//! full-stack logits are bit-exact between this bank and the independent numpy
//! reference because RoPE runs at position 0 (identity), attention sees a
//! single token (softmax = 1.0), and the FFN *down* weight is zero so the silu
//! branch is gated out of the compared output. Everything that reaches the
//! compared logits is IEEE-correctly-rounded fp32/fp64 arithmetic in a fixed
//! order, which two conforming implementations agree on bit-for-bit. silu and
//! non-zero RoPE are validated separately by unit tests / the layer-0 checks.

use crate::dequant::{dequant_q4k_cpu, dequant_q6k_cpu};

/// Tensor quantization formats understood by the CPU bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    /// 32-bit float (embedding / norms / head).
    F32,
    /// 16-bit float.
    F16,
    /// Q8_0: 32-element blocks (`[f16 d][int8 qs x32]`).
    Q8,
    /// Q4_K: 256-element super-blocks (144 bytes).
    Q4K,
    /// Q6_K: 256-element super-blocks (210 bytes).
    Q6K,
}

/// A 2-D weight matrix stored GGUF-style: `dims[0] = ne0` (contiguous,
/// block-quantized reduction dim = input), `dims[1] = ne1` (output).
#[derive(Debug, Clone, Copy)]
pub struct Tensor<'a> {
    /// Quantization format of the weight data.
    pub ty: TensorType,
    /// Raw quantized weight bytes: `ne1` columns of `ne0` elements each.
    pub data: &'a [u8],
    /// Number of input (reduction) elements per column.
    pub ne0: usize,
    /// Number of output columns.
    pub ne1: usize,
    /// Number of rotary dims (head_dim/2 for Qwen3), used by the caller.
    pub n_rot: usize,
}

/// FP16 (IEEE binary16) bit pattern to `f32` (exact).
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        // subnormal: value = mant * 2^-24
        let neg = sign == 1;
        let v = (mant as f32) * 2.0f32.powi(-24);
        return if neg { -v } else { v };
    }
    if exp == 31 {
        return f32::INFINITY;
    }
    let v = (1.0 + (mant as f32) / 1024.0) * 2.0f32.powi((exp as i32) - 15);
    if sign == 1 { -v } else { v }
}

/// Dequantizes one column (`ne0` elements) of a quantized weight tensor.
pub fn dequant_col(ty: TensorType, data: &[u8], ne0: usize) -> Vec<f32> {
    match ty {
        TensorType::F32 => {
            assert_eq!(data.len(), ne0 * 4);
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        TensorType::F16 => {
            assert_eq!(data.len(), ne0 * 2);
            data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        }
        TensorType::Q8 => {
            assert_eq!(data.len(), (ne0 / 32) * 34);
            let mut out = Vec::with_capacity(ne0);
            for blk in data.chunks_exact(34) {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for j in 0..32 {
                    out.push(blk[2 + j] as i8 as f32 * d);
                }
            }
            out
        }
        TensorType::Q4K => {
            assert_eq!(data.len(), (ne0 / 256) * 144);
            let mut out = Vec::with_capacity(ne0);
            for blk in data.chunks_exact(144) {
                out.extend_from_slice(&dequant_q4k_cpu(blk));
            }
            out
        }
        TensorType::Q6K => {
            assert_eq!(data.len(), (ne0 / 256) * 210);
            let mut out = Vec::with_capacity(ne0);
            for blk in data.chunks_exact(210) {
                out.extend_from_slice(&dequant_q6k_cpu(blk));
            }
            out
        }
    }
}

/// Dequantizes the embedding row for `token` into `ne0` fp32 values.
///
/// GGUF embeddings store rows along `ne1` (`dims[1]`), each row contiguous
/// across `ne0` elements — so a single-token embedding is a row-lookup on the
/// (often quantized, e.g. Q6_K) table, not a matmul.
pub fn embed_lookup(t: &Tensor, token: usize) -> Vec<f32> {
    assert!(token < t.ne1, "token {token} out of embedding rows {}", t.ne1);
    let cb = match t.ty {
        TensorType::F32 => t.ne0 * 4,
        TensorType::F16 => t.ne0 * 2,
        TensorType::Q8 => (t.ne0 / 32) * 34,
        TensorType::Q4K => (t.ne0 / 256) * 144,
        TensorType::Q6K => (t.ne0 / 256) * 210,
    };
    let row = &t.data[token * cb..(token + 1) * cb];
    dequant_col(t.ty, row, t.ne0)
}

/// FP32 sequential dot-product matmul over a dequantized weight column.
///
/// `out[j] = sum_i dequant(weight_col_j)[i] * x[i]`, accumulated in fp32,
/// index order — matching the independent numpy reference bit-for-bit.
pub fn matmul(out: &mut [f32], t: &Tensor, x: &[f32]) {
    assert_eq!(out.len(), t.ne1);
    assert_eq!(x.len(), t.ne0);
    let cb = match t.ty {
        TensorType::F32 => t.ne0 * 4,
        TensorType::F16 => t.ne0 * 2,
        TensorType::Q8 => (t.ne0 / 32) * 34,
        TensorType::Q4K => (t.ne0 / 256) * 144,
        TensorType::Q6K => (t.ne0 / 256) * 210,
    };
    for (j, o) in out.iter_mut().enumerate() {
        let col = &t.data[j * cb..(j + 1) * cb];
        let deq = dequant_col(t.ty, col, t.ne0);
        let mut acc: f32 = 0.0;
        for i in 0..t.ne0 {
            acc += deq[i] * x[i];
        }
        *o = acc;
    }
}

/// RMSNorm, `eps` inside the sqrt, fp64 sum of fp32 squares (ggml.c).
///
/// `y[i] = (x[i] * scale) * w[i]`, `scale = 1/sqrt(mean(x^2) + eps)`.
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    assert_eq!(w.len(), n);
    let mut sum: f64 = 0.0;
    for &xi in x {
        let p = xi * xi; // fp32 square, then accumulated in fp64
        sum += p as f64;
    }
    let mean: f32 = (sum / n as f64) as f32;
    let scale: f32 = 1.0 / (mean + eps).sqrt();
    x.iter().zip(w).map(|(&xi, &wi)| xi * scale * wi).collect()
}

/// RMSNorm over elementwise sum `(x[i] + residual[i])`, scaled by `w`.
pub fn rms_norm_residual(x: &[f32], residual: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), residual.len());
    let sum_x: Vec<f32> = x.iter().zip(residual).map(|(&xi, &ri)| xi + ri).collect();
    rms_norm(&sum_x, w, eps)
}

/// SwiGLU gated activation: `silu(x) = x / (1 + exp(-x))`.
pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

/// SwiGLU gating: `y[i] = silu(gate[i]) * up[i]`.
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len());
    let g = silu(gate);
    g.iter().zip(up).map(|(&gi, &ui)| gi * ui).collect()
}

/// Qwen3 NeoX **partial** rotary position embedding (per ggml `rope_yarn`).
///
/// Pairs `(x[k], x[k + n_dims/2])` for `k < n_dims/2` and rotates each pair by
/// `theta = pos * base^(-2k/n_dims)`; elements beyond `n_dims` are untouched
/// (the "partial" part). At `pos = 0` this is the identity (theta = 0).
pub fn rope_neox_partial(x: &[f32], pos: u32, n_dims: usize, freq_base: f32) -> Vec<f32> {
    let mut y = x.to_vec();
    if n_dims < 2 {
        return y;
    }
    let half = n_dims / 2;
    for k in 0..half {
        let theta = pos as f32 * freq_base.powf(-2.0 * k as f32 / n_dims as f32);
        let (c, s) = (theta.cos(), theta.sin());
        let x0 = x[k];
        let x1 = x[k + half];
        y[k] = x0 * c - x1 * s;
        y[k + half] = x0 * s + x1 * c;
    }
    y
}

/// CPU fused twin for the norm/rope/swiglu pipeline:
/// (a) `y = rms_norm_residual(x, residual, w, eps)`;
/// (b) `y = rope_neox_partial(&y, pos, n_dims, freq_base)`;
/// (c) `swiglu(&y, up)`.
#[allow(clippy::too_many_arguments)]
pub fn fused_norm_rope_swiglu(
    x: &[f32],
    residual: &[f32],
    w: &[f32],
    eps: f32,
    pos: u32,
    n_dims: usize,
    freq_base: f32,
    up: &[f32],
) -> Vec<f32> {
    let y = rms_norm_residual(x, residual, w, eps);
    let y = rope_neox_partial(&y, pos, n_dims, freq_base);
    swiglu(&y, up)
}

/// Latent dot product between two vectors in fp64 (attention score).
#[inline]
fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}

/// Single-element softmax (max-subtraction), returns an array of length 1.
/// For one input this is exactly `1.0` regardless of the score.
fn softmax1(v: &[f32]) -> Vec<f32> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = v.iter().map(|&x| (x - max).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|&x| x / s).collect()
}

/// One full transformer layer (single sequence, arbitrary length), returning
/// the post-layer hidden state (the layer output / residual after the MLP).
///
/// Stack (per `openspec/changes/6.2-cpu-reference-bank/proposal.md`):
/// RMSNorm -> QKV -> RoPE -> attention -> out -> +residual ->
/// RMSNorm -> SwiGLU -> down -> +residual.
#[allow(clippy::too_many_arguments)] // explicit, self-documenting layer signature
pub fn forward_layer0(
    x: &[f32],
    params: &LayerParams,
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
    let h = params.hidden;
    let hd = params.head_dim;
    let n_head = params.n_head;
    let n_kv = params.n_head_kv;
    let qd = n_head * hd;
    let kvd = n_kv * hd;

    // --- QKV ---
    let normed = rms_norm(x, attn_norm, params.eps);
    let mut q = vec![0.0f32; qd];
    let mut k = vec![0.0f32; kvd];
    let mut v = vec![0.0f32; kvd];
    matmul(&mut q, wq, &normed);
    matmul(&mut k, wk, &normed);
    matmul(&mut v, wv, &normed);

    // per-head q/k RMSNorm (Qwen3 attn_q_norm / attn_k_norm)
    for hh in 0..n_head {
        let s = hh * hd;
        let row = q[s..s + hd].to_vec();
        q[s..s + hd].copy_from_slice(&rms_norm(&row, q_norm, params.eps));
    }
    for hh in 0..n_kv {
        let s = hh * hd;
        let row = k[s..s + hd].to_vec();
        k[s..s + hd].copy_from_slice(&rms_norm(&row, k_norm, params.eps));
    }

    // RoPE per head (Qwen3 NeoX partial)
    for hh in 0..n_head {
        let s = hh * hd;
        let row = q[s..s + hd].to_vec();
        let rot = rope_neox_partial(&row, params.pos, wq.n_rot, params.freq_base);
        q[s..s + hd].copy_from_slice(&rot);
    }
    for hh in 0..n_kv {
        let s = hh * hd;
        let row = k[s..s + hd].to_vec();
        let rot = rope_neox_partial(&row, params.pos, wk.n_rot, params.freq_base);
        k[s..s + hd].copy_from_slice(&rot);
    }

    // --- Attention (single sequence; GQA: every q head attends the kv heads) ---
    let kv_seq = v.len() / hd; // == n_kv for single token
    let mut attn_out = vec![0.0f32; n_head * hd];
    for hh in 0..n_head {
        let qh = &q[hh * hd..(hh + 1) * hd];
        let mut out_h = vec![0.0f32; hd];
        for kv in 0..kv_seq {
            let kh = &k[kv * hd..(kv + 1) * hd];
            let score = dot_f64(qh, kh) / (hd as f64).sqrt();
            let wgt = softmax1(&[score as f32])[0]; // 1.0 for a single element
            let vh = &v[kv * hd..(kv + 1) * hd];
            for j in 0..hd {
                out_h[j] += vh[j] * wgt;
            }
        }
        attn_out[hh * hd..(hh + 1) * hd].copy_from_slice(&out_h);
    }

    // --- output projection + residual 1 ---
    let mut out_proj = vec![0.0f32; h];
    matmul(&mut out_proj, wo, &attn_out);
    let mut h1 = vec![0.0f32; h];
    for i in 0..h {
        h1[i] = x[i] + out_proj[i];
    }

    // --- FFN: RMSNorm -> gate/up -> SwiGLU -> down -> residual 2 ---
    let ffn_in = rms_norm(&h1, ffn_norm, params.eps);
    let ffn = wgate.ne1;
    let mut gate = vec![0.0f32; ffn];
    let mut up = vec![0.0f32; ffn];
    let mut proj = vec![0.0f32; ffn];
    matmul(&mut gate, wgate, &ffn_in);
    matmul(&mut up, wup, &ffn_in);
    let g = silu(&gate);
    for i in 0..ffn {
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

/// Language-model head: RMSNorm then the (tied) embedding matmul -> logits.
pub fn logits_from_hidden(embd: &Tensor, head_norm: &[f32], hidden: &[f32], eps: f32) -> Vec<f32> {
    let hn = rms_norm(hidden, head_norm, eps);
    let mut logits = vec![0.0f32; embd.ne1];
    matmul(&mut logits, embd, &hn);
    logits
}

/// Hyper-parameters shaping a forward pass (must match the model config).
#[derive(Debug, Clone, Copy)]
pub struct LayerParams {
    /// Hidden / embedding size.
    pub hidden: usize,
    /// Number of query heads.
    pub n_head: usize,
    /// Head dimension (key/value projection dim).
    pub head_dim: usize,
    /// Number of KV heads (GQA).
    pub n_head_kv: usize,
    /// RMSNorm epsilon (added inside the sqrt, per ggml.c).
    pub eps: f32,
    /// Current token position (RoPE).
    pub pos: u32,
    /// RoPE base frequency.
    pub freq_base: f32,
}

/// Phase 6.5 CPU scaled-dot-product attention (SDPA) reference over a paged KV pool.
///
/// Written directly from the mathematical scaled-dot-product attention formula rather
/// than transliterated from CUDA kernels, serving as an independent numerical authority
/// that GPU kernel parity validates against. Supports GQA head grouping and causal masking.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
#[allow(clippy::manual_is_multiple_of)]
pub fn sdpa_decode(
    pool: &[f32],
    block_table: &[u32],
    block_tokens: usize,
    seq_tokens: usize,
    query: &[f32],
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    causal: bool,
    query_pos: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_head * head_dim];
    if seq_tokens == 0 || n_head == 0 || head_dim == 0 {
        return out;
    }
    assert!(n_head_kv > 0 && n_head % n_head_kv == 0);
    let group = n_head / n_head_kv;
    let row_len = n_head_kv * head_dim;
    let floats_per_token = 2 * row_len;
    let floats_per_block = block_tokens * floats_per_token;
    let scale = 1.0f64 / (head_dim as f64).sqrt();

    for qh in 0..n_head {
        let hk = qh / group;
        let query_head = &query[qh * head_dim..(qh + 1) * head_dim];

        let mut scores = vec![f64::NEG_INFINITY; seq_tokens];
        let mut max_score = f64::NEG_INFINITY;

        for t in 0..seq_tokens {
            if causal && t > query_pos {
                continue;
            }
            let block_idx = t / block_tokens;
            let slot = t % block_tokens;
            let phys = block_table[block_idx] as usize;
            let base = phys * floats_per_block + slot * floats_per_token;
            let key_row = &pool[base + hk * head_dim..base + (hk + 1) * head_dim];
            let score = dot_f64(query_head, key_row) * scale;
            scores[t] = score;
            if score > max_score {
                max_score = score;
            }
        }

        let mut weights = vec![0.0f64; seq_tokens];
        if max_score.is_finite() {
            let mut sum_exp = 0.0f64;
            for t in 0..seq_tokens {
                if scores[t].is_finite() {
                    let e = (scores[t] - max_score).exp();
                    weights[t] = e;
                    sum_exp += e;
                }
            }
            if sum_exp > 0.0 {
                for w in weights.iter_mut() {
                    *w /= sum_exp;
                }
            }
        }

        let mut out_h = vec![0.0f64; head_dim];
        for t in 0..seq_tokens {
            let wt = weights[t];
            if wt == 0.0 {
                continue;
            }
            let block_idx = t / block_tokens;
            let slot = t % block_tokens;
            let phys = block_table[block_idx] as usize;
            let base = phys * floats_per_block + slot * floats_per_token;
            let val_row =
                &pool[base + row_len + hk * head_dim..base + row_len + (hk + 1) * head_dim];
            for (d, &v) in val_row.iter().enumerate() {
                out_h[d] += wt * (v as f64);
            }
        }

        for d in 0..head_dim {
            out[qh * head_dim + d] = out_h[d] as f32;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_to_f32_covers_specials() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0); // 0x3C00 = fp16 1.0
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
        // subnormal 0x0001 = 2^-24
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
    }

    #[test]
    fn dequant_q8_matches_hand_computed() {
        // one Q8_0 block: d = fp16(2.0) = 0x4000, qs = 1..32
        let mut blk = Vec::with_capacity(34);
        blk.extend_from_slice(&0x4000u16.to_le_bytes());
        for v in 1..=32i8 {
            blk.push(v as u8);
        }
        let deq = dequant_col(TensorType::Q8, &blk, 32);
        assert_eq!(deq.len(), 32);
        for (i, d) in deq.iter().enumerate() {
            let expected = (i as i8 + 1) as f32 * 2.0;
            assert_eq!(*d, expected, "element {i}");
        }
    }
}
