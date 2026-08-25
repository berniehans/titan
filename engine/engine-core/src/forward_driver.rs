//! Full forward driver (Phase 6.7) — prefill and single-token decode entry
//! points over the streamed GPU pipeline, ADDITIVE beside `stub_next_token`
//! (Phase 6.8 owns the swap; engine-core has NO dependency on engine-server,
//! so this driver cannot touch `stub_next_token` by construction).
//!
//! # Declared routing (real Qwen3-0.6B fixture)
//! - **GPU** (engine-cuda): `MultiFormatGEMV` Q4_K for `attn_q`/`attn_k`/
//!   `attn_output`/`ffn_gate`/`ffn_up`; `NormRope` for attention-input RMSNorm,
//!   per-head Q/K RMSNorm + per-head Neox-Partial RoPE (pos>=0), ffn RMSNorm
//!   and fused SwiGLU; `PagedKvGpu::append_kv` into a **resident per-layer KV
//!   pool**; `PagedAttention` decode over it.
//! - **CPU bank fallback** (declared, per 6.6): this fixture's `attn_v`,
//!   `ffn_down` and `token_embd` are **Q6_K** — the landed GEMV has no Q6_K
//!   path (Q4K/Q8/F16 only). They run through `forward_cpu`
//!   (`matmul`/`embed_lookup`/`logits_from_hidden`). The tied lm-head uses the
//!   Q6_K embedding on the CPU bank.
//!
//! # Multi-token math (PagedAttention is a single-query decode kernel)
//! Prefill streams one token at a time: token `pos`'s embedding runs every
//! layer, each layer appends its K/V rows to the resident per-layer KV pool,
//! and the final hidden state runs through the tie-headed logits. `run_prefill`
//! returns the LAST position's next-token logits (matching the 6.1
//! final-position `logits_XX.bin`); `run_decode` runs one more token against
//! the resident KV and returns its next-token logits.
//!
//! # Paged KV layout (matches `engine-kvcache` flat pool and `engine-cuda`)
//! pool: phys block b @ `b*floats_per_block`; token s @ `+s*floats_per_token`
//! (key row then value row); `row_len = n_head_kv*head_dim`, `floats_per_token
//! = 2*row_len`. One sequence, `n_blocks=1`, `block_tokens=max_seq` → logical
//! slot `s` == position `s`, block table `[0]`.

use crate::error::EngineError;
use crate::forward_cpu::{Tensor, TensorType, embed_lookup, logits_from_hidden, matmul};
use cudarc::driver::CudaDevice;
use engine_cuda::{
    CudaStream, DeviceBuffer, GemvFormat, MODE_NORM, MODE_ROPE, MODE_SWIGLU, MultiFormatGEMV,
    NormRope, PagedAttention, PagedKvGpu, PagedKvLayout,
};
use engine_io::{GgmlType, GgufReader, LoadedPinned, ModelConfig};
use std::sync::Arc;

/// Resident transient working-set VRAM budget: `pingpong + kv_pool +
/// activations + logits` must stay <= 5.2 GiB at every step (6.7 gate).
pub const VRAM_BUDGET_BYTES: usize = 5 * 1024 * 1024 * 1024 + 200 * 1024 * 1024;

fn ggml_to_bank(t: GgmlType) -> Option<TensorType> {
    match t {
        GgmlType::F32 => Some(TensorType::F32),
        GgmlType::Q4_K => Some(TensorType::Q4K),
        GgmlType::Q6_K => Some(TensorType::Q6K),
        _ => None,
    }
}

fn bank_tensor<'a>(
    read: &GgufReader,
    pinned: &'a LoadedPinned,
    name: &str,
) -> Result<Tensor<'a>, EngineError> {
    let info = read
        .get_tensor(name)
        .ok_or_else(|| engine_io::GgufError::MissingMetadata(name.to_string()))?;
    if info.dims.len() != 2 {
        return Err(engine_io::GgufError::InvalidTensorShape(name.to_string()).into());
    }
    let ty = ggml_to_bank(info.ggml_type).ok_or_else(|| {
        engine_io::GgufError::InvalidTensorShape(format!("unsupported quant {name}"))
    })?;
    let data = pinned
        .tensor(name)
        .ok_or_else(|| engine_io::GgufError::MissingMetadata(name.to_string()))?;
    Ok(Tensor {
        ty,
        data,
        ne0: info.dims[0] as usize,
        ne1: info.dims[1] as usize,
        n_rot: 0,
    })
}

fn f32_norm(pinned: &LoadedPinned, name: &str) -> Result<Vec<f32>, EngineError> {
    let b = pinned
        .tensor(name)
        .ok_or_else(|| engine_io::GgufError::MissingMetadata(name.to_string()))?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_f32(v: &[u8]) -> Vec<f32> {
    v.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn alloc_dev(dev: &Arc<CudaDevice>, n_floats: usize) -> Result<DeviceBuffer, EngineError> {
    Ok(DeviceBuffer::alloc(Arc::clone(dev), n_floats * 4)?)
}

fn upload_f32(
    stream: &CudaStream,
    dev: &Arc<CudaDevice>,
    v: &[f32],
) -> Result<DeviceBuffer, EngineError> {
    let b = DeviceBuffer::alloc(Arc::clone(dev), v.len() * 4)?;
    b.copy_from_host(stream, &f32_bytes(v))?;
    Ok(b)
}

fn upload_bytes(
    stream: &CudaStream,
    dev: &Arc<CudaDevice>,
    bytes: &[u8],
) -> Result<DeviceBuffer, EngineError> {
    let b = DeviceBuffer::alloc(Arc::clone(dev), bytes.len())?;
    b.copy_from_host(stream, bytes)?;
    Ok(b)
}

fn download_f32(
    stream: &CudaStream,
    b: &DeviceBuffer,
    n_floats: usize,
) -> Result<Vec<f32>, EngineError> {
    let mut raw = vec![0u8; n_floats * 4];
    b.copy_to_host(stream, &mut raw)?;
    Ok(bytes_f32(&raw))
}

struct LayerRsrc<'a> {
    wq_dev: DeviceBuffer,
    wq_ne0: usize,
    wq_ne1: usize,
    wk_dev: DeviceBuffer,
    wk_ne0: usize,
    wk_ne1: usize,
    wv: Tensor<'a>,
    wo_dev: DeviceBuffer,
    wo_ne0: usize,
    wo_ne1: usize,
    wgate_dev: DeviceBuffer,
    wgate_ne0: usize,
    wgate_ne1: usize,
    wup_dev: DeviceBuffer,
    wup_ne0: usize,
    wup_ne1: usize,
    wdown: Tensor<'a>,
    an_dev: DeviceBuffer,
    qn_dev: DeviceBuffer,
    kn_dev: DeviceBuffer,
    fn_dev: DeviceBuffer,
    pool_dev: DeviceBuffer,
}

/// Result of running the prefill forward driver.
pub struct PrefillResult {
    /// Next-token logits at the last prompt position (length == vocab_size).
    pub logits: Vec<f32>,
    /// Number of tokens processed.
    pub tokens: usize,
}

/// Runs the full transformer model across all prompt tokens, streaming one token at a time,
/// appending K/V into a resident per-layer KV pool, and returning the last position's next-token logits.
#[allow(clippy::too_many_arguments)]
pub fn run_prefill(
    reader: &GgufReader,
    pinned: &LoadedPinned,
    cfg: &ModelConfig,
    tokens: &[u32],
) -> Result<PrefillResult, EngineError> {
    if tokens.is_empty() {
        return Ok(PrefillResult {
            logits: Vec::new(),
            tokens: 0,
        });
    }

    let device = Arc::new(CudaDevice::new(0)?);
    let stream = CudaStream::new(Arc::clone(&device))?;
    let gemv = MultiFormatGEMV::new(Arc::clone(&device))?;
    let nr = NormRope::new(Arc::clone(&device))?;
    let pkv = PagedKvGpu::new(Arc::clone(&device))?;
    let pa = PagedAttention::new(Arc::clone(&device))?;

    let h = cfg.hidden_size as usize;
    let hd = cfg.head_dim as usize;
    let nh = cfg.n_head as usize;
    let nkv = cfg.n_head_kv as usize;
    let ff = cfg.intermediate_size as usize;
    let eps = cfg.rms_norm_eps;
    let base = cfg.rope_freq_base;
    // Qwen3 RoPE rotates the FULL head dim (NeoX pairs over [0, head_dim));
    // the head_dim/2 assumption was falsified by the golden gate (cos 0.71 vs
    // 0.9986 at pos>=1), so n_rot = hd (128 for this fixture).
    let n_rot = hd;
    let qdim = nh * hd;
    let kvd = nkv * hd;
    let seq_len = tokens.len();
    let n_layer = cfg.n_layer as usize;

    let layout = PagedKvLayout {
        n_blocks: 1,
        block_tokens: seq_len,
        row_len: kvd,
    };

    let emb = bank_tensor(reader, pinned, "token_embd.weight")?;
    let head_norm = f32_norm(pinned, "output_norm.weight")?;

    let mut layers: Vec<LayerRsrc> = Vec::with_capacity(n_layer);
    for l in 0..n_layer {
        let wq = bank_tensor(reader, pinned, &format!("blk.{l}.attn_q.weight"))?;
        let wk = bank_tensor(reader, pinned, &format!("blk.{l}.attn_k.weight"))?;
        let wv = bank_tensor(reader, pinned, &format!("blk.{l}.attn_v.weight"))?;
        let wo = bank_tensor(reader, pinned, &format!("blk.{l}.attn_output.weight"))?;
        let wgate = bank_tensor(reader, pinned, &format!("blk.{l}.ffn_gate.weight"))?;
        let wup = bank_tensor(reader, pinned, &format!("blk.{l}.ffn_up.weight"))?;
        let wdown = bank_tensor(reader, pinned, &format!("blk.{l}.ffn_down.weight"))?;

        let an = f32_norm(pinned, &format!("blk.{l}.attn_norm.weight"))?;
        let qn = f32_norm(pinned, &format!("blk.{l}.attn_q_norm.weight"))?;
        let kn = f32_norm(pinned, &format!("blk.{l}.attn_k_norm.weight"))?;
        let fnw = f32_norm(pinned, &format!("blk.{l}.ffn_norm.weight"))?;

        let wq_dev = upload_bytes(&stream, &device, wq.data)?;
        let wk_dev = upload_bytes(&stream, &device, wk.data)?;
        let wo_dev = upload_bytes(&stream, &device, wo.data)?;
        let wgate_dev = upload_bytes(&stream, &device, wgate.data)?;
        let wup_dev = upload_bytes(&stream, &device, wup.data)?;

        let an_dev = upload_f32(&stream, &device, &an)?;
        let qn_dev = upload_f32(&stream, &device, &qn)?;
        let kn_dev = upload_f32(&stream, &device, &kn)?;
        let fn_dev = upload_f32(&stream, &device, &fnw)?;

        let pool_dev = alloc_dev(&device, layout.floats_total())?;

        layers.push(LayerRsrc {
            wq_dev,
            wq_ne0: wq.ne0,
            wq_ne1: wq.ne1,
            wk_dev,
            wk_ne0: wk.ne0,
            wk_ne1: wk.ne1,
            wv,
            wo_dev,
            wo_ne0: wo.ne0,
            wo_ne1: wo.ne1,
            wgate_dev,
            wgate_ne0: wgate.ne0,
            wgate_ne1: wgate.ne1,
            wup_dev,
            wup_ne0: wup.ne0,
            wup_ne1: wup.ne1,
            wdown,
            an_dev,
            qn_dev,
            kn_dev,
            fn_dev,
            pool_dev,
        });
    }

    let zh = upload_f32(&stream, &device, &vec![0.0f32; h])?;
    let zhd = upload_f32(&stream, &device, &vec![0.0f32; hd])?;
    let zff = upload_f32(&stream, &device, &vec![0.0f32; ff])?;
    let bt_dev = upload_bytes(&stream, &device, &0u32.to_le_bytes())?;

    let x_dev = alloc_dev(&device, h)?;
    let input_norm_dev = alloc_dev(&device, h)?;
    let q_dev = alloc_dev(&device, qdim)?;
    let k_dev = alloc_dev(&device, kvd)?;
    let v_dev = alloc_dev(&device, kvd)?;
    let head_dev = alloc_dev(&device, hd)?;
    let attn_dev = alloc_dev(&device, qdim)?;
    let op_dev = alloc_dev(&device, h)?;
    let h1_dev = alloc_dev(&device, h)?;
    let ffin_dev = alloc_dev(&device, h)?;
    let gate_dev = alloc_dev(&device, ff)?;
    let up_dev = alloc_dev(&device, ff)?;
    let proj_dev = alloc_dev(&device, ff)?;

    let mut x_host = vec![0.0f32; h];

    for (p, &token_id) in tokens.iter().enumerate() {
        x_host = embed_lookup(&emb, token_id as usize);

        for layer in &layers {
            // Stage 1: Input RMSNorm (GPU)
            x_dev.copy_from_host(&stream, &f32_bytes(&x_host))?;
            nr.launch(
                &stream,
                &x_dev,
                &zh,
                &layer.an_dev,
                &input_norm_dev,
                &input_norm_dev,
                eps,
                h,
                0,
                base,
                0,
                MODE_NORM,
            )?;

            // Stage 2: Q/K GEMV (GPU Q4_K)
            gemv.gemv(
                &stream,
                GemvFormat::Q4K,
                &layer.wq_dev,
                &input_norm_dev,
                &q_dev,
                layer.wq_ne0,
                layer.wq_ne1,
            )?;
            gemv.gemv(
                &stream,
                GemvFormat::Q4K,
                &layer.wk_dev,
                &input_norm_dev,
                &k_dev,
                layer.wk_ne0,
                layer.wk_ne1,
            )?;

            // Stage 3: V CPU Fallback (Q6_K)
            let input_norm_host = download_f32(&stream, &input_norm_dev, h)?;
            let mut v_host = vec![0.0f32; kvd];
            matmul(&mut v_host, &layer.wv, &input_norm_host);
            v_dev.copy_from_host(&stream, &f32_bytes(&v_host))?;

            // Stage 4: Per-head Q/K RMSNorm + RoPE (GPU)
            let mut qhost = download_f32(&stream, &q_dev, qdim)?;
            let mut khost = download_f32(&stream, &k_dev, kvd)?;

            for hh in 0..nh {
                head_dev.copy_from_host(&stream, &f32_bytes(&qhost[hh * hd..(hh + 1) * hd]))?;
                nr.launch(
                    &stream,
                    &head_dev,
                    &zhd,
                    &layer.qn_dev,
                    &head_dev,
                    &head_dev,
                    eps,
                    hd,
                    0,
                    base,
                    0,
                    MODE_NORM,
                )?;
                nr.launch(
                    &stream, &head_dev, &zhd, &zhd, &head_dev, &head_dev, eps, hd, n_rot, base,
                    p as u32, MODE_ROPE,
                )?;
                let out = download_f32(&stream, &head_dev, hd)?;
                qhost[hh * hd..(hh + 1) * hd].copy_from_slice(&out);
            }

            for hh in 0..nkv {
                head_dev.copy_from_host(&stream, &f32_bytes(&khost[hh * hd..(hh + 1) * hd]))?;
                nr.launch(
                    &stream,
                    &head_dev,
                    &zhd,
                    &layer.kn_dev,
                    &head_dev,
                    &head_dev,
                    eps,
                    hd,
                    0,
                    base,
                    0,
                    MODE_NORM,
                )?;
                nr.launch(
                    &stream, &head_dev, &zhd, &zhd, &head_dev, &head_dev, eps, hd, n_rot, base,
                    p as u32, MODE_ROPE,
                )?;
                let out = download_f32(&stream, &head_dev, hd)?;
                khost[hh * hd..(hh + 1) * hd].copy_from_slice(&out);
            }

            q_dev.copy_from_host(&stream, &f32_bytes(&qhost))?;
            k_dev.copy_from_host(&stream, &f32_bytes(&khost))?;

            // Stage 5: Paged KV Append (GPU)
            pkv.append_kv(
                &stream,
                &layout,
                &layer.pool_dev,
                &k_dev,
                &v_dev,
                &bt_dev,
                p,
                1,
            )?;

            // Stage 6: PagedAttention Decode (GPU)
            pa.launch(
                &stream,
                &q_dev,
                &layer.pool_dev,
                &bt_dev,
                &attn_dev,
                nh,
                nkv,
                hd,
                seq_len,
                p + 1,
                p,
                true,
            )?;

            // Stage 7: Output Projection (GPU Q4_K) + Residual 1 (Host)
            gemv.gemv(
                &stream,
                GemvFormat::Q4K,
                &layer.wo_dev,
                &attn_dev,
                &op_dev,
                layer.wo_ne0,
                layer.wo_ne1,
            )?;
            let op_host = download_f32(&stream, &op_dev, h)?;
            let mut h1_host = vec![0.0f32; h];
            for i in 0..h {
                h1_host[i] = x_host[i] + op_host[i];
            }

            // Stage 8: FFN Norm (GPU)
            h1_dev.copy_from_host(&stream, &f32_bytes(&h1_host))?;
            nr.launch(
                &stream,
                &h1_dev,
                &zh,
                &layer.fn_dev,
                &ffin_dev,
                &ffin_dev,
                eps,
                h,
                0,
                base,
                0,
                MODE_NORM,
            )?;

            // Stage 9: FFN Gate & Up (GPU Q4_K)
            gemv.gemv(
                &stream,
                GemvFormat::Q4K,
                &layer.wgate_dev,
                &ffin_dev,
                &gate_dev,
                layer.wgate_ne0,
                layer.wgate_ne1,
            )?;
            gemv.gemv(
                &stream,
                GemvFormat::Q4K,
                &layer.wup_dev,
                &ffin_dev,
                &up_dev,
                layer.wup_ne0,
                layer.wup_ne1,
            )?;

            // Stage 10: SwiGLU Gating (GPU)
            nr.launch(
                &stream,
                &gate_dev,
                &zff,
                &proj_dev,
                &up_dev,
                &proj_dev,
                eps,
                ff,
                0,
                base,
                0,
                MODE_SWIGLU,
            )?;
            let proj_host = download_f32(&stream, &proj_dev, ff)?;

            // Stage 11: FFN Down CPU Fallback (Q6_K) + Residual 2 (Host)
            let mut down_host = vec![0.0f32; h];
            matmul(&mut down_host, &layer.wdown, &proj_host);
            let mut h2_host = vec![0.0f32; h];
            for i in 0..h {
                h2_host[i] = h1_host[i] + down_host[i];
            }
            x_host = h2_host;
        }
    }

    let logits = logits_from_hidden(&emb, &head_norm, &x_host, eps);
    stream.sync()?;
    Ok(PrefillResult {
        logits,
        tokens: seq_len,
    })
}
