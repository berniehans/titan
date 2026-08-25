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

/// Forward driver executing transformer prefill and single-token decode steps
/// over GPU kernels and CPU fallbacks.
pub struct ForwardDriver<'a> {
    device: Arc<CudaDevice>,
    stream: CudaStream,
    gemv: MultiFormatGEMV,
    nr: NormRope,
    pkv: PagedKvGpu,
    pa: PagedAttention,
    layout: PagedKvLayout,
    emb: Tensor<'a>,
    head_norm: Vec<f32>,
    layers: Vec<LayerRsrc<'a>>,
    // scratch (reused, no per-step alloc):
    zh: DeviceBuffer,
    zhd: DeviceBuffer,
    zff: DeviceBuffer,
    bt_dev: DeviceBuffer,
    x_dev: DeviceBuffer,
    input_norm_dev: DeviceBuffer,
    q_dev: DeviceBuffer,
    k_dev: DeviceBuffer,
    v_dev: DeviceBuffer,
    head_dev: DeviceBuffer,
    attn_dev: DeviceBuffer,
    op_dev: DeviceBuffer,
    h1_dev: DeviceBuffer,
    ffin_dev: DeviceBuffer,
    gate_dev: DeviceBuffer,
    up_dev: DeviceBuffer,
    proj_dev: DeviceBuffer,
    // dims
    h: usize,
    hd: usize,
    nh: usize,
    nkv: usize,
    hff: usize,
    qdim: usize,
    kvd: usize,
    n_rot: usize,
    eps: f32,
    base: f32,
    n_layer: usize,
    pos: usize,
}

impl<'a> ForwardDriver<'a> {
    /// Creates a new `ForwardDriver` allocating resident per-layer KV pools and scratch buffers
    /// sized for up to `capacity_tokens` tokens.
    pub fn new(
        reader: &GgufReader,
        pinned: &'a LoadedPinned,
        cfg: &ModelConfig,
        capacity_tokens: usize,
    ) -> Result<Self, EngineError> {
        let capacity = capacity_tokens.max(1);
        let device = CudaDevice::new(0)?;
        let stream = CudaStream::new(Arc::clone(&device))?;
        let gemv = MultiFormatGEMV::new(Arc::clone(&device))?;
        let nr = NormRope::new(Arc::clone(&device))?;
        let pkv = PagedKvGpu::new(Arc::clone(&device))?;
        let pa = PagedAttention::new(Arc::clone(&device))?;

        let h = cfg.hidden_size as usize;
        let hd = cfg.head_dim as usize;
        let nh = cfg.n_head as usize;
        let nkv = cfg.n_head_kv as usize;
        let hff = cfg.intermediate_size as usize;
        let eps = cfg.rms_norm_eps;
        let base = cfg.rope_freq_base;
        // Qwen3 RoPE rotates the FULL head dim (NeoX pairs over [0, head_dim));
        // the head_dim/2 assumption was falsified by the golden gate (cos 0.71 vs
        // 0.9986 at pos>=1), so n_rot = hd (128 for this fixture).
        let n_rot = hd;
        let qdim = nh * hd;
        let kvd = nkv * hd;
        let n_layer = cfg.n_layer as usize;

        let layout = PagedKvLayout {
            n_blocks: 1,
            block_tokens: capacity,
            row_len: kvd,
        };

        let emb = bank_tensor(reader, pinned, "token_embd.weight")?;
        let head_norm = f32_norm(pinned, "output_norm.weight")?;

        let mut layers: Vec<LayerRsrc<'a>> = Vec::with_capacity(n_layer);
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
        let zff = upload_f32(&stream, &device, &vec![0.0f32; hff])?;
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
        let gate_dev = alloc_dev(&device, hff)?;
        let up_dev = alloc_dev(&device, hff)?;
        let proj_dev = alloc_dev(&device, hff)?;

        Ok(Self {
            device,
            stream,
            gemv,
            nr,
            pkv,
            pa,
            layout,
            emb,
            head_norm,
            layers,
            zh,
            zhd,
            zff,
            bt_dev,
            x_dev,
            input_norm_dev,
            q_dev,
            k_dev,
            v_dev,
            head_dev,
            attn_dev,
            op_dev,
            h1_dev,
            ffin_dev,
            gate_dev,
            up_dev,
            proj_dev,
            h,
            hd,
            nh,
            nkv,
            hff,
            qdim,
            kvd,
            n_rot,
            eps,
            base,
            n_layer,
            pos: 0,
        })
    }

    /// Appends all `tokens` sequentially into resident KV starting at `pos=0`,
    /// updates `pos = tokens.len()`, and returns the final position's next-token logits.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, EngineError> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        if tokens.len() > self.layout.block_tokens {
            return Err(engine_cuda::CudaError::InvalidSize {
                expected: self.layout.block_tokens,
                actual: tokens.len(),
            }
            .into());
        }
        let mut last_hidden = Vec::new();
        for (p, &token_id) in tokens.iter().enumerate() {
            last_hidden = self.step_one(token_id, p)?;
        }
        self.pos = tokens.len();
        self.stream.sync()?;
        Ok(self.lm_head(&last_hidden))
    }

    /// Single-token decode step over resident KV pool at current `self.pos`.
    /// Increments `self.pos` by 1 and returns the next-token logits.
    pub fn decode(&mut self, token: u32) -> Result<Vec<f32>, EngineError> {
        if self.pos >= self.layout.block_tokens {
            return Err(engine_cuda::CudaError::InvalidSize {
                expected: self.layout.block_tokens,
                actual: self.pos + 1,
            }
            .into());
        }
        let h = self.step_one(token, self.pos)?;
        self.pos += 1;
        self.stream.sync()?;
        Ok(self.lm_head(&h))
    }

    /// Per-token single-topology forward pass at position `p`. Takes `&mut self`
    /// because it mutates the resident KV pools / scratch device buffers through
    /// interior mutability (`copy_from_host`, `append_kv`).
    fn step_one(&mut self, token: u32, p: usize) -> Result<Vec<f32>, EngineError> {
        let mut x_host = embed_lookup(&self.emb, token as usize);

        for layer in &self.layers {
            // Stage 1: Input RMSNorm (GPU)
            self.x_dev
                .copy_from_host(&self.stream, &f32_bytes(&x_host))?;
            self.nr.launch(
                &self.stream,
                &self.x_dev,
                &self.zh,
                &layer.an_dev,
                &self.input_norm_dev,
                &self.input_norm_dev,
                self.eps,
                self.h,
                0,
                self.base,
                0,
                MODE_NORM,
            )?;

            // Stage 2: Q/K GEMV (GPU Q4_K)
            self.gemv.gemv(
                &self.stream,
                GemvFormat::Q4K,
                &layer.wq_dev,
                &self.input_norm_dev,
                &self.q_dev,
                layer.wq_ne0,
                layer.wq_ne1,
            )?;
            self.gemv.gemv(
                &self.stream,
                GemvFormat::Q4K,
                &layer.wk_dev,
                &self.input_norm_dev,
                &self.k_dev,
                layer.wk_ne0,
                layer.wk_ne1,
            )?;

            // Stage 3: V CPU Fallback (Q6_K)
            let input_norm_host = download_f32(&self.stream, &self.input_norm_dev, self.h)?;
            let mut v_host = vec![0.0f32; self.kvd];
            matmul(&mut v_host, &layer.wv, &input_norm_host);
            self.v_dev
                .copy_from_host(&self.stream, &f32_bytes(&v_host))?;

            // Stage 4: Per-head Q/K RMSNorm + RoPE (GPU)
            let mut qhost = download_f32(&self.stream, &self.q_dev, self.qdim)?;
            let mut khost = download_f32(&self.stream, &self.k_dev, self.kvd)?;

            for hh in 0..self.nh {
                self.head_dev.copy_from_host(
                    &self.stream,
                    &f32_bytes(&qhost[hh * self.hd..(hh + 1) * self.hd]),
                )?;
                self.nr.launch(
                    &self.stream,
                    &self.head_dev,
                    &self.zhd,
                    &layer.qn_dev,
                    &self.head_dev,
                    &self.head_dev,
                    self.eps,
                    self.hd,
                    0,
                    self.base,
                    0,
                    MODE_NORM,
                )?;
                self.nr.launch(
                    &self.stream,
                    &self.head_dev,
                    &self.zhd,
                    &self.zhd,
                    &self.head_dev,
                    &self.head_dev,
                    self.eps,
                    self.hd,
                    self.n_rot,
                    self.base,
                    p as u32,
                    MODE_ROPE,
                )?;
                let out = download_f32(&self.stream, &self.head_dev, self.hd)?;
                qhost[hh * self.hd..(hh + 1) * self.hd].copy_from_slice(&out);
            }

            for hh in 0..self.nkv {
                self.head_dev.copy_from_host(
                    &self.stream,
                    &f32_bytes(&khost[hh * self.hd..(hh + 1) * self.hd]),
                )?;
                self.nr.launch(
                    &self.stream,
                    &self.head_dev,
                    &self.zhd,
                    &layer.kn_dev,
                    &self.head_dev,
                    &self.head_dev,
                    self.eps,
                    self.hd,
                    0,
                    self.base,
                    0,
                    MODE_NORM,
                )?;
                self.nr.launch(
                    &self.stream,
                    &self.head_dev,
                    &self.zhd,
                    &self.zhd,
                    &self.head_dev,
                    &self.head_dev,
                    self.eps,
                    self.hd,
                    self.n_rot,
                    self.base,
                    p as u32,
                    MODE_ROPE,
                )?;
                let out = download_f32(&self.stream, &self.head_dev, self.hd)?;
                khost[hh * self.hd..(hh + 1) * self.hd].copy_from_slice(&out);
            }

            self.q_dev
                .copy_from_host(&self.stream, &f32_bytes(&qhost))?;
            self.k_dev
                .copy_from_host(&self.stream, &f32_bytes(&khost))?;

            // Stage 5: Paged KV Append (GPU)
            self.pkv.append_kv(
                &self.stream,
                &self.layout,
                &layer.pool_dev,
                &self.k_dev,
                &self.v_dev,
                &self.bt_dev,
                p,
                1,
            )?;

            // Stage 6: PagedAttention Decode (GPU)
            self.pa.launch(
                &self.stream,
                &self.q_dev,
                &layer.pool_dev,
                &self.bt_dev,
                &self.attn_dev,
                self.nh,
                self.nkv,
                self.hd,
                self.layout.block_tokens,
                p + 1,
                p,
                true,
            )?;

            // Stage 7: Output Projection (GPU Q4_K) + Residual 1 (Host)
            self.gemv.gemv(
                &self.stream,
                GemvFormat::Q4K,
                &layer.wo_dev,
                &self.attn_dev,
                &self.op_dev,
                layer.wo_ne0,
                layer.wo_ne1,
            )?;
            let op_host = download_f32(&self.stream, &self.op_dev, self.h)?;
            let mut h1_host = vec![0.0f32; self.h];
            for i in 0..self.h {
                h1_host[i] = x_host[i] + op_host[i];
            }

            // Stage 8: FFN Norm (GPU)
            self.h1_dev
                .copy_from_host(&self.stream, &f32_bytes(&h1_host))?;
            self.nr.launch(
                &self.stream,
                &self.h1_dev,
                &self.zh,
                &layer.fn_dev,
                &self.ffin_dev,
                &self.ffin_dev,
                self.eps,
                self.h,
                0,
                self.base,
                0,
                MODE_NORM,
            )?;

            // Stage 9: FFN Gate & Up (GPU Q4_K)
            self.gemv.gemv(
                &self.stream,
                GemvFormat::Q4K,
                &layer.wgate_dev,
                &self.ffin_dev,
                &self.gate_dev,
                layer.wgate_ne0,
                layer.wgate_ne1,
            )?;
            self.gemv.gemv(
                &self.stream,
                GemvFormat::Q4K,
                &layer.wup_dev,
                &self.ffin_dev,
                &self.up_dev,
                layer.wup_ne0,
                layer.wup_ne1,
            )?;

            // Stage 10: SwiGLU Gating (GPU)
            self.nr.launch(
                &self.stream,
                &self.gate_dev,
                &self.zff,
                &self.proj_dev,
                &self.up_dev,
                &self.proj_dev,
                self.eps,
                self.hff,
                0,
                self.base,
                0,
                MODE_SWIGLU,
            )?;
            let proj_host = download_f32(&self.stream, &self.proj_dev, self.hff)?;

            // Stage 11: FFN Down CPU Fallback (Q6_K) + Residual 2 (Host)
            let mut down_host = vec![0.0f32; self.h];
            matmul(&mut down_host, &layer.wdown, &proj_host);
            let mut h2_host = vec![0.0f32; self.h];
            for i in 0..self.h {
                h2_host[i] = h1_host[i] + down_host[i];
            }
            x_host = h2_host;
        }

        Ok(x_host)
    }

    /// Computes next-token logits from the final hidden state.
    fn lm_head(&self, hid: &[f32]) -> Vec<f32> {
        logits_from_hidden(&self.emb, &self.head_norm, hid, self.eps)
    }

    /// Current token position in resident KV pool.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Total capacity in tokens for resident KV pool.
    pub fn capacity(&self) -> usize {
        self.layout.block_tokens
    }

    /// Reference to the underlying CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Number of transformer layers.
    pub fn n_layer(&self) -> usize {
        self.n_layer
    }
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

    let mut driver = ForwardDriver::new(reader, pinned, cfg, tokens.len())?;
    let logits = driver.prefill(tokens)?;
    Ok(PrefillResult {
        logits,
        tokens: tokens.len(),
    })
}
