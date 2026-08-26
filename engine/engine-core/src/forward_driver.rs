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
use crate::forward_cpu::{Tensor, TensorType, embed_lookup, logits_from_hidden};
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

/// Breakdown of transient and resident device memory footprint tracked by the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramFootprint {
    /// Device memory occupied by uploaded layer weights (bytes).
    pub pingpong_bytes: usize,
    /// Device memory occupied by resident per-layer KV pools (bytes).
    pub kv_pool_bytes: usize,
    /// Device memory occupied by reusable scratch activations buffers (bytes).
    pub activations_bytes: usize,
    /// Device memory required for output logits (bytes).
    pub logits_bytes: usize,
}

impl VramFootprint {
    /// Total VRAM footprint across all four tracked buckets.
    pub fn total(&self) -> usize {
        self.pingpong_bytes + self.kv_pool_bytes + self.activations_bytes + self.logits_bytes
    }

    /// Asserts that the total footprint does not exceed the specified budget.
    pub fn assert_within_budget(&self, budget: usize) -> Result<(), EngineError> {
        if self.total() > budget {
            return Err(engine_cuda::CudaError::InvalidSize {
                expected: budget,
                actual: self.total(),
            }
            .into());
        }
        Ok(())
    }

    /// Logs or prints a formatted budget trace.
    pub fn print_trace(&self) {
        println!("=== VRAM Budget Trace (ForwardDriver) ===");
        println!(
            "  pingpong/weights: {:>10} bytes ({:.2} MB)",
            self.pingpong_bytes,
            self.pingpong_bytes as f64 / (1024.0 * 1024.0)
        );
        println!(
            "  kv_pool:          {:>10} bytes ({:.2} MB)",
            self.kv_pool_bytes,
            self.kv_pool_bytes as f64 / (1024.0 * 1024.0)
        );
        println!(
            "  activations:      {:>10} bytes ({:.2} MB)",
            self.activations_bytes,
            self.activations_bytes as f64 / (1024.0 * 1024.0)
        );
        println!(
            "  logits:           {:>10} bytes ({:.2} MB)",
            self.logits_bytes,
            self.logits_bytes as f64 / (1024.0 * 1024.0)
        );
        println!(
            "  TOTAL:            {:>10} bytes ({:.2} MB / {:.3} GB)",
            self.total(),
            self.total() as f64 / (1024.0 * 1024.0),
            self.total() as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!(
            "  BUDGET:           {:>10} bytes ({:.2} GB)",
            VRAM_BUDGET_BYTES,
            VRAM_BUDGET_BYTES as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
}

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
    stream.sync()?;
    Ok(bytes_f32(&raw))
}

fn ggml_to_gemv(t: TensorType) -> Result<GemvFormat, EngineError> {
    match t {
        TensorType::Q4K => Ok(GemvFormat::Q4K),
        TensorType::Q6K => Ok(GemvFormat::Q6K),
        TensorType::Q8 => Ok(GemvFormat::Q8),
        TensorType::F16 => Ok(GemvFormat::F16),
        other => Err(EngineError::Validation(format!(
            "unsupported GEMV quant format {:?}",
            other
        ))),
    }
}

struct LayerRsrc<'a> {
    wq_dev: DeviceBuffer,
    wq_fmt: GemvFormat,
    wq_ne0: usize,
    wq_ne1: usize,
    wk_dev: DeviceBuffer,
    wk_fmt: GemvFormat,
    wk_ne0: usize,
    wk_ne1: usize,
    wv_dev: DeviceBuffer,
    wv_fmt: GemvFormat,
    wv_ne0: usize,
    wv_ne1: usize,
    wo_dev: DeviceBuffer,
    wo_fmt: GemvFormat,
    wo_ne0: usize,
    wo_ne1: usize,
    wgate_dev: DeviceBuffer,
    wgate_fmt: GemvFormat,
    wgate_ne0: usize,
    wgate_ne1: usize,
    wup_dev: DeviceBuffer,
    wup_fmt: GemvFormat,
    wup_ne0: usize,
    wup_ne1: usize,
    wdown_dev: DeviceBuffer,
    wdown_fmt: GemvFormat,
    wdown_ne0: usize,
    wdown_ne1: usize,
    an_dev: DeviceBuffer,
    qn_dev: DeviceBuffer,
    kn_dev: DeviceBuffer,
    fn_dev: DeviceBuffer,
    pool_dev: DeviceBuffer,
    _marker: std::marker::PhantomData<&'a ()>,
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
    pos_dev: DeviceBuffer,
    zq: DeviceBuffer,
    zk: DeviceBuffer,
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
    decode_graph: Option<engine_cuda::CudaGraphExec>,
    // dims
    h: usize,
    hd: usize,
    nh: usize,
    nkv: usize,
    hff: usize,
    _qdim: usize,
    _kvd: usize,
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
            let fn_ = f32_norm(pinned, &format!("blk.{l}.ffn_norm.weight"))?;

            let wq_dev = upload_bytes(&stream, &device, &wq.data)?;
            let wk_dev = upload_bytes(&stream, &device, &wk.data)?;
            let wv_dev = upload_bytes(&stream, &device, &wv.data)?;
            let wo_dev = upload_bytes(&stream, &device, &wo.data)?;
            let wgate_dev = upload_bytes(&stream, &device, &wgate.data)?;
            let wup_dev = upload_bytes(&stream, &device, &wup.data)?;
            let wdown_dev = upload_bytes(&stream, &device, &wdown.data)?;

            let an_dev = upload_f32(&stream, &device, &an)?;
            let qn_dev = upload_f32(&stream, &device, &qn)?;
            let kn_dev = upload_f32(&stream, &device, &kn)?;
            let fn_dev = upload_f32(&stream, &device, &fn_)?;

            let pool_dev = upload_bytes(
                &stream,
                &device,
                &vec![0u8; layout.floats_total() * 4],
            )?;

            layers.push(LayerRsrc {
                wq_dev,
                wq_fmt: ggml_to_gemv(wq.ty)?,
                wq_ne0: wq.ne0,
                wq_ne1: wq.ne1,
                wk_dev,
                wk_fmt: ggml_to_gemv(wk.ty)?,
                wk_ne0: wk.ne0,
                wk_ne1: wk.ne1,
                wv_dev,
                wv_fmt: ggml_to_gemv(wv.ty)?,
                wv_ne0: wv.ne0,
                wv_ne1: wv.ne1,
                wo_dev,
                wo_fmt: ggml_to_gemv(wo.ty)?,
                wo_ne0: wo.ne0,
                wo_ne1: wo.ne1,
                wgate_dev,
                wgate_fmt: ggml_to_gemv(wgate.ty)?,
                wgate_ne0: wgate.ne0,
                wgate_ne1: wgate.ne1,
                wup_dev,
                wup_fmt: ggml_to_gemv(wup.ty)?,
                wup_ne0: wup.ne0,
                wup_ne1: wup.ne1,
                wdown_dev,
                wdown_fmt: ggml_to_gemv(wdown.ty)?,
                wdown_ne0: wdown.ne0,
                wdown_ne1: wdown.ne1,
                an_dev,
                qn_dev,
                kn_dev,
                fn_dev,
                pool_dev,
                _marker: std::marker::PhantomData,
            });
        }

        let pos_dev = upload_bytes(&stream, &device, &0u32.to_le_bytes())?;
        let zq = upload_f32(&stream, &device, &vec![0.0f32; qdim])?;
        let zk = upload_f32(&stream, &device, &vec![0.0f32; kvd])?;
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

        let driver = Self {
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
            pos_dev,
            zq,
            zk,
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
            decode_graph: None,
            h,
            hd,
            nh,
            nkv,
            hff,
            _qdim: qdim,
            _kvd: kvd,
            n_rot,
            eps,
            base,
            n_layer,
            pos: 0,
        };
        driver
            .vram_footprint()
            .assert_within_budget(VRAM_BUDGET_BYTES)?;
        Ok(driver)
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

    /// Returns the number of transformer layers in this model.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Records the full 28-layer forward decode pass onto `self.stream`.
    fn record_decode_pass(&self) -> Result<(), EngineError> {
        for layer in &self.layers {
            // Stage 1: Input RMSNorm (GPU)
            self.nr.launch(
                &self.stream,
                &self.x_dev,
                &self.zh,
                &layer.an_dev,
                &self.zh,
                &self.input_norm_dev,
                self.eps,
                self.h,
                0,
                self.base,
                0,
                MODE_NORM,
            )?;

            // Stage 2: Q/K GEMV
            self.gemv.gemv(
                &self.stream,
                layer.wq_fmt,
                &layer.wq_dev,
                &self.input_norm_dev,
                &self.q_dev,
                layer.wq_ne0,
                layer.wq_ne1,
            )?;
            self.gemv.gemv(
                &self.stream,
                layer.wk_fmt,
                &layer.wk_dev,
                &self.input_norm_dev,
                &self.k_dev,
                layer.wk_ne0,
                layer.wk_ne1,
            )?;

            // Stage 3: V GEMV
            self.gemv.gemv(
                &self.stream,
                layer.wv_fmt,
                &layer.wv_dev,
                &self.input_norm_dev,
                &self.v_dev,
                layer.wv_ne0,
                layer.wv_ne1,
            )?;

            // Stage 4: Per-head Q/K RMSNorm + RoPE (100% GPU, fused in single launch)
            self.nr.launch_with_pos_ptr(
                &self.stream,
                &self.q_dev,
                &self.zq,
                &layer.qn_dev,
                &self.zq,
                &self.q_dev,
                self.eps,
                self.hd,
                self.n_rot,
                self.base,
                0,
                MODE_NORM | MODE_ROPE,
                Some(&self.pos_dev),
            )?;

            self.nr.launch_with_pos_ptr(
                &self.stream,
                &self.k_dev,
                &self.zk,
                &layer.kn_dev,
                &self.zk,
                &self.k_dev,
                self.eps,
                self.hd,
                self.n_rot,
                self.base,
                0,
                MODE_NORM | MODE_ROPE,
                Some(&self.pos_dev),
            )?;

            // Stage 5: Paged KV Append (GPU)
            self.pkv.append_kv_with_pos_ptr(
                &self.stream,
                &self.layout,
                &layer.pool_dev,
                &self.k_dev,
                &self.v_dev,
                &self.bt_dev,
                0,
                1,
                Some(&self.pos_dev),
            )?;

            // Stage 6: PagedAttention Decode (GPU)
            self.pa.launch_with_pos_ptr(
                &self.stream,
                &self.q_dev,
                &layer.pool_dev,
                &self.bt_dev,
                &self.attn_dev,
                self.nh,
                self.nkv,
                self.hd,
                self.layout.block_tokens,
                1,
                0,
                true,
                Some(&self.pos_dev),
            )?;

            // Stage 7: Output Projection + Residual 1
            self.gemv.gemv(
                &self.stream,
                layer.wo_fmt,
                &layer.wo_dev,
                &self.attn_dev,
                &self.op_dev,
                layer.wo_ne0,
                layer.wo_ne1,
            )?;
            self.nr.launch(
                &self.stream,
                &self.op_dev,
                &self.x_dev,
                &self.zh,
                &self.zh,
                &self.h1_dev,
                self.eps,
                self.h,
                0,
                self.base,
                0,
                0,
            )?;

            // Stage 8: FFN Norm
            self.nr.launch(
                &self.stream,
                &self.h1_dev,
                &self.zh,
                &layer.fn_dev,
                &self.zh,
                &self.ffin_dev,
                self.eps,
                self.h,
                0,
                self.base,
                0,
                MODE_NORM,
            )?;

            // Stage 9: FFN Gate & Up
            self.gemv.gemv(
                &self.stream,
                layer.wgate_fmt,
                &layer.wgate_dev,
                &self.ffin_dev,
                &self.gate_dev,
                layer.wgate_ne0,
                layer.wgate_ne1,
            )?;
            self.gemv.gemv(
                &self.stream,
                layer.wup_fmt,
                &layer.wup_dev,
                &self.ffin_dev,
                &self.up_dev,
                layer.wup_ne0,
                layer.wup_ne1,
            )?;

            // Stage 10: SwiGLU Gating
            self.nr.launch(
                &self.stream,
                &self.gate_dev,
                &self.zff,
                &self.zff,
                &self.up_dev,
                &self.gate_dev,
                self.eps,
                self.hff,
                0,
                self.base,
                0,
                MODE_SWIGLU,
            )?;

            // Stage 11: FFN Down + Residual 2
            self.gemv.gemv(
                &self.stream,
                layer.wdown_fmt,
                &layer.wdown_dev,
                &self.gate_dev,
                &self.proj_dev,
                layer.wdown_ne0,
                layer.wdown_ne1,
            )?;
            self.nr.launch(
                &self.stream,
                &self.proj_dev,
                &self.h1_dev,
                &self.zh,
                &self.zh,
                &self.x_dev,
                self.eps,
                self.h,
                0,
                self.base,
                0,
                0,
            )?;
        }
        Ok(())
    }

    /// Captures the full 28-layer transformer decode pass into a CUDA graph.
    pub fn capture_decode_graph(&mut self) -> Result<(), EngineError> {
        self.stream.begin_capture()?;
        self.record_decode_pass()?;
        let graph = self.stream.end_capture()?;
        let exec = graph.instantiate()?;
        self.decode_graph = Some(exec);
        Ok(())
    }

    /// Fast single-token decode using pre-captured CUDA graph with single driver launch.
    pub fn decode_graph(&mut self, token: u32) -> Result<Vec<f32>, EngineError> {
        if self.decode_graph.is_none() {
            self.capture_decode_graph()?;
        }

        let p = self.pos;
        if p >= self.layout.block_tokens {
            return Err(engine_cuda::CudaError::InvalidSize {
                expected: self.layout.block_tokens,
                actual: p + 1,
            }
            .into());
        }

        let x_host = embed_lookup(&self.emb, token as usize);
        self.x_dev
            .copy_from_host(&self.stream, &f32_bytes(&x_host))?;
        let pos_bytes = (p as u32).to_le_bytes();
        self.pos_dev.copy_from_host(&self.stream, &pos_bytes)?;

        if let Some(ref exec) = self.decode_graph {
            exec.launch(&self.stream)?;
        }

        let last_hidden = download_f32(&self.stream, &self.x_dev, self.h)?;
        self.pos += 1;
        self.stream.sync()?;
        Ok(self.lm_head(&last_hidden))
    }

    /// Single-token decode step over resident KV pool at current `self.pos`.
    /// Executes via persistent single-launch CUDA Graph with device-side dynamic position advancing.
    pub fn decode(&mut self, token: u32) -> Result<Vec<f32>, EngineError> {
        self.decode_graph(token)
    }

    /// Per-token single-topology forward pass at position `p`.
    fn step_one(&mut self, token: u32, p: usize) -> Result<Vec<f32>, EngineError> {
        self.vram_footprint()
            .assert_within_budget(VRAM_BUDGET_BYTES)?;
        let x_host = embed_lookup(&self.emb, token as usize);
        self.x_dev
            .copy_from_host(&self.stream, &f32_bytes(&x_host))?;
        let pos_bytes = (p as u32).to_le_bytes();
        self.pos_dev.copy_from_host(&self.stream, &pos_bytes)?;

        self.record_decode_pass()?;

        let last_hidden = download_f32(&self.stream, &self.x_dev, self.h)?;
        Ok(last_hidden)
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

    /// Computes the VRAM footprint across all layers and transient buffers.
    pub fn vram_footprint(&self) -> VramFootprint {
        let pingpong_bytes: usize = self
            .layers
            .iter()
            .map(|l| {
                l.wq_dev.size()
                    + l.wk_dev.size()
                    + l.wv_dev.size()
                    + l.wo_dev.size()
                    + l.wgate_dev.size()
                    + l.wup_dev.size()
                    + l.wdown_dev.size()
                    + l.an_dev.size()
                    + l.qn_dev.size()
                    + l.kn_dev.size()
                    + l.fn_dev.size()
            })
            .sum();

        let kv_pool_bytes: usize = self.layers.iter().map(|l| l.pool_dev.size()).sum();

        let activations_bytes: usize = self.zh.size()
            + self.zhd.size()
            + self.zff.size()
            + self.bt_dev.size()
            + self.x_dev.size()
            + self.input_norm_dev.size()
            + self.q_dev.size()
            + self.k_dev.size()
            + self.v_dev.size()
            + self.head_dev.size()
            + self.attn_dev.size()
            + self.op_dev.size()
            + self.h1_dev.size()
            + self.ffin_dev.size()
            + self.gate_dev.size()
            + self.up_dev.size()
            + self.proj_dev.size();

        let logits_bytes: usize = self.emb.ne1 * 4;

        VramFootprint {
            pingpong_bytes,
            kv_pool_bytes,
            activations_bytes,
            logits_bytes,
        }
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
