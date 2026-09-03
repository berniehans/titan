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
    BatchedGEMM, CudaStream, DeviceBuffer, DispatchTelemetry, DispatchTelemetrySnapshot,
    FlashAttention2, GemvFormat, MODE_BROADCAST_RESIDUAL, MODE_NORM, MODE_ROPE, MODE_SWIGLU,
    NormRope, PagedAttention, PagedKvGpu, PagedKvLayout,
};
use engine_io::{GgmlType, GgufReader, LoadedPinned, ModelConfig};
use serde::Serialize;

use std::sync::Arc;

const FORWARD_DECODE_PATH_ENV: &str = "TITAN_FORWARD_DECODE_PATH";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodePath {
    F32,
    Q8,
}

impl DecodePath {
    fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q8 => "q8",
        }
    }
}

fn parse_decode_path(value: Option<&str>) -> Result<DecodePath, String> {
    match value.unwrap_or("f32") {
        "f32" => Ok(DecodePath::F32),
        "q8" => Ok(DecodePath::Q8),
        value => Err(format!(
            "invalid {FORWARD_DECODE_PATH_ENV}={value:?}; expected \"f32\" or \"q8\""
        )),
    }
}

#[cfg(test)]
mod decode_path_tests {
    use super::{DecodePath, parse_decode_path};

    #[test]
    fn parses_default_and_rejects_unknown_path() {
        assert_eq!(parse_decode_path(None).unwrap(), DecodePath::F32);
        assert_eq!(parse_decode_path(Some("f32")).unwrap(), DecodePath::F32);
        assert_eq!(parse_decode_path(Some("q8")).unwrap(), DecodePath::Q8);
        let error = parse_decode_path(Some("bogus")).unwrap_err();
        assert!(error.contains("TITAN_FORWARD_DECODE_PATH"));
        assert!(error.contains("f32"));
        assert!(error.contains("q8"));
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct DecodeStageTiming {
    pub elapsed_ms: Option<f64>,
    pub status: String,
}

impl DecodeStageTiming {
    fn unavailable() -> Self {
        Self {
            elapsed_ms: None,
            status: "not_applicable".into(),
        }
    }
    fn measured(value: f64) -> Self {
        Self {
            elapsed_ms: Some(value),
            status: "measured".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DecodeStageTimings {
    pub gemv_gemm: DecodeStageTiming,
    pub attention: DecodeStageTiming,
    pub ffn: DecodeStageTiming,
    pub lm_head: DecodeStageTiming,
    pub copies: DecodeStageTiming,
    pub waits: DecodeStageTiming,
    pub graph_replay: DecodeStageTiming,
}

impl Default for DecodeStageTimings {
    fn default() -> Self {
        Self {
            gemv_gemm: DecodeStageTiming::unavailable(),
            attention: DecodeStageTiming::unavailable(),
            ffn: DecodeStageTiming::unavailable(),
            lm_head: DecodeStageTiming::unavailable(),
            copies: DecodeStageTiming::unavailable(),
            waits: DecodeStageTiming::unavailable(),
            graph_replay: DecodeStageTiming::unavailable(),
        }
    }
}

impl DecodeStageTimings {
    pub fn measured(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> Self {
        Self {
            gemv_gemm: DecodeStageTiming::measured(a),
            attention: DecodeStageTiming::measured(b),
            ffn: DecodeStageTiming::measured(c),
            lm_head: DecodeStageTiming::measured(d),
            copies: DecodeStageTiming::measured(e),
            waits: DecodeStageTiming::measured(f),
            graph_replay: DecodeStageTiming::measured(0.0),
        }
    }
    pub fn measured_with_graph_replay(
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        e: f64,
        f: f64,
        graph: f64,
    ) -> Self {
        let mut value = Self::measured(a, b, c, d, e, f);
        value.graph_replay = DecodeStageTiming::measured(graph);
        value
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct DecodeTelemetryCounters {
    pub graph_launches: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecodeTelemetry {
    pub decode_ms: f64,
    pub stage_timings: DecodeStageTimings,
    pub counters: DecodeTelemetryCounters,
    pub overhead_ms: f64,
    pub overlap_ms: f64,
    pub attribution: &'static str,
    pub reconciliation_tolerance_ms: f64,
}

impl DecodeTelemetry {
    pub fn from_measured_stages(
        decode_ms: f64,
        stages: DecodeStageTimings,
        overhead_ms: f64,
        tolerance: f64,
        graph_launches: usize,
    ) -> Self {
        Self {
            decode_ms,
            stage_timings: stages,
            counters: DecodeTelemetryCounters { graph_launches },
            overhead_ms,
            overlap_ms: 0.0,
            attribution: "cuda_events_plus_host_boundaries",
            reconciliation_tolerance_ms: tolerance,
        }
    }
    pub fn from_measured_stages_with_accounting(
        decode_ms: f64,
        stages: DecodeStageTimings,
        graph_launches: usize,
    ) -> Self {
        let sum = Self::stage_sum(&stages);
        let (overhead, overlap) = if sum >= decode_ms {
            (0.0, sum - decode_ms)
        } else {
            (decode_ms - sum, 0.0)
        };
        let mut result =
            Self::from_measured_stages(decode_ms, stages, overhead, 0.001, graph_launches);
        result.overlap_ms = overlap;
        result
    }
    fn stage_sum(stages: &DecodeStageTimings) -> f64 {
        [
            &stages.gemv_gemm,
            &stages.attention,
            &stages.ffn,
            &stages.lm_head,
            &stages.copies,
            &stages.waits,
            &stages.graph_replay,
        ]
        .into_iter()
        .filter_map(|s| s.elapsed_ms)
        .sum()
    }
    pub fn stage_sum_ms(&self) -> f64 {
        Self::stage_sum(&self.stage_timings)
    }
    pub fn reconciles(&self, tolerance: f64) -> bool {
        (self.stage_sum_ms() + self.overhead_ms - self.overlap_ms - self.decode_ms).abs()
            <= tolerance
    }
}

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

pub(crate) fn bank_tensor<'a>(
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

pub(crate) fn f32_norm(pinned: &LoadedPinned, name: &str) -> Result<Vec<f32>, EngineError> {
    let b = pinned
        .tensor(name)
        .ok_or_else(|| engine_io::GgufError::MissingMetadata(name.to_string()))?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub(crate) fn f32_norm_opt(pinned: &LoadedPinned, name: &str, fallback_len: usize) -> Vec<f32> {
    if let Some(b) = pinned.tensor(name) {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else {
        vec![1.0f32; fallback_len]
    }
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

pub(crate) fn ggml_to_gemv(t: TensorType) -> Result<GemvFormat, EngineError> {
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
    _wup_ne0: usize,
    _wup_ne1: usize,
    wdown_dev: DeviceBuffer,
    wdown_fmt: GemvFormat,
    wdown_ne0: usize,
    wdown_ne1: usize,
    an_dev: DeviceBuffer,
    qn_dev: DeviceBuffer,
    kn_dev: DeviceBuffer,
    fn_dev: DeviceBuffer,
    qb_dev: DeviceBuffer,
    kb_dev: DeviceBuffer,
    vb_dev: DeviceBuffer,
    pool_dev: DeviceBuffer,
    _marker: std::marker::PhantomData<&'a ()>,
}

/// Maximum speculative batch size supported by preallocated device buffers.
pub const MAX_SPEC_K: usize = 8;

/// Forward driver executing transformer prefill and single-token decode steps
/// over GPU kernels and CPU fallbacks.
// GPU scratch buffers are preallocated for optional resident/speculative paths.
#[allow(dead_code)]
pub struct ForwardDriver<'a> {
    device: Arc<CudaDevice>,
    stream: CudaStream,
    batched_gemm: BatchedGEMM,
    decode_path: DecodePath,
    dispatch_telemetry: Option<Arc<DispatchTelemetry>>,
    decode_telemetry: Option<DecodeTelemetry>,
    nr: NormRope,
    pkv: PagedKvGpu,
    pa: PagedAttention,
    flash_attn: FlashAttention2,
    layout: PagedKvLayout,
    emb: Tensor<'a>,
    lm_head_weight: Tensor<'a>,
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
    qx_dev: DeviceBuffer,
    qd_dev: DeviceBuffer,
    qs_dev: DeviceBuffer,
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
    // speculative preallocated working buffers (max_k = 8):
    spec_x_dev: DeviceBuffer,
    spec_qx_dev: DeviceBuffer,
    spec_qd_dev: DeviceBuffer,
    spec_qs_dev: DeviceBuffer,
    spec_norm_dev: DeviceBuffer,
    spec_q_dev: DeviceBuffer,
    spec_k_dev: DeviceBuffer,
    spec_v_dev: DeviceBuffer,
    spec_attn_dev: DeviceBuffer,
    spec_op_dev: DeviceBuffer,
    spec_ffin_dev: DeviceBuffer,
    spec_gate_dev: DeviceBuffer,
    spec_up_dev: DeviceBuffer,
    spec_proj_dev: DeviceBuffer,
    spec_head_normed_dev: DeviceBuffer,
    spec_logits_dev: DeviceBuffer,
    spec_zh: DeviceBuffer,
    spec_zq: DeviceBuffer,
    spec_zk: DeviceBuffer,
    spec_zff: DeviceBuffer,
    // preallocated prefill scratch buffers (max_chunk = 512):
    prefill_tokens_dev: DeviceBuffer,
    prefill_x_dev: DeviceBuffer,
    prefill_norm_dev: DeviceBuffer,
    prefill_q_dev: DeviceBuffer,
    prefill_k_dev: DeviceBuffer,
    prefill_v_dev: DeviceBuffer,
    prefill_attn_dev: DeviceBuffer,
    prefill_op_dev: DeviceBuffer,
    prefill_ffin_dev: DeviceBuffer,
    prefill_gate_dev: DeviceBuffer,
    prefill_up_dev: DeviceBuffer,
    prefill_proj_dev: DeviceBuffer,
    prefill_zh_dev: DeviceBuffer,
    prefill_zq_dev: DeviceBuffer,
    prefill_zk_dev: DeviceBuffer,
    prefill_zff_dev: DeviceBuffer,
    decode_graph: Option<engine_cuda::CudaGraphExec>,
    autonomous_graph: Option<engine_cuda::CudaGraphExec>,
    speculative_graph: Option<engine_cuda::CudaGraphExec>,
    emb_dev: DeviceBuffer,
    token_id_dev: DeviceBuffer,
    selected_token_dev: DeviceBuffer,
    spec_tokens_dev: DeviceBuffer,
    splitk_scratch_dev: DeviceBuffer,
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
    has_qk_norm: bool,
    has_attn_bias: bool,
    lm_head_dev: DeviceBuffer,
    lm_head_fmt: GemvFormat,
    emb_fmt: GemvFormat,
    head_norm_dev: DeviceBuffer,
    head_normed_dev: DeviceBuffer,
    logits_dev: DeviceBuffer,
    vocab_size: usize,
    logits_host: Vec<f32>,
    history_dev: DeviceBuffer,
    step_counter_dev: DeviceBuffer,

    pub radix_tree: engine_kvcache::RadixTree,
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
        let decode_path = parse_decode_path(std::env::var(FORWARD_DECODE_PATH_ENV).ok().as_deref())
            .map_err(EngineError::Validation)?;
        println!("Configured forward decode path: {}", decode_path.as_str());
        let capacity = capacity_tokens.max(1);
        let device = CudaDevice::new(0)?;
        let stream = CudaStream::new(Arc::clone(&device))?;
        let batched_gemm = BatchedGEMM::new(Arc::clone(&device))?;
        let nr = NormRope::new(Arc::clone(&device))?;
        let pkv = PagedKvGpu::new(Arc::clone(&device))?;
        let pa = PagedAttention::new(Arc::clone(&device))?;
        let flash_attn = FlashAttention2::new(Arc::clone(&device))?;

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
            data_type: engine_cuda::KvDataType::F32,
        };

        let emb = bank_tensor(reader, pinned, "token_embd.weight")?;
        let lm_head_weight = if reader.get_tensor("output.weight").is_some() {
            bank_tensor(reader, pinned, "output.weight")?
        } else {
            emb
        };
        let head_norm = f32_norm(pinned, "output_norm.weight")?;

        let has_attn_bias = pinned.tensor("blk.0.attn_q.bias").is_some();
        let has_qk_norm = pinned.tensor("blk.0.attn_q_norm.weight").is_some();
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
            let qn = f32_norm_opt(pinned, &format!("blk.{l}.attn_q_norm.weight"), hd);
            let kn = f32_norm_opt(pinned, &format!("blk.{l}.attn_k_norm.weight"), hd);
            let fn_ = f32_norm(pinned, &format!("blk.{l}.ffn_norm.weight"))?;

            let qb = f32_norm_opt(pinned, &format!("blk.{l}.attn_q.bias"), qdim);
            let kb = f32_norm_opt(pinned, &format!("blk.{l}.attn_k.bias"), kvd);
            let vb = f32_norm_opt(pinned, &format!("blk.{l}.attn_v.bias"), kvd);

            let wq_dev = upload_bytes(&stream, &device, wq.data)?;
            let wk_dev = upload_bytes(&stream, &device, wk.data)?;
            let wv_dev = upload_bytes(&stream, &device, wv.data)?;
            let wo_dev = upload_bytes(&stream, &device, wo.data)?;
            let wgate_dev = upload_bytes(&stream, &device, wgate.data)?;
            let wup_dev = upload_bytes(&stream, &device, wup.data)?;
            let wdown_dev = upload_bytes(&stream, &device, wdown.data)?;

            let an_dev = upload_f32(&stream, &device, &an)?;
            let qn_dev = upload_f32(&stream, &device, &qn)?;
            let kn_dev = upload_f32(&stream, &device, &kn)?;
            let fn_dev = upload_f32(&stream, &device, &fn_)?;

            let qb_dev = upload_f32(&stream, &device, &qb)?;
            let kb_dev = upload_f32(&stream, &device, &kb)?;
            let vb_dev = upload_f32(&stream, &device, &vb)?;

            let pool_dev = upload_bytes(&stream, &device, &vec![0u8; layout.floats_total() * 4])?;

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
                _wup_ne0: wup.ne0,
                _wup_ne1: wup.ne1,
                wdown_dev,
                wdown_fmt: ggml_to_gemv(wdown.ty)?,
                wdown_ne0: wdown.ne0,
                wdown_ne1: wdown.ne1,
                an_dev,
                qn_dev,
                kn_dev,
                fn_dev,
                qb_dev,
                kb_dev,
                vb_dev,
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
        let bt_ids: Vec<u32> = (0..layout.n_blocks as u32).collect();
        let bt_dev = upload_bytes(
            &stream,
            &device,
            &bt_ids
                .iter()
                .flat_map(|id| id.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?;

        let x_dev = alloc_dev(&device, h)?;
        let max_quant_dim = h.max(hff).max(qdim);
        let qx_dev = upload_bytes(&stream, &device, &vec![0u8; max_quant_dim])?;
        let qd_dev = alloc_dev(&device, max_quant_dim / 32)?;
        let qs_dev = alloc_dev(&device, max_quant_dim / 32)?;
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
        let max_splitk_out = (cfg.vocab_size as usize).max(hff).max(h);
        let splitk_scratch_dev = alloc_dev(&device, 4 * max_splitk_out)?;

        const MAX_SPEC_K: usize = 8;
        let spec_x_dev = alloc_dev(&device, MAX_SPEC_K * h)?;
        let spec_qx_dev = upload_bytes(&stream, &device, &vec![0u8; MAX_SPEC_K * max_quant_dim])?;
        let spec_qd_dev = alloc_dev(&device, MAX_SPEC_K * (max_quant_dim / 32))?;
        let spec_qs_dev = alloc_dev(&device, MAX_SPEC_K * (max_quant_dim / 32))?;
        let spec_norm_dev = alloc_dev(&device, MAX_SPEC_K * h)?;
        let spec_q_dev = alloc_dev(&device, MAX_SPEC_K * qdim)?;
        let spec_k_dev = alloc_dev(&device, MAX_SPEC_K * kvd)?;
        let spec_v_dev = alloc_dev(&device, MAX_SPEC_K * kvd)?;
        let spec_attn_dev = alloc_dev(&device, MAX_SPEC_K * qdim)?;
        let spec_op_dev = alloc_dev(&device, MAX_SPEC_K * h)?;
        let spec_ffin_dev = alloc_dev(&device, MAX_SPEC_K * h)?;
        let spec_gate_dev = alloc_dev(&device, MAX_SPEC_K * hff)?;
        let spec_up_dev = alloc_dev(&device, MAX_SPEC_K * hff)?;
        let spec_proj_dev = alloc_dev(&device, MAX_SPEC_K * hff)?;

        let spec_zh = upload_f32(&stream, &device, &vec![0.0f32; MAX_SPEC_K * h])?;
        let spec_zq = upload_f32(&stream, &device, &vec![0.0f32; MAX_SPEC_K * qdim])?;
        let spec_zk = upload_f32(&stream, &device, &vec![0.0f32; MAX_SPEC_K * kvd])?;
        let spec_zff = upload_f32(&stream, &device, &vec![0.0f32; MAX_SPEC_K * hff])?;
        let lm_head_dev = upload_bytes(&stream, &device, lm_head_weight.data)?;
        let lm_head_fmt = ggml_to_gemv(lm_head_weight.ty)?;
        let head_norm_dev = upload_f32(&stream, &device, &head_norm)?;
        let head_normed_dev = alloc_dev(&device, h)?;
        let vocab_size = lm_head_weight.ne1;
        let logits_dev = alloc_dev(&device, vocab_size)?;

        let spec_head_normed_dev = alloc_dev(&device, MAX_SPEC_K * h)?;
        let spec_logits_dev = alloc_dev(&device, MAX_SPEC_K * vocab_size)?;
        let emb_dev = upload_bytes(&stream, &device, emb.data)?;
        let token_id_dev = upload_bytes(&stream, &device, &[0u8; 4])?;
        let selected_token_dev = upload_bytes(&stream, &device, &[0u8; 64])?;
        let spec_tokens_dev = upload_bytes(&stream, &device, &[0u8; 64])?;
        let history_dev = alloc_dev(&device, 2048)?;
        let step_counter_dev = upload_bytes(&stream, &device, &[0u8; 4])?;

        const MAX_PREFILL_CHUNK: usize = 512;
        let prefill_tokens_dev = alloc_dev(&device, MAX_PREFILL_CHUNK)?;
        let prefill_x_dev = alloc_dev(&device, MAX_PREFILL_CHUNK * h)?;
        let prefill_norm_dev = alloc_dev(&device, MAX_PREFILL_CHUNK * h)?;
        let prefill_q_dev = alloc_dev(&device, MAX_PREFILL_CHUNK * qdim)?;
        let prefill_k_dev = alloc_dev(&device, MAX_PREFILL_CHUNK * kvd)?;
        let prefill_v_dev = alloc_dev(&device, MAX_PREFILL_CHUNK * kvd)?;
        let prefill_attn_dev = alloc_dev(&device, MAX_PREFILL_CHUNK * qdim)?;
        let prefill_op_dev = alloc_dev(&device, MAX_PREFILL_CHUNK * h)?;
        let prefill_ffin_dev = alloc_dev(&device, MAX_PREFILL_CHUNK * h)?;
        let prefill_gate_dev = alloc_dev(&device, MAX_PREFILL_CHUNK * hff)?;
        let prefill_up_dev = alloc_dev(&device, MAX_PREFILL_CHUNK * hff)?;
        let prefill_proj_dev = alloc_dev(&device, MAX_PREFILL_CHUNK * hff)?;

        let prefill_zh_dev = upload_f32(&stream, &device, &vec![0.0f32; MAX_PREFILL_CHUNK * h])?;
        let prefill_zq_dev = upload_f32(&stream, &device, &vec![0.0f32; MAX_PREFILL_CHUNK * qdim])?;
        let prefill_zk_dev = upload_f32(&stream, &device, &vec![0.0f32; MAX_PREFILL_CHUNK * kvd])?;
        let prefill_zff_dev = upload_f32(&stream, &device, &vec![0.0f32; MAX_PREFILL_CHUNK * hff])?;

        let driver = Self {
            device,
            stream,
            batched_gemm,
            decode_path,
            nr,
            pkv,
            pa,
            flash_attn,
            layout,
            emb,
            lm_head_weight,
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
            qx_dev,
            qd_dev,
            qs_dev,
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
            spec_x_dev,
            spec_qx_dev,
            spec_qd_dev,
            spec_qs_dev,
            spec_norm_dev,
            spec_q_dev,
            spec_k_dev,
            spec_v_dev,
            spec_attn_dev,
            spec_op_dev,
            spec_ffin_dev,
            spec_gate_dev,
            spec_up_dev,
            spec_proj_dev,
            spec_head_normed_dev,
            spec_logits_dev,
            spec_zh,
            spec_zq,
            spec_zk,
            spec_zff,
            prefill_tokens_dev,
            prefill_x_dev,
            prefill_norm_dev,
            prefill_q_dev,
            prefill_k_dev,
            prefill_v_dev,
            prefill_attn_dev,
            prefill_op_dev,
            prefill_ffin_dev,
            prefill_gate_dev,
            prefill_up_dev,
            prefill_proj_dev,
            prefill_zh_dev,
            prefill_zq_dev,
            prefill_zk_dev,
            prefill_zff_dev,
            decode_graph: None,
            autonomous_graph: None,
            speculative_graph: None,
            emb_dev,
            token_id_dev,
            selected_token_dev,
            spec_tokens_dev,
            splitk_scratch_dev,
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
            has_qk_norm,
            has_attn_bias,
            lm_head_dev,
            lm_head_fmt,
            emb_fmt: ggml_to_gemv(emb.ty)?,
            head_norm_dev,
            head_normed_dev,
            logits_dev,
            vocab_size,
            logits_host: vec![0.0f32; vocab_size],
            history_dev,
            step_counter_dev,
            dispatch_telemetry: None,
            decode_telemetry: None,

            radix_tree: engine_kvcache::RadixTree::new(layout.block_tokens),
        };
        driver
            .vram_footprint()
            .assert_within_budget(VRAM_BUDGET_BYTES)?;
        Ok(driver)
    }

    /// Enables dispatch accounting for benchmark runs only.
    pub fn enable_dispatch_telemetry(&mut self) {
        let telemetry = Arc::new(DispatchTelemetry::new());
        self.batched_gemm
            .set_dispatch_telemetry(Some(Arc::clone(&telemetry)));
        self.dispatch_telemetry = Some(telemetry);
    }

    /// Returns benchmark dispatch telemetry for diagnostic probes.
    pub fn dispatch_telemetry_snapshot(&self) -> Option<DispatchTelemetrySnapshot> {
        self.dispatch_telemetry
            .as_ref()
            .map(|telemetry| telemetry.snapshot())
    }

    /// Takes aggregate decode telemetry when a reliable measurement is available.
    pub fn take_decode_telemetry(&mut self) -> Option<DecodeTelemetry> {
        self.decode_telemetry.take()
    }

    /// Evaluates `tokens` in parallel chunks using batched GEMM and FlashAttention-2,
    /// populating resident KV cache and returning the final position's next-token logits.
    #[allow(clippy::manual_clamp)] // Bounds are fixed and ordered: 1 <= 512.
    pub fn prefill_chunked(
        &mut self,
        tokens: &[u32],
        chunk_size: usize,
    ) -> Result<Vec<f32>, EngineError> {
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

        // 1. Radix Prefix Cache Match
        let match_res = self.radix_tree.match_prefix(tokens);
        let mut current_pos =
            if match_res.matched_tokens > 0 && self.pos == match_res.matched_tokens {
                match_res.matched_tokens
            } else {
                0
            };

        if current_pos == tokens.len() {
            let logits = download_f32(&self.stream, &self.logits_dev, self.vocab_size)?;
            self.stream.sync()?;
            return Ok(logits);
        }

        let chunk_limit = chunk_size.max(1).min(512);
        let mut last_hidden = Vec::new();
        let t_all_start = std::time::Instant::now();
        while current_pos < tokens.len() {
            let chunk_len = (tokens.len() - current_pos).min(chunk_limit);
            let chunk_tokens = &tokens[current_pos..current_pos + chunk_len];

            // 1. Embed tokens
            let mut x_host = Vec::with_capacity(chunk_len * self.h);
            for &t in chunk_tokens {
                let row = embed_lookup(&self.emb, t as usize);
                x_host.extend_from_slice(&row);
            }
            self.prefill_x_dev
                .copy_from_host(&self.stream, &f32_bytes(&x_host))?;
            let t_embed = t_all_start.elapsed();

            let t_layers_start = std::time::Instant::now();
            for layer in self.layers.iter() {
                // a & b. QKV Projection with Precomputed Input RMSNorm
                self.batched_gemm.set_telemetry_tensor_role(Some("qkv"));
                // b. Input RMSNorm with fused bias residual
                self.nr.launch_batched_with_pos_ptr(
                    &self.stream,
                    &self.prefill_x_dev,
                    &self.prefill_zh_dev,
                    &layer.an_dev,
                    &self.prefill_zh_dev,
                    &self.prefill_norm_dev,
                    self.eps,
                    self.h,
                    0,
                    0.0,
                    0,
                    MODE_NORM | MODE_BROADCAST_RESIDUAL,
                    None,
                    chunk_len,
                    1,
                )?;

                self.batched_gemm.gemm(
                    &self.stream,
                    &layer.wq_dev,
                    &self.prefill_norm_dev,
                    &self.prefill_q_dev,
                    self.h,
                    self._qdim,
                    chunk_len,
                    layer.wq_fmt,
                )?;
                self.batched_gemm.gemm(
                    &self.stream,
                    &layer.wk_dev,
                    &self.prefill_norm_dev,
                    &self.prefill_k_dev,
                    self.h,
                    self._kvd,
                    chunk_len,
                    layer.wk_fmt,
                )?;
                self.batched_gemm.gemm(
                    &self.stream,
                    &layer.wv_dev,
                    &self.prefill_norm_dev,
                    &self.prefill_v_dev,
                    self.h,
                    self._kvd,
                    chunk_len,
                    layer.wv_fmt,
                )?;

                // c. Q/K RMSNorm + RoPE (+ optional attention biases)
                let qk_mode = if self.has_qk_norm {
                    MODE_NORM | MODE_ROPE | MODE_BROADCAST_RESIDUAL
                } else {
                    MODE_ROPE | MODE_BROADCAST_RESIDUAL
                };
                let q_res = if self.has_attn_bias {
                    &layer.qb_dev
                } else {
                    &self.prefill_zq_dev
                };
                let k_res = if self.has_attn_bias {
                    &layer.kb_dev
                } else {
                    &self.prefill_zk_dev
                };

                self.nr.launch_batched_with_pos_ptr(
                    &self.stream,
                    &self.prefill_q_dev,
                    q_res,
                    &layer.qn_dev,
                    &self.prefill_zq_dev,
                    &self.prefill_q_dev,
                    self.eps,
                    self.hd,
                    self.n_rot,
                    self.base,
                    current_pos as u32,
                    qk_mode,
                    None,
                    self.nh * chunk_len,
                    self.nh,
                )?;
                self.nr.launch_batched_with_pos_ptr(
                    &self.stream,
                    &self.prefill_k_dev,
                    k_res,
                    &layer.kn_dev,
                    &self.prefill_zk_dev,
                    &self.prefill_k_dev,
                    self.eps,
                    self.hd,
                    self.n_rot,
                    self.base,
                    current_pos as u32,
                    qk_mode,
                    None,
                    self.nkv * chunk_len,
                    self.nkv,
                )?;

                // d. Append K, V chunk into resident paged KV pool
                self.pkv.append_kv(
                    &self.stream,
                    &self.layout,
                    &layer.pool_dev,
                    &self.prefill_k_dev,
                    &self.prefill_v_dev,
                    &self.bt_dev,
                    current_pos,
                    chunk_len,
                )?;

                // e. FlashAttention-2 causal prefill
                self.flash_attn.launch(
                    &self.stream,
                    &self.prefill_q_dev,
                    &layer.pool_dev,
                    &self.bt_dev,
                    &self.prefill_attn_dev,
                    self.nh,
                    self.nkv,
                    self.hd,
                    self.layout.block_tokens,
                    chunk_len,
                    current_pos,
                )?;

                // f. Output projection with Fused In-Place Residual 1
                self.batched_gemm
                    .set_telemetry_tensor_role(Some("attn_output"));
                self.batched_gemm.gemm_with_residual(
                    &self.stream,
                    &layer.wo_dev,
                    &self.prefill_attn_dev,
                    &self.prefill_op_dev,
                    self._qdim,
                    self.h,
                    chunk_len,
                    layer.wo_fmt,
                    Some(&self.prefill_x_dev),
                )?;

                // g & h. FFN gate & up projections with fused SwiGLU
                self.batched_gemm
                    .set_telemetry_tensor_role(Some("ffn_gate_up"));
                self.nr.launch_batched_with_pos_ptr(
                    &self.stream,
                    &self.prefill_op_dev,
                    &self.prefill_zh_dev,
                    &layer.fn_dev,
                    &self.prefill_zh_dev,
                    &self.prefill_ffin_dev,
                    self.eps,
                    self.h,
                    0,
                    0.0,
                    0,
                    MODE_NORM | MODE_BROADCAST_RESIDUAL,
                    None,
                    chunk_len,
                    1,
                )?;

                self.batched_gemm.gemm(
                    &self.stream,
                    &layer.wgate_dev,
                    &self.prefill_ffin_dev,
                    &self.prefill_gate_dev,
                    self.h,
                    self.hff,
                    chunk_len,
                    layer.wgate_fmt,
                )?;
                self.batched_gemm.gemm(
                    &self.stream,
                    &layer.wup_dev,
                    &self.prefill_ffin_dev,
                    &self.prefill_up_dev,
                    self.h,
                    self.hff,
                    chunk_len,
                    layer.wup_fmt,
                )?;
                self.nr.launch_batched_with_pos_ptr(
                    &self.stream,
                    &self.prefill_gate_dev,
                    &self.prefill_zff_dev,
                    &self.prefill_zff_dev,
                    &self.prefill_up_dev,
                    &self.prefill_proj_dev,
                    self.eps,
                    self.hff,
                    0,
                    0.0,
                    0,
                    MODE_SWIGLU,
                    None,
                    chunk_len,
                    1,
                )?;

                // i. Down projection with Fused In-Place Residual 2
                self.batched_gemm
                    .set_telemetry_tensor_role(Some("ffn_down"));
                self.batched_gemm.gemm_with_residual(
                    &self.stream,
                    &layer.wdown_dev,
                    &self.prefill_proj_dev,
                    &self.prefill_x_dev,
                    layer.wdown_ne0,
                    layer.wdown_ne1,
                    chunk_len,
                    layer.wdown_fmt,
                    Some(&self.prefill_op_dev),
                )?;
            }

            self.stream.sync()?;
            let t_layers = t_layers_start.elapsed();

            // Download last token hidden state
            let t_dl_start = std::time::Instant::now();
            let x_out_all = download_f32(&self.stream, &self.prefill_x_dev, chunk_len * self.h)?;
            let last_token_hidden =
                x_out_all[(chunk_len - 1) * self.h..chunk_len * self.h].to_vec();
            last_hidden = last_token_hidden;
            let t_dl = t_dl_start.elapsed();

            println!(
                "  [Prefill Timing] Tokens: {}, Embed: {:.2} ms, 28 Layers: {:.2} ms, DL: {:.2} ms",
                chunk_len,
                t_embed.as_secs_f64() * 1000.0,
                t_layers.as_secs_f64() * 1000.0,
                t_dl.as_secs_f64() * 1000.0
            );

            current_pos += chunk_len;
        }

        self.pos = tokens.len();
        self.pos_dev
            .copy_from_host(&self.stream, &(self.pos as u32).to_le_bytes())?;
        self.x_dev
            .copy_from_host(&self.stream, &f32_bytes(&last_hidden))?;
        self.record_lm_head_pass(&self.x_dev)?;
        let logits = download_f32(&self.stream, &self.logits_dev, self.vocab_size)?;
        self.stream.sync()?;
        self.radix_tree.insert(tokens, &[0], false);
        Ok(logits)
    }

    /// Verifies $K$ proposed candidate tokens in parallel in a single GPU pass.
    ///
    /// `base_token`: The certified token from the previous step to evaluate at `self.pos`.
    /// `candidates`: Slice of $K$ candidate tokens for positions `self.pos + 1 .. self.pos + K`.
    pub fn verify_speculative(
        &mut self,
        base_token: u32,
        candidates: &[u32],
        sampler: &mut crate::sampler::Sampler,
        params: &crate::sampler::SamplerParams,
        context: &[u32],
    ) -> Result<crate::speculative::SpeculativeVerificationResult, EngineError> {
        if candidates.is_empty() {
            let logits = self.decode(base_token)?;
            let next_tok = sampler.sample(&logits, context, params);
            return Ok(crate::speculative::SpeculativeVerificationResult {
                emitted_tokens: vec![next_tok],
                n_accepted: 0,
                bonus_token: next_tok,
                total_emitted: 1,
            });
        }

        let mut batch_tokens = Vec::with_capacity(candidates.len() + 1);
        batch_tokens.push(base_token);
        batch_tokens.extend_from_slice(candidates);

        let k = batch_tokens.len();
        let current_pos = self.pos;

        if current_pos + k > self.layout.block_tokens {
            let logits = self.decode(base_token)?;
            let next_tok = sampler.sample(&logits, context, params);
            return Ok(crate::speculative::SpeculativeVerificationResult {
                emitted_tokens: vec![next_tok],
                n_accepted: 0,
                bonus_token: next_tok,
                total_emitted: 1,
            });
        }

        if k > MAX_SPEC_K {
            let logits = self.decode(base_token)?;
            let next_tok = sampler.sample(&logits, context, params);
            return Ok(crate::speculative::SpeculativeVerificationResult {
                emitted_tokens: vec![next_tok],
                n_accepted: 0,
                bonus_token: next_tok,
                total_emitted: 1,
            });
        }

        if params.temperature <= 1e-6 && k == 4 {
            if self.speculative_graph.is_none() {
                self.capture_speculative_verification_graph(4)?;
            }

            let mut batch_token_bytes = [0u8; 16];
            for (i, &t) in batch_tokens.iter().enumerate() {
                batch_token_bytes[i * 4..(i + 1) * 4].copy_from_slice(&t.to_le_bytes());
            }
            self.spec_tokens_dev
                .copy_from_host(&self.stream, &batch_token_bytes)?;
            let pos_bytes = (current_pos as u32).to_le_bytes();
            self.pos_dev.copy_from_host(&self.stream, &pos_bytes)?;

            if let Some(ref exec) = self.speculative_graph {
                exec.launch(&self.stream)?;
            }

            let mut sampled_bytes = [0u8; 16];
            self.selected_token_dev
                .copy_to_host(&self.stream, &mut sampled_bytes)?;
            self.stream.sync()?;

            let mut target_preds = Vec::with_capacity(4);
            for chunk in sampled_bytes.chunks_exact(4) {
                target_preds.push(u32::from_le_bytes(chunk.try_into().unwrap()));
            }

            let mut n_accepted = 0;
            let mut emitted_tokens = Vec::with_capacity(4);
            for i in 0..candidates.len() {
                if candidates[i] == target_preds[i] {
                    n_accepted += 1;
                    emitted_tokens.push(candidates[i]);
                } else {
                    emitted_tokens.push(target_preds[i]);
                    break;
                }
            }

            let bonus_token = if n_accepted == candidates.len() {
                let bonus = target_preds[candidates.len()];
                emitted_tokens.push(bonus);
                bonus
            } else {
                target_preds[n_accepted]
            };

            let total_emitted = emitted_tokens.len();
            self.pos = current_pos + 1 + n_accepted;

            return Ok(crate::speculative::SpeculativeVerificationResult {
                emitted_tokens,
                n_accepted,
                bonus_token,
                total_emitted,
            });
        }

        // 1. Embed K candidate tokens on host and upload directly to preallocated buffer
        let mut x_host = Vec::with_capacity(k * self.h);
        for &tok in &batch_tokens {
            let emb_vec = embed_lookup(&self.emb, tok as usize);
            x_host.extend_from_slice(&emb_vec);
        }

        self.spec_x_dev
            .copy_from_host(&self.stream, &f32_bytes(&x_host))?;

        for layer in self.layers.iter() {
            // a. Quantize spec_x_dev with input RMSNorm -> (spec_qx_dev, spec_qd_dev, spec_qs_dev)
            self.batched_gemm
                .set_telemetry_tensor_role(Some("activation_quantization"));
            self.batched_gemm.quantize_q8_1_batched(
                &self.stream,
                &self.spec_x_dev,
                Some(&layer.an_dev),
                &self.spec_qx_dev,
                &self.spec_qd_dev,
                &self.spec_qs_dev,
                self.h,
                k,
                self.eps,
            )?;

            // b. Fused QKV Projection
            self.batched_gemm.set_telemetry_tensor_role(Some("qkv"));
            let (qb_opt, kb_opt, vb_opt) = if self.has_attn_bias {
                (
                    Some(&layer.qb_dev),
                    Some(&layer.kb_dev),
                    Some(&layer.vb_dev),
                )
            } else {
                (None, None, None)
            };
            if layer.wq_fmt == GemvFormat::Q4K
                && layer.wk_fmt == GemvFormat::Q4K
                && layer.wv_fmt == GemvFormat::Q4K
            {
                self.batched_gemm.set_telemetry_tensor_role(Some("qkv"));
                self.batched_gemm.gemm_fused_qkv_q4k(
                    &self.stream,
                    &layer.wq_dev,
                    &layer.wk_dev,
                    &layer.wv_dev,
                    &self.spec_qx_dev,
                    &self.spec_qd_dev,
                    &self.spec_qs_dev,
                    &self.spec_q_dev,
                    &self.spec_k_dev,
                    &self.spec_v_dev,
                    self.h,
                    self._qdim,
                    self._kvd,
                    k,
                    qb_opt,
                    kb_opt,
                    vb_opt,
                )?;
            } else if layer.wq_fmt == GemvFormat::Q4K
                && layer.wk_fmt == GemvFormat::Q4K
                && layer.wv_fmt == GemvFormat::Q6K
            {
                self.batched_gemm.gemm_fused_qkv(
                    &self.stream,
                    &layer.wq_dev,
                    &layer.wk_dev,
                    &layer.wv_dev,
                    &self.spec_qx_dev,
                    &self.spec_qd_dev,
                    &self.spec_qs_dev,
                    &self.spec_q_dev,
                    &self.spec_k_dev,
                    &self.spec_v_dev,
                    self.h,
                    self._qdim,
                    self._kvd,
                    k,
                    qb_opt,
                    kb_opt,
                    vb_opt,
                )?;
            } else {
                self.batched_gemm.gemm_q8_act_with_residual(
                    &self.stream,
                    &layer.wq_dev,
                    &self.spec_qx_dev,
                    &self.spec_qd_dev,
                    &self.spec_qs_dev,
                    &self.spec_q_dev,
                    layer.wq_ne0,
                    layer.wq_ne1,
                    k,
                    layer.wq_fmt,
                    None,
                )?;
                self.batched_gemm.gemm_q8_act_with_residual(
                    &self.stream,
                    &layer.wk_dev,
                    &self.spec_qx_dev,
                    &self.spec_qd_dev,
                    &self.spec_qs_dev,
                    &self.spec_k_dev,
                    layer.wk_ne0,
                    layer.wk_ne1,
                    k,
                    layer.wk_fmt,
                    None,
                )?;
                self.batched_gemm.gemm_q8_act_with_residual(
                    &self.stream,
                    &layer.wv_dev,
                    &self.spec_qx_dev,
                    &self.spec_qd_dev,
                    &self.spec_qs_dev,
                    &self.spec_v_dev,
                    layer.wv_ne0,
                    layer.wv_ne1,
                    k,
                    layer.wv_fmt,
                    None,
                )?;
            }

            // c. Q/K RMSNorm + RoPE (+ optional attention biases)
            let qk_mode = if self.has_qk_norm {
                MODE_NORM | MODE_ROPE | MODE_BROADCAST_RESIDUAL
            } else {
                MODE_ROPE | MODE_BROADCAST_RESIDUAL
            };
            let q_res = if self.has_attn_bias {
                &layer.qb_dev
            } else {
                &self.spec_zq
            };
            let k_res = if self.has_attn_bias {
                &layer.kb_dev
            } else {
                &self.spec_zk
            };

            self.nr.launch_batched_with_pos_ptr(
                &self.stream,
                &self.spec_q_dev,
                q_res,
                &layer.qn_dev,
                &self.spec_zq,
                &self.spec_q_dev,
                self.eps,
                self.hd,
                self.n_rot,
                self.base,
                current_pos as u32,
                qk_mode,
                None,
                self.nh * k,
                self.nh,
            )?;
            self.nr.launch_batched_with_pos_ptr(
                &self.stream,
                &self.spec_k_dev,
                k_res,
                &layer.kn_dev,
                &self.spec_zk,
                &self.spec_k_dev,
                self.eps,
                self.hd,
                self.n_rot,
                self.base,
                current_pos as u32,
                qk_mode,
                None,
                self.nkv * k,
                self.nkv,
            )?;

            // d. Append K, V into resident paged KV pool
            self.pkv.append_kv(
                &self.stream,
                &self.layout,
                &layer.pool_dev,
                &self.spec_k_dev,
                &self.spec_v_dev,
                &self.bt_dev,
                current_pos,
                k,
            )?;

            // e. FlashAttention-2 causal kernel
            self.flash_attn.launch(
                &self.stream,
                &self.spec_q_dev,
                &layer.pool_dev,
                &self.bt_dev,
                &self.spec_attn_dev,
                self.nh,
                self.nkv,
                self.hd,
                self.layout.block_tokens,
                k,
                current_pos,
            )?;

            // f. Output projection: quantize spec_attn_dev (no norm) -> gemm_q8_act_with_residual with residual spec_x_dev -> spec_op_dev
            self.batched_gemm
                .set_telemetry_tensor_role(Some("attn_output"));
            self.batched_gemm.quantize_q8_1_batched(
                &self.stream,
                &self.spec_attn_dev,
                None,
                &self.spec_qx_dev,
                &self.spec_qd_dev,
                &self.spec_qs_dev,
                self._qdim,
                k,
                self.eps,
            )?;
            self.batched_gemm.gemm_q8_act_with_residual(
                &self.stream,
                &layer.wo_dev,
                &self.spec_qx_dev,
                &self.spec_qd_dev,
                &self.spec_qs_dev,
                &self.spec_op_dev,
                self._qdim,
                self.h,
                k,
                layer.wo_fmt,
                Some(&self.spec_x_dev),
            )?;

            // g. FFN: quantize spec_op_dev with FFN RMSNorm (layer.fn_dev) -> (spec_qx_dev, spec_qd_dev, spec_qs_dev)
            self.batched_gemm
                .set_telemetry_tensor_role(Some("activation_quantization"));
            self.batched_gemm.quantize_q8_1_batched(
                &self.stream,
                &self.spec_op_dev,
                Some(&layer.fn_dev),
                &self.spec_qx_dev,
                &self.spec_qd_dev,
                &self.spec_qs_dev,
                self.h,
                k,
                self.eps,
            )?;

            // h. Fused Gate + Up SwiGLU: gemm_q4k_fused_gate_up_swiglu_mma -> spec_proj_dev
            self.batched_gemm
                .set_telemetry_tensor_role(Some("ffn_gate_up"));
            if layer.wgate_fmt == GemvFormat::Q4K && layer.wup_fmt == GemvFormat::Q4K {
                self.batched_gemm.gemm_q4k_fused_gate_up_swiglu_mma(
                    &self.stream,
                    &layer.wgate_dev,
                    &layer.wup_dev,
                    &self.spec_qx_dev,
                    &self.spec_qd_dev,
                    &self.spec_qs_dev,
                    &self.spec_proj_dev,
                    self.h,
                    self.hff,
                    k,
                )?;
            } else {
                self.batched_gemm.gemm_q8_act_with_residual(
                    &self.stream,
                    &layer.wgate_dev,
                    &self.spec_qx_dev,
                    &self.spec_qd_dev,
                    &self.spec_qs_dev,
                    &self.spec_gate_dev,
                    layer.wgate_ne0,
                    layer.wgate_ne1,
                    k,
                    layer.wgate_fmt,
                    None,
                )?;
                self.batched_gemm.gemm_q8_act_with_residual(
                    &self.stream,
                    &layer.wup_dev,
                    &self.spec_qx_dev,
                    &self.spec_qd_dev,
                    &self.spec_qs_dev,
                    &self.spec_up_dev,
                    layer.wgate_ne0,
                    layer.wgate_ne1,
                    k,
                    layer.wup_fmt,
                    None,
                )?;
                self.nr.launch_batched_with_pos_ptr(
                    &self.stream,
                    &self.spec_gate_dev,
                    &self.spec_zff,
                    &self.spec_zff,
                    &self.spec_up_dev,
                    &self.spec_proj_dev,
                    self.eps,
                    self.hff,
                    0,
                    self.base,
                    0,
                    MODE_SWIGLU,
                    None,
                    k,
                    1,
                )?;
            }

            // i. Down projection: quantize spec_proj_dev (no norm) -> gemm_q8_act_with_residual with residual spec_op_dev -> spec_x_dev
            self.batched_gemm
                .set_telemetry_tensor_role(Some("ffn_down"));
            self.batched_gemm.quantize_q8_1_batched(
                &self.stream,
                &self.spec_proj_dev,
                None,
                &self.spec_qx_dev,
                &self.spec_qd_dev,
                &self.spec_qs_dev,
                self.hff,
                k,
                self.eps,
            )?;
            self.batched_gemm.gemm_q8_act_with_residual(
                &self.stream,
                &layer.wdown_dev,
                &self.spec_qx_dev,
                &self.spec_qd_dev,
                &self.spec_qs_dev,
                &self.spec_x_dev,
                layer.wdown_ne0,
                layer.wdown_ne1,
                k,
                layer.wdown_fmt,
                Some(&self.spec_op_dev),
            )?;
        }

        // 2. GPU batched final RMSNorm + lm_head projection for all K candidate positions
        self.batched_gemm
            .set_telemetry_tensor_role(Some("activation_quantization"));
        self.batched_gemm.quantize_q8_1_batched(
            &self.stream,
            &self.spec_x_dev,
            Some(&self.head_norm_dev),
            &self.spec_qx_dev,
            &self.spec_qd_dev,
            &self.spec_qs_dev,
            self.h,
            k,
            self.eps,
        )?;

        self.batched_gemm.gemm_q8_act_with_residual(
            &self.stream,
            &self.lm_head_dev,
            &self.spec_qx_dev,
            &self.spec_qd_dev,
            &self.spec_qs_dev,
            &self.spec_logits_dev,
            self.h,
            self.vocab_size,
            k,
            self.lm_head_fmt,
            None,
        )?;

        if params.temperature <= 1e-6 {
            self.batched_gemm.sample_greedy_batched(
                &self.stream,
                &self.spec_logits_dev,
                &self.selected_token_dev,
                self.vocab_size,
                k,
            )?;
            let mut sampled_bytes = vec![0u8; k * 4];
            self.selected_token_dev
                .copy_to_host(&self.stream, &mut sampled_bytes)?;
            self.stream.sync()?;

            let mut target_preds = Vec::with_capacity(k);
            for chunk in sampled_bytes.chunks_exact(4) {
                target_preds.push(u32::from_le_bytes(chunk.try_into().unwrap()));
            }

            let mut n_accepted = 0;
            let mut emitted_tokens = Vec::with_capacity(k);
            for i in 0..candidates.len() {
                if candidates[i] == target_preds[i] {
                    n_accepted += 1;
                    emitted_tokens.push(candidates[i]);
                } else {
                    emitted_tokens.push(target_preds[i]);
                    break;
                }
            }

            let bonus_token = if n_accepted == candidates.len() {
                let bonus = target_preds[candidates.len()];
                emitted_tokens.push(bonus);
                bonus
            } else {
                target_preds[n_accepted]
            };

            let total_emitted = emitted_tokens.len();
            self.pos = current_pos + 1 + n_accepted;

            return Ok(crate::speculative::SpeculativeVerificationResult {
                emitted_tokens,
                n_accepted,
                bonus_token,
                total_emitted,
            });
        }

        let raw_logits = download_f32(&self.stream, &self.spec_logits_dev, k * self.vocab_size)?;
        let mut target_logits_refs: Vec<&[f32]> = Vec::with_capacity(k);
        for i in 0..k {
            target_logits_refs.push(&raw_logits[i * self.vocab_size..(i + 1) * self.vocab_size]);
        }

        // 3. Verify candidates against target logits
        let verif_res = crate::speculative::SpeculativeVerifier::verify_stochastic(
            candidates,
            &[],
            &target_logits_refs,
            sampler,
            params,
            context,
        );

        // 4. Update sequence position to match committed tokens
        self.pos = current_pos + 1 + verif_res.n_accepted;
        self.stream.sync()?;

        Ok(verif_res)
    }

    /// Appends all `tokens` sequentially into resident KV starting at `pos=0`,
    /// updates `pos = tokens.len()`, and returns the final position's next-token logits.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, EngineError> {
        self.prefill_chunked(tokens, 256)
    }

    /// Sets the sequence position (e.g. for speculative rollback / synchronization).
    pub fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
        let _ = self
            .pos_dev
            .copy_from_host(&self.stream, &(pos as u32).to_le_bytes());
    }

    /// Returns the number of transformer layers in this model.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    fn record_decode_pass(&self) -> Result<(), EngineError> {
        for layer in &self.layers {
            // Stage 1 & 2: Fused QKV Projection with Fused Input RMSNorm (Projects Q, K, V simultaneously in 1 kernel)
            let (qb_opt, kb_opt, vb_opt) = if self.has_attn_bias {
                (
                    Some(&layer.qb_dev),
                    Some(&layer.kb_dev),
                    Some(&layer.vb_dev),
                )
            } else {
                (None, None, None)
            };

            self.batched_gemm
                .set_telemetry_tensor_role(Some("activation_quantization"));
            self.batched_gemm.quantize_q8_1(
                &self.stream,
                &self.x_dev,
                Some(&layer.an_dev),
                &self.qx_dev,
                &self.qd_dev,
                &self.qs_dev,
                self.h,
                self.eps,
            )?;

            if layer.wq_fmt == GemvFormat::Q4K
                && layer.wk_fmt == GemvFormat::Q4K
                && layer.wv_fmt == GemvFormat::Q4K
            {
                self.batched_gemm.set_telemetry_tensor_role(Some("qkv"));
                self.batched_gemm.gemm_fused_qkv_q4k(
                    &self.stream,
                    &layer.wq_dev,
                    &layer.wk_dev,
                    &layer.wv_dev,
                    &self.qx_dev,
                    &self.qd_dev,
                    &self.qs_dev,
                    &self.q_dev,
                    &self.k_dev,
                    &self.v_dev,
                    self.h,
                    self._qdim,
                    self._kvd,
                    1,
                    qb_opt,
                    kb_opt,
                    vb_opt,
                )?;
            } else if layer.wq_fmt == GemvFormat::Q4K
                && layer.wk_fmt == GemvFormat::Q4K
                && layer.wv_fmt == GemvFormat::Q6K
            {
                self.batched_gemm.set_telemetry_tensor_role(Some("qkv"));
                self.batched_gemm.gemm_fused_qkv(
                    &self.stream,
                    &layer.wq_dev,
                    &layer.wk_dev,
                    &layer.wv_dev,
                    &self.qx_dev,
                    &self.qd_dev,
                    &self.qs_dev,
                    &self.q_dev,
                    &self.k_dev,
                    &self.v_dev,
                    self.h,
                    self._qdim,
                    self._kvd,
                    1,
                    qb_opt,
                    kb_opt,
                    vb_opt,
                )?;
            } else {
                self.batched_gemm.set_telemetry_tensor_role(Some("qkv"));
                self.batched_gemm.gemm_q8_act_with_residual(
                    &self.stream,
                    &layer.wq_dev,
                    &self.qx_dev,
                    &self.qd_dev,
                    &self.qs_dev,
                    &self.q_dev,
                    layer.wq_ne0,
                    layer.wq_ne1,
                    1,
                    layer.wq_fmt,
                    None,
                )?;
                self.batched_gemm.set_telemetry_tensor_role(Some("qkv"));
                self.batched_gemm.gemm_q8_act_with_residual(
                    &self.stream,
                    &layer.wk_dev,
                    &self.qx_dev,
                    &self.qd_dev,
                    &self.qs_dev,
                    &self.k_dev,
                    layer.wk_ne0,
                    layer.wk_ne1,
                    1,
                    layer.wk_fmt,
                    None,
                )?;
                self.batched_gemm.set_telemetry_tensor_role(Some("qkv"));
                self.batched_gemm.gemm_q8_act_with_residual(
                    &self.stream,
                    &layer.wv_dev,
                    &self.qx_dev,
                    &self.qd_dev,
                    &self.qs_dev,
                    &self.v_dev,
                    layer.wv_ne0,
                    layer.wv_ne1,
                    1,
                    layer.wv_fmt,
                    None,
                )?;
            }

            // Stage 3 & 4: Per-head Q/K RMSNorm + RoPE + Paged KV Append (Fused in 1 single kernel launch!)
            let qk_mode = if self.has_qk_norm {
                MODE_NORM | MODE_ROPE
            } else {
                MODE_ROPE
            };

            self.nr.launch_fused_qk(
                &self.stream,
                &self.q_dev,
                &self.k_dev,
                &layer.qn_dev,
                &layer.kn_dev,
                self.nh,
                self.nkv,
                self.hd,
                self.n_rot,
                self.base,
                self.eps,
                qk_mode,
                Some(&self.pos_dev),
                Some(&self.v_dev),
                Some(&layer.pool_dev),
                Some(&self.bt_dev),
                self.layout.block_tokens,
            )?;

            // Stage 5: FlashDecoding Split-KV Attention (GPU)
            self.pa.launch_flash_decoding(
                &self.stream,
                &self.q_dev,
                &layer.pool_dev,
                &self.bt_dev,
                &self.attn_dev,
                self.nh,
                self.nkv,
                self.hd,
                self.layout.block_tokens,
                self.pos + 1,
                self.pos,
                true,
                Some(&self.pos_dev),
            )?;

            // Stage 6: Output Projection with Fused In-Place Residual 1 (attn_dev -> h1_dev with + x_dev)
            self.batched_gemm
                .set_telemetry_tensor_role(Some("activation_quantization"));
            self.batched_gemm.quantize_q8_1(
                &self.stream,
                &self.attn_dev,
                None,
                &self.qx_dev,
                &self.qd_dev,
                &self.qs_dev,
                layer.wo_ne0,
                0.0,
            )?;
            if layer.wo_fmt == GemvFormat::Q4K {
                self.batched_gemm
                    .set_telemetry_tensor_role(Some("attn_output"));
                self.batched_gemm.gemm_q4k_mma(
                    &self.stream,
                    &layer.wo_dev,
                    &self.qx_dev,
                    &self.qd_dev,
                    &self.qs_dev,
                    &self.h1_dev,
                    layer.wo_ne0,
                    layer.wo_ne1,
                    1,
                    Some(&self.x_dev),
                )?;
            } else {
                self.batched_gemm
                    .set_telemetry_tensor_role(Some("attn_output"));
                self.batched_gemm.gemm_q8_act_with_residual(
                    &self.stream,
                    &layer.wo_dev,
                    &self.qx_dev,
                    &self.qd_dev,
                    &self.qs_dev,
                    &self.h1_dev,
                    layer.wo_ne0,
                    layer.wo_ne1,
                    1,
                    layer.wo_fmt,
                    Some(&self.x_dev),
                )?;
            }

            // Stage 7: FFN Gate & Up (+ Fused SwiGLU & FFN RMSNorm)
            self.batched_gemm
                .set_telemetry_tensor_role(Some("activation_quantization"));
            self.batched_gemm.quantize_q8_1(
                &self.stream,
                &self.h1_dev,
                Some(&layer.fn_dev),
                &self.qx_dev,
                &self.qd_dev,
                &self.qs_dev,
                self.h,
                self.eps,
            )?;

            if layer.wgate_fmt == GemvFormat::Q4K && layer.wup_fmt == GemvFormat::Q4K {
                self.batched_gemm
                    .set_telemetry_tensor_role(Some("ffn_gate_up"));
                self.batched_gemm.gemm_q4k_fused_gate_up_swiglu_mma(
                    &self.stream,
                    &layer.wgate_dev,
                    &layer.wup_dev,
                    &self.qx_dev,
                    &self.qd_dev,
                    &self.qs_dev,
                    &self.gate_dev,
                    layer.wgate_ne0,
                    layer.wgate_ne1,
                    1,
                )?;
            } else {
                self.batched_gemm
                    .set_telemetry_tensor_role(Some("ffn_gate_up"));
                self.batched_gemm.gemm_q8_act_with_residual(
                    &self.stream,
                    &layer.wgate_dev,
                    &self.qx_dev,
                    &self.qd_dev,
                    &self.qs_dev,
                    &self.gate_dev,
                    layer.wgate_ne0,
                    layer.wgate_ne1,
                    1,
                    layer.wgate_fmt,
                    None,
                )?;
                self.batched_gemm
                    .set_telemetry_tensor_role(Some("ffn_gate_up"));
                self.batched_gemm.gemm_q8_act_with_residual(
                    &self.stream,
                    &layer.wup_dev,
                    &self.qx_dev,
                    &self.qd_dev,
                    &self.qs_dev,
                    &self.up_dev,
                    layer.wgate_ne0,
                    layer.wgate_ne1,
                    1,
                    layer.wup_fmt,
                    None,
                )?;
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
            }

            // Stage 11: FFN Down with Fused In-Place Residual 2 (gate_dev -> x_dev with + h1_dev)
            self.batched_gemm
                .set_telemetry_tensor_role(Some("activation_quantization"));
            self.batched_gemm.quantize_q8_1(
                &self.stream,
                &self.gate_dev,
                None,
                &self.qx_dev,
                &self.qd_dev,
                &self.qs_dev,
                layer.wdown_ne0,
                0.0,
            )?;
            if layer.wdown_fmt == GemvFormat::Q4K {
                self.batched_gemm
                    .set_telemetry_tensor_role(Some("ffn_down"));
                self.batched_gemm.gemm_q4k_mma(
                    &self.stream,
                    &layer.wdown_dev,
                    &self.qx_dev,
                    &self.qd_dev,
                    &self.qs_dev,
                    &self.x_dev,
                    layer.wdown_ne0,
                    layer.wdown_ne1,
                    1,
                    Some(&self.h1_dev),
                )?;
            } else {
                self.batched_gemm
                    .set_telemetry_tensor_role(Some("ffn_down"));
                self.batched_gemm.gemm_q8_act_with_residual(
                    &self.stream,
                    &layer.wdown_dev,
                    &self.qx_dev,
                    &self.qd_dev,
                    &self.qs_dev,
                    &self.x_dev,
                    layer.wdown_ne0,
                    layer.wdown_ne1,
                    1,
                    layer.wdown_fmt,
                    Some(&self.h1_dev),
                )?;
            }
        }
        Ok(())
    }

    /// Records the full 28-layer forward decode pass onto `self.stream`.
    fn record_decode_pass_f32(&mut self) -> Result<(), EngineError> {
        for layer in self.layers.iter() {
            // Stage 1 & 2: correctness-first F32-input QKV projections.
            // Keep activation quantization out of the integrated decode path.

            self.nr.launch(
                &self.stream,
                &self.x_dev,
                &self.zff,
                &layer.an_dev,
                &self.zff,
                &self.input_norm_dev,
                self.eps,
                self.h,
                0,
                self.base,
                0,
                MODE_NORM,
            )?;
            self.batched_gemm.set_telemetry_tensor_role(Some("qkv"));
            self.batched_gemm.gemm(
                &self.stream,
                &layer.wq_dev,
                &self.input_norm_dev,
                &self.q_dev,
                layer.wq_ne0,
                layer.wq_ne1,
                1,
                layer.wq_fmt,
            )?;
            self.batched_gemm.set_telemetry_tensor_role(Some("qkv"));
            self.batched_gemm.gemm(
                &self.stream,
                &layer.wk_dev,
                &self.input_norm_dev,
                &self.k_dev,
                layer.wk_ne0,
                layer.wk_ne1,
                1,
                layer.wk_fmt,
            )?;
            self.batched_gemm.set_telemetry_tensor_role(Some("qkv"));
            self.batched_gemm.gemm(
                &self.stream,
                &layer.wv_dev,
                &self.input_norm_dev,
                &self.v_dev,
                layer.wv_ne0,
                layer.wv_ne1,
                1,
                layer.wv_fmt,
            )?;

            // Stage 3 & 4: Per-head Q/K RMSNorm + RoPE + Paged KV Append (Fused in 1 single kernel launch!)
            let qk_mode = if self.has_qk_norm {
                MODE_NORM | MODE_ROPE
            } else {
                MODE_ROPE
            };

            self.nr.launch_fused_qk(
                &self.stream,
                &self.q_dev,
                &self.k_dev,
                &layer.qn_dev,
                &layer.kn_dev,
                self.nh,
                self.nkv,
                self.hd,
                self.n_rot,
                self.base,
                self.eps,
                qk_mode,
                Some(&self.pos_dev),
                Some(&self.v_dev),
                Some(&layer.pool_dev),
                Some(&self.bt_dev),
                self.layout.block_tokens,
            )?;

            // Stage 5: FlashDecoding Split-KV Attention (GPU)
            self.pa.launch_flash_decoding(
                &self.stream,
                &self.q_dev,
                &layer.pool_dev,
                &self.bt_dev,
                &self.attn_dev,
                self.nh,
                self.nkv,
                self.hd,
                self.layout.block_tokens,
                self.pos + 1,
                self.pos,
                true,
                Some(&self.pos_dev),
            )?;

            // Stage 6: Direct F32-input output projection with residual.
            self.batched_gemm
                .set_telemetry_tensor_role(Some("attn_output"));
            self.batched_gemm.gemm_with_residual(
                &self.stream,
                &layer.wo_dev,
                &self.attn_dev,
                &self.h1_dev,
                layer.wo_ne0,
                layer.wo_ne1,
                1,
                layer.wo_fmt,
                Some(&self.x_dev),
            )?;

            // Stage 7: FFN Gate & Up (+ Fused SwiGLU & FFN RMSNorm)
            self.nr.launch(
                &self.stream,
                &self.h1_dev,
                &self.zff,
                &layer.fn_dev,
                &self.zff,
                &self.ffin_dev,
                self.eps,
                self.h,
                0,
                self.base,
                0,
                MODE_NORM,
            )?;
            self.batched_gemm
                .set_telemetry_tensor_role(Some("ffn_gate_up"));
            self.batched_gemm.gemm(
                &self.stream,
                &layer.wgate_dev,
                &self.ffin_dev,
                &self.gate_dev,
                layer.wgate_ne0,
                layer.wgate_ne1,
                1,
                layer.wgate_fmt,
            )?;
            self.batched_gemm
                .set_telemetry_tensor_role(Some("ffn_gate_up"));
            self.batched_gemm.gemm(
                &self.stream,
                &layer.wup_dev,
                &self.ffin_dev,
                &self.up_dev,
                layer._wup_ne0,
                layer._wup_ne1,
                1,
                layer.wup_fmt,
            )?;
            self.nr.launch(
                &self.stream,
                &self.gate_dev,
                &self.zff,
                &self.zff,
                &self.up_dev,
                &self.proj_dev,
                self.eps,
                self.hff,
                0,
                self.base,
                0,
                MODE_SWIGLU,
            )?;

            // Stage 11: Direct F32-input FFN down projection with residual.
            self.batched_gemm
                .set_telemetry_tensor_role(Some("activation_quantization"));

            self.batched_gemm
                .set_telemetry_tensor_role(Some("ffn_down"));
            self.batched_gemm.gemm_with_residual(
                &self.stream,
                &layer.wdown_dev,
                &self.proj_dev,
                &self.x_dev,
                layer.wdown_ne0,
                layer.wdown_ne1,
                1,
                layer.wdown_fmt,
                Some(&self.h1_dev),
            )?;
        }
        Ok(())
    }

    pub fn record_lm_head_pass(&self, hidden_dev: &DeviceBuffer) -> Result<(), EngineError> {
        self.batched_gemm
            .set_telemetry_tensor_role(Some("activation_quantization"));
        self.batched_gemm.quantize_q8_1(
            &self.stream,
            hidden_dev,
            Some(&self.head_norm_dev),
            &self.qx_dev,
            &self.qd_dev,
            &self.qs_dev,
            self.h,
            self.eps,
        )?;
        if self.lm_head_fmt == GemvFormat::Q4K {
            self.batched_gemm.set_telemetry_tensor_role(Some("lm_head"));
            self.batched_gemm.gemm_q4k_mma(
                &self.stream,
                &self.lm_head_dev,
                &self.qx_dev,
                &self.qd_dev,
                &self.qs_dev,
                &self.logits_dev,
                self.h,
                self.vocab_size,
                1,
                None,
            )?;
        } else {
            self.batched_gemm.set_telemetry_tensor_role(Some("lm_head"));
            self.batched_gemm.gemm_q8_act_with_residual(
                &self.stream,
                &self.lm_head_dev,
                &self.qx_dev,
                &self.qd_dev,
                &self.qs_dev,
                &self.logits_dev,
                self.h,
                self.vocab_size,
                1,
                self.lm_head_fmt,
                None,
            )?;
        }
        Ok(())
    }

    /// Records the correctness-first direct-F32 lm_head projection.
    pub fn record_lm_head_pass_f32(&self, hidden_dev: &DeviceBuffer) -> Result<(), EngineError> {
        self.nr.launch(
            &self.stream,
            hidden_dev,
            &self.zff,
            &self.head_norm_dev,
            &self.zff,
            &self.head_normed_dev,
            self.eps,
            self.h,
            0,
            self.base,
            0,
            MODE_NORM,
        )?;
        self.batched_gemm.set_telemetry_tensor_role(Some("lm_head"));
        self.batched_gemm.gemm(
            &self.stream,
            &self.lm_head_dev,
            &self.head_normed_dev,
            &self.logits_dev,
            self.h,
            self.vocab_size,
            1,
            self.lm_head_fmt,
        )?;
        Ok(())
    }

    /// Captures the full 28-layer transformer decode pass + lm_head into a CUDA graph.
    pub fn capture_decode_graph(&mut self) -> Result<(), EngineError> {
        self.stream.begin_capture()?;
        match self.decode_path {
            DecodePath::F32 => self.record_decode_pass_f32()?,
            DecodePath::Q8 => self.record_decode_pass()?,
        }
        match self.decode_path {
            DecodePath::F32 => self.record_lm_head_pass_f32(&self.x_dev)?,
            DecodePath::Q8 => self.record_lm_head_pass(&self.x_dev)?,
        }
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
        if p >= self.layout.total_tokens() {
            return Err(engine_cuda::CudaError::InvalidSize {
                expected: self.layout.total_tokens(),
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

        let raw_slice = unsafe {
            std::slice::from_raw_parts_mut(
                self.logits_host.as_mut_ptr() as *mut u8,
                self.vocab_size * std::mem::size_of::<f32>(),
            )
        };
        self.logits_dev.copy_to_host(&self.stream, raw_slice)?;
        self.pos += 1;
        self.stream.sync()?;
        Ok(self.logits_host.clone())
    }

    /// Zero-copy fast decode that returns a borrowed slice of the persistent logits buffer.
    pub fn decode_graph_slice(&mut self, token: u32) -> Result<&[f32], EngineError> {
        if self.decode_graph.is_none() {
            self.capture_decode_graph()?;
        }

        let p = self.pos;
        if p >= self.layout.total_tokens() {
            return Err(engine_cuda::CudaError::InvalidSize {
                expected: self.layout.total_tokens(),
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

        let raw_slice = unsafe {
            std::slice::from_raw_parts_mut(
                self.logits_host.as_mut_ptr() as *mut u8,
                self.vocab_size * std::mem::size_of::<f32>(),
            )
        };
        self.logits_dev.copy_to_host(&self.stream, raw_slice)?;
        self.pos += 1;
        self.stream.sync()?;
        Ok(&self.logits_host)
    }

    /// Captures the complete end-to-end decode pipeline (Embedding -> 28 Transformer Layers -> LM Head -> Greedy Sampler -> History Advance)
    /// into a single autonomous GPU CUDA Graph.
    pub fn capture_autonomous_decode_graph(&mut self) -> Result<(), EngineError> {
        self.stream.begin_capture()?;
        self.batched_gemm.get_rows(
            &self.stream,
            &self.emb_dev,
            &self.token_id_dev,
            &self.x_dev,
            self.h,
            1,
            self.emb_fmt,
        )?;
        match self.decode_path {
            DecodePath::F32 => self.record_decode_pass_f32()?,
            DecodePath::Q8 => self.record_decode_pass()?,
        }
        match self.decode_path {
            DecodePath::F32 => self.record_lm_head_pass_f32(&self.x_dev)?,
            DecodePath::Q8 => self.record_lm_head_pass(&self.x_dev)?,
        }
        self.batched_gemm.sample_greedy(
            &self.stream,
            &self.logits_dev,
            &self.selected_token_dev,
            self.vocab_size,
        )?;
        self.batched_gemm.advance_token_step(
            &self.stream,
            &self.selected_token_dev,
            &self.token_id_dev,
            &self.pos_dev,
            Some(&self.history_dev),
            Some(&self.step_counter_dev),
        )?;
        let graph = self.stream.end_capture()?;
        let exec = graph.instantiate()?;
        self.autonomous_graph = Some(exec);
        Ok(())
    }

    /// Captures the complete speculative verification pipeline for $K$ tokens into a pre-allocated CUDA Graph.
    pub fn capture_speculative_verification_graph(&mut self, k: usize) -> Result<(), EngineError> {
        self.stream.begin_capture()?;
        self.batched_gemm.get_rows(
            &self.stream,
            &self.emb_dev,
            &self.spec_tokens_dev,
            &self.spec_x_dev,
            self.h,
            k,
            self.emb_fmt,
        )?;
        for layer in &self.layers {
            // a. Quantize spec_x_dev with input RMSNorm
            self.batched_gemm
                .set_telemetry_tensor_role(Some("activation_quantization"));
            self.batched_gemm.quantize_q8_1_batched(
                &self.stream,
                &self.spec_x_dev,
                Some(&layer.an_dev),
                &self.spec_qx_dev,
                &self.spec_qd_dev,
                &self.spec_qs_dev,
                self.h,
                k,
                self.eps,
            )?;

            // b. Fused QKV
            let (qb_opt, kb_opt, vb_opt) = if self.has_attn_bias {
                (
                    Some(&layer.qb_dev),
                    Some(&layer.kb_dev),
                    Some(&layer.vb_dev),
                )
            } else {
                (None, None, None)
            };
            if layer.wq_fmt == GemvFormat::Q4K
                && layer.wk_fmt == GemvFormat::Q4K
                && layer.wv_fmt == GemvFormat::Q4K
            {
                self.batched_gemm.set_telemetry_tensor_role(Some("qkv"));
                self.batched_gemm.gemm_fused_qkv_q4k(
                    &self.stream,
                    &layer.wq_dev,
                    &layer.wk_dev,
                    &layer.wv_dev,
                    &self.spec_qx_dev,
                    &self.spec_qd_dev,
                    &self.spec_qs_dev,
                    &self.spec_q_dev,
                    &self.spec_k_dev,
                    &self.spec_v_dev,
                    self.h,
                    self._qdim,
                    self._kvd,
                    k,
                    qb_opt,
                    kb_opt,
                    vb_opt,
                )?;
            } else if layer.wq_fmt == GemvFormat::Q4K
                && layer.wk_fmt == GemvFormat::Q4K
                && layer.wv_fmt == GemvFormat::Q6K
            {
                self.batched_gemm.gemm_fused_qkv(
                    &self.stream,
                    &layer.wq_dev,
                    &layer.wk_dev,
                    &layer.wv_dev,
                    &self.spec_qx_dev,
                    &self.spec_qd_dev,
                    &self.spec_qs_dev,
                    &self.spec_q_dev,
                    &self.spec_k_dev,
                    &self.spec_v_dev,
                    self.h,
                    self._qdim,
                    self._kvd,
                    k,
                    qb_opt,
                    kb_opt,
                    vb_opt,
                )?;
            } else {
                self.batched_gemm.gemm_q8_act_with_residual(
                    &self.stream,
                    &layer.wq_dev,
                    &self.spec_qx_dev,
                    &self.spec_qd_dev,
                    &self.spec_qs_dev,
                    &self.spec_q_dev,
                    layer.wq_ne0,
                    layer.wq_ne1,
                    k,
                    layer.wq_fmt,
                    qb_opt,
                )?;
                self.batched_gemm.gemm_q8_act_with_residual(
                    &self.stream,
                    &layer.wk_dev,
                    &self.spec_qx_dev,
                    &self.spec_qd_dev,
                    &self.spec_qs_dev,
                    &self.spec_k_dev,
                    layer.wk_ne0,
                    layer.wk_ne1,
                    k,
                    layer.wk_fmt,
                    kb_opt,
                )?;
                self.batched_gemm.gemm_q8_act_with_residual(
                    &self.stream,
                    &layer.wv_dev,
                    &self.spec_qx_dev,
                    &self.spec_qd_dev,
                    &self.spec_qs_dev,
                    &self.spec_v_dev,
                    layer.wv_ne0,
                    layer.wv_ne1,
                    k,
                    layer.wv_fmt,
                    vb_opt,
                )?;
            }

            // c, d, e. Fused Q/K Norm + RoPE + Paged KV Append (1 single kernel!)
            let qk_mode = if self.has_qk_norm {
                MODE_NORM | MODE_ROPE
            } else {
                MODE_ROPE
            };
            self.nr.launch_fused_qk(
                &self.stream,
                &self.spec_q_dev,
                &self.spec_k_dev,
                &layer.qn_dev,
                &layer.kn_dev,
                self.nh,
                self.nkv,
                self.hd,
                self.n_rot,
                self.base,
                self.eps,
                qk_mode,
                Some(&self.pos_dev),
                Some(&self.spec_v_dev),
                Some(&layer.pool_dev),
                Some(&self.bt_dev),
                self.layout.block_tokens,
            )?;

            // e. FlashDecoding Attention
            self.pa.launch_with_pos_ptr(
                &self.stream,
                &self.spec_q_dev,
                &layer.pool_dev,
                &self.bt_dev,
                &self.spec_attn_dev,
                self.nh,
                self.nkv,
                self.hd,
                self.layout.block_tokens,
                1,
                0,
                true,
                Some(&self.pos_dev),
            )?;

            // f. Output projection
            self.batched_gemm
                .set_telemetry_tensor_role(Some("activation_quantization"));
            self.batched_gemm.quantize_q8_1_batched(
                &self.stream,
                &self.spec_attn_dev,
                None,
                &self.spec_qx_dev,
                &self.spec_qd_dev,
                &self.spec_qs_dev,
                self._qdim,
                k,
                self.eps,
            )?;
            if layer.wo_fmt == GemvFormat::Q4K {
                self.batched_gemm.gemm_q4k_mma(
                    &self.stream,
                    &layer.wo_dev,
                    &self.spec_qx_dev,
                    &self.spec_qd_dev,
                    &self.spec_qs_dev,
                    &self.spec_op_dev,
                    self._qdim,
                    self.h,
                    k,
                    Some(&self.spec_x_dev),
                )?;
            } else {
                self.batched_gemm.gemm_q8_act_with_residual(
                    &self.stream,
                    &layer.wo_dev,
                    &self.spec_qx_dev,
                    &self.spec_qd_dev,
                    &self.spec_qs_dev,
                    &self.spec_op_dev,
                    self._qdim,
                    self.h,
                    k,
                    layer.wo_fmt,
                    Some(&self.spec_x_dev),
                )?;
            }

            // g. FFN RMSNorm
            self.batched_gemm
                .set_telemetry_tensor_role(Some("activation_quantization"));
            self.batched_gemm.quantize_q8_1_batched(
                &self.stream,
                &self.spec_op_dev,
                Some(&layer.fn_dev),
                &self.spec_qx_dev,
                &self.spec_qd_dev,
                &self.spec_qs_dev,
                self.h,
                k,
                self.eps,
            )?;

            // h. SwiGLU FFN
            if layer.wgate_fmt == GemvFormat::Q4K && layer.wup_fmt == GemvFormat::Q4K {
                self.batched_gemm.gemm_q4k_fused_gate_up_swiglu_mma(
                    &self.stream,
                    &layer.wgate_dev,
                    &layer.wup_dev,
                    &self.spec_qx_dev,
                    &self.spec_qd_dev,
                    &self.spec_qs_dev,
                    &self.spec_proj_dev,
                    self.h,
                    self.hff,
                    k,
                )?;
            } else {
                self.batched_gemm.gemm_q8_act_with_residual(
                    &self.stream,
                    &layer.wgate_dev,
                    &self.spec_qx_dev,
                    &self.spec_qd_dev,
                    &self.spec_qs_dev,
                    &self.spec_gate_dev,
                    layer.wgate_ne0,
                    layer.wgate_ne1,
                    k,
                    layer.wgate_fmt,
                    None,
                )?;
                self.batched_gemm.gemm_q8_act_with_residual(
                    &self.stream,
                    &layer.wup_dev,
                    &self.spec_qx_dev,
                    &self.spec_qd_dev,
                    &self.spec_qs_dev,
                    &self.spec_up_dev,
                    layer.wgate_ne0,
                    layer.wgate_ne1,
                    k,
                    layer.wup_fmt,
                    None,
                )?;
                self.nr.launch_batched_with_pos_ptr(
                    &self.stream,
                    &self.spec_gate_dev,
                    &self.spec_zff,
                    &self.spec_zff,
                    &self.spec_up_dev,
                    &self.spec_proj_dev,
                    self.eps,
                    self.hff,
                    0,
                    self.base,
                    0,
                    MODE_SWIGLU,
                    None,
                    k,
                    1,
                )?;
            }

            // i. Down projection
            self.batched_gemm
                .set_telemetry_tensor_role(Some("activation_quantization"));
            self.batched_gemm.quantize_q8_1_batched(
                &self.stream,
                &self.spec_proj_dev,
                None,
                &self.spec_qx_dev,
                &self.spec_qd_dev,
                &self.spec_qs_dev,
                self.hff,
                k,
                self.eps,
            )?;
            if layer.wdown_fmt == GemvFormat::Q4K {
                self.batched_gemm.gemm_q4k_mma(
                    &self.stream,
                    &layer.wdown_dev,
                    &self.spec_qx_dev,
                    &self.spec_qd_dev,
                    &self.spec_qs_dev,
                    &self.spec_x_dev,
                    layer.wdown_ne0,
                    layer.wdown_ne1,
                    k,
                    Some(&self.spec_op_dev),
                )?;
            } else {
                self.batched_gemm.gemm_q8_act_with_residual(
                    &self.stream,
                    &layer.wdown_dev,
                    &self.spec_qx_dev,
                    &self.spec_qd_dev,
                    &self.spec_qs_dev,
                    &self.spec_x_dev,
                    layer.wdown_ne0,
                    layer.wdown_ne1,
                    k,
                    layer.wdown_fmt,
                    Some(&self.spec_op_dev),
                )?;
            }
        }

        // LM Head
        self.batched_gemm
            .set_telemetry_tensor_role(Some("activation_quantization"));
        self.batched_gemm.quantize_q8_1_batched(
            &self.stream,
            &self.spec_x_dev,
            Some(&self.head_norm_dev),
            &self.spec_qx_dev,
            &self.spec_qd_dev,
            &self.spec_qs_dev,
            self.h,
            k,
            self.eps,
        )?;
        if self.lm_head_fmt == GemvFormat::Q4K {
            self.batched_gemm.gemm_q4k_mma(
                &self.stream,
                &self.lm_head_dev,
                &self.spec_qx_dev,
                &self.spec_qd_dev,
                &self.spec_qs_dev,
                &self.spec_logits_dev,
                self.h,
                self.vocab_size,
                k,
                None,
            )?;
        } else if self.lm_head_fmt == GemvFormat::Q6K {
            self.batched_gemm.gemm_q6k(
                &self.stream,
                &self.lm_head_dev,
                &self.spec_qx_dev,
                &self.spec_qd_dev,
                &self.spec_qs_dev,
                &self.spec_logits_dev,
                self.h,
                self.vocab_size,
                k,
                None,
            )?;
        } else {
            self.batched_gemm.gemm_q8_act_with_residual(
                &self.stream,
                &self.lm_head_dev,
                &self.spec_qx_dev,
                &self.spec_qd_dev,
                &self.spec_qs_dev,
                &self.spec_logits_dev,
                self.h,
                self.vocab_size,
                k,
                self.lm_head_fmt,
                None,
            )?;
        }
        self.batched_gemm.sample_greedy_batched(
            &self.stream,
            &self.spec_logits_dev,
            &self.selected_token_dev,
            self.vocab_size,
            k,
        )?;

        let graph = self.stream.end_capture()?;
        let exec = graph.instantiate()?;
        self.speculative_graph = Some(exec);
        Ok(())
    }

    /// Autonomous GPU-native decode step: Uploads only token_id (4 bytes) and pos (4 bytes),
    /// launches the full GPU pipeline CUDA Graph, and downloads ONLY the sampled token_id (4 bytes).
    /// Zero CPU vector allocations, zero CPU dequantization, and zero large PCIe memory traffic!
    pub fn decode_step_autonomous(&mut self, token: u32) -> Result<u32, EngineError> {
        if self.autonomous_graph.is_none() {
            self.capture_autonomous_decode_graph()?;
        }

        let p = self.pos;
        if p >= self.layout.total_tokens() {
            return Err(engine_cuda::CudaError::InvalidSize {
                expected: self.layout.total_tokens(),
                actual: p + 1,
            }
            .into());
        }

        let tok_bytes = token.to_le_bytes();
        self.token_id_dev.copy_from_host(&self.stream, &tok_bytes)?;
        let pos_bytes = (p as u32).to_le_bytes();
        self.pos_dev.copy_from_host(&self.stream, &pos_bytes)?;
        let zero_step = 0u32.to_le_bytes();
        self.step_counter_dev
            .copy_from_host(&self.stream, &zero_step)?;

        if let Some(ref exec) = self.autonomous_graph {
            exec.launch(&self.stream)?;
        }

        let mut out_bytes = [0u8; 4];
        self.selected_token_dev
            .copy_to_host(&self.stream, &mut out_bytes)?;
        self.pos += 1;
        self.stream.sync()?;
        Ok(u32::from_le_bytes(out_bytes))
    }

    /// High-throughput autonomous multi-token GPU stream generation.
    /// Runs $N$ decode iterations entirely on GPU without intermediate CPU synchronizations,
    /// syncing once at the end and returning the generated tokens!
    pub fn generate_autonomous_gpu(
        &mut self,
        start_token: u32,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>, EngineError> {
        if max_new_tokens == 0 {
            return Ok(Vec::new());
        }

        if self.autonomous_graph.is_none() {
            self.capture_autonomous_decode_graph()?;
        }

        let p = self.pos;
        if p + max_new_tokens > self.layout.total_tokens() {
            return Err(engine_cuda::CudaError::InvalidSize {
                expected: self.layout.total_tokens(),
                actual: p + max_new_tokens,
            }
            .into());
        }

        let tok_bytes = start_token.to_le_bytes();
        self.token_id_dev.copy_from_host(&self.stream, &tok_bytes)?;
        let pos_bytes = (p as u32).to_le_bytes();
        self.pos_dev.copy_from_host(&self.stream, &pos_bytes)?;
        let zero_step = 0u32.to_le_bytes();
        self.step_counter_dev
            .copy_from_host(&self.stream, &zero_step)?;
        if let Some(ref exec) = self.autonomous_graph {
            for _ in 0..max_new_tokens {
                exec.launch(&self.stream)?;
            }
        }
        self.pos += max_new_tokens;
        let mut out_bytes = vec![0u8; max_new_tokens * 4];
        self.history_dev
            .copy_to_host(&self.stream, &mut out_bytes)?;
        self.stream.sync()?;

        let mut generated_tokens = Vec::with_capacity(max_new_tokens);
        for chunk in out_bytes.chunks_exact(4) {
            let tok = u32::from_le_bytes(chunk.try_into().unwrap());
            generated_tokens.push(tok);
        }

        Ok(generated_tokens)
    }

    /// Single-token decode step over resident KV pool at current `self.pos`.
    pub fn decode(&mut self, token: u32) -> Result<Vec<f32>, EngineError> {
        let p = self.pos;
        if p >= self.layout.total_tokens() {
            return Err(engine_cuda::CudaError::InvalidSize {
                expected: self.layout.total_tokens(),
                actual: p + 1,
            }
            .into());
        }

        let x_host = embed_lookup(&self.emb, token as usize);
        self.x_dev
            .copy_from_host(&self.stream, &f32_bytes(&x_host))?;
        let pos_bytes = (p as u32).to_le_bytes();
        self.pos_dev.copy_from_host(&self.stream, &pos_bytes)?;

        let t_start = std::time::Instant::now();
        match self.decode_path {
            DecodePath::F32 => self.record_decode_pass_f32()?,
            DecodePath::Q8 => self.record_decode_pass()?,
        }
        match self.decode_path {
            DecodePath::F32 => self.record_lm_head_pass_f32(&self.x_dev)?,
            DecodePath::Q8 => self.record_lm_head_pass(&self.x_dev)?,
        }

        let logits = download_f32(&self.stream, &self.logits_dev, self.vocab_size)?;
        self.pos += 1;
        self.stream.sync()?;
        let t_dec = t_start.elapsed();
        println!(
            "    [Decode Step Profile] Total: {:.2} ms",
            t_dec.as_secs_f64() * 1000.0
        );
        Ok(logits)
    }

    /// Computes next-token logits from hidden state on CPU (used for speculative slices).
    pub fn lm_head(&self, hid: &[f32]) -> Vec<f32> {
        logits_from_hidden(&self.lm_head_weight, &self.head_norm, hid, self.eps)
    }

    /// Current token position in resident KV pool.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Resets the current sequence position and KV pools.
    pub fn reset_pos(&mut self) {
        self.pos = 0;
        self.radix_tree = engine_kvcache::RadixTree::new(self.layout.block_tokens);
        let zero_bytes = 0u32.to_le_bytes();
        let _ = self.pos_dev.copy_from_host(&self.stream, &zero_bytes);
        for layer in &self.layers {
            let _ = layer
                .pool_dev
                .copy_from_host(&self.stream, &vec![0u8; self.layout.floats_total() * 4]);
        }
        let _ = self.stream.sync();
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
            .sum::<usize>()
            + self.lm_head_dev.size()
            + self.head_norm_dev.size();

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
            + self.proj_dev.size()
            + self.head_normed_dev.size();

        let logits_bytes: usize = self.logits_dev.size();

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
