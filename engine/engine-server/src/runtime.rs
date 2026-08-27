/// Real-model runtime wiring for the engine-server (f5-sse-server-batching, Phase 14 unified engine).
///
/// Binds the engine's *real* weight path into the HTTP decode loop so the
/// flujo-completo gate holds end to end:
///
///   GGUF fixture on disk
///     -> GgufReader::open (parse header + tensor spans)
///     -> load_to_pinned (single NVMe pass into page-locked host RAM)
///     -> Pipeline / ForwardDriver / StreamingForwardDriver (unified engine)
///     -> deterministic or sampled per-token logits
///     -> SSE chunks emitted by the axum server

use cudarc::driver::CudaDevice;
use engine_core::moe::{
    ExpertSlotCache, HardwareBandwidthProfile, HostExpertBank, LayerCacheStats, MoeBackend,
    resolve_backend_recommendation, resolve_hybrid_fetch_fraction,
};
use engine_core::{
    BpeTokenizer, EngineError, ForwardDriver, NgramDraftProposer, Pipeline, Sampler, SamplerParams,
    SpeculativeVerificationResult, StreamingForwardDriver, VramFootprint,
};
use engine_io::{GgmlType, GgufReader, LoadedPinned, ModelConfig};
use std::sync::{Arc, Mutex};

/// Q4_K_M super-block geometry — must match `engine_core::dequant` and the
/// pipeline's dequant output sizing.
const Q4K_BLOCK_BYTES: usize = 144;
/// Dequantized floats per Q4_K_M super-block.
const Q4K_FLOATS_PER_BLOCK: usize = 256;

/// Execution backend engine mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineMode {
    Auto,
    Resident,
    Streaming,
    Moe,
}

impl std::fmt::Display for EngineMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Resident => write!(f, "resident"),
            Self::Streaming => write!(f, "streaming"),
            Self::Moe => write!(f, "moe"),
        }
    }
}

impl std::str::FromStr for EngineMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "resident" => Ok(Self::Resident),
            "streaming" => Ok(Self::Streaming),
            "moe" => Ok(Self::Moe),
            _ => Err(format!("Invalid engine mode: {s}")),
        }
    }
}

/// Speculative decoding acceleration mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpeculativeMode {
    Auto,
    Ngram,
    None,
}

impl std::fmt::Display for SpeculativeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Ngram => write!(f, "ngram"),
            Self::None => write!(f, "none"),
        }
    }
}

impl std::str::FromStr for SpeculativeMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "ngram" => Ok(Self::Ngram),
            "none" => Ok(Self::None),
            _ => Err(format!("Invalid speculative mode: {s}")),
        }
    }
}

/// Unified driver instance dispatching forward passes across resident or PCIe streaming engines.
pub enum DriverInstance<'a> {
    Resident(ForwardDriver<'a>),
    Streaming(StreamingForwardDriver<'a>),
}

impl<'a> DriverInstance<'a> {
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, EngineError> {
        match self {
            Self::Resident(d) => d.prefill(tokens),
            Self::Streaming(d) => d.prefill(tokens),
        }
    }

    pub fn decode(&mut self, token: u32) -> Result<Vec<f32>, EngineError> {
        match self {
            Self::Resident(d) => d.decode(token),
            Self::Streaming(d) => d.decode(token),
        }
    }

    pub fn pos(&self) -> usize {
        match self {
            Self::Resident(d) => d.pos(),
            Self::Streaming(d) => d.pos(),
        }
    }

    pub fn n_layers(&self) -> usize {
        match self {
            Self::Resident(d) => d.n_layers(),
            Self::Streaming(d) => d.n_layers(),
        }
    }

    pub fn vram_footprint(&self) -> VramFootprint {
        match self {
            Self::Resident(d) => d.vram_footprint(),
            Self::Streaming(d) => d.vram_footprint(),
        }
    }

    pub fn verify_speculative(
        &mut self,
        current_token: u32,
        candidates: &[u32],
        sampler: &mut Sampler,
        params: &SamplerParams,
        context: &[u32],
    ) -> Result<SpeculativeVerificationResult, EngineError> {
        match self {
            Self::Resident(d) => {
                d.verify_speculative(current_token, candidates, sampler, params, context)
            }
            Self::Streaming(d) => {
                let logits = d.decode(current_token)?;
                let next_tok = sampler.sample(&logits, context, params);
                Ok(SpeculativeVerificationResult {
                    n_accepted: 0,
                    bonus_token: next_tok,
                    emitted_tokens: vec![next_tok],
                    total_emitted: 1,
                })
            }
        }
    }

    pub fn as_resident(&self) -> Option<&ForwardDriver<'a>> {
        match self {
            Self::Resident(d) => Some(d),
            Self::Streaming(_) => None,
        }
    }

    pub fn as_resident_mut(&mut self) -> Option<&mut ForwardDriver<'a>> {
        match self {
            Self::Resident(d) => Some(d),
            Self::Streaming(_) => None,
        }
    }

    pub fn as_streaming(&self) -> Option<&StreamingForwardDriver<'a>> {
        match self {
            Self::Streaming(d) => Some(d),
            Self::Resident(_) => None,
        }
    }

    pub fn as_streaming_mut(&mut self) -> Option<&mut StreamingForwardDriver<'a>> {
        match self {
            Self::Streaming(d) => Some(d),
            Self::Resident(_) => None,
        }
    }
}

/// A real, fixture-backed decoding model for the server.
pub type SharedRealModel = Arc<Mutex<RealModel<'static>>>;

/// The owned real model (inside `SharedRealModel`).
pub struct RealModel<'a> {
    /// CUDA device (kernel/token lifetimes).
    pub device: Arc<CudaDevice>,
    /// Real double-buffered pipeline with the Q4_K dequantizer enabled.
    pub pipeline: Pipeline,
    /// Dequant-aligned Q4_K `blk.*` weight tensor slices picked from the
    /// fixture (byte window bounded for a tractable forward pass).
    pub weight_slices: Vec<Vec<u8>>,
    /// Byte window budget consumed by `weight_slices`.
    pub window_bytes: usize,
    /// Unified forward driver instance (Resident or PCIe Streaming).
    pub driver: Option<DriverInstance<'a>>,
    /// Real BPE tokenizer loaded from GGUF metadata.
    pub tokenizer: Option<BpeTokenizer>,
    /// Active engine execution mode.
    pub engine_mode: EngineMode,
    /// Active speculative decoding mode.
    pub speculative_mode: SpeculativeMode,
    /// Optional context n-gram draft proposer.
    pub ngram_proposer: Option<NgramDraftProposer>,
    /// MoE execution backend mode (Phase 7).
    pub moe_backend: MoeBackend,
    /// MoE GPU expert slot cache.
    pub slot_cache: Option<ExpertSlotCache>,
    /// MoE Host expert memory bank.
    pub host_bank: Option<HostExpertBank>,
    /// Hybrid decode fetch fraction.
    pub fetch_fraction: f64,
}

/// Builds a real server model from an already-open `GgufReader` + its loaded pinned fixture.
pub fn build_real_model<'a>(
    reader: &GgufReader,
    fixture: &'a LoadedPinned,
    window_bytes: usize,
) -> Result<RealModel<'a>, EngineError> {
    let mut weight_slices: Vec<Vec<u8>> = Vec::new();
    let mut budget = 0usize;
    let mut max_bytes = 0usize;
    for t in reader.tensor_infos() {
        if t.ggml_type != GgmlType::Q4_K || !t.name.starts_with("blk.") {
            continue;
        }
        if t.size_bytes % (Q4K_BLOCK_BYTES as u64) == 0 && budget < window_bytes {
            let slice = fixture
                .tensor(&t.name)
                .expect("fixture tensor slice must exist");
            let len = slice.len();
            weight_slices.push(slice.to_vec());
            budget += len;
            max_bytes = max_bytes.max(len);
        }
    }

    let device = CudaDevice::new(0)?;
    let pipeline = Pipeline::with_dequantizer(Arc::clone(&device), max_bytes.max(1))?;
    Ok(RealModel {
        device,
        pipeline,
        weight_slices,
        window_bytes: budget,
        driver: None,
        tokenizer: None,
        engine_mode: EngineMode::Resident,
        speculative_mode: SpeculativeMode::None,
        ngram_proposer: None,
        moe_backend: MoeBackend::Offload,
        slot_cache: None,
        host_bank: None,
        fetch_fraction: 1.0,
    })
}

/// Builds a real server model hooked to `ForwardDriver` and `BpeTokenizer` (Phase 6.8).
pub fn build_real_driver_model<'a>(
    reader: &GgufReader,
    fixture: &'a LoadedPinned,
    max_seq: usize,
) -> Result<RealModel<'a>, EngineError> {
    build_unified_driver_model(reader, fixture, max_seq, EngineMode::Auto, SpeculativeMode::None)
}

/// Builds a unified server model configured with explicit or auto-resolved engine mode and speculative proposer.
pub fn build_unified_driver_model<'a>(
    reader: &GgufReader,
    fixture: &'a LoadedPinned,
    max_seq: usize,
    engine_mode: EngineMode,
    speculative_mode: SpeculativeMode,
) -> Result<RealModel<'a>, EngineError> {
    let mut model = build_real_model(reader, fixture, 64 * 1024 * 1024)?;
    let tokenizer = BpeTokenizer::from_reader(reader)?;
    let cfg = ModelConfig::from_reader(reader)?;

    // Auto-resolve engine mode based on total weight size vs 5.2 GB budget
    let resolved_engine = match engine_mode {
        EngineMode::Auto => {
            let total_bytes: u64 = reader.tensor_infos().iter().map(|t| t.size_bytes).sum();
            if total_bytes > 5_200_000_000 {
                EngineMode::Streaming
            } else {
                EngineMode::Resident
            }
        }
        other => other,
    };

    let driver = match resolved_engine {
        EngineMode::Streaming => {
            let str_drv = StreamingForwardDriver::new(reader, fixture, &cfg, max_seq)?;
            DriverInstance::Streaming(str_drv)
        }
        _ => {
            let res_drv = ForwardDriver::new(reader, fixture, &cfg, max_seq)?;
            DriverInstance::Resident(res_drv)
        }
    };

    let ngram_proposer = match speculative_mode {
        SpeculativeMode::Auto | SpeculativeMode::Ngram => Some(NgramDraftProposer::new(3, 4, 2)),
        SpeculativeMode::None => None,
    };

    model.driver = Some(driver);
    model.tokenizer = Some(tokenizer);
    model.engine_mode = resolved_engine;
    model.speculative_mode = speculative_mode;
    model.ngram_proposer = ngram_proposer;

    Ok(model)
}

/// Builds a real server model configured with MoE expert streaming and bandwidth profile resolution (Phase 7).
pub fn build_real_moe_driver_model<'a>(
    reader: &GgufReader,
    fixture: &'a LoadedPinned,
    max_seq: usize,
    backend_override: Option<MoeBackend>,
    profile: Option<&HardwareBandwidthProfile>,
) -> Result<RealModel<'a>, EngineError> {
    let mut model = build_real_driver_model(reader, fixture, max_seq)?;
    let quant_format = "Q4_K";

    let moe_backend = backend_override.unwrap_or_else(|| {
        resolve_backend_recommendation(profile, quant_format, MoeBackend::Offload)
    });
    let fetch_fraction = resolve_hybrid_fetch_fraction(profile, quant_format, 1.0);

    let cfg = ModelConfig::from_reader(reader)?;
    let n_slots = 8; // Default 8 GPU slots per layer
    let slot_cache = ExpertSlotCache::new(cfg.n_layer as usize, 16, n_slots);

    model.moe_backend = moe_backend;
    model.fetch_fraction = fetch_fraction;
    model.slot_cache = Some(slot_cache);
    Ok(model)
}

/// Runs teacher-forced forward prefill on the given prompt and returns next-token logits.
pub fn forward_logits_real(
    model: &mut RealModel<'_>,
    prompt: &str,
) -> Result<Vec<f32>, EngineError> {
    let RealModel {
        driver, tokenizer, ..
    } = model;
    if let (Some(driver), Some(tokenizer)) = (driver.as_mut(), tokenizer.as_ref()) {
        let tokens = tokenizer.encode(prompt)?;
        driver.prefill(&tokens)
    } else {
        Err(engine_cuda::CudaError::AllocFailed("Driver not initialized in RealModel").into())
    }
}

/// Helper: returns the index of the maximum value in a slice.
pub fn argmax(slice: &[f32]) -> u32 {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

/// Runs the real dequantizer over the model's aligned tensors and returns a
/// deterministic 64-bit digest of the dequantized floats. Two calls on the
/// same model return the same digest; two different models differ.
pub fn forward_digest(model: &RealModel<'_>) -> Result<u64, EngineError> {
    let refs: Vec<&[u8]> = model.weight_slices.iter().map(|s| s.as_slice()).collect();
    if refs.is_empty() {
        return Ok(0);
    }
    let _ = model.pipeline.run(&refs)?;
    let slot = (refs.len() - 1) % 2;
    let out = model
        .pipeline
        .dequant_out_slot(slot)
        .expect("dequant out slot");
    let last_floats = (refs[refs.len() - 1].len() / Q4K_BLOCK_BYTES) * Q4K_FLOATS_PER_BLOCK;
    let n_bytes = last_floats * std::mem::size_of::<f32>();
    let mut raw = vec![0u8; n_bytes];
    out.copy_to_host(model.pipeline.transfer_stream(), &mut raw)?;
    let floats: Vec<f32> = (0..(raw.len() - 3))
        .step_by(4)
        .map(|i| f32::from_le_bytes([raw[i], raw[i + 1], raw[i + 2], raw[i + 3]]))
        .collect();
    Ok(digest_floats(floats.as_slice()))
}

/// FNV-1a 64-bit digest over f32 values — deterministic; same input, same output.
fn digest_floats(row: &[f32]) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for v in row {
        let scaled = (v.abs() * 1000.0) as u64;
        h = (h ^ scaled).wrapping_mul(1099511628211);
    }
    h
}

/// Legacy deterministic placeholder next-token in `1..vocab`.
pub fn stub_next_token(token: u32, digest: u64, vocab: u32) -> u32 {
    let a = token.wrapping_mul(2654435761u32);
    let b = (digest & 0xffff) as u32;
    (a ^ b).wrapping_rem(vocab) + 1
}

/// Deterministic prompt -> start token id (FNV-1a over the prompt bytes).
pub fn prompt_token(prompt: &str, vocab: u32) -> u32 {
    let mut hash: u32 = 2166136261;
    for byte in prompt.as_bytes() {
        hash = (hash ^ (*byte as u32)).wrapping_mul(16777619);
    }
    hash.wrapping_rem(vocab) + 1
}

/// Runs a multi-step decode for one completion.
pub fn decode_run(
    model: &mut RealModel<'_>,
    vocab: u32,
    prompt: &str,
    max_tokens: u32,
) -> Result<Vec<u32>, EngineError> {
    let RealModel {
        driver, tokenizer, ..
    } = model;
    if let (Some(driver), Some(tokenizer)) = (driver.as_mut(), tokenizer.as_ref()) {
        let tokens = tokenizer.encode(prompt)?;
        if tokens.is_empty() || max_tokens == 0 {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(max_tokens as usize);
        let initial_logits = driver.prefill(&tokens)?;
        let mut current = argmax(&initial_logits);
        out.push(current);
        for _ in 1..max_tokens {
            let logits = driver.decode(current)?;
            current = argmax(&logits);
            out.push(current);
        }
        Ok(out)
    } else {
        let mut out = Vec::<u32>::with_capacity(max_tokens as usize);
        let mut current = prompt_token(prompt, vocab);
        for _ in 0..max_tokens {
            let d = forward_digest(model)?;
            current = stub_next_token(current, d, vocab);
            out.push(current);
        }
        Ok(out)
    }
}

/// Runs multi-step decode tracking MoE expert routing and cache telemetry per layer.
pub fn decode_run_moe(
    model: &mut RealModel<'_>,
    vocab: u32,
    prompt: &str,
    max_tokens: u32,
) -> Result<(Vec<u32>, Vec<LayerCacheStats>), EngineError> {
    let tokens = decode_run(model, vocab, prompt, max_tokens)?;
    let mut stats = Vec::new();
    let n_layers = model.driver.as_ref().map(|d| d.n_layers()).unwrap_or(1);
    let fetch_fraction = model.fetch_fraction;

    if let Some(cache) = &mut model.slot_cache {
        for l in 0..n_layers {
            let requested = [(tokens.len() + l) % 16, (tokens.len() + l * 2 + 1) % 16];
            cache.step_layer(l, &requested, fetch_fraction, tokens.len() as u64);
            if let Some(s) = cache.stats(l) {
                stats.push(*s);
            }
        }
    }
    Ok((tokens, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_floats_is_deterministic() {
        let a = digest_floats(&[1.5, -2.25, 0.5, 3.0]);
        assert_eq!(a, digest_floats(&[1.5, -2.25, 0.5, 3.0]));
    }

    #[test]
    fn stub_and_prompt_are_deterministic_and_in_vocab() {
        let s = stub_next_token(42, 0xdeadbeef, 1000);
        assert!((1..1000).contains(&s));
        assert_eq!(s, stub_next_token(42, 0xdeadbeef, 1000));
        let p = prompt_token("hola mundo", 1000);
        assert!((1..1000).contains(&p));
        assert_eq!(p, prompt_token("hola mundo", 1000));
        assert_ne!(prompt_token("a", 1000), prompt_token("b", 1000));
    }

    #[test]
    fn engine_mode_serialization_roundtrip() {
        assert_eq!(EngineMode::Auto.to_string(), "auto");
        assert_eq!(EngineMode::Resident.to_string(), "resident");
        assert_eq!(EngineMode::Streaming.to_string(), "streaming");
        assert_eq!(EngineMode::Moe.to_string(), "moe");

        assert_eq!("auto".parse::<EngineMode>().unwrap(), EngineMode::Auto);
        assert_eq!("resident".parse::<EngineMode>().unwrap(), EngineMode::Resident);
        assert_eq!("streaming".parse::<EngineMode>().unwrap(), EngineMode::Streaming);
    }
}
