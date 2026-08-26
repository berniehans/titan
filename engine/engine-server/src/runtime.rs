/// Real-model runtime wiring for the engine-server (f5-sse-server-batching).
///
/// Binds the engine's *real* weight path into the HTTP decode loop so the
/// flujo-completo gate holds end to end:
///
///   GGUF fixture on disk
///     -> GgufReader::open (parse header + tensor spans)
///     -> load_to_pinned (single NVMe pass into page-locked host RAM)
///     -> Pipeline::with_dequantizer (real GPU Q4_K kernel, compute stage)
///     -> deterministic per-token logits derived from the dequantized weights
///     -> SSE chunks emitted by the axum server
///
/// # Honesty constraint
///
/// The model "forward pass" between embedding and vocabulary is still a
/// deterministic placeholder (`stub_next_token`). It depends on the *actual*
/// dequantized weights of the loaded fixture plus the token id, so a real
/// model produces a real (and reproducible) output sequence — but no claim of
/// model quality is made.
///
/// # Usage
/// - `#[ignore]` GPU E2E: `build_real_model` + `decode_run` drive `/v1/completions`
///   through the real pipeline. Requires a live CUDA device + `nvrtc64_*.dll`
///   on PATH via `engine-server/tests/e2e_full_stack.rs`.
/// - CI-safe tests keep the synthetic `KV` decode path; nothing here runs in
///   `cargo test --workspace` (GPU + fixture required).
use cudarc::driver::CudaDevice;
use engine_core::{BpeTokenizer, EngineError, ForwardDriver, Pipeline};
use engine_io::{GgmlType, GgufReader, LoadedPinned, ModelConfig};
use std::sync::{Arc, Mutex};

/// Q4_K_M super-block geometry — must match `engine_core::dequant` and the
/// pipeline's dequant output sizing.
const Q4K_BLOCK_BYTES: usize = 144;
/// Dequantized floats per Q4_K_M super-block.
const Q4K_FLOATS_PER_BLOCK: usize = 256;

/// A real, fixture-backed decoding model for the server.
///
/// Owns the live CUDA device, the double-buffered dequantizer pipeline, and
/// the dequant-aligned Q4_K weight tensor slices that the generation loop
/// streams to the GPU (copied out of the loader's pinned fixture).
///
/// When initialized via `build_real_driver_model`, also owns the full `ForwardDriver`
/// and `BpeTokenizer` for real end-to-end model execution.
///
/// `Mutex`-wrapped because axum serves concurrent requests on tokio worker
/// threads but a single CUDA pipeline is not re-entrant across threads.
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
    /// Full forward driver over the streamed GPU pipeline (Phase 6.7/6.8).
    pub driver: Option<ForwardDriver<'a>>,
    /// Real BPE tokenizer loaded from GGUF metadata.
    pub tokenizer: Option<BpeTokenizer>,
}

/// Builds a real server model from an already-open `GgufReader` + its loaded
/// pinned fixture.
///
/// Collects a bounded window of dequant-aligned Q4_K `blk.*` weight tensors
/// and constructs `Pipeline::with_dequantizer` sized to the largest slice.
/// Requires a local CUDA device and NVRTC on PATH. Loading (which returns
/// `GgufError`, not `EngineError`) stays with the caller (`#[ignore]` GPU test).
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
    })
}

/// Builds a real server model hooked to `ForwardDriver` and `BpeTokenizer` (Phase 6.8).
pub fn build_real_driver_model<'a>(
    reader: &GgufReader,
    fixture: &'a LoadedPinned,
    max_seq: usize,
) -> Result<RealModel<'a>, EngineError> {
    let mut model = build_real_model(reader, fixture, 64 * 1024 * 1024)?;
    let tokenizer = BpeTokenizer::from_reader(reader)?;
    let cfg = ModelConfig::from_reader(reader)?;
    let driver = ForwardDriver::new(reader, fixture, &cfg, max_seq)?;
    model.driver = Some(driver);
    model.tokenizer = Some(tokenizer);
    Ok(model)
}

/// Runs teacher-forced forward prefill on the given prompt and returns next-token logits.
pub fn forward_logits_real(
    model: &mut RealModel<'_>,
    prompt: &str,
) -> Result<Vec<f32>, EngineError> {
    if let (Some(driver), Some(tokenizer)) = (model.driver.as_mut(), model.tokenizer.as_ref()) {
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
    // Read only the last tensor's dequantized floats (the ping-pong slot holds
    // the last-processed layer's output).
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

/// Deterministic placeholder next-token in `1..vocab`. Mirrors
/// `engine_server::session::stub_next_token` so the real and synthetic paths
/// stay semantically comparable.
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
/// When `driver` and `tokenizer` are hooked, runs real forward prefill and single-token
/// decode steps over the streamed GPU pipeline.
/// When not hooked, falls back to the deterministic placeholder stub path.
pub fn decode_run(
    model: &mut RealModel,
    vocab: u32,
    prompt: &str,
    max_tokens: u32,
) -> Result<Vec<u32>, EngineError> {
    if let (Some(driver), Some(tokenizer)) = (model.driver.as_mut(), model.tokenizer.as_ref()) {
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
}
