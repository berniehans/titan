//! Typed model hyperparameters extracted from GGUF metadata.
//!
//! `ModelConfig` turns the raw `GgufReader` metadata key-value map into typed,
//! validated fields (n_layer, n_head, head_dim, hidden_size, ...). Required
//! hyperparameters hard-fail when absent (`GgufError::MissingMetadata`); optional
//! ones (rope freq scale, head/value dims when absent) fall back to the same
//! defaults llama.cpp uses, never to silent garbage.
//!
//! Keys read (Qwen3/Qwen family):
//! - `general.architecture`
//! - `qwen3.block_count` (n_layer)
//! - `qwen3.embedding_length` (hidden_size / n_embd)
//! - `qwen3.feed_forward_length` (n_ff / intermediate_size)
//! - `qwen3.attention.head_count` (n_head)
//! - `qwen3.attention.head_count_kv` (n_head_kv, GQA)
//! - `qwen3.attention.key_length` / `value_length` (head dims)
//! - `qwen3.context_length`
//! - `qwen3.rope.freq_base` / `freq_scale`
//! - `qwen3.attention.layer_norm_rms_epsilon`
//! - `tokenizer.ggml.tokens` (derives vocab_size)

use crate::error::GgufError;
use crate::reader::GgufReader;
use crate::types::GgufValue;

/// Number of decoder layers.
pub const DEFAULT_N_LAYER: u32 = 0;
/// Default RoPE base frequency (llama.cpp default when `rope.freq_base` absent).
pub const DEFAULT_ROPE_FREQ_BASE: f32 = 10_000.0;
/// Default RMS norm epsilon (llama.cpp `LLAMA_DEFAULT_RMS_EPS`).
pub const DEFAULT_RMS_EPS: f32 = 1e-5;
/// `general.architecture` value expected for Qwen3.
pub const ARCH_QWEN3: &str = "qwen3";

/// Typed hyperparameters of a transformer model, derived from GGUF metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    /// Architecture name (e.g. "qwen3").
    pub architecture: String,
    /// Number of decoder layers.
    pub n_layer: u32,
    /// Number of query heads.
    pub n_head: u32,
    /// Number of KV heads (GQA); equal to n_head without GQA.
    pub n_head_kv: u32,
    /// Attention head dimension (key/value projection dim).
    pub head_dim: u32,
    /// Value projection head dim (falls back to head_dim).
    pub value_dim: u32,
    /// Hidden / embedding size.
    pub hidden_size: u32,
    /// Feed-forward intermediate size.
    pub intermediate_size: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Maximum context length.
    pub context_length: u32,
    /// RoPE base frequency.
    pub rope_freq_base: f32,
    /// RoPE frequency scaling factor (1.0 = none).
    pub rope_freq_scale: f32,
    /// RMS norm epsilon.
    pub rms_norm_eps: f32,
    /// Tokenizer family ("bpe", "spm", ...) mapped from `tokenizer.ggml.model`.
    pub tokenizer_model: String,
    /// End-of-sequence token id.
    pub eos_token_id: u32,
    /// Padding token id.
    pub padding_token_id: u32,
    /// Whether the model prepends a BOS token.
    pub add_bos: bool,
}

impl ModelConfig {
    /// Parses a typed `ModelConfig` from the reader's GGUF metadata.
    pub fn from_reader(reader: &GgufReader) -> Result<Self, GgufError> {
        let m = reader.metadata();

        let architecture = read_string(m, "general.architecture")?;
        let arch = architecture.as_str();

        let get_u32 = |suffix: &str| -> Result<u32, GgufError> {
            let key = format!("{arch}.{suffix}");
            if let Some(v) = m.get(&key).and_then(extract_u32) {
                return Ok(v);
            }
            // Fallback aliases
            for alt in &["qwen3", "qwen2", "qwen", "llama"] {
                let alt_key = format!("{alt}.{suffix}");
                if let Some(v) = m.get(&alt_key).and_then(extract_u32) {
                    return Ok(v);
                }
            }
            Err(GgufError::MissingMetadata(key))
        };

        let get_f32 = |suffix: &str| -> Option<f32> {
            let key = format!("{arch}.{suffix}");
            if let Some(v) = m.get(&key).and_then(extract_f32) {
                return Some(v);
            }
            for alt in &["qwen3", "qwen2", "qwen", "llama"] {
                let alt_key = format!("{alt}.{suffix}");
                if let Some(v) = m.get(&alt_key).and_then(extract_f32) {
                    return Some(v);
                }
            }
            None
        };

        let n_layer = get_u32("block_count")?;
        let hidden_size = get_u32("embedding_length")?;
        let intermediate_size = get_u32("feed_forward_length")?;
        let n_head = get_u32("attention.head_count")?;
        let n_head_kv = get_u32("attention.head_count_kv").unwrap_or(n_head);
        let context_length = get_u32("context_length").unwrap_or(4096);

        // Optional head dims; derive from n_embd/n_head when absent.
        let head_dim =
            get_u32("attention.key_length").unwrap_or(hidden_size.checked_div(n_head).unwrap_or(0));
        let value_dim = get_u32("attention.value_length").unwrap_or(head_dim);

        // Optional rope / norm defaults mirror llama.cpp.
        let rope_freq_base = get_f32("rope.freq_base").unwrap_or(DEFAULT_ROPE_FREQ_BASE);
        let rope_freq_scale = get_f32("rope.freq_scale").unwrap_or(1.0);
        let rms_norm_eps = get_f32("attention.layer_norm_rms_epsilon").unwrap_or(DEFAULT_RMS_EPS);

        // Vocab size comes from the token array length.
        let vocab_size = match m.get("tokenizer.ggml.tokens").and_then(GgufValue::as_array) {
            Some(tokens) => tokens.len() as u32,
            None => {
                return Err(GgufError::MissingMetadata(
                    "tokenizer.ggml.tokens".to_string(),
                ));
            }
        };

        // Tokenizer family mapping ("gpt2" is the BPE family used by Qwen3).
        let tokenizer_model = match m.get("tokenizer.ggml.model").and_then(GgufValue::as_str) {
            Some("gpt2") | Some("llama3") | Some("deepseek-llm") => "bpe".to_string(),
            Some(other) => other.to_string(),
            None => "bpe".to_string(),
        };

        let eos_token_id = m
            .get("tokenizer.ggml.eos_token_id")
            .and_then(extract_u32)
            .unwrap_or(0);
        let padding_token_id = m
            .get("tokenizer.ggml.padding_token_id")
            .and_then(extract_u32)
            .unwrap_or(0);
        let add_bos = m
            .get("tokenizer.ggml.add_bos_token")
            .and_then(GgufValue::as_bool)
            .unwrap_or(false);

        Ok(Self {
            architecture,
            n_layer,
            n_head,
            n_head_kv,
            head_dim,
            value_dim,
            hidden_size,
            intermediate_size,
            vocab_size,
            context_length,
            rope_freq_base,
            rope_freq_scale,
            rms_norm_eps,
            tokenizer_model,
            eos_token_id,
            padding_token_id,
            add_bos,
        })
    }

    /// A coherent generic-transformer default config for an architecture.
    ///
    /// Used as a fallback/scratch baseline where a real header is not required.
    /// Every optional field sits at a sane, non-zero default (head_dim derives
    /// from hidden_size/n_head like the `from_reader` missing-key branch).
    pub fn defaults_for_architecture(architecture: &str) -> Self {
        let n_head = 12u32;
        let hidden_size = 768u32;
        Self {
            architecture: architecture.to_string(),
            n_layer: 12,
            n_head,
            n_head_kv: 12,
            head_dim: hidden_size / n_head,
            value_dim: hidden_size / n_head,
            hidden_size,
            intermediate_size: 3072,
            vocab_size: 32_000,
            context_length: 2048,
            rope_freq_base: DEFAULT_ROPE_FREQ_BASE,
            rope_freq_scale: 1.0,
            rms_norm_eps: DEFAULT_RMS_EPS,
            tokenizer_model: "bpe".to_string(),
            eos_token_id: 0,
            padding_token_id: 0,
            add_bos: false,
        }
    }
}

fn read_string(
    m: &std::collections::HashMap<String, GgufValue>,
    key: &str,
) -> Result<String, GgufError> {
    m.get(key)
        .and_then(GgufValue::as_str)
        .map(str::to_string)
        .ok_or_else(|| GgufError::MissingMetadata(key.to_string()))
}

#[allow(dead_code)]
fn read_u32(m: &std::collections::HashMap<String, GgufValue>, key: &str) -> Result<u32, GgufError> {
    m.get(key)
        .and_then(extract_u32)
        .ok_or_else(|| GgufError::MissingMetadata(key.to_string()))
}

/// Accepts any integral GGUF value variant (U32/U64/I32/I64) as u32.
fn extract_u32(v: &GgufValue) -> Option<u32> {
    match v {
        GgufValue::U32(x) => Some(*x),
        GgufValue::U64(x) => u32::try_from(*x).ok(),
        GgufValue::I32(x) => u32::try_from(*x).ok(),
        GgufValue::I64(x) => u32::try_from(*x).ok(),
        _ => None,
    }
}

/// Accepts F32/F64 (and integral values that fit) as f32.
fn extract_f32(v: &GgufValue) -> Option<f32> {
    match v {
        GgufValue::F32(x) => Some(*x),
        GgufValue::F64(x) => Some(*x as f32),
        GgufValue::U32(x) => Some(*x as f32),
        _ => extract_u32(v).map(|x| x as f32),
    }
}
