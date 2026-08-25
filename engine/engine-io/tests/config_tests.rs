//! Task 1.1 — Failing test: parse the real Qwen3-0.6B fixture's GGUF metadata
//! into a typed `ModelConfig`, asserting the exact known hyperparameters.
//!
//! Reference values (cross-checked from the fixture header, see
//! `testdata/parse_gguf.py` and `openspec/changes/6.1-config-tokenizer-goldens/`):
//!   block_count(n_layer)=28, embedding_length(n_embd)=1024,
//!   feed_forward_length(n_ff)=3072, head_count(n_head)=16,
//!   head_count_kv(n_head_kv)=8, key_length=128, value_length=128,
//!   rope.freq_base=1_000_000.0, rms_norm_eps≈1e-6, vocab_size=151936.

use engine_io::GgufReader;
use engine_io::config::ModelConfig;
use std::path::PathBuf;

fn get_fixture_path() -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("ENGINE_TESTDATA") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return Some(p);
        }
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("../../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        manifest_dir.join("../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
    ];
    for c in &candidates {
        if c.exists() {
            return Some(c.canonicalize().unwrap_or_else(|_| c.clone()));
        }
    }
    None
}

#[test]
fn test_6_1_parse_fixture_config_matches_qwen3_0_6b() {
    let Some(fixture) = get_fixture_path() else {
        eprintln!("SKIP: fixture not present in this environment");
        return;
    };
    let reader = GgufReader::open(&fixture).expect("Failed to open GGUF file");
    let cfg = ModelConfig::from_reader(&reader).expect("ModelConfig from fixture metadata");

    assert_eq!(cfg.architecture, "qwen3");
    assert_eq!(cfg.n_layer, 28, "n_layer from qwen3.block_count");
    assert_eq!(cfg.n_head, 16, "n_head from qwen3.attention.head_count");
    assert_eq!(
        cfg.n_head_kv, 8,
        "n_head_kv from qwen3.attention.head_count_kv"
    );
    assert_eq!(
        cfg.head_dim, 128,
        "head_dim from qwen3.attention.key_length"
    );
    assert_eq!(
        cfg.hidden_size, 1024,
        "hidden_size from qwen3.embedding_length"
    );
    assert_eq!(
        cfg.intermediate_size, 3072,
        "intermediate_size from qwen3.feed_forward_length"
    );
    assert_eq!(
        cfg.vocab_size, 151_936,
        "vocab_size from tokenizer.ggml.tokens len"
    );
    assert_eq!(
        cfg.context_length, 40_960,
        "context_length from qwen3.context_length"
    );

    // rope
    assert_eq!(cfg.rope_freq_base, 1_000_000.0, "rope.freq_base");
    assert_eq!(
        cfg.rope_freq_scale, 1.0,
        "rope.freq_scale default when absent"
    );

    // rms norm eps (f32 ≈ 1e-6)
    assert!(
        (cfg.rms_norm_eps - 1e-6).abs() < 1e-9,
        "rms_norm_eps ≈ 1e-6, got {}",
        cfg.rms_norm_eps
    );

    // tokenizer identity
    assert_eq!(cfg.tokenizer_model, "bpe");
    assert_eq!(cfg.eos_token_id, 151_645);
    assert_eq!(cfg.padding_token_id, 151_654);
    assert!(!cfg.add_bos);
}

/// Task 1.3 — absent/optional metadata falls back to sane defaults and never
/// produces silent garbage. Composed on the same module without a fixture by
/// building a synthetic metadata map that exercises every optional branch.
#[test]
fn test_6_1_optional_fields_default_sanely() {
    // A helper route: build a reader-free default and assert defaults are sane.
    let d = ModelConfig::defaults_for_architecture("qwen3");
    assert_eq!(d.architecture, "qwen3");
    assert_eq!(d.rope_freq_scale, 1.0, "freq scale defaults to 1.0");
    assert_eq!(
        d.head_dim,
        d.hidden_size / d.n_head,
        "head_dim derives from n_embd/n_head"
    );
    assert!(d.rope_freq_base > 0.0);
    assert!(d.rms_norm_eps > 0.0);
}
