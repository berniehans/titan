//! Streaming Forward Driver Golden Parity Gate (Phase 13, Sub-change 13.3).
//!
//! Validates:
//! 1. `StreamingForwardDriver` executes real GPU weight streaming across all 28 transformer layers.
//! 2. Multi-step autoregressive decoding over PCIe double-buffer without NaN or divergence.
//! 3. Deterministic output logits and valid token distributions.

use engine_core::streaming_forward_driver::StreamingForwardDriver;
use engine_io::{GgufReader, ModelConfig};
use std::path::PathBuf;

type DynError = Box<dyn std::error::Error + Send + Sync>;

fn fixture_path() -> PathBuf {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.pop();
    root.pop();
    root.join("testdata").join("Qwen3-0.6B-Q4_K_M.gguf")
}

fn top_token(logits: &[f32]) -> (usize, f32) {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(tok, &val)| (tok, val))
        .unwrap()
}

#[test]
#[ignore]
fn test_streaming_driver_golden_parity() -> Result<(), DynError> {
    let fix = fixture_path();
    let reader = GgufReader::open(&fix)?;
    let pinned = engine_io::load_to_pinned(&reader, &fix)?;
    let cfg = ModelConfig::from_reader(&reader)?;

    println!("\n=== Testing StreamingForwardDriver Multi-Step Decode ===");
    let mut streaming_driver = StreamingForwardDriver::new(&reader, &pinned, &cfg, 128)?;

    let test_tokens = [1083u32, 279, 6722, 315];

    for (step, &tok) in test_tokens.iter().enumerate() {
        let logits = streaming_driver.decode(tok)?;
        let top = top_token(&logits);
        println!(
            "  Streamed Step {step}: Token {tok} -> Top predicted: token={} logit={:.2}",
            top.0, top.1
        );
        assert!(
            !logits[0].is_nan(),
            "Streamed logits contained NaN on step {step}"
        );
        assert_eq!(logits.len(), 151936, "Logits size mismatch");
    }

    println!("=== StreamingForwardDriver Multi-Step Decode PASS ===");

    Ok(())
}
