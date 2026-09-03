//! Chunked Prefill & FlashAttention-2 Parity Gate (Phase 11, Task 3.3).
//!
//! Validates:
//! 1. `ForwardDriver::prefill_chunked` processes prompts in parallel chunks via batched GEMM & FlashAttention-2.
//! 2. Compares output logits against standard `ForwardDriver::prefill` (`cos-sim >= 0.997`).
//! 3. Verifies across prompt fixtures.

use engine_core::ForwardDriver;
use engine_core::tokenizer::BpeTokenizer;
use engine_io::{GgufReader, ModelConfig};
use std::path::PathBuf;

type DynError = Box<dyn std::error::Error + Send + Sync>;

fn fixture_path() -> PathBuf {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.pop();
    root.pop();
    root.join("testdata").join("Qwen3-0.6B-Q4_K_M.gguf")
}

fn prompts_path() -> PathBuf {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.pop();
    root.pop();
    root.join("tests").join("fixtures").join("prompts.txt")
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[test]
#[ignore]
fn test_chunked_prefill_parity() -> Result<(), DynError> {
    engine_cuda::ensure_cuda_dll_paths();
    let fix = fixture_path();
    let reader = GgufReader::open(&fix)?;
    let pinned = engine_io::load_to_pinned(&reader, &fix)?;
    let cfg = ModelConfig::from_reader(&reader)?;
    let tok = BpeTokenizer::from_reader(&reader)?;

    let prompts_txt = std::fs::read_to_string(prompts_path())?;
    let prompts: Vec<&str> = prompts_txt
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    for (i, prompt) in prompts.iter().take(3).enumerate() {
        let token_ids = tok.encode(prompt)?;
        if token_ids.len() < 2 {
            continue;
        }

        println!(
            "\nEvaluating Prompt {i:02} ({:?}): {} tokens",
            prompt,
            token_ids.len()
        );

        // 1. Serial prefill baseline
        let mut drv_serial = ForwardDriver::new(&reader, &pinned, &cfg, token_ids.len())?;
        let logits_serial = drv_serial.prefill(&token_ids)?;

        // 2. Chunked prefill with chunk_size = 16
        let mut drv_chunked = ForwardDriver::new(&reader, &pinned, &cfg, token_ids.len())?;
        let logits_chunked = drv_chunked.prefill_chunked(&token_ids, 16)?;

        assert_eq!(logits_serial.len(), logits_chunked.len());
        assert_eq!(logits_serial.len(), 151936);

        let cs = cosine_similarity(&logits_serial, &logits_chunked);
        println!("Prompt {i:02} Chunked Prefill Cosine Similarity = {cs:.6}");

        assert!(
            cs >= 0.997,
            "Cosine similarity {cs} below threshold 0.997 for prompt {i}"
        );

        // Top token prediction agreement
        let top_serial = logits_serial
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let top_chunked = logits_chunked
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        println!(
            "  Serial Top Token: {} ({:.2}) | Chunked Top Token: {} ({:.2})",
            top_serial.0, top_serial.1, top_chunked.0, top_chunked.1
        );
        assert_eq!(
            top_serial.0, top_chunked.0,
            "Predicted top token mismatch on prompt {i}"
        );
    }

    Ok(())
}
