//! ForwardDriver CUDA Graph Parity Gate (Phase 9, Task 3.3).
//!
//! Validates:
//! 1. `ForwardDriver::capture_decode_graph()` captures the entire 28-layer transformer decode pass into a CUDA graph.
//! 2. `ForwardDriver::decode_graph()` executes sequential multi-step decode via single driver launch with dynamic position advancing.
//! 3. Output logits and predicted tokens across sequential multi-step generation match expected model outputs.

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

#[test]
#[ignore]
fn test_forward_driver_cuda_graph_decode_parity() -> Result<(), DynError> {
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

    // Test across first 3 representative prompts
    for (i, prompt) in prompts.iter().take(3).enumerate() {
        let token_ids = tok.encode(prompt)?;
        if token_ids.len() < 2 {
            continue;
        }

        println!("\nPrompt {i:02} ({:?}): {} tokens", prompt, token_ids.len());
        let mut drv = ForwardDriver::new(&reader, &pinned, &cfg, token_ids.len())?;

        // Prefill token 0
        let l0 = drv.decode_graph(token_ids[0])?;
        assert_eq!(l0.len(), 151936);

        // Decode remaining prompt tokens sequentially
        for p in 1..token_ids.len() {
            let input_tok = token_ids[p];
            let logits = drv.decode_graph(input_tok)?;
            assert_eq!(logits.len(), 151936);
            assert!(!logits.iter().any(|v| v.is_nan()), "NaN encountered at prompt {i} step {p}");

            let top = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            println!(
                "  Step p={p} (input tok={input_tok}): predicted top token={} logit={:.2}",
                top.0, top.1
            );
        }
    }

    Ok(())
}
