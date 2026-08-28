//! Local inference quality demonstration (Phase 6/7 ForwardDriver).
//!
//! Evaluates real model inference across diverse prompts on the local Qwen3-0.6B fixture.

use cudarc::driver::CudaDevice;
use engine_core::BpeTokenizer;
use engine_io::{GgufReader, LoadedPinned, load_to_pinned};
use engine_server::runtime;
use std::path::PathBuf;
use std::time::Instant;

type DynError = Box<dyn std::error::Error + Send + Sync>;

fn fixture_path() -> Option<PathBuf> {
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
#[ignore]
fn test_local_inference_quality_samples() -> Result<(), DynError> {
    engine_cuda::ensure_cuda_dll_paths();
    let fixture = fixture_path().ok_or("fixture not present (GPU test)")?;
    println!("\nLoading model fixture: {:?}", fixture);
    let start_load = Instant::now();
    let reader = GgufReader::open(&fixture)?;
    let pinned: &'static LoadedPinned = Box::leak(Box::new(load_to_pinned(&reader, &fixture)?));
    let _device = CudaDevice::new(0)?;
    println!(
        "Loaded to pinned RAM in {:.2} ms",
        start_load.elapsed().as_secs_f64() * 1000.0
    );

    let mut model = runtime::build_real_driver_model(&reader, pinned, 128)?;
    let tokenizer = BpeTokenizer::from_reader(&reader)?;

    let prompts = [
        "The capital of France is",
        "Hello, my name is",
        "def fibonacci(n):",
        "2 + 2 =",
    ];

    println!("\n=======================================================");
    println!("      TITAN LOCAL INFERENCE QUALITY EVALUATION         ");
    println!("=======================================================");

    for prompt in prompts {
        println!("\n[PROMPT]: {:?}", prompt);
        let start_eval = Instant::now();

        let driver = model.driver.as_mut().expect("driver initialized");
        let prompt_tokens = tokenizer.encode(prompt)?;

        println!(
            "  Encoded input tokens (n={}): {:?}",
            prompt_tokens.len(),
            prompt_tokens
        );

        let initial_logits = driver.prefill(&prompt_tokens)?;
        let mut current_token = runtime::argmax(&initial_logits);
        let mut generated_tokens = vec![current_token];
        let mut generated_text = String::new();

        if let Ok(piece) = tokenizer.decode(&[current_token]) {
            generated_text.push_str(&piece);
        }

        const GEN_TOKENS: usize = 6;
        for _ in 1..GEN_TOKENS {
            let logits = driver.decode(current_token)?;
            current_token = runtime::argmax(&logits);
            generated_tokens.push(current_token);
            if let Ok(piece) = tokenizer.decode(&[current_token]) {
                generated_text.push_str(&piece);
            }
        }

        let elapsed = start_eval.elapsed().as_secs_f64();
        println!(
            "  Generated tokens (n={}): {:?}",
            generated_tokens.len(),
            generated_tokens
        );
        println!("  Decoded output text: {:?}", generated_text);
        println!("  Full completion: {:?} + {:?}", prompt, generated_text);
        println!(
            "  Time elapsed: {:.2} s ({:.2} tok/s)",
            elapsed,
            (prompt_tokens.len() + GEN_TOKENS) as f64 / elapsed
        );
    }

    println!("\n=======================================================\n");

    Ok(())
}
