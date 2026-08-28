//! Grammar-Constrained Tool Call Generation Test (Milestone 7).
//!
//! Spawns the real Titan engine on GPU with JSON Grammar constrained sampling
//! and verifies that generated structured output strictly conforms to valid JSON.

use cudarc::driver::CudaDevice;
use engine_core::grammar::JsonGrammar;
use engine_core::sampler::{Sampler, SamplerParams};
use engine_io::{GgufReader, LoadedPinned, load_to_pinned};
use engine_server::runtime::{self, RealModel};
use std::path::PathBuf;

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
fn test_grammar_constrained_json_generation() -> Result<(), DynError> {
    engine_cuda::ensure_cuda_dll_paths();
    let Some(fixture) = fixture_path() else {
        eprintln!("SKIPPING test_grammar_constrained_json_generation: fixture not found");
        return Ok(());
    };

    println!("\n================================================================================");
    println!(">>> TESTING GRAMMAR-GUIDED CONSTRAINED JSON DECODING ON GPU");
    println!("================================================================================");

    let reader = GgufReader::open(&fixture)?;
    let pinned: &'static LoadedPinned = Box::leak(Box::new(load_to_pinned(&reader, &fixture)?));
    let _device = CudaDevice::new(0)?;

    let mut model: RealModel<'static> = runtime::build_real_driver_model(&reader, pinned, 512)?;
    let tokenizer = model.tokenizer.take().expect("tokenizer");
    let mut driver = model.driver.take().expect("driver");

    let prompt = "<|im_start|>system\nYou are a tool calling agent. Call the function `get_weather` with argument `city` in JSON format.<|im_end|>\n<|im_start|>user\nWhat is the weather in Tokyo?<|im_end|>\n<|im_start|>assistant\n```json\n{";
    let prompt_tokens = tokenizer.encode(prompt)?;

    let mut logits = driver.prefill(&prompt_tokens)?;
    let mut sampler = Sampler::new(42);
    let params = SamplerParams::greedy();

    let mut generated_tokens = Vec::new();
    let mut grammar = JsonGrammar::inside_object().with_keys(&["city", "weather", "temperature"]);

    print!("  [Generated Stream]: {{");
    for step in 0..30 {
        // Sample with JSON Grammar constraint
        let tok = sampler.sample_constrained(&logits, &generated_tokens, &params, |cand_tok| {
            let cand_str = tokenizer.decode_piece(cand_tok);
            grammar.is_token_valid(&cand_str)
        });

        if tok == 0 || tok == 151645 || tok == 151643 { // EOS tokens
            break;
        }

        let tok_str = tokenizer.decode_piece(tok);
        grammar.advance(&tok_str);
        print!("{tok_str}");

        generated_tokens.push(tok);
        logits = driver.decode(tok)?;

        if grammar.is_complete() && step >= 5 {
            println!("\n  [Grammar] Reached complete JSON document state at step {}", step);
            break;
        }
    }

    let raw_generated = tokenizer.decode(&generated_tokens)?;
    let generated_json_text = format!("{{{raw_generated}");
    println!("\n  [Final Generated Output]:\n{}", generated_json_text.trim());

    // Verify output strictly parses as JSON
    let parsed: serde_json::Value = serde_json::from_str(generated_json_text.trim())?;
    println!("  [Serde Parse Verification]: {:?}", parsed);
    assert_eq!(parsed["city"], "Tokyo");

    println!("\n>>> SUCCESS: Grammar-Constrained JSON decoding produced 100% valid JSON!");
    println!("================================================================================\n");

    Ok(())
}
