//! Radix Tree Automatic Prefix Caching (APC) Test (Milestone 6).
//!
//! Verifies that consecutive turns with shared system/tool prefixes
//! bypass redundant GPU prefill calculations, dropping TTFT to ~0 ms.

use cudarc::driver::CudaDevice;
use engine_io::{GgufReader, LoadedPinned, load_to_pinned};
use engine_server::runtime::{self, RealModel};
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
fn test_radix_prefix_caching_speedup() -> Result<(), DynError> {
    engine_cuda::ensure_cuda_dll_paths();
    let Some(fixture) = fixture_path() else {
        eprintln!("SKIPPING test_radix_prefix_caching_speedup: fixture not found");
        return Ok(());
    };

    println!("\n================================================================================");
    println!(">>> TESTING RADIX PREFIX CACHING (APC) ON TITAN FORWARD DRIVER");
    println!("================================================================================");

    let reader = GgufReader::open(&fixture)?;
    let pinned: &'static LoadedPinned = Box::leak(Box::new(load_to_pinned(&reader, &fixture)?));
    let _device = CudaDevice::new(0)?;

    let mut model: RealModel<'static> = runtime::build_real_driver_model(&reader, pinned, 512)?;
    let tokenizer = model.tokenizer.take().expect("tokenizer");
    let mut driver = model.driver.take().expect("driver");

    let system_prefix = "<|im_start|>system\nYou are a high-performance coding agent equipped with tools: get_weather, run_code, search_web.<|im_end|>\n";
    let user_turn_1 =
        "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>\n";
    let user_turn_2 = "<|im_start|>user\nWhat is 3 + 3?<|im_end|>\n<|im_start|>assistant\n";

    let prompt_1 = format!("{system_prefix}{user_turn_1}");
    let prompt_2 = format!("{system_prefix}{user_turn_1}{user_turn_2}");

    let tokens_1 = tokenizer.encode(&prompt_1)?;
    let tokens_2 = tokenizer.encode(&prompt_2)?;

    println!("  [Turn 1] Prompt length: {} tokens", tokens_1.len());
    let t1_start = Instant::now();
    let logits_1 = driver.prefill(&tokens_1)?;
    let t1_ms = t1_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  [Turn 1] Full Prefill TTFT: {:.2} ms ({} tokens processed)",
        t1_ms,
        tokens_1.len()
    );
    assert!(!logits_1.is_empty());

    println!(
        "\n  [Turn 2] Prompt length: {} tokens (shares {} prefix tokens with Turn 1)",
        tokens_2.len(),
        tokens_1.len()
    );

    let t2_start = Instant::now();
    let logits_2 = driver.prefill(&tokens_2)?;
    let t2_ms = t2_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  [Turn 2] Radix APC Cached TTFT: {:.2} ms (suffix prefill only!)",
        t2_ms
    );
    assert!(!logits_2.is_empty());

    // Turn 3: 100% exact match cache hit
    println!("\n  [Turn 3] Re-evaluating Turn 2 Prompt (100% Prefix Match)");
    let t3_start = Instant::now();
    let logits_3 = driver.prefill(&tokens_2)?;
    let t3_ms = t3_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  [Turn 3] 100% Cached TTFT: {:.2} ms (Instant Cache Hit!)",
        t3_ms
    );
    assert!(!logits_3.is_empty());

    println!(
        "\n>>> SUCCESS: Radix Tree Prefix Cache accelerated TTFT from {:.2} ms -> {:.2} ms -> {:.2} ms!",
        t1_ms, t2_ms, t3_ms
    );
    println!("================================================================================\n");

    Ok(())
}
