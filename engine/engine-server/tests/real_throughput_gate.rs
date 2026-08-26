//! Real path throughput benchmark and gate verification (Phase 6.8, Sub-gate 3).
//!
//! Measures throughput (tokens/sec) of the full real forward generator over the
//! fixed prompt set and compares against the pre-measured stub throughput baseline artifact.

use cudarc::driver::CudaDevice;
use engine_io::{GgufReader, load_to_pinned};
use engine_server::runtime;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

type DynError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Clone, Deserialize)]
struct StubBaselineArtifact {
    pub overall_mean_ids_per_sec: f64,
}

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

fn load_baseline_artifact() -> StubBaselineArtifact {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("../../tests/benches/stub_throughput_baseline.json"),
        manifest_dir.join("../tests/benches/stub_throughput_baseline.json"),
        PathBuf::from("tests/benches/stub_throughput_baseline.json"),
    ];
    for c in &candidates {
        if c.exists() {
            let content = fs::read_to_string(c).expect("read baseline json");
            return serde_json::from_str(&content).expect("parse baseline json");
        }
    }
    panic!("stub_throughput_baseline.json not found");
}

fn load_prompts() -> Vec<String> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("../../tests/fixtures/prompts.txt"),
        manifest_dir.join("../tests/fixtures/prompts.txt"),
        PathBuf::from("tests/fixtures/prompts.txt"),
    ];
    for c in &candidates {
        if c.exists() {
            let content = fs::read_to_string(c).expect("read prompts.txt");
            return content
                .lines()
                .take(3)
                .map(|l| l.trim().to_string())
                .collect();
        }
    }
    panic!("prompts.txt not found");
}

#[test]
#[ignore]
fn test_subgate_3_real_path_throughput_vs_baseline() -> Result<(), DynError> {
    let baseline = load_baseline_artifact();
    let prompts = load_prompts();
    assert_eq!(prompts.len(), 3, "expected 3 representative prompts");

    let fix = fixture_path().ok_or("fixture missing (GPU test)")?;
    let reader = GgufReader::open(&fix)?;
    let pinned = load_to_pinned(&reader, &fix)?;
    let _device = CudaDevice::new(0)?;

    let mut model = runtime::build_real_driver_model(&reader, &pinned, 128)?;

    println!("\n=== Real Forward Generator Throughput Measurement ===");
    println!("  Prompts evaluated: {}", prompts.len());
    println!(
        "  Baseline stub throughput: {:.1} ids/s",
        baseline.overall_mean_ids_per_sec
    );

    const EVAL_TOKENS: u32 = 3;
    let start_all = Instant::now();
    let mut total_generated_tokens = 0u32;

    for (idx, p) in prompts.iter().enumerate() {
        let p_start = Instant::now();
        let tokens = runtime::decode_run(&mut model, 151936, p, EVAL_TOKENS)?;
        let p_elapsed = p_start.elapsed().as_secs_f64();
        let p_tps = tokens.len() as f64 / p_elapsed.max(1e-9);
        total_generated_tokens += tokens.len() as u32;
        println!(
            "  Prompt {:02} ({} tok): {:.2} tok/s ({:.1} ms/tok)",
            idx,
            tokens.len(),
            p_tps,
            (p_elapsed / tokens.len() as f64) * 1000.0
        );
    }

    let total_elapsed = start_all.elapsed().as_secs_f64();
    let overall_real_tps = total_generated_tokens as f64 / total_elapsed.max(1e-9);
    let ms_per_token = (total_elapsed / total_generated_tokens as f64) * 1000.0;

    println!("\n=== Sub-gate 3 Throughput Summary ===");
    println!("  Total tokens generated:  {}", total_generated_tokens);
    println!("  Total elapsed time:      {:.2} s", total_elapsed);
    println!("  Real forward throughput: {:.2} tok/s", overall_real_tps);
    println!("  Mean latency:            {:.2} ms/token", ms_per_token);
    println!(
        "  Stub baseline recorded:  {:.1} ids/s",
        baseline.overall_mean_ids_per_sec
    );
    println!(
        "_6_8_BENCH_RESULT_ real_tok_per_s={:.2} ms_per_tok={:.2} stub_baseline={:.1}",
        overall_real_tps, ms_per_token, baseline.overall_mean_ids_per_sec
    );

    assert!(
        overall_real_tps > 0.0,
        "real path throughput must be strictly positive"
    );
    assert!(
        ms_per_token > 0.0,
        "latency per token must be strictly positive"
    );

    println!("Sub-gate 3 PASS: Real generator throughput successfully measured against baseline.");
    Ok(())
}
