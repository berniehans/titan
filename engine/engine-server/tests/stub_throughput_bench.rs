//! Stub-path throughput benchmark and baseline generator (Phase 6.8, Group 0).
//!
//! Measures throughput (ids/s) of the pre-swap stub generation path across
//! the 12 fixed prompts from `tests/fixtures/prompts.txt`.
//!
//! Generates and commits the baseline artifact `tests/benches/stub_throughput_baseline.json`
//! and asserts that throughput is stable across 3 runs with spread < 5%.

use engine_kvcache::PagedKvCacheConfig;
use engine_server::scheduler::BatchScheduler;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const TOKENS_PER_PROMPT: u32 = 128;
const REPEATS_PER_PROMPT: usize = 50;
const NUM_RUNS: usize = 3;
const VOCAB_SIZE: u32 = 1000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptBaseline {
    pub prompt_index: usize,
    pub prompt: String,
    pub n_tokens: u32,
    pub mean_duration_ms: f64,
    pub mean_ids_per_sec: f64,
    pub runs_ids_per_sec: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StubThroughputBaseline {
    pub timestamp: String,
    pub n_prompts: usize,
    pub tokens_per_prompt: u32,
    pub total_tokens_per_run: u32,
    pub overall_mean_ids_per_sec: f64,
    pub runs_overall_ids_per_sec: Vec<f64>,
    pub spread_pct: f64,
    pub prompts: Vec<PromptBaseline>,
}

fn prompts_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("../../tests/fixtures/prompts.txt"),
        manifest_dir.join("../tests/fixtures/prompts.txt"),
        PathBuf::from("tests/fixtures/prompts.txt"),
    ];
    for c in &candidates {
        if c.exists() {
            return c.clone();
        }
    }
    panic!("prompts.txt not found");
}

fn load_prompts() -> Vec<String> {
    let content = fs::read_to_string(prompts_path()).expect("read prompts.txt");
    content
        .lines()
        .take(12)
        .map(|l| l.trim().to_string())
        .collect()
}

fn prompt_token(prompt: &str, vocab: u32) -> u32 {
    let mut hash: u32 = 2166136261;
    for byte in prompt.as_bytes() {
        hash = (hash ^ (*byte as u32)).wrapping_mul(16777619);
    }
    hash.wrapping_rem(vocab) + 1
}

fn run_stub_decode_batch(scheduler: &mut BatchScheduler, prompt: &str, max_tokens: u32, repeats: usize) -> (usize, f64) {
    let start_tok = prompt_token(prompt, VOCAB_SIZE);
    for _ in 0..repeats {
        scheduler.add(VOCAB_SIZE, start_tok, max_tokens).expect("session");
    }

    let t0 = Instant::now();
    let mut token_count = 0;
    while scheduler.active_count() > 0 {
        token_count += scheduler.advance().len();
    }
    let elapsed = t0.elapsed().as_secs_f64();
    (token_count, elapsed)
}

#[test]
fn test_measure_stub_throughput_baseline_artifact() {
    let prompts = load_prompts();
    assert_eq!(prompts.len(), 12, "expected 12 golden prompts");

    let cfg = PagedKvCacheConfig {
        n_blocks: REPEATS_PER_PROMPT * 2,
        block_tokens: TOKENS_PER_PROMPT as usize + 1,
        heads: 1,
        head_dim: 1,
    };
    let mut scheduler = BatchScheduler::new(cfg).expect("kv pool");

    // Warm-up runs to settle CPU caches and JIT/OS scheduling
    for _ in 0..3 {
        for p in &prompts {
            let _ = run_stub_decode_batch(&mut scheduler, p, TOKENS_PER_PROMPT, 10);
        }
    }

    println!("\n=== Measuring Stub-Path Throughput Baseline (3 runs) ===");

    let mut run_overall_tps = Vec::with_capacity(NUM_RUNS);
    let mut prompt_run_tps: Vec<Vec<f64>> = vec![Vec::with_capacity(NUM_RUNS); prompts.len()];
    let mut prompt_run_durations: Vec<Vec<f64>> = vec![Vec::with_capacity(NUM_RUNS); prompts.len()];

    for run_idx in 0..NUM_RUNS {
        let mut total_tokens = 0u32;
        let mut total_elapsed = 0.0;

        for (p_idx, p_str) in prompts.iter().enumerate() {
            let (count, elapsed) = run_stub_decode_batch(&mut scheduler, p_str, TOKENS_PER_PROMPT, REPEATS_PER_PROMPT);
            assert_eq!(count, (TOKENS_PER_PROMPT as usize) * REPEATS_PER_PROMPT);
            total_tokens += count as u32;
            total_elapsed += elapsed;

            let tps = count as f64 / elapsed.max(1e-9);
            prompt_run_tps[p_idx].push(tps);
            prompt_run_durations[p_idx].push(elapsed * 1000.0);
        }

        let overall_tps = total_tokens as f64 / total_elapsed.max(1e-9);
        run_overall_tps.push(overall_tps);
        println!(
            "  Run {}: total tokens={}, elapsed={:.3}ms, overall throughput={:.1} ids/s",
            run_idx + 1,
            total_tokens,
            total_elapsed * 1000.0,
            overall_tps
        );
    }

    let min_tps = run_overall_tps.iter().copied().fold(f64::INFINITY, f64::min);
    let max_tps = run_overall_tps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean_tps = run_overall_tps.iter().sum::<f64>() / NUM_RUNS as f64;
    let spread_pct = ((max_tps - min_tps) / mean_tps) * 100.0;

    println!("\n=== Stub Baseline Results ===");
    println!("  Mean throughput: {:.1} ids/s", mean_tps);
    println!("  Min throughput:  {:.1} ids/s", min_tps);
    println!("  Max throughput:  {:.1} ids/s", max_tps);
    println!("  Spread:          {:.2}%", spread_pct);

    assert!(mean_tps > 0.0, "mean throughput must be positive");
    assert!(
        spread_pct < 5.0,
        "spread across 3 runs ({spread_pct:.2}%) must be < 5.0%"
    );

    let mut prompt_baselines = Vec::new();
    for (p_idx, p_str) in prompts.iter().enumerate() {
        let mean_dur = prompt_run_durations[p_idx].iter().sum::<f64>() / NUM_RUNS as f64;
        let mean_p_tps = prompt_run_tps[p_idx].iter().sum::<f64>() / NUM_RUNS as f64;
        prompt_baselines.push(PromptBaseline {
            prompt_index: p_idx,
            prompt: p_str.clone(),
            n_tokens: TOKENS_PER_PROMPT,
            mean_duration_ms: mean_dur,
            mean_ids_per_sec: mean_p_tps,
            runs_ids_per_sec: prompt_run_tps[p_idx].clone(),
        });
    }

    let baseline_data = StubThroughputBaseline {
        timestamp: "2026-08-26T10:10:00Z".to_string(),
        n_prompts: prompts.len(),
        tokens_per_prompt: TOKENS_PER_PROMPT,
        total_tokens_per_run: TOKENS_PER_PROMPT * (prompts.len() as u32) * (REPEATS_PER_PROMPT as u32),
        overall_mean_ids_per_sec: mean_tps,
        runs_overall_ids_per_sec: run_overall_tps,
        spread_pct,
        prompts: prompt_baselines,
    };

    // Save artifact to tests/benches/stub_throughput_baseline.json
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_dir = manifest_dir.join("../../tests/benches");
    fs::create_dir_all(&out_dir).expect("create tests/benches dir");
    let out_path = out_dir.join("stub_throughput_baseline.json");

    let json_bytes = serde_json::to_string_pretty(&baseline_data).expect("serialize baseline");
    fs::write(&out_path, json_bytes).expect("write baseline JSON");
    println!("Baseline artifact written to: {}", out_path.display());
}
