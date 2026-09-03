//! Speculative Decoding Speedup Benchmark Gate (Phase 12, Sub-change 12.4).
//!
//! Measures:
//! 1. Generation throughput (tok/s) for Standard vs Speculative decoding.
//! 2. Candidate acceptance rate ($\alpha$).
//! 3. Total latency speedup factor across structured and repetitive prompts.

use engine_core::ForwardDriver;
use engine_core::ngram_draft::NgramDraftProposer;
use engine_core::sampler::{Sampler, SamplerParams};
use engine_core::tokenizer::BpeTokenizer;
use engine_io::{GgufReader, ModelConfig};
use std::path::PathBuf;
use std::time::Instant;

type DynError = Box<dyn std::error::Error + Send + Sync>;

fn fixture_path() -> PathBuf {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.pop();
    root.pop();
    root.join("testdata").join("Qwen3-0.6B-Q4_K_M.gguf")
}

#[test]
#[ignore]
fn test_speculative_speedup_benchmark() -> Result<(), DynError> {
    let candidates_dll = [
        r"C:\Users\niber\AppData\Local\hermes\hermes-agent\venv\Lib\site-packages\torch\lib",
        r"C:\Users\niber\.unsloth\studio\unsloth_studio\Lib\site-packages\torch\lib",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
    ];
    let mut extra_paths = Vec::new();
    for c in &candidates_dll {
        if std::path::Path::new(c).exists() {
            extra_paths.push(*c);
        }
    }
    let current_path = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{};{}", extra_paths.join(";"), current_path);
    unsafe {
        std::env::set_var("PATH", new_path);
    }

    let fix = fixture_path();
    let reader = GgufReader::open(&fix)?;
    let pinned = engine_io::load_to_pinned(&reader, &fix)?;
    let cfg = ModelConfig::from_reader(&reader)?;
    let tok = BpeTokenizer::from_reader(&reader)?;

    println!("\n==========================================================================");
    println!("  TITAN PHASE 12 BENCHMARK: SPECULATIVE DECODING SPEEDUP (28 LAYERS)     ");
    println!("==========================================================================");

    let prompts = [
        (
            "Repetitive Pattern",
            "struct Point { x: f32, y: f32, z: f32 }\nstruct Vector { x: f32, y: f32, z: f32 }\nstruct Matrix { x: f32, y: f32, z: f32 }\nstruct",
        ),
        (
            "Structured Code",
            "fn fibonacci(n: u64) -> u64 {\n    if n <= 1 {\n        return n;\n    }\n    fibonacci(n - 1) + fibonacci(n - 2)\n}\n\nfn fibonacci_fast(",
        ),
    ];

    let gen_target = 16;
    let params = SamplerParams::greedy();

    for (name, prompt) in &prompts {
        let prompt_tokens = tok.encode(prompt)?;

        // 1. Standard Autoregressive Decode
        let mut drv_std = ForwardDriver::new(&reader, &pinned, &cfg, 256)?;
        let l0 = drv_std.prefill(&prompt_tokens)?;
        let mut sampler_std = Sampler::new(42);
        let mut std_tokens = prompt_tokens.clone();

        let t_std_start = Instant::now();
        let first_tok = sampler_std.sample(&l0, &std_tokens, &params);
        std_tokens.push(first_tok);
        let mut cur_tok = first_tok;
        for _ in 1..gen_target {
            let logits = drv_std.decode(cur_tok)?;
            let next_tok = sampler_std.sample(&logits, &std_tokens, &params);
            std_tokens.push(next_tok);
            cur_tok = next_tok;
        }
        let std_elapsed_ms = t_std_start.elapsed().as_secs_f64() * 1000.0;
        let std_tok_per_sec = (gen_target as f64) / (std_elapsed_ms / 1000.0);

        // 2. Speculative Decode
        let mut drv_spec = ForwardDriver::new(&reader, &pinned, &cfg, 256)?;
        let l0_spec = drv_spec.prefill(&prompt_tokens)?;
        let mut sampler_spec = Sampler::new(42);
        let mut spec_tokens = prompt_tokens.clone();
        let proposer = NgramDraftProposer::new(3, 4, 2);

        let mut total_proposed = 0;
        let mut total_accepted = 0;

        let t_spec_start = Instant::now();
        let first_spec_tok = sampler_spec.sample(&l0_spec, &spec_tokens, &params);
        spec_tokens.push(first_spec_tok);
        let mut last_tok = first_spec_tok;

        while spec_tokens.len() < prompt_tokens.len() + gen_target {
            let candidates = proposer.propose(&spec_tokens);

            if candidates.is_empty() {
                let logits = drv_spec.decode(last_tok)?;
                let next_tok = sampler_spec.sample(&logits, &spec_tokens, &params);
                spec_tokens.push(next_tok);
                last_tok = next_tok;
            } else {
                total_proposed += candidates.len();
                let verif = drv_spec.verify_speculative(
                    last_tok,
                    &candidates,
                    &mut sampler_spec,
                    &params,
                    &spec_tokens,
                )?;
                total_accepted += verif.n_accepted;
                spec_tokens.extend_from_slice(&verif.emitted_tokens);
                last_tok = *verif.emitted_tokens.last().unwrap();
            }
        }
        let spec_elapsed_ms = t_spec_start.elapsed().as_secs_f64() * 1000.0;
        let spec_tok_per_sec = (gen_target as f64) / (spec_elapsed_ms / 1000.0);

        let acceptance_rate = if total_proposed > 0 {
            (total_accepted as f64) / (total_proposed as f64) * 100.0
        } else {
            0.0
        };

        let speedup = std_elapsed_ms / spec_elapsed_ms.max(0.001);

        println!(
            "| Prompt: {:<18} | Std: {:>5.1} tok/s ({:>6.1} ms) | Spec: {:>5.1} tok/s ({:>6.1} ms) | Accept: {:>4.1}% | Speedup: {:>4.2}x |",
            name,
            std_tok_per_sec,
            std_elapsed_ms,
            spec_tok_per_sec,
            spec_elapsed_ms,
            acceptance_rate,
            speedup
        );
    }
    println!("==========================================================================\n");

    Ok(())
}
