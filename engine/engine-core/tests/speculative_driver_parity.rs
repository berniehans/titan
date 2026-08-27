//! Speculative ForwardDriver GPU Parity Test (Phase 12, Sub-change 12.3).
//!
//! Validates:
//! 1. Speculative decoding (`verify_speculative` + `NgramDraftProposer`) generates
//!    bit-identical token sequences compared to standard single-token autoregressive decoding.
//! 2. Tested on real Qwen3-0.6B weights across 28 layers with greedy sampling.

use engine_core::ForwardDriver;
use engine_core::ngram_draft::NgramDraftProposer;
use engine_core::sampler::{Sampler, SamplerParams};
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

#[test]
#[ignore]
fn test_speculative_driver_exact_parity() -> Result<(), DynError> {
    let fix = fixture_path();
    let reader = GgufReader::open(&fix)?;
    let pinned = engine_io::load_to_pinned(&reader, &fix)?;
    let cfg = ModelConfig::from_reader(&reader)?;
    let tok = BpeTokenizer::from_reader(&reader)?;

    let prompt = "The capital of France is Paris. The capital of France is";
    let prompt_tokens = tok.encode(prompt)?;
    let max_gen = 10;

    let params = SamplerParams::greedy();

    // 1. Baseline: Standard Autoregressive Generation
    let mut drv_std = ForwardDriver::new(&reader, &pinned, &cfg, 128)?;
    let l0 = drv_std.prefill(&prompt_tokens)?;
    let mut sampler_std = Sampler::new(42);
    let mut std_tokens = prompt_tokens.clone();

    let first_tok = sampler_std.sample(&l0, &std_tokens, &params);
    std_tokens.push(first_tok);

    let mut cur_tok = first_tok;
    for _ in 1..max_gen {
        let logits = drv_std.decode(cur_tok)?;
        let next_tok = sampler_std.sample(&logits, &std_tokens, &params);
        std_tokens.push(next_tok);
        cur_tok = next_tok;
    }

    let std_generated = &std_tokens[prompt_tokens.len()..];
    println!("Standard Generated Tokens: {:?}", std_generated);

    // 2. Speculative Generation with NgramDraftProposer
    let mut drv_spec = ForwardDriver::new(&reader, &pinned, &cfg, 128)?;
    let l0_spec = drv_spec.prefill(&prompt_tokens)?;
    let mut sampler_spec = Sampler::new(42);
    let mut spec_tokens = prompt_tokens.clone();
    let proposer = NgramDraftProposer::new(3, 4, 2);

    let first_spec_tok = sampler_spec.sample(&l0_spec, &spec_tokens, &params);
    spec_tokens.push(first_spec_tok);

    let mut last_tok = first_spec_tok;

    while spec_tokens.len() < prompt_tokens.len() + max_gen {
        let candidates = proposer.propose(&spec_tokens);

        if candidates.is_empty() {
            // Standard single-step decode fallback
            let logits = drv_spec.decode(last_tok)?;
            let next_tok = sampler_spec.sample(&logits, &spec_tokens, &params);
            spec_tokens.push(next_tok);
            last_tok = next_tok;
        } else {
            // Multi-token speculative verification: base_token + candidates
            let verif = drv_spec.verify_speculative(
                last_tok,
                &candidates,
                &mut sampler_spec,
                &params,
                &spec_tokens,
            )?;
            println!(
                "  Speculative step: base {} + candidates {:?} -> accepted {} candidates (emitted {:?})",
                last_tok, candidates, verif.n_accepted, verif.emitted_tokens
            );
            spec_tokens.extend_from_slice(&verif.emitted_tokens);
            last_tok = *verif.emitted_tokens.last().unwrap();
        }
    }

    spec_tokens.truncate(prompt_tokens.len() + max_gen);
    let spec_generated = &spec_tokens[prompt_tokens.len()..];
    println!("Speculative Generated Tokens: {:?}", spec_generated);

    assert_eq!(
        std_generated, spec_generated,
        "Speculative decoding output must match standard autoregressive decode bit-for-bit"
    );

    println!("Speculative decoding parity PASS: Exact token identity verified!");
    Ok(())
}
