use engine_core::forward_driver::ForwardDriver;
use engine_core::sampler::{Sampler, SamplerParams};
use engine_core::tokenizer::BpeTokenizer;
use engine_io::config::ModelConfig;
use engine_io::loader::load_to_pinned;
use engine_io::reader::GgufReader;
use std::path::PathBuf;
use std::time::Instant;

#[test]
#[ignore]
fn test_speculative_speedup_1b_to_3b() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();

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

    let models_dir = workspace_root.join("models");
    let draft_path = models_dir.join("Llama-3.2-1B-Instruct-Q4_K_M.gguf");
    let target_path = models_dir.join("Llama-3.2-3B-Instruct-Q4_K_M.gguf");

    if !draft_path.exists() || !target_path.exists() {
        println!(
            "Skipping speculative bench: models not found ({:?}, {:?})",
            draft_path, target_path
        );
        return Ok(());
    }

    println!("\n================================================================================");
    println!(">>> BENCHMARKING MULTI-MODEL SPECULATIVE DECODING (1B Draft -> 3B Target)");
    println!("================================================================================");

    // 1. Load Draft Model (1B) into GPU VRAM
    println!("Loading Draft Model (Llama 3.2 1B)...");
    let draft_reader = GgufReader::open(&draft_path)?;
    let draft_cfg = ModelConfig::from_reader(&draft_reader)?;
    let draft_pinned = load_to_pinned(&draft_reader, &draft_path)?;
    let mut draft_driver = ForwardDriver::new(&draft_reader, &draft_pinned, &draft_cfg, 512)?;
    let _ = draft_driver.capture_autonomous_decode_graph();

    // 2. Load Target Model (3B) into GPU VRAM
    println!("Loading Target Model (Llama 3.2 3B)...");
    let target_reader = GgufReader::open(&target_path)?;
    let target_cfg = ModelConfig::from_reader(&target_reader)?;
    let target_pinned = load_to_pinned(&target_reader, &target_path)?;
    let tokenizer = BpeTokenizer::from_reader(&target_reader)?;
    let mut target_driver = ForwardDriver::new(&target_reader, &target_pinned, &target_cfg, 512)?;
    let _ = target_driver.capture_autonomous_decode_graph();

    let raw_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 28 Aug 2026\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWrite a short explanation of how photosynthesizing plants convert sunlight into energy.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    let prompt_tokens = tokenizer.encode(raw_prompt)?;

    let mut sampler = Sampler::new(42);
    let params = SamplerParams::greedy();

    // -------------------------------------------------------------------------
    // BASELINE: Target 3B Single-Model Autonomous Decode
    // -------------------------------------------------------------------------
    println!("\n[1/2] Running Target 3B Baseline (Autonomous Decode)...");
    target_driver.reset_pos();
    let init_logits = target_driver.prefill(&prompt_tokens)?;
    let cur_token = {
        let mut best_i = 0;
        let mut best_val = f32::NEG_INFINITY;
        for (idx, &v) in init_logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_i = idx;
            }
        }
        best_i as u32
    };
    println!(
        "  Target 3B cur_token in [1/2]: {} ({:?})",
        cur_token,
        tokenizer.decode(&[cur_token])
    );

    let n_steps = 30;
    let t_start_base = Instant::now();
    let gen_tokens = target_driver.generate_autonomous_gpu(cur_token, n_steps)?;
    let elapsed_base = t_start_base.elapsed().as_secs_f64();
    let base_tok_per_sec = gen_tokens.len() as f64 / elapsed_base;
    let base_ms_per_tok = (elapsed_base * 1000.0) / gen_tokens.len() as f64;
    println!(
        "  Target 3B Baseline: {:.1} tok/s ({:.2} ms/tok) | {} tokens generated in {:.2} ms",
        base_tok_per_sec,
        base_ms_per_tok,
        gen_tokens.len(),
        elapsed_base * 1000.0
    );
    println!("  Baseline text: {:?}", tokenizer.decode(&gen_tokens));

    // -------------------------------------------------------------------------
    // SPECULATIVE DECODING: 1B Draft -> 3B Target (K = 3)
    // -------------------------------------------------------------------------
    println!("\n[2/2] Running Multi-Model Speculative Decoding (1B Draft -> 3B Target, K=3)...");
    draft_driver.reset_pos();
    target_driver.reset_pos();
    let draft_init_logits = draft_driver.prefill(&prompt_tokens)?;
    let target_init_logits = target_driver.prefill(&prompt_tokens)?;

    let draft_init_token = Sampler::argmax(&draft_init_logits) as u32;
    let target_init_token = Sampler::argmax(&target_init_logits) as u32;

    println!(
        "  Draft initial token from prefill: {} ({:?})",
        draft_init_token,
        tokenizer.decode(&[draft_init_token])
    );
    println!(
        "  Target initial token from prefill: {} ({:?})",
        target_init_token,
        tokenizer.decode(&[target_init_token])
    );

    let mut current_token = target_init_token;
    let mut spec_emitted_tokens = vec![current_token];
    let mut total_accepted = 0;
    let mut total_proposed = 0;
    let k = 3;
    let spec_target_tokens = 30;

    let t_start_spec = Instant::now();
    while spec_emitted_tokens.len() < spec_target_tokens {
        // Step A: Draft model emits K candidates autonomously on GPU
        let t_draft_start = Instant::now();
        let candidates = draft_driver.generate_autonomous_gpu(current_token, k)?;
        let t_draft = t_draft_start.elapsed();
        total_proposed += k;

        // Step B: Target model verifies candidates in 1 parallel GPU pass
        let t_verif_start = Instant::now();
        let verif = target_driver.verify_speculative(
            current_token,
            &candidates,
            &mut sampler,
            &params,
            &spec_emitted_tokens,
        )?;
        let t_verif = t_verif_start.elapsed();

        if spec_emitted_tokens.len() < 15 {
            let cand_str: Vec<(u32, String)> = candidates
                .iter()
                .map(|&c| (c, tokenizer.decode(&[c]).unwrap_or_default()))
                .collect();
            let emit_str: Vec<(u32, String)> = verif
                .emitted_tokens
                .iter()
                .map(|&c| (c, tokenizer.decode(&[c]).unwrap_or_default()))
                .collect();
            println!(
                "  [DEBUG] cur: {} ({:?}) | Draft cand: {:?} | Target emitted: {:?} | Acc: {} | Draft: {:.2}ms, Verif: {:.2}ms",
                current_token,
                tokenizer.decode(&[current_token]).unwrap_or_default(),
                cand_str,
                emit_str,
                verif.n_accepted,
                t_draft.as_secs_f64() * 1000.0,
                t_verif.as_secs_f64() * 1000.0
            );
        }

        total_accepted += verif.n_accepted;
        for &tok in &verif.emitted_tokens {
            spec_emitted_tokens.push(tok);
        }

        // Step C: Synchronize draft sequence position with target
        draft_driver.set_pos(target_driver.pos());
        current_token = verif.bonus_token;
    }
    let elapsed_spec = t_start_spec.elapsed().as_secs_f64();
    let spec_tokens_generated = spec_emitted_tokens.len() - 1;
    let spec_tok_per_sec = spec_tokens_generated as f64 / elapsed_spec;
    let spec_ms_per_tok = (elapsed_spec * 1000.0) / spec_tokens_generated as f64;
    let acceptance_rate = (total_accepted as f64 / total_proposed as f64) * 100.0;
    let speedup = spec_tok_per_sec / base_tok_per_sec;

    println!(
        "  Speculative Decoding: {:.1} tok/s ({:.2} ms/tok) | {} tokens in {:.2} ms",
        spec_tok_per_sec,
        spec_ms_per_tok,
        spec_tokens_generated,
        elapsed_spec * 1000.0
    );
    println!(
        "  Acceptance Rate: {:.1}% ({}/{} candidates accepted)",
        acceptance_rate, total_accepted, total_proposed
    );
    println!("  Effective Speedup: {:.2}x vs Target 3B Baseline", speedup);

    println!("\n================================================================================");
    println!("===                   SPECULATIVE DECODING SPEEDUP SUMMARY                   ===");
    println!("================================================================================");
    println!("Target Model:           Llama 3.2 3B Instruct");
    println!("Draft Model:            Llama 3.2 1B Instruct");
    println!("Baseline 3B Throughput: {:.1} tok/s", base_tok_per_sec);
    println!("Speculative Throughput: {:.1} tok/s", spec_tok_per_sec);
    println!("Acceleration Ratio:     {:.2}x", speedup);
    println!("================================================================================\n");

    Ok(())
}
