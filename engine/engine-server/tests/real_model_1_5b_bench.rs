use engine_core::{BpeTokenizer, ForwardDriver, NgramDraftProposer, Sampler, SamplerParams};
use engine_io::{load_to_pinned, GgufReader, ModelConfig};
use std::path::PathBuf;
use std::time::Instant;

fn model_path() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("../../models/qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        manifest_dir.join("../models/qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        PathBuf::from("models/qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        PathBuf::from("../models/qwen2.5-1.5b-instruct-q4_k_m.gguf"),
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
fn test_real_qwen2_5_1_5b_resident_inference() -> Result<(), Box<dyn std::error::Error>> {
    engine_cuda::ensure_cuda_dll_paths();

    let p = match model_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: models/qwen2.5-1.5b-instruct-q4_k_m.gguf not found");
            return Ok(());
        }
    };

    println!("\n=======================================================");
    println!("=== Testing Titan GPU Resident Inference on Real Model ===");
    println!("Model: {}", p.display());

    let t0 = Instant::now();
    let reader = GgufReader::open(&p)?;
    let cfg = ModelConfig::from_reader(&reader)?;
    println!(
        "Hyperparams: arch={}, layers={}, hidden={}, heads={}/{}, intermediate={}, vocab={}",
        cfg.architecture,
        cfg.n_layer,
        cfg.hidden_size,
        cfg.n_head,
        cfg.n_head_kv,
        cfg.intermediate_size,
        cfg.vocab_size
    );

    let pinned = load_to_pinned(&reader, &p)?;
    let load_dur = t0.elapsed();
    println!(
        "Host RAM load in {:.3}s ({:.2} MB)",
        load_dur.as_secs_f64(),
        pinned.host().as_slice().len() as f64 / (1024.0 * 1024.0)
    );
    println!("Tensors in model:");
    for t in reader.tensor_infos() {
        if !t.name.starts_with("blk.") || t.name.starts_with("blk.0.") || t.name.starts_with("blk.6.") {
            println!("  - {}: ty={:?}, shape={:?}, size={}", t.name, t.ggml_type, t.dims, t.size_bytes);
        }
    }
    let tokenizer = BpeTokenizer::from_reader(&reader)?;
    let t_driver = Instant::now();
    let mut driver = ForwardDriver::new(&reader, &pinned, &cfg, 512)?;
    let vram_mb = driver.vram_footprint().total() as f64 / (1024.0 * 1024.0);
    println!(
        "GPU Driver initialized in {:.3}s (Total VRAM Footprint: {:.1} MB)",
        t_driver.elapsed().as_secs_f64(),
        vram_mb
    );

    let test_prompts = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain what is quantum computing in one short sentence.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
    ];

    let mut sampler = Sampler::new(42);
    let params = SamplerParams {
        temperature: 0.7,
        top_p: 0.9,
        ..Default::default()
    };
    let stop_tokens = [151645u32, 151643u32];

    for (i, &raw_prompt) in test_prompts.iter().enumerate() {
        driver.reset_pos();
        println!("\n--- Prompt #{} ---", i + 1);
        let prompt_tokens = tokenizer.encode(raw_prompt)?;
        println!("Prompt length: {} tokens", prompt_tokens.len());

        let t_prefill = Instant::now();
        let initial_logits = driver.prefill(&prompt_tokens)?;
        let ttft = t_prefill.elapsed();
        let ttft_ms = ttft.as_secs_f64() * 1000.0;
        let prefill_tok_s = prompt_tokens.len() as f64 / ttft.as_secs_f64();
        println!("TTFT (Prefill): {:.2} ms ({:.1} tok/s)", ttft_ms, prefill_tok_s);
        println!("Prompt tokens: {:?}", prompt_tokens);
        let mut indexed: Vec<(usize, f32)> = initial_logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        println!("Top 5 logits after prefill:");
        for &(id, logit) in indexed.iter().take(5) {
            let piece = tokenizer.decode(&[id as u32]).unwrap_or_default();
            println!("  token {} ({:?}): logit = {:.4}", id, piece, logit);
        }

        let mut context = prompt_tokens.clone();
        let first_tok = sampler.sample(&initial_logits, &context, &params);
        let first_piece = tokenizer.decode(&[first_tok]).unwrap_or_default();
        print!("Generated: {}", first_piece);
        context.push(first_tok);

        let mut current_tok = first_tok;
        let mut decode_tokens = 1;
        let t_gen = Instant::now();

        for step in 0..40 {
            if stop_tokens.contains(&current_tok) {
                break;
            }
            let next_tok = if i == 1 {
                // Use 100% autonomous GPU decoding for Prompt #2
                driver.decode_step_autonomous(current_tok)?
            } else {
                let logits = driver.decode_graph_slice(current_tok)?;
                if step == 0 {
                    println!("\n[Decode Step 0] logits has NaN: {}", logits.iter().any(|f| f.is_nan()));
                    let mut d_idx: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
                    d_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    for &(id, logit) in d_idx.iter().take(5) {
                        let piece = tokenizer.decode(&[id as u32]).unwrap_or_default();
                        println!("  token {} ({:?}): logit = {:.4}", id, piece, logit);
                    }
                }
                sampler.sample(logits, &context, &params)
            };
            let next_piece = tokenizer.decode(&[next_tok]).unwrap_or_default();
            print!("{}", next_piece);
            context.push(next_tok);
            current_tok = next_tok;
            decode_tokens += 1;
            if stop_tokens.contains(&next_tok) {
                break;
            }
        }
        let gen_dur = t_gen.elapsed();
        let gen_tok_s = decode_tokens as f64 / gen_dur.as_secs_f64();
        let mode_label = if i == 1 { "Autonomous GPU Graph" } else { "Host-Synchronized Slice" };
        println!(
            "\n[Metrics - {}] Generated {} tokens in {:.3}s -> Decode Throughput: {:.1} tok/s",
            mode_label,
            decode_tokens,
            gen_dur.as_secs_f64(),
            gen_tok_s
        );
    }

    // --- Prompt #3 (Speculative Decoding Benchmark with N-Gram Proposer) ---
    {
        driver.reset_pos();
        println!("\n--- Prompt #3 (Speculative Decoding Engine) ---");
        let spec_prompt = "<|im_start|>system\nYou are a coding assistant.<|im_end|>\n<|im_start|>user\nWrite a Rust function to calculate fibonacci numbers.<|im_end|>\n<|im_start|>assistant\n";
        let prompt_tokens = tokenizer.encode(spec_prompt)?;
        println!("Prompt length: {} tokens", prompt_tokens.len());

        let t_prefill = Instant::now();
        let initial_logits = driver.prefill(&prompt_tokens)?;
        let ttft = t_prefill.elapsed();
        println!("TTFT: {:.2} ms", ttft.as_secs_f64() * 1000.0);

        let proposer = NgramDraftProposer::new(4, 4, 2);
        let mut context = prompt_tokens.clone();
        let first_tok = sampler.sample(&initial_logits, &context, &params);
        context.push(first_tok);
        let first_piece = tokenizer.decode(&[first_tok]).unwrap_or_default();
        print!("Generated: {}", first_piece);

        let mut current_tok = first_tok;
        let mut total_generated = 1;
        let mut speculative_steps = 0;
        let mut accepted_drafts = 0;
        let t_spec = Instant::now();

        while total_generated < 45 {
            if stop_tokens.contains(&current_tok) {
                break;
            }
            let candidates = proposer.propose(&context);
            if candidates.is_empty() {
                let next_tok = driver.decode_step_autonomous(current_tok)?;
                let piece = tokenizer.decode(&[next_tok]).unwrap_or_default();
                print!("{}", piece);
                context.push(next_tok);
                current_tok = next_tok;
                total_generated += 1;
            } else {
                speculative_steps += 1;
                let res = driver.verify_speculative(current_tok, &candidates, &mut sampler, &params, &context)?;
                accepted_drafts += res.n_accepted;
                for &tok in &res.emitted_tokens {
                    let piece = tokenizer.decode(&[tok]).unwrap_or_default();
                    print!("{}", piece);
                    context.push(tok);
                }
                total_generated += res.total_emitted;
                current_tok = res.bonus_token;
            }
        }
        let spec_dur = t_spec.elapsed();
        let spec_tok_s = total_generated as f64 / spec_dur.as_secs_f64();
        println!(
            "\n[Speculative Metrics] Generated {} tokens in {:.3}s (Steps: {}, Drafts Accepted: {}) -> Throughput: {:.1} tok/s",
            total_generated,
            spec_dur.as_secs_f64(),
            speculative_steps,
            accepted_drafts,
            spec_tok_s
        );
    }

    println!("\n=======================================================");
    println!("=== Real Model Inference Benchmark COMPLETED SUCCESSFULLY ===");
    println!("=======================================================\n");

    Ok(())
}
