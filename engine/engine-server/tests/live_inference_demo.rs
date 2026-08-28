use engine_core::{BpeTokenizer, ForwardDriver, Sampler, SamplerParams};
use engine_io::{GgufReader, load_to_pinned, ModelConfig};
use std::path::PathBuf;
use std::time::Instant;

#[test]
#[ignore]
fn test_live_text_inference_generation() -> Result<(), Box<dyn std::error::Error>> {
    engine_cuda::ensure_cuda_dll_paths();
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let models_dir = workspace_root.join("models");

    let models = [
        ("Qwen 2.5 1.5B Instruct", models_dir.join("qwen2.5-1.5b-instruct-q4_k_m.gguf")),
        ("Llama 3.2 1B Instruct", models_dir.join("Llama-3.2-1B-Instruct-Q4_K_M.gguf")),
    ];

    let prompts = [
        "The capital of France is",
        "The largest planet in our solar system is",
        "The chemical formula for water is",
        "def add(a, b):",
    ];

    let greedy_params = SamplerParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
        repetition_penalty: 1.1,
        seed: None,
    };

    for (model_name, model_path) in &models {
        if !model_path.exists() {
            println!("Skipping {}: model not found at {:?}", model_name, model_path);
            continue;
        }

        println!("\n================================================================================");
        println!(">>> TITAN LIVE REAL-HARDWARE INFERENCE EVALUATION: {}", model_name);
        println!("================================================================================");

        let t_load = Instant::now();
        let reader = GgufReader::open(model_path)?;
        let cfg = ModelConfig::from_reader(&reader)?;
        let pinned = load_to_pinned(&reader, model_path)?;
        let tokenizer = BpeTokenizer::from_reader(&reader)?;
        let mut driver = ForwardDriver::new(&reader, &pinned, &cfg, 512)?;
        println!("Loaded model into GPU VRAM in {:.2} ms\n", t_load.elapsed().as_secs_f64() * 1000.0);

        let mut sampler = Sampler::new(42);

        for (idx, &prompt) in prompts.iter().enumerate() {
            driver.reset_pos();
            let prompt_tokens = tokenizer.encode(prompt)?;

            let t_prefill = Instant::now();
            let initial_logits = driver.prefill(&prompt_tokens)?;
            let prefill_dur = t_prefill.elapsed();

            let mut current_token = sampler.sample(&initial_logits, &prompt_tokens, &greedy_params);
            let mut generated_tokens = vec![current_token];

            let t_decode = Instant::now();
            for _ in 1..15 {
                let logits = driver.decode(current_token)?;
                current_token = sampler.sample(&logits, &generated_tokens, &greedy_params);
                generated_tokens.push(current_token);
            }
            let decode_dur = t_decode.elapsed();

            let gen_text = tokenizer.decode(&generated_tokens).unwrap_or_else(|_| "<decoding error>".into());
            let decode_speed = (generated_tokens.len() - 1) as f64 / decode_dur.as_secs_f64();

            println!("--------------------------------------------------------------------------------");
            println!("[Prompt #{}] \"{}\"", idx + 1, prompt);
            println!("  [Generated Output]: \"{}\"", gen_text.trim());
            println!("  [Prefill]:  {:.2} ms ({} tokens -> {:.1} tok/s)",
                prefill_dur.as_secs_f64() * 1000.0, prompt_tokens.len(),
                prompt_tokens.len() as f64 / prefill_dur.as_secs_f64());
            println!("  [Decode]:   {:.1} tok/s ({:.2} ms/tok)",
                decode_speed, (decode_dur.as_secs_f64() * 1000.0) / (generated_tokens.len() - 1) as f64);
            println!("--------------------------------------------------------------------------------\n");
        }
    }

    Ok(())
}