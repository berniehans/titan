use engine_core::{BpeTokenizer, ForwardDriver};
use engine_io::{load_to_pinned, GgufReader, ModelConfig};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

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

#[derive(Serialize)]
struct LlamaReq<'a> {
    prompt: &'a str,
    n_predict: usize,
    temperature: f32,
}

#[derive(Deserialize, Debug)]
struct LlamaTimings {
    prompt_n: usize,
    prompt_ms: f64,
    prompt_per_second: f64,
    predicted_n: usize,
    predicted_ms: f64,
    predicted_per_second: f64,
}

#[derive(Deserialize, Debug)]
struct LlamaResp {
    content: String,
    timings: LlamaTimings,
}

struct LlamaServerProcess {
    child: Child,
}

impl Drop for LlamaServerProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

async fn start_llama_server(model_p: &std::path::Path, port: u16) -> Result<Option<LlamaServerProcess>, Box<dyn std::error::Error>> {
    let ollama_dir = PathBuf::from(r"C:\Users\niber\AppData\Local\Programs\Ollama\lib\ollama");
    let cuda_dir = ollama_dir.join("cuda_v12");
    let server_exe = ollama_dir.join("llama-server.exe");

    if !server_exe.exists() || !cuda_dir.exists() {
        return Ok(None);
    }

    let path_env = format!(
        "{};{};{}",
        cuda_dir.display(),
        ollama_dir.display(),
        std::env::var("PATH").unwrap_or_default()
    );

    let child = Command::new(&server_exe)
        .current_dir(&cuda_dir)
        .env("PATH", &path_env)
        .arg("-m")
        .arg(model_p)
        .arg("-ngl")
        .arg("99")
        .arg("--port")
        .arg(port.to_string())
        .arg("-c")
        .arg("2048")
        .arg("--no-warmup")
        .spawn()?;

    // Poll until healthy
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(500))
        .build()?;
    let health_url = format!("http://127.0.0.1:{}/health", port);
    let start = Instant::now();
    let mut ready = false;

    while start.elapsed() < Duration::from_secs(15) {
        if let Ok(resp) = client.get(&health_url).send().await {
            if resp.status().is_success() {
                ready = true;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }

    if !ready {
        return Ok(None);
    }

    Ok(Some(LlamaServerProcess { child }))
}

#[tokio::test]
#[ignore]
async fn test_titan_vs_llama_cpp_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    engine_cuda::ensure_cuda_dll_paths();

    let p = match model_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: models/qwen2.5-1.5b-instruct-q4_k_m.gguf not found");
            return Ok(());
        }
    };

    println!("\n================================================================================");
    println!("===           TITAN GPU RESIDENT ENGINE vs LLAMA.CPP BENCHMARK               ===");
    println!("================================================================================");
    println!("Model: {}", p.display());
    println!("Device: NVIDIA GeForce RTX 3060 Laptop GPU (Pure CUDA)");

    let test_prompts = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nGive me a numbered list of 5 historical dates.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain quantum computing in three concise bullet points.<|im_end|>\n<|im_start|>assistant\n",
    ];

    let n_tokens_to_generate = 41;
    let port = 8099;

    // -------------------------------------------------------------
    // 1. LLAMA.CPP BENCHMARK RUN
    // -------------------------------------------------------------
    println!("\n[1/2] Starting official llama.cpp (CUDA 12 + Graphs enabled)...");
    let llama_proc = start_llama_server(&p, port).await?;

    let mut llama_prefill_speeds = Vec::new();
    let mut llama_decode_speeds = Vec::new();
    let mut llama_decode_lats = Vec::new();

    if llama_proc.is_some() {
        let client = reqwest::Client::new();
        let comp_url = format!("http://127.0.0.1:{}/completion", port);

        for (i, &prompt) in test_prompts.iter().enumerate() {
            let req = LlamaReq {
                prompt,
                n_predict: n_tokens_to_generate,
                temperature: 0.0,
            };
            let body_str = serde_json::to_string(&req)?;
            let resp_text = client
                .post(&comp_url)
                .header("content-type", "application/json")
                .body(body_str)
                .send()
                .await?
                .text()
                .await?;
            let resp: LlamaResp = serde_json::from_str(&resp_text)?;

            println!("  - Prompt #{}: Prefill = {:.1} tok/s ({:.2} ms), Decode = {:.1} tok/s ({:.2} ms/tok)",
                i + 1,
                resp.timings.prompt_per_second,
                resp.timings.prompt_ms,
                resp.timings.predicted_per_second,
                resp.timings.predicted_ms / resp.timings.predicted_n as f64
            );
            llama_prefill_speeds.push(resp.timings.prompt_per_second);
            llama_decode_speeds.push(resp.timings.predicted_per_second);
            llama_decode_lats.push(resp.timings.predicted_ms / resp.timings.predicted_n as f64);
        }
    } else {
        println!("  [!] llama.cpp server could not be started or found, skipping live llama.cpp run.");
    }
    drop(llama_proc);

    // -------------------------------------------------------------
    // 2. TITAN GPU RESIDENT RUN
    // -------------------------------------------------------------
    println!("\n[2/2] Running Titan GPU Resident Engine (100% Pure Rust / CUDA JIT)...");
    let reader = GgufReader::open(&p)?;
    let cfg = ModelConfig::from_reader(&reader)?;
    let pinned = load_to_pinned(&reader, &p)?;
    let tokenizer = BpeTokenizer::from_reader(&reader)?;
    let mut driver = ForwardDriver::new(&reader, &pinned, &cfg, 512)?;
    let _ = driver.capture_autonomous_decode_graph();

    let mut titan_prefill_speeds = Vec::new();
    let mut titan_decode_speeds = Vec::new();
    let mut titan_decode_lats = Vec::new();

    for (i, &raw_prompt) in test_prompts.iter().enumerate() {
        driver.reset_pos();
        let prompt_tokens = tokenizer.encode(raw_prompt)?;

        // Prefill
        let t_prefill = Instant::now();
        let initial_logits = driver.prefill(&prompt_tokens)?;
        let ttft = t_prefill.elapsed();
        let prefill_speed = prompt_tokens.len() as f64 / ttft.as_secs_f64();
        titan_prefill_speeds.push(prefill_speed);

        // Decode
        let mut cur_token = {
            let mut best_i = 0;
            let mut best_val = f32::NEG_INFINITY;
            for (idx, &v) in initial_logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_i = idx;
                }
            }
            best_i as u32
        };

        // Profile single-step decode
        let _ = driver.decode(cur_token)?;

        // 2a. Streamed GPU Decode
        let t_decode = Instant::now();
        let gen_tokens = driver.generate_autonomous_gpu(cur_token, n_tokens_to_generate)?;
        let decode_dur = t_decode.elapsed();
        let decode_speed = gen_tokens.len() as f64 / decode_dur.as_secs_f64();
        let decode_lat_ms = (decode_dur.as_secs_f64() * 1000.0) / gen_tokens.len() as f64;

        titan_decode_speeds.push(decode_speed);
        titan_decode_lats.push(decode_lat_ms);

        println!("  - Prompt #{}: Prefill = {:.1} tok/s ({:.2} ms), GPU Stream Decode = {:.1} tok/s ({:.2} ms/tok)",
            i + 1,
            prefill_speed,
            ttft.as_secs_f64() * 1000.0,
            decode_speed,
            decode_lat_ms
        );
    }

    // -------------------------------------------------------------
    // 3. COMPARISON TABLE
    // -------------------------------------------------------------
    let avg_llama_prefill: f64 = if !llama_prefill_speeds.is_empty() { llama_prefill_speeds.iter().sum::<f64>() / llama_prefill_speeds.len() as f64 } else { 0.0 };
    let avg_llama_decode: f64 = if !llama_decode_speeds.is_empty() { llama_decode_speeds.iter().sum::<f64>() / llama_decode_speeds.len() as f64 } else { 0.0 };
    let avg_llama_lat: f64 = if !llama_decode_lats.is_empty() { llama_decode_lats.iter().sum::<f64>() / llama_decode_lats.len() as f64 } else { 0.0 };

    let avg_titan_prefill: f64 = titan_prefill_speeds.iter().sum::<f64>() / titan_prefill_speeds.len() as f64;
    let avg_titan_decode: f64 = titan_decode_speeds.iter().sum::<f64>() / titan_decode_speeds.len() as f64;
    let avg_titan_lat: f64 = titan_decode_lats.iter().sum::<f64>() / titan_decode_lats.len() as f64;

    println!("\n================================================================================");
    println!("===                         HEAD-TO-HEAD COMPARISON                          ===");
    println!("================================================================================");
    println!("{:<30} | {:<18} | {:<18} | {:<12}", "Metric", "llama.cpp (C++)", "Titan (Pure Rust)", "Ratio");
    println!("{:-<30}-|-{:-<18}-|-{:-<18}-|-{:-<12}", "", "", "", "");
    println!("{:<30} | {:<15.1} tok/s | {:<15.1} tok/s | {:<10.2}x", "Decode Throughput (Higher)", avg_llama_decode, avg_titan_decode, avg_titan_decode / avg_llama_decode.max(0.001));
    println!("{:<30} | {:<15.2} ms    | {:<15.2} ms    | {:<10.2}x", "Decode Latency (Lower)", avg_llama_lat, avg_titan_lat, avg_titan_lat / avg_llama_lat.max(0.001));
    println!("{:<30} | {:<15.1} tok/s | {:<15.1} tok/s | {:<10.2}x", "Prefill Throughput (TTFT)", avg_llama_prefill, avg_titan_prefill, avg_titan_prefill / avg_llama_prefill.max(0.001));
    println!("{:<30} | {:<18} | {:<18} | {:<12}", "Architecture / Toolchain", "C++ / CMake / DLLs", "100% Pure Rust", "Native");
    println!("{:<30} | {:<18} | {:<18} | {:<12}", "Driver Interface", "C ABI Bindings", "Autonomous CUDA JIT", "Zero-Copy");
    println!("================================================================================\n");

    Ok(())
}
