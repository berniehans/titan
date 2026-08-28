use engine_core::{BpeTokenizer, ForwardDriver};
use engine_io::{load_to_pinned, GgufReader, ModelConfig};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

fn resolve_model_path(rel: &str) -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("../../").join(rel),
        manifest_dir.join("../").join(rel),
        manifest_dir.join(rel),
        PathBuf::from(rel),
        PathBuf::from("../").join(rel),
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

async fn start_llama_server(
    model_p: &std::path::Path,
    port: u16,
) -> Result<Option<LlamaServerProcess>, Box<dyn std::error::Error>> {
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

struct BenchResult {
    model_name: String,
    llama_decode_toks: f64,
    llama_decode_lat: f64,
    titan_decode_toks: f64,
    titan_decode_lat: f64,
    ratio: f64,
}

async fn run_model_benchmark(
    name: &str,
    p: PathBuf,
    test_prompts: &[&str],
    port: u16,
) -> Result<BenchResult, Box<dyn std::error::Error>> {
    println!("\n================================================================================");
    println!(">>> BENCHMARKING MODEL: {} ({})", name, p.display());
    println!("================================================================================");

    let n_tokens_to_generate = 41;

    // 1. LLAMA.CPP
    println!("[1/2] Benchmarking official llama.cpp server...");
    let mut llama_decode_speeds = Vec::new();
    let mut llama_decode_lats = Vec::new();

    let llama_proc = start_llama_server(&p, port).await?;
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

            let lat = resp.timings.predicted_ms / resp.timings.predicted_n.max(1) as f64;
            println!(
                "  llama.cpp Prompt #{}: Prefill = {:.1} tok/s, Decode = {:.1} tok/s ({:.2} ms/tok)",
                i + 1,
                resp.timings.prompt_per_second,
                resp.timings.predicted_per_second,
                lat
            );
            llama_decode_speeds.push(resp.timings.predicted_per_second);
            llama_decode_lats.push(lat);
        }
    } else {
        println!("  [!] llama.cpp server failed to start for {}", name);
    }
    drop(llama_proc);

    // Give time for port and memory cleanup
    tokio::time::sleep(Duration::from_millis(500)).await;

    // 2. TITAN GPU RESIDENT
    println!("\n[2/2] Benchmarking Titan GPU Resident Engine (100% Pure Rust / CUDA JIT)...");
    let reader = GgufReader::open(&p)?;
    let cfg = ModelConfig::from_reader(&reader)?;
    let pinned = load_to_pinned(&reader, &p)?;
    let tokenizer = BpeTokenizer::from_reader(&reader)?;
    let mut driver = ForwardDriver::new(&reader, &pinned, &cfg, 512)?;
    let _ = driver.capture_autonomous_decode_graph();

    let mut titan_decode_speeds = Vec::new();
    let mut titan_decode_lats = Vec::new();

    for (i, &raw_prompt) in test_prompts.iter().enumerate() {
        driver.reset_pos();
        let prompt_tokens = tokenizer.encode(raw_prompt)?;

        // Prefill
        let initial_logits = driver.prefill(&prompt_tokens)?;

        let cur_token = {
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

        let _ = driver.decode(cur_token)?;

        let t_decode = Instant::now();
        let gen_tokens = driver.generate_autonomous_gpu(cur_token, n_tokens_to_generate)?;
        let decode_dur = t_decode.elapsed();
        let decode_speed = gen_tokens.len() as f64 / decode_dur.as_secs_f64();
        let decode_lat_ms = (decode_dur.as_secs_f64() * 1000.0) / gen_tokens.len() as f64;

        titan_decode_speeds.push(decode_speed);
        titan_decode_lats.push(decode_lat_ms);

        println!(
            "  Titan Prompt #{}: GPU Stream Decode = {:.1} tok/s ({:.2} ms/tok)",
            i + 1,
            decode_speed,
            decode_lat_ms
        );
    }

    let avg_llama_decode: f64 = if !llama_decode_speeds.is_empty() {
        llama_decode_speeds.iter().sum::<f64>() / llama_decode_speeds.len() as f64
    } else {
        0.0
    };
    let avg_llama_lat: f64 = if !llama_decode_lats.is_empty() {
        llama_decode_lats.iter().sum::<f64>() / llama_decode_lats.len() as f64
    } else {
        0.0
    };

    let avg_titan_decode: f64 =
        titan_decode_speeds.iter().sum::<f64>() / titan_decode_speeds.len() as f64;
    let avg_titan_lat: f64 =
        titan_decode_lats.iter().sum::<f64>() / titan_decode_lats.len() as f64;

    Ok(BenchResult {
        model_name: name.to_string(),
        llama_decode_toks: avg_llama_decode,
        llama_decode_lat: avg_llama_lat,
        titan_decode_toks: avg_titan_decode,
        titan_decode_lat: avg_titan_lat,
        ratio: avg_titan_decode / avg_llama_decode.max(0.001),
    })
}

#[tokio::test]
#[ignore]
async fn test_multi_model_head_to_head_bench() -> Result<(), Box<dyn std::error::Error>> {
    engine_cuda::ensure_cuda_dll_paths();

    let models = [
        (
            "Qwen 2.5 1.5B Instruct",
            "models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            [
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nGive me a numbered list of 5 historical dates.<|im_end|>\n<|im_start|>assistant\n",
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain quantum computing in three concise bullet points.<|im_end|>\n<|im_start|>assistant\n",
            ],
        ),
        (
            "Llama 3.2 1B Instruct",
            "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            [
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\nGive me a numbered list of 5 historical dates.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\nExplain quantum computing in three concise bullet points.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
            ],
        ),
        (
            "Llama 3.2 3B Instruct",
            "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            [
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\nGive me a numbered list of 5 historical dates.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\nExplain quantum computing in three concise bullet points.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
            ],
        ),
        (
            "DeepSeek-R1-Distill 1.5B",
            "models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
            [
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nGive me a numbered list of 5 historical dates.<|im_end|>\n<|im_start|>assistant\n",
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nExplain quantum computing in three concise bullet points.<|im_end|>\n<|im_start|>assistant\n",
            ],
        ),
    ];

    let mut results = Vec::new();
    let port = 8098;

    for (name, rel_path, prompts) in &models {
        if let Some(path) = resolve_model_path(rel_path) {
            let res = run_model_benchmark(name, path, prompts, port).await?;
            results.push(res);
        } else {
            println!("SKIP: {} (path not found: {})", name, rel_path);
        }
    }

    println!("\n=========================================================================================================");
    println!("===                                MULTI-MODEL HEAD-TO-HEAD COMPARISON                                ===");
    println!("=========================================================================================================");
    println!(
        "{:<28} | {:<20} | {:<20} | {:<12}",
        "Model Name", "llama.cpp (C++)", "Titan (Pure Rust)", "Ratio (Titan / llama.cpp)"
    );
    println!("{:-<28}-|-{:-<20}-|-{:-<20}-|-{:-<12}", "", "", "", "");

    for r in &results {
        println!(
            "{:<28} | {:<6.1} tok/s ({:<5.2} ms) | {:<6.1} tok/s ({:<5.2} ms) | {:<6.2}x",
            r.model_name,
            r.llama_decode_toks,
            r.llama_decode_lat,
            r.titan_decode_toks,
            r.titan_decode_lat,
            r.ratio
        );
    }
    println!("=========================================================================================================\n");

    Ok(())
}
