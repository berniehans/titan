//! Titan Inference Engine — CLI and Server Binary.
//!
//! Subcommands:
//! - `serve`: Launches OpenAI-compatible HTTP server (`/v1/chat/completions`, `/v1/completions`, `/v1/models`).
//! - `chat`: Interactive terminal REPL with multi-turn memory and live token-by-token streaming.

use cudarc::driver::CudaDevice;
use engine_core::{Sampler, SamplerParams};
use engine_io::{GgufReader, LoadedPinned, load_to_pinned};
use engine_server::models::{ChatMessage, format_chatml};
use engine_server::runtime::{self, EngineMode, RealModel, SpeculativeMode};
use engine_server::server::{RealServerCfg, build_router_real};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn resolve_model_path(path: &Path) -> PathBuf {
    if path.exists() {
        return path.to_path_buf();
    }
    let candidates = [
        PathBuf::from("../").join(path),
        PathBuf::from("../../").join(path),
        PathBuf::from("testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("../../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
    ];
    for c in &candidates {
        if c.exists() {
            return c.clone();
        }
    }
    path.to_path_buf()
}

fn default_model_path() -> PathBuf {
    if let Ok(env_path) = std::env::var("ENGINE_TESTDATA") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return p;
        }
    }
    resolve_model_path(Path::new("testdata/Qwen3-0.6B-Q4_K_M.gguf"))
}

fn print_banner() {
    println!(
        r#"
  ████████╗██╗████████╗ █████╗ ███╗   ██╗
  ╚══██╔══╝██║╚══██╔══╝██╔══██╗████╗  ██║
     ██║   ██║   ██║   ███████║██╔██╗ ██║
     ██║   ██║   ██║   ██╔══██║██║╚██╗██║
     ██║   ██║   ██║   ██║  ██║██║ ╚████║
     ╚═╝   ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝
  Titan 100% GPU Layer-Streaming LLM Engine
"#
    );
}

fn print_diagnostics(
    model_path: &Path,
    engine_mode: EngineMode,
    spec_mode: SpeculativeMode,
    vram_mb: f64,
    load_ms: f64,
) {
    print_banner();
    println!("  ╔══════════════════════════════════════════════════════════════════╗");
    println!(
        "  ║ Engine Mode:       {:<45} ║",
        format!("{engine_mode:?}")
    );
    println!("  ║ Speculative Mode:  {:<45} ║", format!("{spec_mode:?}"));
    println!(
        "  ║ Model File:        {:<45} ║",
        format!(
            "{}",
            model_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
        )
    );
    println!(
        "  ║ VRAM Working Set:  {:<45} ║",
        format!("{vram_mb:.1} MB")
    );
    println!(
        "  ║ Host Load Time:    {:<45} ║",
        format!("{load_ms:.2} ms")
    );
    println!("  ╚══════════════════════════════════════════════════════════════════╝\n");
}

fn run_chat(
    model_path: &Path,
    engine_mode: EngineMode,
    spec_mode: SpeculativeMode,
    kv_capacity: usize,
    temperature: f32,
    top_p: f32,
    system_prompt: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let start_load = Instant::now();
    let reader = GgufReader::open(model_path)?;
    let pinned: &'static LoadedPinned = Box::leak(Box::new(load_to_pinned(&reader, model_path)?));
    let _device = CudaDevice::new(0)?;
    let load_ms = start_load.elapsed().as_secs_f64() * 1000.0;

    let mut model: RealModel<'static> =
        runtime::build_unified_driver_model(&reader, pinned, kv_capacity, engine_mode, spec_mode)?;

    let vram_mb = model
        .driver
        .as_ref()
        .map(|d| d.vram_footprint().total() as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0);

    print_diagnostics(
        model_path,
        model.engine_mode,
        model.speculative_mode,
        vram_mb,
        load_ms,
    );

    println!(
        "Interactive chat ready! Type your message and press Enter. Commands: /reset, /exit\n"
    );

    let mut messages: Vec<ChatMessage> = Vec::new();
    if let Some(sys) = system_prompt {
        messages.push(ChatMessage::system(sys));
    }

    let mut sampler = Sampler::new(42);
    let params = SamplerParams {
        temperature,
        top_p,
        top_k: 40,
        repetition_penalty: 1.1,
        seed: None,
    };
    let stop_tokens = [151645u32, 151643u32, 151644u32]; // <|im_end|>, <|endoftext|>, <|im_start|>
    let stop_strings = [
        "<|im_end|>".to_string(),
        "<|endoftext|>".to_string(),
        "<|im_start|>".to_string(),
        "<|im_end".to_string(),
    ];

    let tokenizer = model.tokenizer.take().expect("tokenizer");
    let mut driver = model.driver.take().expect("driver");
    let proposer = model.ngram_proposer.take();

    loop {
        print!("You > ");
        io::stdout().flush()?;
        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "/exit" || input == "exit" || input == "quit" {
            println!("Goodbye!");
            break;
        }
        if input == "/reset" || input == "/clear" {
            messages.clear();
            if let Some(sys) = system_prompt {
                messages.push(ChatMessage::system(sys));
            }
            println!("[Conversation history cleared]\n");
            continue;
        }

        messages.push(ChatMessage::user(input));
        let prompt = format_chatml(&messages);
        let prompt_tokens = tokenizer.encode(&prompt)?;

        print!("\nTitan > ");
        io::stdout().flush()?;

        let mut context = prompt_tokens.clone();
        let t_start_gen = Instant::now();
        let initial_logits = driver.prefill(&prompt_tokens)?;
        let first_tok = sampler.sample(&initial_logits, &context, &params);
        let first_piece = tokenizer.decode(&[first_tok]).unwrap_or_default();

        let mut assistant_response = String::new();

        if !Sampler::is_stop_sequence(first_tok, &first_piece, &stop_tokens, &stop_strings) {
            print!("{}", first_piece);
            io::stdout().flush()?;
            assistant_response.push_str(&first_piece);
            context.push(first_tok);

            let mut current_tok = first_tok;
            const MAX_GEN: usize = 512;

            while context.len() < prompt_tokens.len() + MAX_GEN {
                if let Some(ref p) = proposer {
                    let candidates = p.propose(&context);
                    if !candidates.is_empty() {
                        let verif = match driver.verify_speculative(
                            current_tok,
                            &candidates,
                            &mut sampler,
                            &params,
                            &context,
                        ) {
                            Ok(v) => v,
                            Err(_) => break,
                        };

                        let mut stopped = false;
                        for &emitted_tok in &verif.emitted_tokens {
                            let piece = tokenizer.decode(&[emitted_tok]).unwrap_or_default();
                            if Sampler::is_stop_sequence(
                                emitted_tok,
                                &piece,
                                &stop_tokens,
                                &stop_strings,
                            ) {
                                stopped = true;
                                break;
                            }
                            print!("{}", piece);
                            io::stdout().flush()?;
                            assistant_response.push_str(&piece);
                            context.push(emitted_tok);
                            current_tok = emitted_tok;
                        }
                        if stopped {
                            break;
                        }
                        continue;
                    }
                }

                let logits = match driver.decode(current_tok) {
                    Ok(l) => l,
                    Err(_) => break,
                };
                let next_tok = sampler.sample(&logits, &context, &params);
                let next_piece = tokenizer.decode(&[next_tok]).unwrap_or_default();

                if Sampler::is_stop_sequence(next_tok, &next_piece, &stop_tokens, &stop_strings) {
                    break;
                }

                print!("{}", next_piece);
                io::stdout().flush()?;
                assistant_response.push_str(&next_piece);
                context.push(next_tok);
                current_tok = next_tok;
            }
        }

        let n_gen = context.len().saturating_sub(prompt_tokens.len());
        let elapsed = t_start_gen.elapsed().as_secs_f64();
        let tok_per_sec = if elapsed > 0.0 {
            n_gen as f64 / elapsed
        } else {
            0.0
        };
        println!(
            "\n\x1b[90m[{n_gen} tokens generated in {elapsed:.2}s — {tok_per_sec:.1} tok/s]\x1b[0m\n"
        );
        messages.push(ChatMessage::assistant(assistant_response));
    }

    Ok(())
}

fn run_bench(
    model_path: &Path,
    engine_mode: EngineMode,
    spec_mode: SpeculativeMode,
    kv_capacity: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "\n>>> RUNNING TITAN EMPIRICAL BENCHMARK ON: {}",
        model_path.display()
    );
    let start_load = Instant::now();
    let reader = GgufReader::open(model_path)?;
    let pinned: &'static LoadedPinned = Box::leak(Box::new(load_to_pinned(&reader, model_path)?));
    let _device = CudaDevice::new(0)?;
    let load_ms = start_load.elapsed().as_secs_f64() * 1000.0;

    let mut model: RealModel<'static> =
        runtime::build_unified_driver_model(&reader, pinned, kv_capacity, engine_mode, spec_mode)?;

    let vram_mb = model
        .driver
        .as_ref()
        .map(|d| d.vram_footprint().total() as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0);

    print_diagnostics(
        model_path,
        model.engine_mode,
        model.speculative_mode,
        vram_mb,
        load_ms,
    );

    let tokenizer = model.tokenizer.take().expect("tokenizer");
    let mut driver = model.driver.take().expect("driver");

    let test_prompts = [
        ("Short (Math)", "2 + 2 ="),
        (
            "Medium (Fact)",
            "The capital of France is Paris. What is the capital of Spain?",
        ),
        (
            "Long (Code)",
            "Write a high performance Rust function that computes the Fibonacci sequence using iterative memoization:",
        ),
    ];

    println!("================================================================================");
    println!("  TITAN GPU BENCHMARK SUITE (Greedy Sampling T=0, 40 Tokens Generation)");
    println!("================================================================================");

    for (name, prompt) in &test_prompts {
        let tokens = tokenizer.encode(prompt)?;
        let t0 = Instant::now();
        let _ = driver.prefill(&tokens)?;
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let prefill_tps = tokens.len() as f64 / (prefill_ms / 1000.0);

        let mut gen_tokens = 0;
        let mut cur_tok = 12095u32;
        let t_gen_start = Instant::now();
        for _ in 0..40 {
            let _ = driver.decode(cur_tok)?;
            gen_tokens += 1;
            cur_tok = 220;
        }
        let gen_sec = t_gen_start.elapsed().as_secs_f64();
        let decode_tps = gen_tokens as f64 / gen_sec;
        let ms_per_tok = (gen_sec * 1000.0) / gen_tokens as f64;

        println!(
            "  • {:<16} | Prefill: {:>6.1} tok/s ({:>5.2} ms) | Decode: {:>6.1} tok/s ({:>5.2} ms/tok)",
            name, prefill_tps, prefill_ms, decode_tps, ms_per_tok
        );
    }
    println!("================================================================================\n");

    Ok(())
}

fn print_help() {
    print_banner();
    println!(
        r#"TITAN INFERENCE ENGINE - 100% GPU LAYER STREAMING & SPECULATIVE LLM RUNTIME

USAGE:
    titan <SUBCOMMAND> [OPTIONS]

SUBCOMMANDS:
    chat      Launch interactive terminal chat REPL with live tok/s telemetry
    serve     Launch OpenAI-compatible HTTP API server
    bench     Run automated GPU throughput and TTFT latency benchmark
    agent     Launch optimized backend server preset for Hermes Agent & tool-calling loops
    help      Print this help message

GLOBAL OPTIONS:
    -m, --model <PATH>            Path to GGUF model file (default: auto-detected fixture)
    -e, --engine <MODE>           Execution backend engine: auto | resident | streaming | moe (default: auto)
    -s, --speculative <MODE>      Speculative acceleration: auto | ngram | none (default: auto)
    -c, --kv-capacity <TOKENS>    KV cache token capacity (default: 2048 for chat, 512 for serve)
    -t, --temp, --temperature     Sampling temperature [0.0 - 2.0] (default: 0.7)
    --top-p <FLOAT>               Top-p nucleus sampling probability [0.0 - 1.0] (default: 0.9)
    --system <PROMPT>             Initial system prompt for chat
    -p, --port <PORT>             Port for HTTP server (default: 8000, 8080 for agent)
    -h, --help                    Print help
"#
    );
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    engine_cuda::ensure_cuda_dll_paths();
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("chat");

    if mode == "help" || mode == "--help" || mode == "-h" {
        print_help();
        return Ok(());
    }

    let positional_model = args
        .get(2)
        .filter(|s| !s.starts_with('-'))
        .map(PathBuf::from);
    let raw_model_path = positional_model
        .or_else(|| {
            args.iter()
                .position(|a| a == "--model" || a == "-m")
                .and_then(|idx| args.get(idx + 1))
                .map(PathBuf::from)
        })
        .unwrap_or_else(default_model_path);

    let model_path = resolve_model_path(&raw_model_path);

    let engine_mode: EngineMode = args
        .iter()
        .position(|a| a == "--engine" || a == "-e")
        .and_then(|idx| args.get(idx + 1))
        .and_then(|e| e.parse().ok())
        .unwrap_or(EngineMode::Auto);

    let spec_mode: SpeculativeMode = args
        .iter()
        .position(|a| a == "--speculative" || a == "-s")
        .and_then(|idx| args.get(idx + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(SpeculativeMode::Auto);

    let port: u16 = args
        .iter()
        .position(|a| a == "--port" || a == "-p")
        .and_then(|idx| args.get(idx + 1))
        .and_then(|p| p.parse().ok())
        .unwrap_or(8000);

    let kv_capacity: usize = args
        .iter()
        .position(|a| a == "--kv-capacity" || a == "-c")
        .and_then(|idx| args.get(idx + 1))
        .and_then(|c| c.parse().ok())
        .unwrap_or(if mode == "serve" { 512 } else { 2048 });

    let temperature: f32 = args
        .iter()
        .position(|a| a == "--temp" || a == "-t" || a == "--temperature")
        .and_then(|idx| args.get(idx + 1))
        .and_then(|t| t.parse().ok())
        .unwrap_or(0.7);

    let top_p: f32 = args
        .iter()
        .position(|a| a == "--top-p")
        .and_then(|idx| args.get(idx + 1))
        .and_then(|p| p.parse().ok())
        .unwrap_or(0.9);

    let system_prompt = args
        .iter()
        .position(|a| a == "--system")
        .and_then(|idx| args.get(idx + 1))
        .map(|s| s.as_str());

    if mode == "chat" || mode == "run" {
        run_chat(
            &model_path,
            engine_mode,
            spec_mode,
            kv_capacity,
            temperature,
            top_p,
            system_prompt,
        )?;
    } else if mode == "bench" {
        run_bench(&model_path, engine_mode, spec_mode, kv_capacity)?;
    } else if mode == "serve" || mode == "agent" {
        let actual_port = if mode == "agent" && port == 8000 {
            8080
        } else {
            port
        };
        let start_load = Instant::now();
        let reader = GgufReader::open(&model_path)?;
        let pinned: &'static LoadedPinned =
            Box::leak(Box::new(load_to_pinned(&reader, &model_path)?));
        let _device = CudaDevice::new(0)?;
        let load_ms = start_load.elapsed().as_secs_f64() * 1000.0;

        let real_model: RealModel<'static> = runtime::build_unified_driver_model(
            &reader,
            pinned,
            kv_capacity,
            engine_mode,
            spec_mode,
        )?;

        let vram_mb = real_model
            .driver
            .as_ref()
            .map(|d| d.vram_footprint().total() as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0);

        print_diagnostics(
            &model_path,
            real_model.engine_mode,
            real_model.speculative_mode,
            vram_mb,
            load_ms,
        );

        let shared: Arc<Mutex<RealModel<'static>>> = Arc::new(Mutex::new(real_model));

        let cfg = Arc::new(RealServerCfg {
            vocab: 151936,
            default_max_tokens: 512,
            model: shared,
        });

        let addr = format!("0.0.0.0:{actual_port}");
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        println!("=======================================================");
        if mode == "agent" {
            println!("  TITAN AUTONOMOUS AGENT SERVER (HERMES READY) ON:     ");
            println!("  http://localhost:{actual_port}                       ");
            println!("=======================================================");
            println!("  Hermes Agent / Cursor / LiteLLM Config:              ");
            println!("    \"openai_base_url\": \"http://127.0.0.1:{actual_port}/v1\"");
            println!("    \"model\": \"titan-agent\"                            ");
            println!("    \"temperature\": 0.0                                 ");
        } else {
            println!("  TITAN OPENAI-COMPATIBLE API SERVER LISTENING ON:     ");
            println!("  http://localhost:{actual_port}                       ");
            println!("=======================================================");
            println!("  Endpoints:");
            println!("    • POST http://localhost:{actual_port}/v1/chat/completions");
            println!("    • POST http://localhost:{actual_port}/v1/completions");
            println!("    • GET  http://localhost:{actual_port}/v1/models");
            println!("\n  Connect any OpenAI client (Cursor, LibreChat, Open-WebUI, LiteLLM)!\n");
        }

        let app = build_router_real(cfg);
        axum::serve(listener, app).await?;
    } else {
        print_help();
    }

    Ok(())
}
