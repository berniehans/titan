//! Titan Inference Engine — CLI and Server Binary.
//!
//! Subcommands:
//! - `serve`: Launches OpenAI-compatible HTTP server (`/v1/chat/completions`, `/v1/completions`, `/v1/models`).
//! - `chat`: Interactive terminal REPL with multi-turn memory and live token-by-token streaming.

use cudarc::driver::CudaDevice;
use engine_core::{BpeTokenizer, ForwardDriver, Sampler, SamplerParams};
use engine_io::{GgufReader, LoadedPinned, ModelConfig, load_to_pinned};
use engine_server::models::{ChatMessage, format_chatml};
use engine_server::runtime::{self, RealModel};
use engine_server::server::{RealServerCfg, build_router_real};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn default_model_path() -> PathBuf {
    if let Ok(env_path) = std::env::var("ENGINE_TESTDATA") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return p;
        }
    }
    let candidates = [
        PathBuf::from("testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("../../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
    ];
    for c in &candidates {
        if c.exists() {
            return c.clone();
        }
    }
    PathBuf::from("testdata/Qwen3-0.6B-Q4_K_M.gguf")
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

fn run_chat(model_path: &Path, temperature: f32, top_p: f32) -> Result<(), Box<dyn std::error::Error>> {
    print_banner();
    println!("Loading model from: {:?}", model_path);
    let start_load = Instant::now();
    let reader = GgufReader::open(model_path)?;
    let pinned: &'static LoadedPinned = Box::leak(Box::new(load_to_pinned(&reader, model_path)?));
    let cfg = ModelConfig::from_reader(&reader)?;
    let tokenizer = BpeTokenizer::from_reader(&reader)?;
    let _device = CudaDevice::new(0)?;
    println!(
        "Model weights loaded to pinned RAM in {:.2} ms",
        start_load.elapsed().as_secs_f64() * 1000.0
    );

    let max_seq_tokens = 2048;
    let mut driver = ForwardDriver::new(&reader, pinned, &cfg, max_seq_tokens)?;
    println!("CUDA Graph decode engine initialized (28 layers, resident KV cache).");
    println!("Type your message and press Enter. Commands: /reset, /exit\n");

    let mut messages: Vec<ChatMessage> = Vec::new();
    let mut sampler = Sampler::new(42);
    let params = SamplerParams {
        temperature,
        top_p,
        top_k: 40,
        repetition_penalty: 1.1,
        seed: None,
    };
    let stop_tokens = [151645u32, 151643u32]; // <|im_end|>, <|endoftext|>

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
            println!("[Conversation history cleared]\n");
            continue;
        }

        messages.push(ChatMessage::user(input));
        let prompt = format_chatml(&messages);
        let prompt_tokens = tokenizer.encode(&prompt)?;

        print!("\nTitan > ");
        io::stdout().flush()?;

        let mut context = prompt_tokens.clone();
        let initial_logits = driver.prefill(&prompt_tokens)?;
        let first_tok = sampler.sample(&initial_logits, &context, &params);
        let first_piece = tokenizer.decode(&[first_tok]).unwrap_or_default();

        let mut assistant_response = String::new();

        if !Sampler::is_stop_sequence(first_tok, &first_piece, &stop_tokens, &[]) {
            print!("{}", first_piece);
            io::stdout().flush()?;
            assistant_response.push_str(&first_piece);
            context.push(first_tok);

            let mut current_tok = first_tok;
            const MAX_GEN: usize = 512;
            for _ in 1..MAX_GEN {
                let logits = match driver.decode(current_tok) {
                    Ok(l) => l,
                    Err(_) => break,
                };
                let next_tok = sampler.sample(&logits, &context, &params);
                let next_piece = tokenizer.decode(&[next_tok]).unwrap_or_default();

                if Sampler::is_stop_sequence(next_tok, &next_piece, &stop_tokens, &[]) {
                    break;
                }

                print!("{}", next_piece);
                io::stdout().flush()?;
                assistant_response.push_str(&next_piece);
                context.push(next_tok);
                current_tok = next_tok;
            }
        }

        println!("\n");
        messages.push(ChatMessage::assistant(assistant_response));
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("chat");

    let model_path = args
        .iter()
        .position(|a| a == "--model" || a == "-m")
        .and_then(|idx| args.get(idx + 1))
        .map(PathBuf::from)
        .unwrap_or_else(default_model_path);

    let port: u16 = args
        .iter()
        .position(|a| a == "--port" || a == "-p")
        .and_then(|idx| args.get(idx + 1))
        .and_then(|p| p.parse().ok())
        .unwrap_or(8000);

    let temperature: f32 = args
        .iter()
        .position(|a| a == "--temp" || a == "-t")
        .and_then(|idx| args.get(idx + 1))
        .and_then(|t| t.parse().ok())
        .unwrap_or(0.7);

    let top_p: f32 = args
        .iter()
        .position(|a| a == "--top-p")
        .and_then(|idx| args.get(idx + 1))
        .and_then(|p| p.parse().ok())
        .unwrap_or(0.9);

    if mode == "chat" {
        run_chat(&model_path, temperature, top_p)?;
    } else if mode == "serve" {
        print_banner();
        println!("Loading model for OpenAI Server from: {:?}", model_path);
        let start_load = Instant::now();
        let reader = GgufReader::open(&model_path)?;
        let pinned: &'static LoadedPinned = Box::leak(Box::new(load_to_pinned(&reader, &model_path)?));
        let _device = CudaDevice::new(0)?;
        println!(
            "Model weights loaded to pinned RAM in {:.2} ms",
            start_load.elapsed().as_secs_f64() * 1000.0
        );

        let real_model: RealModel<'static> = runtime::build_real_driver_model(&reader, pinned, 128)?;
        let shared: Arc<Mutex<RealModel<'static>>> = Arc::new(Mutex::new(real_model));

        let cfg = Arc::new(RealServerCfg {
            vocab: 151936,
            default_max_tokens: 512,
            model: shared,
        });

        let addr = format!("0.0.0.0:{port}");
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        println!("\n=======================================================");
        println!("  TITAN OPENAI-COMPATIBLE API SERVER LISTENING ON:    ");
        println!("  http://localhost:{port}                              ");
        println!("=======================================================");
        println!("  Endpoints:");
        println!("    • POST http://localhost:{port}/v1/chat/completions");
        println!("    • POST http://localhost:{port}/v1/completions");
        println!("    • GET  http://localhost:{port}/v1/models");
        println!("\n  Connect any OpenAI client (Cursor, LibreChat, Open-WebUI, LiteLLM)!");

        let app = build_router_real(cfg);
        axum::serve(listener, app).await?;
    } else {
        println!("Usage: titan [chat|serve] [--model <path>] [--port <port>] [--temp <0.7>] [--top-p <0.9>]");
    }

    Ok(())
}
