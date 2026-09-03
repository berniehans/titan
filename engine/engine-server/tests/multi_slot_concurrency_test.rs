//! Multi-Slot Concurrency Test (Milestone 5).
//!
//! Spawns a real ephemeral server and issues 4 concurrent client requests
//! in parallel to verify continuous multi-slot inference.

use cudarc::driver::CudaDevice;
use engine_io::{GgufReader, LoadedPinned, load_to_pinned};
use engine_server::models::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage};
use engine_server::runtime::{self, RealModel};
use engine_server::server::{RealServerCfg, build_router_real};
use reqwest::Client;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

type DynError = Box<dyn std::error::Error + Send + Sync>;

fn fixture_path() -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("ENGINE_TESTDATA") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return Some(p);
        }
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("../../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        manifest_dir.join("../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
    ];
    for c in &candidates {
        if c.exists() {
            return Some(c.canonicalize().unwrap_or_else(|_| c.clone()));
        }
    }
    None
}

async fn spawn_real_server(fixture: PathBuf) -> Result<(Client, String), DynError> {
    let reader = GgufReader::open(&fixture)?;
    let pinned: &'static LoadedPinned = Box::leak(Box::new(load_to_pinned(&reader, &fixture)?));
    let _device = CudaDevice::new(0)?;

    let real_model: RealModel<'static> = runtime::build_real_driver_model(&reader, pinned, 512)?;
    let shared: Arc<Mutex<RealModel<'static>>> = Arc::new(Mutex::new(real_model));

    let cfg = Arc::new(RealServerCfg {
        vocab: 151936,
        default_max_tokens: 32,
        model: shared,
    });

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    let app = build_router_real(cfg);

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = Client::new();
    let base_url = format!("http://127.0.0.1:{port}");
    Ok((client, base_url))
}

#[tokio::test]
#[ignore]
async fn test_multi_slot_concurrent_requests() -> Result<(), DynError> {
    engine_cuda::ensure_cuda_dll_paths();
    let Some(fixture) = fixture_path() else {
        eprintln!("SKIPPING test_multi_slot_concurrent_requests: fixture not found");
        return Ok(());
    };

    println!("\n================================================================================");
    println!(">>> TESTING MULTI-SLOT CONCURRENT INFERENCE ON TITAN ENGINE");
    println!("================================================================================");

    let (client, base_url) = spawn_real_server(fixture).await?;

    let prompts = [
        "What is 2 + 2?",
        "The capital of France is",
        "Explain gravity in one sentence.",
        "Hello! How are you?",
    ];

    let t0 = Instant::now();
    let mut handles = Vec::new();

    for (i, prompt) in prompts.iter().enumerate() {
        let client_clone = client.clone();
        let url = format!("{base_url}/v1/chat/completions");
        let prompt_str = prompt.to_string();

        let handle = tokio::spawn(async move {
            let req = ChatCompletionRequest {
                model: Some("titan-slot-test".to_string()),
                messages: vec![ChatMessage::user(&prompt_str)],
                temperature: Some(0.0),
                top_p: Some(1.0),
                top_k: None,
                repetition_penalty: None,
                max_tokens: 16,
                stop: None,
                stream: false,
                tools: None,
                tool_choice: None,
                response_format: None,
            };

            let res = client_clone
                .post(&url)
                .header("content-type", "application/json")
                .body(serde_json::to_string(&req).unwrap())
                .send()
                .await
                .expect("request send");

            assert_eq!(res.status(), 200);
            let text = res.text().await.expect("text response");
            let body: ChatCompletionResponse = serde_json::from_str(&text).expect("json parse");
            let content = body.choices[0].message.content.clone();
            (i, prompt_str, content)
        });

        handles.push(handle);
    }

    for handle in handles {
        let (idx, prompt, response) = handle.await?;
        println!(
            "  Slot #{idx} [Prompt: \"{prompt}\"] -> Response: \"{}\"",
            response.trim()
        );
        assert!(
            !response.is_empty(),
            "Response from slot #{idx} must not be empty"
        );
    }

    let elapsed = t0.elapsed();
    println!(
        "\n>>> SUCCESS: All 4 concurrent slots completed in {:.2}s!",
        elapsed.as_secs_f64()
    );
    println!("================================================================================\n");

    Ok(())
}
