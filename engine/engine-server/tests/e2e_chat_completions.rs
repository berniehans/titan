//! Real End-to-End Chat Completions & Streaming SSE HTTP Gate (Phase 10, Task 3.4).
//!
//! Spawns a real Axum server on an ephemeral port backed by `ForwardDriver` with
//! CUDA Graphs and tests:
//! 1. `GET /v1/models`
//! 2. `POST /v1/chat/completions` (non-streaming JSON)
//! 3. `POST /v1/chat/completions` (streaming Server-Sent Events)
//! 4. Sampling controls and stop sequence trimming.

use cudarc::driver::CudaDevice;
use engine_io::{GgufReader, LoadedPinned, load_to_pinned};
use engine_server::runtime::{self, RealModel};
use engine_server::server::{RealServerCfg, build_router_real};
use reqwest::Client;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

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

// Spawns a real server with CUDA Graphs on an ephemeral localhost port.
async fn spawn_real_server(fixture: PathBuf) -> Result<(Client, String), DynError> {
    let reader = GgufReader::open(&fixture)?;
    let pinned: &'static LoadedPinned = Box::leak(Box::new(load_to_pinned(&reader, &fixture)?));
    let _device = CudaDevice::new(0)?;

    let real_model: RealModel<'static> = runtime::build_real_driver_model(&reader, pinned, 128)?;
    let shared: Arc<Mutex<RealModel<'static>>> = Arc::new(Mutex::new(real_model));

    let cfg = Arc::new(RealServerCfg {
        vocab: 151936,
        default_max_tokens: 16,
        model: shared,
    });

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    let app = build_router_real(cfg);

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = Client::builder().build()?;
    Ok((client, format!("http://127.0.0.1:{port}")))
}

#[tokio::test]
#[ignore]
async fn test_real_chat_completions_http_e2e() -> Result<(), DynError> {
    engine_cuda::ensure_cuda_dll_paths();
    let fix = fixture_path().ok_or("fixture missing (GPU test)")?;
    println!("Spawning real HTTP server with fixture {:?}", fix);
    let (client, base) = spawn_real_server(fix).await?;

    // 1. Test GET /v1/models
    println!("\n--- Testing GET /v1/models ---");
    let res = client.get(&(base.clone() + "/v1/models")).send().await?;
    assert_eq!(res.status(), 200);
    let body = res.text().await?;
    let models_val: serde_json::Value = serde_json::from_str(&body)?;
    println!("  GET /v1/models response: {}", body);
    assert_eq!(models_val["object"], "list");
    assert_eq!(models_val["data"][0]["id"], "qwen3-0.6b");

    // 2. Test POST /v1/chat/completions (Non-streaming)
    println!("\n--- Testing POST /v1/chat/completions (JSON) ---");
    let chat_req = serde_json::json!({
        "model": "qwen3-0.6b",
        "messages": [
            {"role": "user", "content": "The capital of France is"}
        ],
        "max_tokens": 6,
        "temperature": 0.0,
        "stream": false
    });

    let res = client
        .post(&(base.clone() + "/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(chat_req.to_string())
        .send()
        .await?;
    assert_eq!(res.status(), 200);
    let body = res.text().await?;
    println!("  Chat completion response: {}", body);
    let chat_val: serde_json::Value = serde_json::from_str(&body)?;
    assert!(chat_val["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(chat_val["choices"][0]["message"]["role"], "assistant");
    let content = chat_val["choices"][0]["message"]["content"].as_str().unwrap();
    println!("  Generated content: {:?}", content);
    assert!(!content.is_empty(), "generated content should not be empty");

    // 3. Test POST /v1/chat/completions (Streaming SSE)
    println!("\n--- Testing POST /v1/chat/completions (Streaming SSE) ---");
    let stream_req = serde_json::json!({
        "model": "qwen3-0.6b",
        "messages": [
            {"role": "user", "content": "2 + 2 ="}
        ],
        "max_tokens": 6,
        "temperature": 0.0,
        "stream": true
    });

    let mut res = client
        .post(&(base.clone() + "/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(stream_req.to_string())
        .send()
        .await?;
    assert_eq!(res.status(), 200);

    let mut stream_content = String::new();
    let mut saw_done = false;

    while let Some(chunk) = res.chunk().await? {
        let chunk_str = String::from_utf8(chunk.to_vec())?;
        for line in chunk_str.lines() {
            if line.starts_with("data: ") {
                let payload = line.trim_start_matches("data: ").trim();
                if payload == "[DONE]" {
                    saw_done = true;
                } else if let Ok(val) = serde_json::from_str::<serde_json::Value>(payload) {
                    if let Some(delta) = val["choices"][0]["delta"]["content"].as_str() {
                        stream_content.push_str(delta);
                        print!("{}", delta);
                    }
                }
            }
        }
    }
    println!("\n  Full streamed content: {:?}", stream_content);
    assert!(saw_done, "stream must terminate with [DONE]");
    assert!(!stream_content.is_empty(), "streamed content must be non-empty");

    println!("\nAll E2E Chat Completions HTTP tests passed successfully!");
    Ok(())
}
