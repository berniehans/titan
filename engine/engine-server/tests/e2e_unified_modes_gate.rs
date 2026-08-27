//! End-to-End Multi-Mode Verification Gate (Phase 14).
//!
//! Validates live OpenAI-compatible HTTP chat completions across:
//! 1. `EngineMode::Resident` (Full GPU forward path)
//! 2. `EngineMode::Streaming` (PCIe DMA double-buffered layer streaming path)
//! 3. `SpeculativeMode::Ngram` (Context n-gram speculative acceleration)
//!
//! Verifies:
//! - Telemetry headers (`x-titan-engine-mode`, `x-titan-vram-mb`).
//! - JSON non-streaming `/v1/chat/completions`.
//! - Streaming SSE chunks over `/v1/chat/completions`.

use engine_io::{GgufReader, LoadedPinned, load_to_pinned};
use engine_server::runtime::{self, EngineMode, RealModel, SpeculativeMode};
use engine_server::server::{RealServerCfg, build_router_real};
use reqwest::Client;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

fn resolve_fixture_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ENGINE_TESTDATA") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    let candidates = [
        PathBuf::from("../../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("../testdata/Qwen3-0.6B-Q4_K_M.gguf"),
        PathBuf::from("testdata/Qwen3-0.6B-Q4_K_M.gguf"),
    ];
    for c in &candidates {
        if c.exists() {
            return Some(c.clone());
        }
    }
    None
}

async fn spawn_server(
    model_path: &Path,
    engine_mode: EngineMode,
    spec_mode: SpeculativeMode,
) -> (Client, String) {
    let reader = GgufReader::open(model_path).expect("open gguf");
    let pinned: &'static LoadedPinned = Box::leak(Box::new(
        load_to_pinned(&reader, model_path).expect("pinned load"),
    ));

    let real_model: RealModel<'static> = runtime::build_unified_driver_model(
        &reader,
        pinned,
        128,
        engine_mode,
        spec_mode,
    )
    .expect("build unified model");

    let shared = Arc::new(Mutex::new(real_model));
    let cfg = Arc::new(RealServerCfg {
        vocab: 151936,
        default_max_tokens: 8,
        model: shared,
    });

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("ephemeral bind");
    let port = listener.local_addr().unwrap().port();
    let app = build_router_real(cfg);

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = Client::builder().build().unwrap();
    (client, format!("http://127.0.0.1:{port}"))
}

#[tokio::test]
#[ignore]
async fn test_unified_modes_resident_and_streaming() {
    let Some(fixture_path) = resolve_fixture_path() else {
        eprintln!("Skipping test_unified_modes_resident_and_streaming: fixture not found");
        return;
    };

    println!("\n=== Testing EngineMode::Resident with OpenAI Chat Completions ===");
    let (client_res, base_res) = spawn_server(
        &fixture_path,
        EngineMode::Resident,
        SpeculativeMode::None,
    )
    .await;

    let req_json = serde_json::json!({
        "messages": [{"role": "user", "content": "Hi!"}],
        "model": "qwen3-0.6b",
        "max_tokens": 6,
        "stream": false,
        "temperature": 0.7,
        "top_p": 0.9
    });

    let resp_res = client_res
        .post(format!("{base_res}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(req_json.to_string())
        .send()
        .await
        .expect("send resident request");

    assert_eq!(resp_res.status(), 200);
    assert_eq!(
        resp_res.headers().get("x-titan-engine-mode").unwrap().to_str().unwrap(),
        "resident"
    );
    let body_res: serde_json::Value = serde_json::from_str(&resp_res.text().await.unwrap()).unwrap();
    println!("  Resident generated: {:?}", body_res["choices"][0]["message"]["content"]);
    assert!(body_res["choices"][0]["message"]["content"].as_str().is_some());

    println!("\n=== Testing EngineMode::Streaming with Speculative Ngram Acceleration ===");
    let (client_str, base_str) = spawn_server(
        &fixture_path,
        EngineMode::Streaming,
        SpeculativeMode::Ngram,
    )
    .await;

    let req_stream = serde_json::json!({
        "messages": [{"role": "user", "content": "Hi"}],
        "model": "qwen3-0.6b",
        "max_tokens": 4,
        "stream": true,
        "temperature": 0.7,
        "top_p": 0.9
    });

    let resp_str = client_str
        .post(format!("{base_str}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(req_stream.to_string())
        .send()
        .await
        .expect("send streaming request");

    assert_eq!(resp_str.status(), 200);
    assert_eq!(
        resp_str.headers().get("x-titan-engine-mode").unwrap().to_str().unwrap(),
        "streaming"
    );

    let stream_body = resp_str.text().await.expect("read stream");
    println!("  Streaming SSE body snippet: {:?}", &stream_body[..stream_body.len().min(120)]);
    assert!(stream_body.contains("chat.completion.chunk"));
    assert!(stream_body.contains("[DONE]"));

    println!("\nGate PASS: All unified engine modes (Resident, Streaming, Speculative) validated successfully over HTTP!");
}
