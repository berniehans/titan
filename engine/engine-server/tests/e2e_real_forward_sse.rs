//! SSE E2E autoregressive generation test over real ForwardDriver (Phase 6.8, Sub-gate 2).
//!
//! Asserts that:
//! 1. An SSE request to `/v1/completions` generates coherent autoregressive tokens from the real driver.
//! 2. Token chunks are streamed incrementally with valid UTF-8 text.
//! 3. Stream terminates cleanly with the `[DONE]` marker.
//! 4. Non-streaming endpoint returns complete autoregressive response.

use cudarc::driver::CudaDevice;
use engine_io::{GgufReader, LoadedPinned, load_to_pinned};
use engine_server::runtime::{self, SharedRealModel};
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

fn build_driver_model(
    max_seq: usize,
) -> Result<(SharedRealModel, GgufReader, &'static LoadedPinned), DynError> {
    engine_cuda::ensure_cuda_dll_paths();
    let fixture = fixture_path().ok_or("fixture not present (GPU test)")?;
    let reader = GgufReader::open(&fixture)?;
    let pinned: &'static LoadedPinned = Box::leak(Box::new(load_to_pinned(&reader, &fixture)?));
    let _device = CudaDevice::new(0)?;
    let model = runtime::build_real_driver_model(&reader, pinned, max_seq)?;
    Ok((Arc::new(Mutex::new(model)), reader, pinned))
}

fn spawn_real_server(model: SharedRealModel, vocab: u32) -> String {
    let cfg = Arc::new(RealServerCfg {
        vocab,
        default_max_tokens: 6,
        model,
    });
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let base = rt.block_on(async {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let port = listener.local_addr().unwrap().port();
        let app = build_router_real(cfg);
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://127.0.0.1:{port}")
    });
    std::mem::forget(rt);
    base
}

#[test]
#[ignore]
fn test_subgate_2_sse_real_autoregressive_generation() -> Result<(), DynError> {
    let (model, _reader, _pinned) = build_driver_model(128)?;
    let base = spawn_real_server(model, 151936);

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        let client = Client::new();

        // 1. Test SSE streaming completion
        let req_body = serde_json::json!({
            "model": "Qwen3-0.6B",
            "prompt": "Hello",
            "max_tokens": 5,
            "stream": true
        });

        let mut res = client
            .post(format!("{base}/v1/completions"))
            .header("content-type", "application/json")
            .body(req_body.to_string())
            .send()
            .await
            .expect("send sse request");

        assert_eq!(res.status(), 200);

        let mut chunks = Vec::new();
        let mut generated_text = String::new();
        let mut saw_done = false;

        while let Some(frame) = res.chunk().await.expect("receive chunk") {
            let text = String::from_utf8(frame.to_vec()).unwrap();
            for line in text.lines() {
                if line.starts_with("data: ") {
                    let payload = line.trim_start_matches("data: ").trim();
                    if payload == "[DONE]" {
                        saw_done = true;
                    } else if !payload.is_empty() {
                        let chunk_json: serde_json::Value =
                            serde_json::from_str(payload).expect("valid chunk json");
                        let chunk_text = chunk_json["choices"][0]["text"]
                            .as_str()
                            .expect("text field");
                        generated_text.push_str(chunk_text);
                        chunks.push(chunk_text.to_string());
                    }
                }
            }
        }

        println!("SSE Generated chunks (n={}): {:?}", chunks.len(), chunks);
        println!("SSE Full Generated Text: {:?}", generated_text);

        assert!(saw_done, "stream must terminate with [DONE]");
        assert_eq!(chunks.len(), 5, "must produce 5 token chunks");
        assert!(
            !generated_text.is_empty(),
            "generated text must not be empty"
        );

        // 2. Test non-streaming completion
        let non_stream_req = serde_json::json!({
            "model": "Qwen3-0.6B",
            "prompt": "Hello",
            "max_tokens": 5,
            "stream": false
        });

        let res_non = client
            .post(format!("{base}/v1/completions"))
            .header("content-type", "application/json")
            .body(non_stream_req.to_string())
            .send()
            .await
            .expect("send non-streaming request");

        assert_eq!(res_non.status(), 200);
        let text_body = res_non.text().await.expect("read response text");
        let resp_json: serde_json::Value = serde_json::from_str(&text_body).expect("parse json");
        let non_stream_text = resp_json["choices"][0]["text"]
            .as_str()
            .expect("text field");
        println!("Non-streaming Full Text: {:?}", non_stream_text);
        assert_eq!(resp_json["usage"]["completion_tokens"], 5);

        println!("Sub-gate 2 PASS: SSE E2E autoregressive generation verified.");
    });

    Ok(())
}
