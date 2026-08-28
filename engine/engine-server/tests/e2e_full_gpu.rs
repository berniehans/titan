//! E2E full-stack flow — GPU fixture gate (f5-sse-server-batching, task 4.2).
//!
//! The flujo-completo gate: a real GGUF fixture flows through the *entire*
//! engine stack into the HTTP server, and two *concurrent* SSE clients stream
//! deterministic, per-session-correct completions while a non-streaming request
//! is also verified.
//!
//! Wiring exercised end to end:
//!   testdata/Qwen3-0.6B-Q4_K_M.gguf
//!     -> GgufReader::open          (engine-io: GGUF v3 parser)
//!     -> load_to_pinned           (engine-io: single NVMe pass into pinned RAM)
//!     -> build_real_model         (engine-server: picks aligned Q4_K tensors)
//!     -> Pipeline::with_dequantizer (engine-core + engine-cuda: real GPU dequant)
//!     -> runtime::decode_run      (per-step streaming + digest -> deterministic)
//!     -> /v1/completions (axum)   (SSE + JSON)
//!
//! Requires: a local CUDA device and the NVRTC DLL discoverable on PATH
//! (e.g. `%LOCALAPPDATA%/Temp/nvrtc64_120_0.dll` / `nvrtc-builtins64_130.dll`).
//! `#[ignore]`d so it does not run in CI; run with
//! `cargo test -p engine-server --test e2e_full_stack_gpu -- --ignored`.

use cudarc::driver::CudaDevice;
use engine_io::{GgufReader, LoadedPinned, load_to_pinned};
use engine_server::runtime::{self, SharedRealModel};
use engine_server::server::{RealServerCfg, build_router_real};
use reqwest::Client;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

type DynError = Box<dyn std::error::Error + Send + Sync>;

/// Locates the fixture across the standard candidate paths (mirrors the other
/// crates' fixture helper). Returns `None` when absent (CI skip).
fn fixture_path() -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("ENGINE_TESTDATA") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return Some(p);
        }
    }
    let manifest_dir = PathBuf::from(env!["CARGO_MANIFEST_DIR"]);
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

/// Builds a real model from the fixture (loader -> pinned -> GPU dequant).
fn build_model(
    window_bytes: usize,
) -> Result<(SharedRealModel, GgufReader, &'static LoadedPinned), DynError> {
    engine_cuda::ensure_cuda_dll_paths();
    let fixture = fixture_path().ok_or("fixture not present (GPU E2E requires the GGUF)")?;
    let reader = GgufReader::open(&fixture)?;
    let pinned: &'static LoadedPinned = Box::leak(Box::new(load_to_pinned(&reader, &fixture)?));
    let _device = CudaDevice::new(0)?;
    let model = runtime::build_real_model(&reader, pinned, window_bytes)?;
    Ok((Arc::new(Mutex::new(model)), reader, pinned))
}

/// Binds the real model into an axum server on an ephemeral port and returns a
/// base URL. Uses `tokio::runtime` so the server task can outlive the helper.
fn spawn_real(model: SharedRealModel, vocab: u32) -> String {
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

fn body_json(prompt: &str, max_tokens: u32, stream: bool) -> String {
    format!(r#"{{"model":"test","prompt":"{prompt}","max_tokens":{max_tokens},"stream":{stream}}}"#)
}

/// Collects one SSE completion stream fully, returning the per-session chunk
/// payloads plus the terminal `[DONE]` flag. `collect_stream` asserts framing.
async fn collect_stream(client: Client, base: String, prompt: &str) -> (Vec<String>, bool) {
    let mut res = client
        .post(&(base + "/v1/completions"))
        .header("content-type", "application/json")
        .body(body_json(prompt, 4, true))
        .send()
        .await
        .expect("collect stream send");
    assert_eq!(res.status(), 200);
    let mut chunks = Vec::<String>::new();
    let mut saw_done = false;
    while let Some(frame) = res.chunk().await.expect("chunk") {
        let text = String::from_utf8(frame.to_vec()).unwrap();
        assert!(text.starts_with("data: "), "SSE frame: {text:?}");
        if text.contains("[DONE]") {
            saw_done = true;
        } else {
            let json = text.trim_start_matches("data: ").trim_end();
            serde_json::from_str::<serde_json::Value>(json)
                .unwrap_or_else(|e| panic!("token chunk JSON: {e}"));
            chunks.push(text);
        }
    }
    (chunks, saw_done)
}

/// The flujo-completo gate: real fixture -> real pipeline -> concurrent SSE.
#[test]
#[ignore]
fn e2e_real_full_stack_concurrent_sse_and_non_streaming() -> Result<(), DynError> {
    let (model, _reader, _pinned) = build_model(/* window */ 64 * 1024 * 1024)?;
    let vocab = 1000; // stub vocab bound; the digest depends on the real weights.
    let base = spawn_real(model, vocab);

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let (a, b, a_solo, non_stream_text) = rt.block_on(async {
        // Two concurrent sessions, distinct prompts, one live real-model server.
        let handle_a = tokio::spawn(collect_stream(
            Client::new(),
            base.clone(),
            "real model concurrent alpha",
        ));
        let handle_b = tokio::spawn(collect_stream(
            Client::new(),
            base.clone(),
            "real model concurrent beta",
        ));
        let a = handle_a.await.expect("session a");
        let b = handle_b.await.expect("session b");
        // Determinism: re-run A alone against the same live real model.
        let a_solo =
            collect_stream(Client::new(), base.clone(), "real model concurrent alpha").await;

        // Non-streaming request correctness.
        let non_stream = Client::new()
            .post(&(base + "/v1/completions"))
            .header("content-type", "application/json")
            .body(body_json("real non streaming", 4, false))
            .send()
            .await
            .expect("non-streaming send")
            .text()
            .await
            .expect("non-streaming text");

        (a, b, a_solo, non_stream)
    });
    std::mem::forget(rt);

    // Framing + completeness per session.
    assert!(!a.0.is_empty(), "real session a must produce token chunks");
    assert!(!b.0.is_empty(), "real session b must produce token chunks");
    assert_eq!(a.0.len(), 4, "real session a: one chunk per step");
    assert_eq!(b.0.len(), 4, "real session b: one chunk per step");
    assert!(a.1 && b.1, "both real streams must terminate with [DONE]");

    // Determinism (the crux of the flujo completo): the real-model sequence for
    // a given prompt is stable regardless of concurrency.
    assert_eq!(
        a.0, a_solo.0,
        "real session A must be deterministic across runs"
    );

    // Sessions with different prompts diverge (no cross-session leakage).
    assert_ne!(
        a.0, b.0,
        "distinct prompts must yield distinct real streams"
    );

    // Non-streaming correctness.
    let v: serde_json::Value = serde_json::from_str(&non_stream_text)?;
    assert_eq!(v["object"], "text_completion");
    let text = v["choices"][0]["text"].as_str().expect("text present");
    assert!(
        text.starts_with(" token-"),
        "real non-streaming text callout: {text}"
    );
    assert_eq!(v["usage"]["completion_tokens"].as_u64().unwrap(), 4);

    Ok(())
}
