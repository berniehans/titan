//! E2E full-stack flow — CI-safe variant (f5-sse-server-batching, task 4.3).
//!
//! Same E2E as the GPU fixture test but against the engine's synthetic
//! in-memory decode layout — no GGUF fixture and no CUDA device required, so it
//! runs in plain `cargo test --workspace` (GitHub CI): two *concurrent* SSE
//! clients stream completions from one live axum server, each receiving its own
//! per-session-correct deterministic output (chunks arrive interleaved but
//! per-session order + framing are correct), plus one non-streaming request is
//! verified end to end.
//!
//! The synthetic layout stands in for the loaded model: `KVConfig` drives the
//! deterministic stub decode over an in-memory paged KV pool (no disk, no GPU).
//! This is the CI-safe floor that proves the HTTP/SSE/batching machinery end to
//! end without the ~400 MB fixture or a CUDA device.

use engine_server::{KVConfig, build_router};
use reqwest::Client;
use std::sync::Arc;

/// Spawns a real axum server on an ephemeral 127.0.0.1 port inside `rt` and
/// returns a client + base URL. The server task outlives this fn because the
/// shared runtime is held by the test.
fn spawn(cfg: Arc<KVConfig>, base: &str) -> Client {
    let (client, _) = build_base(cfg, base);
    client
}

fn build_base(cfg: Arc<KVConfig>, port_out: &str) -> (Client, String) {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let r = rt.block_on(async {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let port = listener.local_addr().unwrap().port();
        let app = build_router(cfg);
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let client = Client::builder().build().unwrap();
        (client, format!("http://127.0.0.1:{port}"))
    });
    let _ = port_out;
    std::mem::forget(rt);
    r
}

fn body_json(prompt: &str, max_tokens: u32, stream: bool) -> String {
    format!(r#"{{"model":"test","prompt":"{prompt}","max_tokens":{max_tokens},"stream":{stream}}}"#)
}

/// Reads an SSE stream completely, returning the per-session `data:` chunk
/// payloads in arrival order (token chunks) plus whether `[DONE]` was seen.
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

#[test]
fn e2e_two_concurrent_sse_sessions_per_session_correct() {
    let (_client, base) = build_base(
        Arc::new(KVConfig {
            vocab: 1000,
            default_max_tokens: 6,
        }),
        "",
    );

    // Two concurrent sessions with *different* prompts. Under tokio + axum these
    // are served interleaved (the server multiplexes the scheduler), but each
    // session must come back with its own correct, deterministic token stream.
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let (a, b, a_solo) = rt.block_on(async {
        let client = Client::new();
        let handle_a = tokio::spawn(collect_stream(
            Client::new(),
            base.clone(),
            "concurrency probe alpha",
        ));
        let handle_b = tokio::spawn(collect_stream(
            Client::new(),
            base.clone(),
            "concurrency probe beta",
        ));
        let a = handle_a.await.expect("session a");
        let b = handle_b.await.expect("session b");
        // Determinism: re-run session A alone *after* the concurrent pair, on the
        // same live server.
        let a_solo = collect_stream(Client::new(), base, "concurrency probe alpha").await;
        (a, b, a_solo)
    });
    std::mem::forget(rt);

    assert!(!a.0.is_empty(), "session a must produce token chunks");
    assert!(!b.0.is_empty(), "session b must produce token chunks");
    assert_eq!(a.0.len(), 4, "session a: one chunk per step (max_tokens=4)");
    assert_eq!(b.0.len(), 4, "session b: one chunk per step (max_tokens=4)");
    assert!(
        a.1 && b.1,
        "each concurrent session must observe the [DONE] terminator"
    );

    // Per-session determinism: the concurrent session A's chunk sequence must be
    // byte-identical to the solo re-run (same prompt, same model layout).
    assert_eq!(
        a.0, a_solo.0,
        "session A output must be deterministic regardless of concurrency"
    );

    // Distinct prompts diverge: sessions don't leak into each other.
    assert_ne!(
        a.0, b.0,
        "different prompts must yield different token streams"
    );
}

#[test]
fn e2e_non_streaming_request_returns_correct_completion() {
    let (_client, base) = build_base(
        Arc::new(KVConfig {
            vocab: 1000,
            default_max_tokens: 6,
        }),
        "",
    );

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let body = rt.block_on(async {
        let client = Client::new();
        client
            .post(&(base + "/v1/completions"))
            .header("content-type", "application/json")
            .body(body_json("non streaming probe", 4, false))
            .send()
            .await
            .expect("non-streaming send")
            .text()
            .await
            .expect("non-streaming text")
    });
    std::mem::forget(rt);

    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "text_completion");
    let text = v["choices"][0]["text"].as_str().expect("text present");
    // The non-streaming response concatenates the deterministic token texts.
    assert!(text.starts_with(" token-"), "text callout: {text}");
    assert!(v["usage"]["completion_tokens"].as_u64().unwrap() == 4);
}
