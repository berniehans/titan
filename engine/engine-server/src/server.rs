//! axum wiring: OpenAI-compatible completion route with SSE streaming
//! (task-3 milestone).
//!
//! - Non-streaming: `build_completion` returns the OpenAI-compatible JSON.
//! - Streaming: `stream_events` emits one SSE `data: {json}\n\n` chunk per
//!   generated token followed by the terminal `data: [DONE]\n\n`, all driven
//!   by the deterministic decode loop (`BatchScheduler` + `GenerationSession`).
//!
//! A client that drops the SSE connection cancels its session; see
//! `scheduler::cancel` which frees the session's KV blocks.

use crate::models::{
    ChunkChoice, CompletionChoice, CompletionChunk, CompletionRequest, CompletionResponse,
    CompletionUsage,
};
use crate::scheduler::BatchScheduler;
use crate::sse::{data_event, done_event, to_sse};
use axum::Router;
use axum::extract::{Json, State};
use axum::response::sse::Event;
use axum::response::{IntoResponse, Json as JsonResponse, Response};
use axum::routing::post;
use engine_kvcache::PagedKvCacheConfig;
use std::sync::Arc;

/// Engine scalar configuration for the CPU reference path.
pub struct KVConfig {
    /// Stub vocabulary bound (token ids live in [1, vocab)).
    pub vocab: u32,
    /// Completion budget applied when a request omits max_tokens.
    pub default_max_tokens: u32,
}

/// Deterministic stub token to generated text.
fn token_text(token: u32) -> String {
    format!(" token-{token}")
}

/// Deterministic stub prompt to starting token id.
fn prompt_token(prompt: &str, vocab: u32) -> u32 {
    let mut hash: u32 = 2166136261;
    for byte in prompt.as_bytes() {
        hash = (hash ^ (*byte as u32)).wrapping_mul(16777619);
    }
    hash.wrapping_rem(vocab) + 1
}

/// Deterministic request id for a prompt.
fn completion_id(prompt: &str) -> String {
    let mut hash: u64 = 1469598103934665603;
    for byte in prompt.as_bytes() {
        hash = (hash ^ (*byte as u64)).wrapping_mul(1099511628211);
    }
    format!("cmpl-{hash:x}")
}

/// Serializes a serde model to a compact JSON string.
fn serialize<T>(value: &T) -> String
where
    T: serde::Serialize,
{
    let bytes = serde_json::to_vec(value).expect("serialize json");
    String::from_utf8(bytes).expect("utf8")
}

/// Runs one completion to completion on a fresh scheduler; returns the
/// generated token sequence (excluding the prompt token).
fn decode_tokens(vocab: u32, prompt: &str, max_tokens: u32) -> Vec<u32> {
    let cfg = PagedKvCacheConfig {
        n_blocks: 1,
        block_tokens: max_tokens as usize + 1,
        heads: 1,
        head_dim: 1,
    };
    let mut scheduler = BatchScheduler::new(cfg).expect("kv pool");
    scheduler
        .add(vocab, prompt_token(prompt, vocab), max_tokens)
        .expect("session");
    let mut tokens = Vec::<u32>::with_capacity(max_tokens as usize);
    while scheduler.active_count() > 0 {
        tokens.extend(scheduler.advance());
    }
    tokens
}

/// Builds the SSE event sequence for a streaming completion from a token list:
/// one `data: {json}` event per token, then the `data: [DONE]` terminal event.
fn tokens_to_events(prompt: &str, model: &str, tokens: &[u32]) -> Vec<Event> {
    let length = "length".to_string();

    let mut events = Vec::with_capacity(tokens.len() + 1);
    let last = tokens.len() - 1;
    for (i, t) in tokens.iter().enumerate() {
        let choices = vec![ChunkChoice {
            index: i as u32,
            text: token_text(*t),
            finish_reason: if i == last {
                Some(length.clone())
            } else {
                None
            },
        }];
        let chunk = CompletionChunk {
            id: completion_id(prompt),
            object: "text_completion".to_string(),
            created: 1u64,
            model: model.to_string(),
            choices,
        };
        events.push(data_event(serialize(&chunk)));
    }
    events.push(done_event());
    events
}

/// Builds the SSE event sequence for a streaming completion (synthetic decode).
pub fn stream_events(cfg: Arc<KVConfig>, body: &CompletionRequest) -> Vec<Event> {
    let max_tokens = if body.max_tokens == 0 {
        cfg.default_max_tokens
    } else {
        body.max_tokens
    };
    let tokens = decode_tokens(cfg.vocab, &body.prompt, max_tokens);
    tokens_to_events(&body.prompt, &body.model, &tokens)
}

/// Builds a non-streaming completion response from a token list.
fn tokens_to_completion(prompt: &str, model: &str, tokens: &[u32]) -> CompletionResponse {
    let mut text = String::new();
    for t in tokens {
        text.push_str(&token_text(*t));
    }
    let completion_tokens = tokens.len() as u32;
    let choices = vec![CompletionChoice {
        text,
        index: 0,
        finish_reason: "length".to_string(),
    }];
    CompletionResponse {
        id: completion_id(prompt),
        object: "text_completion".to_string(),
        created: 1u64,
        model: model.to_string(),
        choices,
        usage: CompletionUsage {
            prompt_tokens: 1,
            completion_tokens,
            total_tokens: 1 + completion_tokens,
        },
    }
}

/// Builds a non-streaming completion response via the scheduler decode loop.
pub fn build_completion(cfg: Arc<KVConfig>, body: &CompletionRequest) -> CompletionResponse {
    let max_tokens = if body.max_tokens == 0 {
        cfg.default_max_tokens
    } else {
        body.max_tokens
    };
    let tokens = decode_tokens(cfg.vocab, &body.prompt, max_tokens);
    tokens_to_completion(&body.prompt, &body.model, &tokens)
}

async fn handle_completion(
    State(cfg): State<Arc<KVConfig>>,
    Json(body): Json<CompletionRequest>,
) -> Response {
    if body.stream {
        to_sse(stream_events(cfg, &body)).into_response()
    } else {
        JsonResponse::<CompletionResponse>(build_completion(cfg, &body)).into_response()
    }
}

/// Runtime-backed server config: a real, fixture-derived model behind the
/// OpenAI-compatible surface. The `Mutex` serialises the (non re-entrant) CUDA
/// pipeline across axum worker threads.
pub struct RealServerCfg {
    /// Stub vocabulary border for token ids.
    pub vocab: u32,
    /// Completion budget applied when a request omits max_tokens.
    pub default_max_tokens: u32,
    /// The real model runtime (device + dequantizer pipeline + loaded weights).
    pub model: crate::runtime::SharedRealModel,
}

use crate::runtime::RealModel;

/// Runs one completion through the real model runtime, returning both token IDs and decoded text chunks.
fn decode_tokens_and_texts_real(
    cfg: &RealServerCfg,
    prompt: &str,
    max_tokens: u32,
) -> (Vec<u32>, Vec<String>) {
    let mut model_guard = cfg.model.lock().expect("real model lock");
    let RealModel {
        driver, tokenizer, ..
    } = &mut *model_guard;

    if let (Some(driver), Some(tokenizer)) = (driver.as_mut(), tokenizer.as_ref()) {
        let prompt_tokens = match tokenizer.encode(prompt) {
            Ok(t) => t,
            Err(_) => return (Vec::new(), Vec::new()),
        };
        if prompt_tokens.is_empty() || max_tokens == 0 {
            return (Vec::new(), Vec::new());
        }
        let mut tokens = Vec::with_capacity(max_tokens as usize);
        let mut texts = Vec::with_capacity(max_tokens as usize);

        let initial_logits = driver.prefill(&prompt_tokens).expect("driver prefill");
        let mut current = crate::runtime::argmax(&initial_logits);
        let text_piece = tokenizer
            .decode(&[current])
            .unwrap_or_else(|_| token_text(current));
        tokens.push(current);
        texts.push(text_piece);

        for _ in 1..max_tokens {
            let logits = driver.decode(current).expect("driver decode");
            current = crate::runtime::argmax(&logits);
            let piece = tokenizer
                .decode(&[current])
                .unwrap_or_else(|_| token_text(current));
            tokens.push(current);
            texts.push(piece);
        }
        (tokens, texts)
    } else {
        let tokens = crate::runtime::decode_run(&mut model_guard, cfg.vocab, prompt, max_tokens)
            .expect("real decode run");
        let texts = tokens.iter().map(|&t| token_text(t)).collect();
        (tokens, texts)
    }
}

/// Builds the SSE event sequence from tokens and decoded text chunks.
fn tokens_and_texts_to_events(
    prompt: &str,
    model: &str,
    tokens: &[u32],
    texts: &[String],
) -> Vec<Event> {
    let length = "length".to_string();
    let mut events = Vec::with_capacity(tokens.len() + 1);
    let last = tokens.len().saturating_sub(1);
    for (i, t) in tokens.iter().enumerate() {
        let choices = vec![ChunkChoice {
            index: i as u32,
            text: texts.get(i).cloned().unwrap_or_else(|| token_text(*t)),
            finish_reason: if i == last {
                Some(length.clone())
            } else {
                None
            },
        }];
        let chunk = CompletionChunk {
            id: completion_id(prompt),
            object: "text_completion".to_string(),
            created: 1u64,
            model: model.to_string(),
            choices,
        };
        events.push(data_event(serialize(&chunk)));
    }
    events.push(done_event());
    events
}

/// Builds a non-streaming completion response from tokens and decoded text chunks.
fn tokens_and_texts_to_completion(
    prompt: &str,
    model: &str,
    tokens: &[u32],
    texts: &[String],
) -> CompletionResponse {
    let text = texts.concat();
    let completion_tokens = tokens.len() as u32;
    let choices = vec![CompletionChoice {
        text,
        index: 0,
        finish_reason: "length".to_string(),
    }];
    CompletionResponse {
        id: completion_id(prompt),
        object: "text_completion".to_string(),
        created: 1u64,
        model: model.to_string(),
        choices,
        usage: CompletionUsage {
            prompt_tokens: 1,
            completion_tokens,
            total_tokens: 1 + completion_tokens,
        },
    }
}

async fn handle_real_completion(
    State(cfg): State<Arc<RealServerCfg>>,
    Json(body): Json<CompletionRequest>,
) -> Response {
    let max_tokens = if body.max_tokens == 0 {
        cfg.default_max_tokens
    } else {
        body.max_tokens
    };
    let (tokens, texts) = decode_tokens_and_texts_real(&cfg, &body.prompt, max_tokens);
    if body.stream {
        to_sse(tokens_and_texts_to_events(
            &body.prompt,
            &body.model,
            &tokens,
            &texts,
        ))
        .into_response()
    } else {
        JsonResponse::<CompletionResponse>(tokens_and_texts_to_completion(
            &body.prompt,
            &body.model,
            &tokens,
            &texts,
        ))
        .into_response()
    }
}

/// Builds the axum router serving /v1/completions.
pub fn build_router(cfg: Arc<KVConfig>) -> Router {
    Router::new()
        .route("/v1/completions", post(handle_completion))
        .with_state(cfg)
}

/// Builds the axum router serving /v1/completions backed by a real model
/// runtime (the flujo-completo path exercised by the `#[ignore]` GPU E2E).
pub fn build_router_real(cfg: Arc<RealServerCfg>) -> Router {
    Router::new()
        .route("/v1/completions", post(handle_real_completion))
        .with_state(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::Client;

    fn test_router() -> Router {
        build_router(Arc::new(KVConfig {
            vocab: 1000,
            default_max_tokens: 6,
        }))
    }

    // Spawns a real server on an ephemeral 127.0.0.1 port.
    async fn spawn() -> (Client, String) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let port = listener.local_addr().unwrap().port();
        assert_ne!(port, 0, "ephemeral bind expected");
        let app = test_router();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let client = Client::builder().build().unwrap();
        (client, format!("http://127.0.0.1:{port}"))
    }

    #[tokio::test]
    async fn completions_returns_openai_compatible_json() {
        let (client, base) = spawn().await;
        let res = client
            .post(&(base + "/v1/completions"))
            .header("content-type", "application/json")
            .body(r#"{"model":"test","prompt":"hello world","max_tokens":4,"stream":false}"#)
            .send()
            .await;
        let status = res.unwrap();
        assert_eq!(status.status(), 200);

        let body = status.text().await.unwrap();
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(v["id"].as_str().is_some(), "id present");
        assert_eq!(v["object"], "text_completion");
        let text = v["choices"][0]["text"].as_str().unwrap();
        assert_ne!(text.len(), 0, "choices[0].text non-empty");
        assert!(v["usage"]["total_tokens"].is_u64(), "usage present");
    }

    #[tokio::test]
    async fn unknown_route_returns_404() {
        let (client, base) = spawn().await;
        let res = client.post(&(base + "/v1/does-not-exist")).send().await;
        assert_eq!(res.unwrap().status(), 404);
    }

    #[tokio::test]
    async fn malformed_body_returns_400() {
        let (client, base) = spawn().await;
        let res = client
            .post(&(base + "/v1/completions"))
            .header("content-type", "application/json")
            .body("{ this is not json")
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 400);
    }

    #[tokio::test]
    async fn streaming_frames_chunks_then_done() {
        let (client, base) = spawn().await;
        let mut res = client
            .post(&(base + "/v1/completions"))
            .header("content-type", "application/json")
            .body(r#"{"model":"test","prompt":"stream me","max_tokens":4,"stream":true}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);

        let mut saw_data = false;
        let mut saw_done = false;
        while let Some(chunk) = res.chunk().await.unwrap() {
            let text = String::from_utf8(chunk.to_vec()).unwrap();
            // Each event body is a JSON object serialized on a data: line.
            assert!(
                text.starts_with("data: "),
                "expected `data: ...` framing, got: {text:?}"
            );
            if text.contains("[DONE]") {
                saw_done = true;
            } else {
                // A token chunk: data: {"id":...,"choices":[...]}\n\n
                let json = text.trim_start_matches("data: ").trim_end();
                let _: serde_json::Value =
                    serde_json::from_str(json).expect("token chunk should be JSON");
                saw_data = true;
            }
        }
        assert!(saw_data, "expected at least one token chunk");
        assert!(saw_done, "stream must terminate with [DONE] marker");
    }
}
