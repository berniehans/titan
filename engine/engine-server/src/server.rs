//! axum wiring: OpenAI-compatible completion route (task-1 skeleton).
//!
//! Serves `POST /v1/completions` returning the OpenAI-compatible JSON shape
//! (id, object, choices[0].text, usage) from a deterministic stub. A later
//! milestone wires the real decode loop (session + batch scheduler).

use crate::models::{CompletionChoice, CompletionRequest, CompletionResponse, CompletionUsage};
use axum::Router;
use axum::extract::{Json, State};
use axum::response::{IntoResponse, Json as JsonResponse, Response};
use axum::routing::post;
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

/// Deterministic request id for a prompt.
fn completion_id(prompt: &str) -> String {
    let mut hash: u64 = 1469598103934665603;
    for byte in prompt.as_bytes() {
        hash = (hash ^ (*byte as u64)).wrapping_mul(1099511628211);
    }
    format!("cmpl-{hash:x}")
}

/// Builds a completion response from the placeholder decode loop (task 1).
///
/// Deterministic stub producing exactly `max_tokens` tokens in [1, vocab) so
/// the endpoint has a reproducible response shape before the batch scheduler
/// lands.
pub fn build_completion(cfg: Arc<KVConfig>, body: &CompletionRequest) -> CompletionResponse {
    let max_tokens = if body.max_tokens == 0 {
        cfg.default_max_tokens
    } else {
        body.max_tokens
    };
    let tokens: Vec<u32> = (0..max_tokens).map(|i| (i % cfg.vocab) + 1).collect();

    let mut text = String::new();
    for t in &tokens {
        text.push_str(&token_text(*t));
    }
    let completion_tokens = tokens.len() as u32;
    let choices = vec![CompletionChoice {
        text,
        index: 0,
        finish_reason: "length".to_string(),
    }];
    CompletionResponse {
        id: completion_id(&body.prompt),
        object: "text_completion".to_string(),
        created: 1u64,
        model: body.model.clone(),
        choices,
        usage: CompletionUsage {
            prompt_tokens: 1,
            completion_tokens,
            total_tokens: 1 + completion_tokens,
        },
    }
}

async fn handle_completion(
    State(cfg): State<Arc<KVConfig>>,
    Json(body): Json<CompletionRequest>,
) -> Response {
    JsonResponse::<CompletionResponse>(build_completion(cfg, &body)).into_response()
}

/// Builds the axum router serving /v1/completions.
pub fn build_router(cfg: Arc<KVConfig>) -> Router {
    Router::new()
        .route("/v1/completions", post(handle_completion))
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
}
