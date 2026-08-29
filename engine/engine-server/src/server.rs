//! axum wiring: OpenAI-compatible chat & completion routes with SSE streaming.
//!
//! Exposes:
//! - `POST /v1/chat/completions` (OpenAI chat wire format, streaming SSE & JSON).
//! - `POST /v1/completions` (legacy text completion route, streaming SSE & JSON).
//! - `GET /v1/models` (list loaded models).
//!
//! Backed by:
//! - Synthetic decode loop (`KVConfig`) for fast unit testing.
//! - Real GPU runtime (`RealServerCfg`) running `ForwardDriver` with CUDA Graph decode and `Sampler`.

use crate::models::{
    ChatChoice, ChatChunkChoice, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessage, ChunkChoice, CompletionChoice, CompletionChunk,
    CompletionRequest, CompletionResponse, CompletionUsage, DeltaMessage, ModelCard,
    ModelListResponse, format_chatml,
};
use crate::runtime::RealModel;
use crate::scheduler::BatchScheduler;
use crate::sse::{data_event, done_event, to_sse};
use axum::Router;
use axum::extract::{Json, State};
use axum::response::sse::Event;
use axum::response::{IntoResponse, Json as JsonResponse, Response};
use axum::routing::{get, post};
use engine_core::{Sampler, SamplerParams};
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

/// Deterministic chat completion id.
fn chat_completion_id(prompt: &str) -> String {
    let mut hash: u64 = 1469598103934665603;
    for byte in prompt.as_bytes() {
        hash = (hash ^ (*byte as u64)).wrapping_mul(1099511628211);
    }
    format!("chatcmpl-{hash:x}")
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
        let step_tokens = scheduler.advance();
        tokens.extend_from_slice(&step_tokens);
    }
    tokens
}

/// Builds the SSE event sequence for a streaming completion (synthetic decode).
pub fn stream_events(cfg: Arc<KVConfig>, body: &CompletionRequest) -> Vec<Event> {
    let max_tokens = if body.max_tokens == 0 {
        cfg.default_max_tokens
    } else {
        body.max_tokens
    };
    let model = body.model.as_deref().unwrap_or("qwen3-0.6b");
    let tokens = decode_tokens(cfg.vocab, &body.prompt, max_tokens);
    let texts: Vec<String> = tokens.iter().map(|&t| token_text(t)).collect();
    tokens_and_texts_to_events(&body.prompt, model, &tokens, &texts)
}

/// Builds a non-streaming completion response via the scheduler decode loop.
pub fn build_completion(cfg: Arc<KVConfig>, body: &CompletionRequest) -> CompletionResponse {
    let max_tokens = if body.max_tokens == 0 {
        cfg.default_max_tokens
    } else {
        body.max_tokens
    };
    let model = body.model.as_deref().unwrap_or("qwen3-0.6b");
    let tokens = decode_tokens(cfg.vocab, &body.prompt, max_tokens);
    let texts: Vec<String> = tokens.iter().map(|&t| token_text(t)).collect();
    tokens_and_texts_to_completion(&body.prompt, model, &tokens, &texts)
}

async fn handle_models() -> Response {
    let response = ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelCard {
            id: "qwen3-0.6b".to_string(),
            object: "model".to_string(),
            created: 1720000000,
            owned_by: "titan".to_string(),
        }],
    };
    JsonResponse(response).into_response()
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

async fn handle_chat_completion(
    State(cfg): State<Arc<KVConfig>>,
    Json(body): Json<ChatCompletionRequest>,
) -> Response {
    let prompt = format_chatml(&body.messages);
    let max_tokens = if body.max_tokens == 0 {
        cfg.default_max_tokens
    } else {
        body.max_tokens
    };
    let model = body.model.as_deref().unwrap_or("qwen3-0.6b");
    let tokens = decode_tokens(cfg.vocab, &prompt, max_tokens);
    let texts: Vec<String> = tokens.iter().map(|&t| token_text(t)).collect();

    if body.stream {
        to_sse(chat_tokens_and_texts_to_events(&prompt, model, &tokens, &texts, "length")).into_response()
    } else {
        JsonResponse::<ChatCompletionResponse>(chat_tokens_and_texts_to_response(&prompt, model, &tokens, &texts, "length")).into_response()
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

/// Extracts `SamplerParams` from chat / completion request options.
fn extract_sampler_params(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repetition_penalty: Option<f32>,
) -> SamplerParams {
    let mut p = SamplerParams::default();
    if let Some(t) = temperature {
        p.temperature = t;
    }
    if let Some(tp) = top_p {
        p.top_p = tp;
    }
    if let Some(tk) = top_k {
        p.top_k = tk;
    }
    if let Some(rp) = repetition_penalty {
        p.repetition_penalty = rp;
    }
    p
}

/// Runs generation through the real model runtime with advanced sampling, grammar validation, and stop sequence checking.
fn decode_prompt_real(
    cfg: &RealServerCfg,
    prompt: &str,
    max_tokens: u32,
    params: SamplerParams,
    stop_strings: &[String],
    response_format: Option<&serde_json::Value>,
) -> (Vec<u32>, Vec<String>, String) {
    let mut model_guard = cfg.model.lock().expect("real model lock");
    let RealModel {
        driver,
        tokenizer,
        ngram_proposer,
        ..
    } = &mut *model_guard;

    let mut grammar = if let Some(rf) = response_format {
        let ty = rf.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if ty == "json_object" || ty == "json_schema" {
            Some(engine_core::grammar::JsonGrammar::new())
        } else {
            None
        }
    } else {
        None
    };

    if let (Some(driver), Some(tokenizer)) = (driver.as_mut(), tokenizer.as_ref()) {
        let prompt_tokens = match tokenizer.encode(prompt) {
            Ok(t) => t,
            Err(_) => return (Vec::new(), Vec::new(), "error".to_string()),
        };
        if prompt_tokens.is_empty() || max_tokens == 0 {
            return (Vec::new(), Vec::new(), "length".to_string());
        }

        let mut tokens = Vec::with_capacity(max_tokens as usize);
        let mut texts = Vec::with_capacity(max_tokens as usize);
        let mut context = prompt_tokens.clone();
        let mut sampler = Sampler::new(params.seed.unwrap_or(42));

        let initial_logits = match driver.prefill(&prompt_tokens) {
            Ok(l) => l,
            Err(_) => return (Vec::new(), Vec::new(), "error".to_string()),
        };

        let stop_tokens = [151645u32, 151643u32]; // <|im_end|>, <|endoftext|>

        let first_tok = sampler.sample(&initial_logits, &context, &params);
        let first_piece = tokenizer
            .decode(&[first_tok])
            .unwrap_or_else(|_| token_text(first_tok));

        if Sampler::is_stop_sequence(first_tok, &first_piece, &stop_tokens, stop_strings) {
            return (tokens, texts, "stop".to_string());
        }

        if let Some(ref mut g) = grammar {
            g.advance(&first_piece);
        }

        tokens.push(first_tok);
        texts.push(first_piece);
        context.push(first_tok);

        let mut current_tok = first_tok;
        let mut finish_reason = "length".to_string();

        while tokens.len() < max_tokens as usize {
            if let Some(ref g) = grammar {
                if g.is_complete() {
                    finish_reason = "stop".to_string();
                    break;
                }
            }

            if let Some(proposer) = ngram_proposer {
                let candidates = proposer.propose(&context);
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
                        let piece = tokenizer
                            .decode(&[emitted_tok])
                            .unwrap_or_else(|_| token_text(emitted_tok));
                        if Sampler::is_stop_sequence(emitted_tok, &piece, &stop_tokens, stop_strings) {
                            finish_reason = "stop".to_string();
                            stopped = true;
                            break;
                        }
                        if let Some(ref mut g) = grammar {
                            g.advance(&piece);
                        }
                        tokens.push(emitted_tok);
                        texts.push(piece);
                        context.push(emitted_tok);
                        current_tok = emitted_tok;
                        if tokens.len() >= max_tokens as usize {
                            break;
                        }
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
            let next_piece = tokenizer
                .decode(&[next_tok])
                .unwrap_or_else(|_| token_text(next_tok));

            if Sampler::is_stop_sequence(next_tok, &next_piece, &stop_tokens, stop_strings) {
                finish_reason = "stop".to_string();
                break;
            }

            if let Some(ref mut g) = grammar {
                g.advance(&next_piece);
            }

            tokens.push(next_tok);
            texts.push(next_piece);
            context.push(next_tok);
            current_tok = next_tok;
        }

        (tokens, texts, finish_reason)
    } else {
        let tokens = crate::runtime::decode_run(&mut model_guard, cfg.vocab, prompt, max_tokens)
            .expect("real decode run");
        let texts = tokens.iter().map(|&t| token_text(t)).collect();
        (tokens, texts, "length".to_string())
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

/// Builds streaming SSE event sequence for chat completions.
fn chat_tokens_and_texts_to_events(
    prompt: &str,
    model: &str,
    tokens: &[u32],
    texts: &[String],
    finish_reason: &str,
) -> Vec<Event> {
    let mut events = Vec::with_capacity(tokens.len() + 2);
    let id = chat_completion_id(prompt);

    // Initial chunk emitting role: assistant
    let initial_chunk = ChatCompletionChunk {
        id: id.clone(),
        object: "chat.completion.chunk".to_string(),
        created: 1u64,
        model: model.to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: DeltaMessage {
                role: Some("assistant".to_string()),
                content: Some(String::new()),
            },
            finish_reason: None,
        }],
    };
    events.push(data_event(serialize(&initial_chunk)));

    let last = tokens.len().saturating_sub(1);
    for (i, _) in tokens.iter().enumerate() {
        let piece = texts.get(i).cloned().unwrap_or_default();
        let chunk = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: 1u64,
            model: model.to_string(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: DeltaMessage {
                    role: None,
                    content: Some(piece),
                },
                finish_reason: if i == last {
                    Some(finish_reason.to_string())
                } else {
                    None
                },
            }],
        };
        events.push(data_event(serialize(&chunk)));
    }
    events.push(done_event());
    events
}

/// Builds non-streaming ChatCompletionResponse from tokens and text chunks.
fn chat_tokens_and_texts_to_response(
    prompt: &str,
    model: &str,
    tokens: &[u32],
    texts: &[String],
    finish_reason: &str,
) -> ChatCompletionResponse {
    let content = texts.concat();
    let completion_tokens = tokens.len() as u32;
    let choices = vec![ChatChoice {
        index: 0,
        message: ChatMessage::assistant(content),
        finish_reason: finish_reason.to_string(),
    }];
    ChatCompletionResponse {
        id: chat_completion_id(prompt),
        object: "chat.completion".to_string(),
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
    let model = body.model.as_deref().unwrap_or("qwen3-0.6b");
    let params = extract_sampler_params(
        body.temperature,
        body.top_p,
        body.top_k,
        body.repetition_penalty,
    );
    let stop_strings = body.stop.unwrap_or_default();

    let (engine_mode, vram_mb) = {
        let guard = cfg.model.lock().unwrap();
        let mode = guard.engine_mode.to_string();
        let vram = guard
            .driver
            .as_ref()
            .map(|d| d.vram_footprint().total() as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0);
        (mode, vram)
    };

    let (tokens, texts, _) = decode_prompt_real(&cfg, &body.prompt, max_tokens, params, &stop_strings, None);
    let mut resp = if body.stream {
        to_sse(tokens_and_texts_to_events(
            &body.prompt,
            model,
            &tokens,
            &texts,
        ))
        .into_response()
    } else {
        JsonResponse::<CompletionResponse>(tokens_and_texts_to_completion(
            &body.prompt,
            model,
            &tokens,
            &texts,
        ))
        .into_response()
    };

    resp.headers_mut().insert("x-titan-engine-mode", engine_mode.parse().unwrap());
    resp.headers_mut().insert("x-titan-vram-mb", format!("{vram_mb:.1}").parse().unwrap());
    resp
}

async fn handle_real_chat_completion(
    State(cfg): State<Arc<RealServerCfg>>,
    Json(body): Json<ChatCompletionRequest>,
) -> Response {
    let prompt = crate::models::format_chatml_with_tools(&body.messages, body.tools.as_deref());
    let max_tokens = if body.max_tokens == 0 {
        cfg.default_max_tokens
    } else {
        body.max_tokens
    };
    let model = body.model.as_deref().unwrap_or("qwen3-0.6b");
    let params = extract_sampler_params(
        body.temperature,
        body.top_p,
        body.top_k,
        body.repetition_penalty,
    );
    let stop_strings = body.stop.unwrap_or_default();

    let (engine_mode, vram_mb) = {
        let guard = cfg.model.lock().unwrap();
        let mode = guard.engine_mode.to_string();
        let vram = guard
            .driver
            .as_ref()
            .map(|d| d.vram_footprint().total() as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0);
        (mode, vram)
    };

    let (tokens, texts, finish_reason) = decode_prompt_real(
        &cfg,
        &prompt,
        max_tokens,
        params,
        &stop_strings,
        body.response_format.as_ref(),
    );
    let mut resp = if body.stream {
        to_sse(chat_tokens_and_texts_to_events(
            &prompt,
            model,
            &tokens,
            &texts,
            &finish_reason,
        ))
        .into_response()
    } else {
        JsonResponse::<ChatCompletionResponse>(chat_tokens_and_texts_to_response(
            &prompt,
            model,
            &tokens,
            &texts,
            &finish_reason,
        ))
        .into_response()
    };

    resp.headers_mut().insert("x-titan-engine-mode", engine_mode.parse().unwrap());
    resp.headers_mut().insert("x-titan-vram-mb", format!("{vram_mb:.1}").parse().unwrap());
    resp
}

/// Builds the axum router serving `/v1/chat/completions`, `/v1/completions`, and `/v1/models`.
pub fn build_router(cfg: Arc<KVConfig>) -> Router {
    Router::new()
        .route("/v1/models", get(handle_models))
        .route("/v1/chat/completions", post(handle_chat_completion))
        .route("/v1/completions", post(handle_completion))
        .with_state(cfg)
}

/// Builds the axum router serving `/v1/chat/completions`, `/v1/completions`, and `/v1/models`
/// backed by a real model runtime.
pub fn build_router_real(cfg: Arc<RealServerCfg>) -> Router {
    Router::new()
        .route("/v1/models", get(handle_models))
        .route("/v1/chat/completions", post(handle_real_chat_completion))
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
    async fn models_list_returns_json() {
        let (client, base) = spawn().await;
        let res = client
            .get(&(base + "/v1/models"))
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);
        let body = res.text().await.unwrap();
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"][0]["id"], "qwen3-0.6b");
    }

    #[tokio::test]
    async fn chat_completions_returns_openai_json() {
        let (client, base) = spawn().await;
        let res = client
            .post(&(base + "/v1/chat/completions"))
            .header("content-type", "application/json")
            .body(r#"{"model":"qwen3-0.6b","messages":[{"role":"user","content":"hello"}],"max_tokens":4,"stream":false}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);

        let body = res.text().await.unwrap();
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(v["id"].as_str().unwrap().starts_with("chatcmpl-"));
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        let text = v["choices"][0]["message"]["content"].as_str().unwrap();
        assert_ne!(text.len(), 0);
    }

    #[tokio::test]
    async fn chat_completions_streaming_sse() {
        let (client, base) = spawn().await;
        let mut res = client
            .post(&(base + "/v1/chat/completions"))
            .header("content-type", "application/json")
            .body(r#"{"model":"qwen3-0.6b","messages":[{"role":"user","content":"stream me"}],"max_tokens":4,"stream":true}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(res.status(), 200);

        let mut saw_data = false;
        let mut saw_done = false;
        while let Some(chunk) = res.chunk().await.unwrap() {
            let text = String::from_utf8(chunk.to_vec()).unwrap();
            assert!(text.starts_with("data: "));
            if text.contains("[DONE]") {
                saw_done = true;
            } else {
                let json = text.trim_start_matches("data: ").trim_end();
                let _: serde_json::Value = serde_json::from_str(json).expect("valid JSON chunk");
                saw_data = true;
            }
        }
        assert!(saw_data);
        assert!(saw_done);
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
