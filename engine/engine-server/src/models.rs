/// OpenAI-compatible request/response models for the engine-server.
///
/// The wire contract intentionally mirrors the OpenAI `/v1/completions`
/// endpoint so existing clients can point at the local engine unchanged.
use serde::{Deserialize, Serialize};

/// Body of a `POST /v1/completions` request.
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    /// Model identifier echoed back on responses.
    ///
    /// The engine ignores the model for now; it is accepted so OpenAI-shaped
    /// clients keep working. Placeholder model: no trained weights yet.
    pub model: String,
    /// Prompt text. The engine maps it to a start token via a stub provider.
    pub prompt: String,
    /// Completion token budget.
    pub max_tokens: u32,
    /// When true, respond with a `text/event-stream` (SSE) sequence.
    pub stream: bool,
}

/// One non-streaming completion choice.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    /// Completed token text for this choice.
    pub text: String,
    /// Zero-based index of the choice.
    pub index: u32,
    /// Terminal reason: `stop`, `length`, or `error`.
    pub finish_reason: String,
}

/// Token accounting for a completion response.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Non-streaming `POST /v1/completions` response.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: CompletionUsage,
}

/// One streaming chunk choice.
#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<String>,
}

/// One SSE payload for a generated token.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}
