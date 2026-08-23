/// engine-server: OpenAI-compatible HTTP inference surface (Phase 5).
///
/// Milestone scaffold: the `POST /v1/completions` endpoint with the
/// OpenAI-compatible JSON shape. The decode machinery (session + batch
/// scheduler) and SSE streaming are wired in later milestones of this change.
pub mod models;
pub mod server;

pub use models::{
    ChunkChoice, CompletionChoice, CompletionChunk, CompletionRequest, CompletionResponse,
    CompletionUsage,
};
pub use server::KVConfig;

pub fn version() -> &'static str {
    "0.1.0"
}
