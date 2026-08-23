/// engine-server: OpenAI-compatible HTTP inference surface (Phase 5).
///
/// Milestone 2: the /v1/completions endpoint is wired to the deterministic
/// decode loop (`session` + `scheduler`). SSE streaming lands in the next
/// milestone.
pub mod models;
pub mod scheduler;
pub mod server;
pub mod session;

pub use models::{
    ChunkChoice, CompletionChoice, CompletionChunk, CompletionRequest, CompletionResponse,
    CompletionUsage,
};
pub use scheduler::BatchScheduler;
pub use server::KVConfig;
pub use session::GenerationSession;

pub fn version() -> &'static str {
    "0.1.0"
}
