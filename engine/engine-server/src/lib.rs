/// engine-server: OpenAI-compatible HTTP inference surface (Phase 5).
///
/// Exposes the deterministic decode machinery as a local HTTP server:
/// - POST /v1/completions (OpenAI-compatible JSON, and SSE when stream=true)
///
/// The forward pass is a deterministic placeholder (see `session`).
pub mod models;
pub mod runtime;
pub mod scheduler;
pub mod server;
pub mod session;
pub mod sse;

pub use models::{
    ChunkChoice, CompletionChoice, CompletionChunk, CompletionRequest, CompletionResponse,
    CompletionUsage,
};
pub use scheduler::BatchScheduler;
pub use server::{KVConfig, RealServerCfg, build_router, build_router_real};
pub use session::GenerationSession;
pub use sse::{chunk_frame, done_frame};

pub fn version() -> &'static str {
    "0.1.0"
}
