pub mod dequant;
pub mod error;
pub mod forward_cpu;
pub mod forward_driver;
pub mod pipeline;
pub mod tokenizer;
pub mod vram_accounting;

pub use error::EngineError;
pub use forward_driver::{ForwardDriver, VramFootprint};
pub use pipeline::{Pipeline, PipelineStats};
pub use tokenizer::BpeTokenizer;
pub use vram_accounting::{VramStageBreakdown, compute_static_vram_map};

pub fn version() -> &'static str {
    "0.1.0"
}
