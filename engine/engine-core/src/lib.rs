pub mod dequant;
pub mod error;
pub mod pipeline;

pub use error::EngineError;
pub use pipeline::{Pipeline, PipelineStats};

pub fn version() -> &'static str {
    "0.1.0"
}
