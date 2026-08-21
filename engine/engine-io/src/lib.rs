//! # engine-io
//!
//! I/O library for reading GGUF models and binary weights.

pub mod error;
pub mod layer;
pub mod reader;
pub mod types;

pub use error::GgufError;
pub use layer::{classify_layer, LayerIndex};
pub use reader::GgufReader;
pub use types::{GgmlType, GgufHeader, GgufType, GgufValue, TensorInfo};

/// Crate version.
pub fn version() -> &'static str {
    "0.1.0"
}
