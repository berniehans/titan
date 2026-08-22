//! # engine-io
//!
//! I/O library for reading GGUF models and binary weights.

pub mod error;
pub mod layer;
pub mod loader;
pub mod reader;
pub mod types;

pub use error::GgufError;
pub use layer::{LayerIndex, classify_layer};
pub use loader::{LoadedLayout, LoadedPinned, load_to_pinned};
pub use reader::GgufReader;
pub use types::{GgmlType, GgufHeader, GgufType, GgufValue, TensorInfo};

/// Crate version.
pub fn version() -> &'static str {
    "0.1.0"
}
