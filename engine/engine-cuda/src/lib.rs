pub mod error;
pub mod pinned_host;
pub mod streams;

pub use error::CudaError;
pub use pinned_host::PinnedHost;
pub use streams::CudaStream;

pub fn version() -> &'static str {
    "0.1.0"
}
