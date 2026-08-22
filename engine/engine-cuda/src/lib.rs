pub mod error;
pub mod pinned_host;

pub use error::CudaError;
pub use pinned_host::PinnedHost;

pub fn version() -> &'static str {
    "0.1.0"
}
